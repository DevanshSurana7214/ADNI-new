"""Masked-attention multimodal fusion model (PIPELINE_v2 §3.3, §5).

Per modality ``k`` a small tabular encoder produces a 128-d token ``h_k``. Tokens are
``t_k = h_k + type_k`` with a learnable ``[CLS]`` prepended, and **absent modalities are omitted**
from attention through a key-padding mask (they are never fed as zeros or a substitute token). Two
pre-LN Transformer layers (d=128, 4 heads, FFN 256, dropout 0.1) produce ``h = [CLS]``, and a linear
head maps it to 3 logits.

Modality dropout (training only) drops each **observed** modality independently with probability
``p``, resampling whenever a row would become empty; the effective mask is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

MODALITIES: tuple[str, ...] = ("mri", "fdg", "amy", "tau")


@dataclass
class ModelConfig:
    """Architecture hyperparameters for A0 (§3.3 / §5)."""

    feat_dims: dict[str, int] = field(default_factory=dict)  # kept feature count d_k per modality
    d_model: int = 128
    enc_hidden: int = 256
    enc_dropout: float = 0.2
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    attn_dropout: float = 0.1
    n_classes: int = 3
    modality_dropout: float = 0.25

    def d_in(self, k: str) -> int:
        """Encoder input width for block ``k`` = 2·d_k (z-scored values + missing indicators)."""
        return 2 * int(self.feat_dims[k])


class Encoder(nn.Module):
    """``LN -> Linear(2·d_k, 256) -> GELU -> Dropout(0.2) -> Linear(256, 128) -> LN``."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float) -> None:
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(dropout)
        self.ln_out = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return self.ln_out(x)


class FusionModel(nn.Module):
    """Shared-encoder masked-attention fusion over the four modality blocks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({k: Encoder(cfg.d_in(k), cfg.enc_hidden, cfg.d_model,
                                                  cfg.enc_dropout) for k in MODALITIES})
        self.type_embeddings = nn.Parameter(torch.zeros(len(MODALITIES), cfg.d_model))
        nn.init.normal_(self.type_embeddings, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim,
            dropout=cfg.attn_dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers,
                                                  enable_nested_tensor=False)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        feats: dict[str, torch.Tensor],
        mask: torch.Tensor,
        apply_modality_dropout: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, h, effective_mask)``.

        ``feats[k]`` is ``[B, 2·d_k]``, ``mask`` is a bool tensor ``[B, len(MODALITIES)]``. Absent
        modalities contribute a key-padding-masked token, so their encoder output never reaches the
        attention output.
        """
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        mask = mask.to(dtype=torch.bool)
        eff = mask
        if apply_modality_dropout:
            eff = self.modality_dropout(mask, generator=generator)

        b = mask.shape[0]
        tokens = [self.cls_token.expand(b, -1, -1)]
        for i, k in enumerate(MODALITIES):
            h = self.encoders[k](feats[k])                       # [B, d]
            tokens.append((h + self.type_embeddings[i]).unsqueeze(1))
        tokens = torch.cat(tokens, dim=1)                        # [B, 1 + n_mod, d]

        # key-padding mask: True = ignore. [CLS] is never masked.
        key_pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=mask.device),
                             ~eff.to(mask.device)], dim=1)
        out = self.transformer(tokens, src_key_padding_mask=key_pad)
        h = out[:, 0]                                            # [CLS]
        logits = self.head(h)
        return logits, h, eff

    # ------------------------------------------------------------------ modality dropout
    def modality_dropout(
        self,
        mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Drop each observed modality independently with p, resampling rows that would be empty.

        Returns a bool tensor with at least one ``True`` per row (rows that were empty stay empty —
        the natural mask never has an all-zero row in this cohort).
        """
        p = self.cfg.modality_dropout
        mask = mask.to(dtype=torch.bool)
        keep_prob = torch.rand(mask.shape, generator=generator, device=mask.device) >= p
        eff = mask & keep_prob
        # only rows that had at least one observed modality are resampled
        empty = (~eff.any(dim=1)) & mask.any(dim=1)
        for _ in range(1000):
            if not bool(empty.any()):
                break
            idx = empty.nonzero(as_tuple=False).squeeze(1)
            cand = mask[idx]
            k2 = torch.rand(cand.shape, generator=generator, device=mask.device) >= p
            eff[idx] = cand & k2
            empty = (~eff.any(dim=1)) & mask.any(dim=1)
        return eff

    # ------------------------------------------------------------------ reporting
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

