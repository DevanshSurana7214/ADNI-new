"""Neural model for multimodal AD diagnosis (PIPELINE_v2 §5).

Encoder f_k: LayerNorm -> Linear(2d_k, 256) -> GELU -> Dropout(0.2) -> Linear(256, 128) -> LayerNorm
Tokens: t_k = h_k + type_embedding_k; learnable [CLS]
Absent modalities: OMITTED from attention via key padding mask
Transformer: 2 pre-LN layers, d=128, 4 heads, FFN 256, dropout 0.1
Head: Linear(128, 3) on [CLS]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Encode one modality's features to a 128-d token."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalTransformer(nn.Module):
    """Multimodal transformer for AD diagnosis.
    
    Args:
        modalities: dict mapping modality name to input dimension
        dropout_p: modality dropout probability (training only)
    """
    
    MODALITY_ORDER = ["mri", "fdg", "amy", "tau"]
    
    def __init__(self, modalities: dict[str, int], dropout_p: float = 0.25):
        super().__init__()
        self.dropout_p = dropout_p
        self.modalities = modalities
        
        # Per-modality encoders
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim) for name, dim in modalities.items()
        })
        
        # Type embeddings for each modality + CLS
        self.type_emb = nn.Embedding(len(modalities) + 1, 128)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            norm_first=True,  # pre-LN
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head
        self.head = nn.Linear(128, 3)
    
    def forward(
        self,
        features: dict[str, torch.Tensor],
        mask: dict[str, torch.Tensor] | None = None,
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            features: dict of modality -> [B, d_k] tensor
            mask: dict of modality -> [B] boolean tensor (True=present), or None
            training: whether in training mode (for modality dropout)
        
        Returns:
            logits: [B, 3]
            cls_token: [B, 128] (the [CLS] output)
            effective_mask: [B, 4] boolean (which modalities were actually used)
        """
        B = list(features.values())[0].shape[0]
        device = list(features.values())[0].device
        
        # Modality dropout (training only)
        if training and self.dropout_p > 0:
            effective_mask = self._apply_modality_dropout(features, device)
        else:
            effective_mask = torch.stack([
                mask[m].to(device) if mask is not None else 
                torch.ones(B, dtype=torch.bool, device=device)
                for m in self.MODALITY_ORDER
            ], dim=1)
        
        # Build tokens for present modalities
        tokens_list = []
        token_mask = []  # True = valid token
        
        # CLS token
        cls_token = self.type_emb.weight[-1].unsqueeze(0).expand(B, -1)
        tokens_list.append(cls_token)
        token_mask.append(torch.ones(B, dtype=torch.bool, device=device))
        
        for i, mod_name in enumerate(self.MODALITY_ORDER):
            if mod_name not in features:
                continue
            feat = features[mod_name]
            present = effective_mask[:, i]
            
            if present.any():
                # Encode features
                h = self.encoders[mod_name](feat)
                # Add type embedding
                t = self.type_emb.weight[i]
                token = h + t
                
                tokens_list.append(token)
                token_mask.append(present)
        
        # Concatenate all tokens
        tokens = torch.cat(tokens_list, dim=0)  # [total_valid, 128]
        token_mask = torch.cat(token_mask, dim=0)  # [total_valid]
        
        # Key padding mask: True = pad (masked), False = valid
        key_padding_mask = ~token_mask
        
        # Transformer
        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        
        # CLS is always first
        cls_output = encoded[:B]  # [B, 128]
        
        logits = self.head(cls_output)
        
        return logits, cls_output, effective_mask
    
    def _apply_modality_dropout(
        self,
        features: dict[str, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Apply modality dropout, ensuring at least one modality remains."""
        B = list(features.values())[0].shape[0]
        n_mod = len(self.MODALITY_ORDER)
        
        # Random dropout mask
        drop_mask = torch.rand(B, n_mod, device=device) < self.dropout_p
        
        # Ensure at least one modality per sample (loop until no empty)
        any_present = drop_mask.any(dim=1)
        while not any_present.all():
            empty_mask = ~any_present
            n_empty = empty_mask.sum().item()
            # Resample dropout for empty samples
            retry_mask = torch.rand(n_empty, n_mod, device=device) < self.dropout_p
            # If any retry still has all dropped, force first modality to stay
            still_empty = retry_mask.all(dim=1)
            if still_empty.any():
                retry_mask[still_empty, 0] = True
            drop_mask[empty_mask] = retry_mask
            any_present = drop_mask.any(dim=1)
        
        # Effective mask: True = present (not dropped)
        effective_mask = ~drop_mask
        
        return effective_mask


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
