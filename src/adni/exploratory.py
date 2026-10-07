"""Step 11b exploratory arms A5-A8 (EXPLORATORY; pre-registered in configs/step11b.yaml).

All four arms build on the A4 loss (tau = 1) and the R2 training recipe:

  A5  A4 + pattern conditioning: FiLM (gamma, beta) from a 16-d embedding of the EFFECTIVE mask
      pattern applied to the pooled CLS representation before the head.
  A6  A5 + cross-view consistency: two independent modality-dropout views per batch;
      L = A4_CE(view 1) + lambda * [ -sum_c p2(c) log p1(c) ] with lambda = 0.1 and a
      stop-gradient on view 2 (p2 carries no gradient).
  A7  A6 + pattern-aware MoE: K = 4 residual-MLP experts over the pooled representation; the
      router sees [pooled h, pattern embedding, mask bits] only - NO label-dependent input.
  A8  A6 + class-dependent prototypes: mu_(c, m) = mu_c + B(h(mu_c) * e_m) with
      h(x) = W_h x (a learned 16-d map), W_e: 16 -> d, B: d -> d, and a prototype loss term of
      weight 0.1: L_prot = CE( -||h_cls - mu_(c, m)||^2 , y ).

The v1 "gated additive" prototype form mu_(c,m) = mu_c + B(h(mu_c) * e_m) differs from the older
additive form mu_c + B(e_m): the learned h(mu_c) couples the class identity with the pattern
embedding, so prototypes separate ACROSS CLASSES for a fixed pattern (unit-tested).

Every arm shares the A0 backbone (encoders, type embeddings, [CLS], transformer); only the head /
training loss changes. Absent modalities never enter attention (key-padding mask), so a pattern
embedding built ONLY from mask bits is invariant to the CONTENT of absent modalities by
construction; the unit tests verify the end-to-end property.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import Priors, adjusted_logits, mask_to_pattern
from .metrics import pn_balanced_accuracy
from .models import MODALITIES, ModelConfig
from .train import set_seed

N_PATTERNS = 2 ** len(MODALITIES)          # 16 possible availability patterns
PATTERN_TO_INDEX = {}                      # bit-string -> row of the embedding table
for _i in range(N_PATTERNS):
    PATTERN_TO_INDEX[format(_i, f"0{len(MODALITIES)}b")] = _i
del _i

EXPLORATORY_ARMS: tuple[str, ...] = ("A5", "A6", "A7", "A8")


def pattern_index(patterns) -> torch.Tensor:
    """Map pattern bit-strings (bit order = MODALITIES) to embedding-table rows."""
    idx = [PATTERN_TO_INDEX[str(p)] for p in patterns]
    return torch.as_tensor(idx, dtype=torch.long)


@dataclass
class ExploratoryConfig:
    """Hyperparameters of the exploratory heads (pre-registered values; do not tune)."""

    d_pattern: int = 16
    n_experts: int = 4
    lambda_consistency: float = 0.1
    lambda_prototype: float = 0.1
    film: bool = True
    moe: bool = False
    prototypes: bool = False
    # per-arm presets resolved from configs/step11b.yaml at run time
    @classmethod
    def for_arm(cls, arm: str) -> "ExploratoryConfig":
        if arm == "A5":
            return cls(film=True)
        if arm == "A6":
            return cls(film=True, lambda_consistency=0.1)
        if arm == "A7":
            return cls(film=True, lambda_consistency=0.1, moe=True)
        if arm == "A8":
            return cls(film=True, lambda_consistency=0.1, prototypes=True)
        raise ValueError(f"unknown exploratory arm {arm!r}; expected one of {EXPLORATORY_ARMS}")


class PatternEmbedding(nn.Module):
    """Learned embedding of the 16 availability patterns (no feature content, mask bits only)."""

    def __init__(self, d_pattern: int = 16) -> None:
        super().__init__()
        self.table = nn.Embedding(N_PATTERNS, d_pattern)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, patterns: list[str]) -> torch.Tensor:
        idx = pattern_index(patterns).to(self.table.weight.device)
        return self.table(idx)


class FiLM(nn.Module):
    """Feature-wise linear modulation: x -> (1 + gamma) * x + beta from the pattern embedding."""

    def __init__(self, d_model: int, d_pattern: int = 16) -> None:
        super().__init__()
        self.to_gamma = nn.Linear(d_pattern, d_model)
        self.to_beta = nn.Linear(d_pattern, d_model)

    def forward(self, x: torch.Tensor, e_m: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(e_m)
        beta = self.to_beta(e_m)
        return (1.0 + gamma) * x + beta


class ResidualExpert(nn.Module):
    """A residual MLP expert: x + W2(gelu(W1 x))."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class PatternMoE(nn.Module):
    """Pattern-aware mixture of experts; router input = [h_pooled, e_m, mask bits].

    The router NEVER sees labels or the predictions - pre-registered and unit-tested.
    """

    def __init__(self, d_model: int, d_pattern: int = 16, n_experts: int = 4,
                 d_hidden: int = 256) -> None:
        super().__init__()
        self.experts = nn.ModuleList([ResidualExpert(d_model, d_hidden) for _ in range(n_experts)])
        self.router = nn.Linear(d_model + d_pattern + len(MODALITIES), n_experts)

    def forward(self, h: torch.Tensor, e_m: torch.Tensor, mask: torch.Tensor):
        router_in = torch.cat([h, e_m, mask.to(dtype=h.dtype)], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)             # [B, K]
        stack = torch.stack([expert(h) for expert in self.experts], dim=1)  # [B, K, d]
        return (gates.unsqueeze(-1) * stack).sum(dim=1), gates


class PrototypeHead(nn.Module):
    """Class-dependent prototypes mu_(c, m) = mu_c + B(h(mu_c) * e_m).

    ``h`` is a learned linear map d_model -> d_pattern, ``e_m`` the pattern embedding,
    ``B`` a learned map d_pattern -> d_model. Because h(mu_c) depends on the CLASS,
    prototypes separate across classes even for one fixed pattern (unlike the additive
    v1 form mu_c + B(e_m)).
    """

    def __init__(self, d_model: int, n_classes: int, d_pattern: int = 16) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(n_classes, d_model))
        nn.init.normal_(self.mu, std=0.02)
        self.h = nn.Linear(d_model, d_pattern)
        self.B = nn.Linear(d_pattern, d_model)

    def forward(self, e_m: torch.Tensor) -> torch.Tensor:
        """Return per-row prototypes ``[B, n_classes, d_model]``.

        Each episode has its own pattern embedding, so row ``b`` gets
        ``mu_(c, m_b) = mu_c + B(h(mu_c) * e_m_b)`` for every class ``c``.
        """
        h_mu = self.h(self.mu)                                  # [C, d_pattern]
        prod = h_mu.unsqueeze(0) * e_m.unsqueeze(1)             # [B, C, d_pattern]
        return self.mu.unsqueeze(0) + self.B(prod)              # [B, C, d_model]


def _embed_and_condition(backbone_out: torch.Tensor, eff_mask: torch.Tensor,
                         pattern_emb: PatternEmbedding, film: FiLM | None) -> tuple[torch.Tensor, torch.Tensor]:
    e_m = pattern_emb(mask_to_pattern(eff_mask))
    if film is not None:
        backbone_out = film(backbone_out, e_m)
    return backbone_out, e_m

class ExploratoryModel(nn.Module):
    """A0 backbone + the A5-A8 heads. Returns the raw (un-adjusted) class logits.

    ``forward`` supports two modality-dropout views of the same batch (A6-A8): view 2 is a fresh
    dropout draw whose outputs are detached (stop-gradient), used only by the consistency loss.
    """

    def __init__(self, cfg: ModelConfig, xcfg: ExploratoryConfig) -> None:
        super().__init__()
        from .models import FusionModel

        self.backbone = FusionModel(cfg)
        self.xcfg = xcfg
        d = cfg.d_model
        self.pattern_emb = PatternEmbedding(xcfg.d_pattern)
        self.film = FiLM(d, xcfg.d_pattern) if xcfg.film else None
        self.moe = PatternMoE(d, xcfg.d_pattern, xcfg.n_experts) if xcfg.moe else None
        self.prototype_head = PrototypeHead(d, cfg.n_classes, xcfg.d_pattern) if xcfg.prototypes else None
        self.head = self.backbone.head
        if self.moe is not None:
            self.head = nn.Linear(d, cfg.n_classes)

    # ------------------------------------------------------------------ helpers
    def _encode(self, feats, mask, apply_dropout: bool, generator=None):
        logits_bb, h, eff = self.backbone(feats, mask, apply_modality_dropout=apply_dropout,
                                          generator=generator)
        h_c, e_m = _embed_and_condition(h, eff, self.pattern_emb, self.film)
        if self.moe is not None:
            h_c, _gates = self.moe(h_c, e_m, eff)
        logits = self.head(h_c)
        return logits, h_c, eff, e_m

    # ------------------------------------------------------------------ forward
    def forward(self, feats, mask, apply_modality_dropout: bool = False,
                generator: torch.Generator | None = None, two_views: bool = False):
        """Return view-1 outputs; with ``two_views`` also the detached view-2 probabilities.

        Returns ``(logits1, h1, eff1, e_m1[, p2_detached])``.
        """
        logits, h, eff, e_m = self._encode(feats, mask, apply_modality_dropout, generator)
        if not two_views:
            return logits, h, eff, e_m
        # view 2: a fresh dropout draw; stop-gradient via torch.no_grad() (view 2 carries no grad)
        with torch.no_grad():
            logits2, _h2, _eff2, _e2 = self._encode(feats, mask, True, generator)
            p2 = torch.softmax(logits2, dim=1)
        return logits, h, eff, e_m, p2

    def prototype_logits(self, h: torch.Tensor, e_m: torch.Tensor) -> torch.Tensor:
        """Class scores from the negative squared Euclidean distance to each prototype."""
        if self.prototype_head is None:
            raise ValueError("this arm has no prototype head")
        protos = self.prototype_head(e_m)                          # [B, C, d]
        d2 = (h.unsqueeze(1) - protos).pow(2).sum(dim=-1)          # [B, C]
        return -d2


# --------------------------------------------------------------------- losses
def exploratory_loss(
    arm: str,
    model: ExploratoryModel,
    feats,
    mask,
    y: torch.Tensor,
    priors: Priors,
    tau: float = 1.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    """Training loss for one batch under an exploratory arm.

    A5  L = CE(s_v1 + tau*log pi(c|m_v1), y)                       (A4 term on view 1)
    A6  L = A5 + lambda_c * [ -sum_c p2(c) log p1(c) ]             (view 2 detached)
    A7  L = A6                                                     (MoE lives in the model)
    A8  L = A6 + 0.1 * CE( -||h_v1 - mu_(c, m_v1)||^2 , y )

    Returns ``(loss, info)`` with the batch-mean components for logging.
    """
    two_views = arm in ("A6", "A7", "A8")
    out = model(feats, mask, apply_modality_dropout=True, generator=generator, two_views=two_views)
    logits, h, eff, e_m = out[:4]
    p2 = out[4] if two_views else None
    patterns = mask_to_pattern(eff)
    ce = F.cross_entropy(adjusted_logits(logits, patterns, priors, tau), y)
    info = {"ce": float(ce.detach().item())}
    loss = ce
    if two_views:
        log_p1 = torch.log_softmax(logits, dim=1)
        cons = -(p2 * log_p1).sum(dim=1).mean()
        loss = loss + model.xcfg.lambda_consistency * cons
        info["consistency"] = float(cons.detach().item())
    if arm == "A8":
        proto_logits = model.prototype_logits(h, e_m)
        proto = F.cross_entropy(proto_logits, y)
        loss = loss + model.xcfg.lambda_prototype * proto
        info["prototype"] = float(proto.detach().item())
    info["total"] = float(loss.detach().item())
    return loss, info

# ------------------------------------------------------------------- trainer
class ExploratoryTrainer:
    """Trainer for A5-A8: identical protocol to ``adni.train.Trainer`` (both checkpoint rules,
    early stopping on validation macro-F1), but the batch loss comes from
    :func:`exploratory_loss` and view 2 is drawn inside it."""

    def __init__(self, model: ExploratoryModel, cfg, priors: Priors, val_weights=None) -> None:
        import adni.train as tr

        self._tr = tr
        self.model = model
        self.cfg = cfg
        self.priors = priors
        self.val_weights = None if val_weights is None else np.asarray(val_weights, dtype=float)
        self.device = cfg.device
        self.model.to(self.device)

    @torch.no_grad()
    def _eval_logits(self, data):
        self.model.eval()
        logits, _, _, _ = self.model(data.feats, data.mask, apply_modality_dropout=False)
        return logits

    @torch.no_grad()
    def _val_metrics(self, data) -> dict:
        from sklearn.metrics import f1_score, roc_auc_score

        logits = self._eval_logits(data)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y = data.y.cpu().numpy()
        present = [c for c in range(proba.shape[1]) if (y == c).any()]
        auroc = float("nan") if len(present) < 2 else float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=present))
        out = {"loss": float(torch.nn.functional.cross_entropy(logits, data.y).item()),
               "macro_f1": float(f1_score(y, proba.argmax(axis=1), average="macro",
                                          labels=[0, 1, 2], zero_division=0)),
               "macro_auroc": auroc, "pn_bacc": float("nan")}
        if self.val_weights is not None:
            out["pn_bacc"] = float(pn_balanced_accuracy(y, proba.argmax(axis=1), self.val_weights))
        return out

    def fit(self, train, val) -> dict:
        import copy

        cfg = self.cfg
        set_seed(cfg.seed, cfg.deterministic)
        train = train.to(self.device)
        val = val.to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = len(train)
        history: list[dict] = []
        best = {"macro_f1": -1.0, "epoch": -1}
        best_state = copy.deepcopy(self.model.state_dict())
        best_pn = {"pn_bacc": -1.0, "epoch": -1}
        best_pn_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0
        last_info: dict = {}

        for epoch in range(cfg.max_epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            comps = {"ce": 0.0, "consistency": 0.0, "prototype": 0.0, "total": 0.0}
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                batch = train.index(idx)
                opt.zero_grad(set_to_none=True)
                loss, info = exploratory_loss(arm=self.cfg.arm, model=self.model,
                                              feats=batch.feats, mask=batch.mask, y=batch.y,
                                              priors=self.priors, tau=self.cfg.tau)
                loss.backward()
                opt.step()
                for k in comps:
                    comps[k] += info.get(k, 0.0) * len(idx)
                last_info = info
            vm = self._val_metrics(val)
            val_pn = float(vm["pn_bacc"])
            history.append({"epoch": epoch, "train_loss": comps["total"] / n,
                            "train_ce": comps["ce"] / n,
                            "train_consistency": comps["consistency"] / n,
                            "train_prototype": comps["prototype"] / n,
                            "val_loss": vm["loss"], "val_macro_f1": vm["macro_f1"],
                            "val_macro_auroc": vm["macro_auroc"],
                            "val_pn_bacc": val_pn if np.isfinite(val_pn) else None})
            improved = vm["macro_f1"] > best["macro_f1"] + 1e-12
            if improved:
                best = {"macro_f1": vm["macro_f1"], "epoch": epoch}
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if np.isfinite(vm["pn_bacc"]) and vm["pn_bacc"] > best_pn["pn_bacc"] + 1e-12:
                best_pn = {"pn_bacc": vm["pn_bacc"], "epoch": epoch}
                best_pn_state = copy.deepcopy(self.model.state_dict())
            if not improved and epochs_no_improve >= cfg.patience:
                break

        self.model.load_state_dict(best_state)
        return {"history": history, "best": best, "best_pn_bacc": best_pn,
                "epochs_run": len(history), "best_state": best_state,
                "best_pn_bacc_state": best_pn_state, "last_batch_info": last_info}


def train_exploratory_arm(arm: str, feat_dims, train, val, cfg, model_cfg_kwargs=None,
                          priors: Priors | None = None, val_weights=None,
                          xcfg: ExploratoryConfig | None = None):
    """Fit an exploratory arm on one fold. Returns (model, result_dict, wall_clock_seconds).

    Seeds *before* constructing the model (same rule as ``adni.train.train_arm``), so every
    (arm, seed, fold) is deterministic.
    """
    from dataclasses import replace as dc_replace
    import time

    from .models import ModelConfig

    if priors is None:
        raise ValueError(f"exploratory arm {arm} needs training-split priors (A4 loss)")
    cfg = dc_replace(cfg, arm=arm)
    xcfg = xcfg if xcfg is not None else ExploratoryConfig.for_arm(arm)
    set_seed(cfg.seed, cfg.deterministic)
    model = ExploratoryModel(ModelConfig(feat_dims=feat_dims, **(model_cfg_kwargs or {})), xcfg)
    trainer = ExploratoryTrainer(model, cfg, priors, val_weights)
    t0 = time.time()
    result = trainer.fit(train, val)
    result["seconds"] = time.time() - t0
    result["n_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["arm"] = arm
    return model, result, result["seconds"]


# ------------------------------------------------------------------ diagnostics
@torch.no_grad()
def embedding_swap_probs(model: ExploratoryModel, data, donor_rows: np.ndarray, device: str):
    """Step 11b embedding-swap test: replace every test episode's pattern embedding by the
    embedding of a random OTHER test episode (class-agnostic), keeping everything else fixed.

    FiLM and the MoE router are recomputed with the SWAPPED embedding; the backbone features and
    the mask (hence the key-padding mask in attention) are unchanged. Returns ``(p_orig, p_swap,
    patterns_self, patterns_donor)`` - softmax probabilities at the raw-logit (rho uniform)
    operating point and the effective patterns of the episodes and their donors.
    """
    model.eval()
    logits_o, _, eff_o, _ = model(data.feats, data.mask, apply_modality_dropout=False)
    p_o = torch.softmax(logits_o, dim=1)
    h_raw = model.backbone(data.feats, data.mask, apply_modality_dropout=False)[1]
    don = torch.as_tensor(np.asarray(donor_rows), dtype=torch.long, device=data.mask.device)
    e_swapped = model.pattern_emb(mask_to_pattern(eff_o[don]))
    h = model.film(h_raw, e_swapped) if model.film is not None else h_raw
    if model.moe is not None:
        h, _g = model.moe(h, e_swapped, eff_o)
    logits_s = model.head(h)
    p_s = torch.softmax(logits_s, dim=1)
    return (p_o.cpu().numpy(), p_s.cpu().numpy(),
            mask_to_pattern(eff_o), mask_to_pattern(eff_o[don]))
