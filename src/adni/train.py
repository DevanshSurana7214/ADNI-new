"""Deterministic trainer for the A0 fusion model (PIPELINE_v2 §6).

AdamW (lr 3e-4, weight decay 0.05), batch 256, up to 100 epochs, early stopping with patience 15 on
the **validation macro-F1** (the same rule for every arm). Seeds are fixed for python/numpy/torch and
deterministic algorithms are requested where supported. The loss in this step is plain cross-entropy
(arm A0).

The transformed tensors for a fold are kept on the target device to avoid per-batch host/device
copies or DataLoader overhead.
"""
from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable

# Must be set before CUDA initialises so cuBLAS pick-deterministic kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from .losses import Priors, arm_loss, mask_to_pattern
from .metrics import pn_balanced_accuracy
from .models import MODALITIES, FusionModel


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 256
    max_epochs: int = 100
    patience: int = 15
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deterministic: bool = True
    arm: str = "A0"          # loss arm A0-A4 (PIPELINE_v2 section 6); one config switch for all arms
    tau: float = 1.0         # A4 logit-adjustment temperature (1.0 per the spec)


@dataclass
class Tensors:
    """A fixed fold split held on the device: per-block features, natural mask, labels."""

    feats: dict[str, torch.Tensor]     # k -> [N, 2·d_k]
    mask: torch.Tensor                 # [N, n_mod] bool
    y: torch.Tensor                    # [N] int64

    def to(self, device: str) -> "Tensors":
        return Tensors({k: v.to(device) for k, v in self.feats.items()},
                       self.mask.to(device), self.y.to(device))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def index(self, idx: torch.Tensor) -> "Tensors":
        return Tensors({k: v[idx] for k, v in self.feats.items()}, self.mask[idx], self.y[idx])


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - depends on build
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_tensors(pp, episode_ids: Iterable[int], ep) -> Tensors:
    """Build :class:`Tensors` from a fitted :class:`~adni.preprocess.Preprocessor`.

    ``pp.transform`` returns rows sorted by episode_id, so we sort the ids first to align labels.
    """
    ids = np.unique(np.asarray(list(episode_ids), dtype=np.int64))
    out = pp.transform(ids)
    feats = {k: torch.tensor(out[k][0], dtype=torch.float32) for k in MODALITIES}
    mask = torch.stack([torch.tensor(out[k][1], dtype=torch.bool) for k in MODALITIES], dim=1)
    y = torch.tensor(ep["y"].loc[ids].to_numpy(), dtype=torch.int64)
    return Tensors(feats, mask, y)


def _macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    return float(f1_score(y, proba.argmax(axis=1), average="macro", labels=[0, 1, 2], zero_division=0))


def _macro_auroc(y: np.ndarray, proba: np.ndarray) -> float:
    present = [c for c in range(proba.shape[1]) if (y == c).any()]
    if len(present) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=present))


class Trainer:
    """Shared trainer for arms A0-A4; only the loss (and its training-split priors) changes."""

    def __init__(
        self,
        model: FusionModel,
        cfg: TrainConfig,
        priors: Priors | None = None,
        val_weights: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.priors = priors
        # Fixed per-episode PN weights for the VALIDATION split (rho uniform, from the fold's
        # TRAIN-split pi_hat). They do not depend on the epoch, so they are computed once outside.
        self.val_weights = None if val_weights is None else np.asarray(val_weights, dtype=float)
        self.device = cfg.device
        self.model.to(self.device)

    @torch.no_grad()
    def predict_proba(self, data: Tensors) -> np.ndarray:
        self.model.eval()
        logits, _, _ = self.model(data.feats, data.mask, apply_modality_dropout=False)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def _val_metrics(self, data: Tensors) -> dict:
        self.model.eval()
        logits, _, _ = self.model(data.feats, data.mask, apply_modality_dropout=False)
        loss = float(nn.functional.cross_entropy(logits, data.y).item())
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        y = data.y.cpu().numpy()
        out = {"loss": loss, "macro_f1": _macro_f1(y, proba), "macro_auroc": _macro_auroc(y, proba),
               "pn_bacc": float("nan")}
        if self.val_weights is not None:
            out["pn_bacc"] = float(pn_balanced_accuracy(y, proba.argmax(axis=1), self.val_weights))
        return out

    def fit(self, train: Tensors, val: Tensors) -> dict:
        cfg = self.cfg
        set_seed(cfg.seed, cfg.deterministic)
        train = train.to(self.device)
        val = val.to(self.device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        n = len(train)
        history: list[dict] = []
        # PRIMARY checkpoint rule: best validation macro-F1 (drives early stopping / patience).
        best = {"macro_f1": -1.0, "epoch": -1}
        best_state = copy.deepcopy(self.model.state_dict())
        # SECONDARY checkpoint rule: best validation PN balanced accuracy (rho uniform), tracked on
        # every epoch the run actually trains, but never used to stop training.
        best_pn = {"pn_bacc": -1.0, "epoch": -1}
        best_pn_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        for epoch in range(cfg.max_epochs):
            self.model.train()
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = perm[start:start + cfg.batch_size]
                batch = train.index(idx)
                opt.zero_grad(set_to_none=True)
                logits, _, eff = self.model(batch.feats, batch.mask, apply_modality_dropout=True)
                patterns = mask_to_pattern(eff)
                loss = arm_loss(cfg.arm, logits, batch.y, patterns, self.priors, cfg.tau)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * len(idx)
            train_loss = epoch_loss / n

            vm = self._val_metrics(val)
            # ``val_pn_bacc`` is NaN when not computable (no val_weights). NaN is not self-equal,
            # so it must not be stored in the per-epoch history, or two identical runs would
            # compare unequal. Store ``None`` ("not computed") instead; the metric stays NaN
            # inside ``vm`` so the finite guard below still works.
            val_pn_bacc = float(vm["pn_bacc"])
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": vm["loss"],
                            "val_macro_f1": vm["macro_f1"], "val_macro_auroc": vm["macro_auroc"],
                            "val_pn_bacc": val_pn_bacc if np.isfinite(val_pn_bacc) else None})
            improved = vm["macro_f1"] > best["macro_f1"] + 1e-12
            if improved:
                best = {"macro_f1": vm["macro_f1"], "epoch": epoch}
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            # Track the secondary state on every epoch, before any early-stopping break.
            if np.isfinite(vm["pn_bacc"]) and vm["pn_bacc"] > best_pn["pn_bacc"] + 1e-12:
                best_pn = {"pn_bacc": vm["pn_bacc"], "epoch": epoch}
                best_pn_state = copy.deepcopy(self.model.state_dict())
            if not improved and epochs_no_improve >= cfg.patience:
                break

        self.model.load_state_dict(best_state)
        return {"history": history, "best": best, "best_pn_bacc": best_pn, "epochs_run": len(history),
                "best_state": best_state, "best_pn_bacc_state": best_pn_state}


def train_arm(
    arm: str,
    feat_dims: dict[str, int],
    train: Tensors,
    val: Tensors,
    cfg: TrainConfig,
    model_cfg_kwargs: dict | None = None,
    priors: Priors | None = None,
    val_weights: np.ndarray | None = None,
) -> tuple[FusionModel, dict, float]:
    """Fit loss arm *arm* on one fold. Returns (model, result_dict, wall_clock_seconds).

    This is the REAL entry point for every arm; ``train_a0`` delegates to it. It seeds *before*
    constructing the model so that weight initialisation is reproducible, and ``Trainer.fit``
    re-seeds again so the dropout/permutation stream is independent of the init draw. A fold is
    therefore fully deterministic for a given ``cfg.seed``. ``val_weights`` (optional) are the
    fixed validation PN weights for the SECONDARY checkpoint rule.
    """
    from dataclasses import replace

    from .models import ModelConfig

    cfg = replace(cfg, arm=arm)
    set_seed(cfg.seed, cfg.deterministic)
    model = FusionModel(ModelConfig(feat_dims=feat_dims, **(model_cfg_kwargs or {})))
    trainer = Trainer(model, cfg, priors, val_weights)
    t0 = time.time()
    result = trainer.fit(train, val)
    result["seconds"] = time.time() - t0
    result["n_parameters"] = model.num_parameters()
    result["arm"] = arm
    return model, result, result["seconds"]


def train_a0(
    feat_dims: dict[str, int],
    train: Tensors,
    val: Tensors,
    cfg: TrainConfig,
    model_cfg_kwargs: dict | None = None,
) -> tuple[FusionModel, dict, float]:
    """Fit A0 (plain cross-entropy) on one fold. Equivalent to ``train_arm('A0', ...)``."""
    return train_arm("A0", feat_dims, train, val, cfg, model_cfg_kwargs)
