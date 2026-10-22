"""Models for PIPELINE v6 (section 4).

- LGBM: LightGBM multiclass (section 4.1).
- MLP late fusion: one encoder per modality (Linear-ReLU-Dropout0.3-Linear-ReLU,
  hidden 64), concatenate embeddings, linear head to 3 classes (section 4.2).
- M0 shortcut probe: multinomial logistic regression on the mask bits only (section 4.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

# LightGBM params from config; mirrored here for type clarity.
LGBM_PARAMS_DEFAULT: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "n_jobs": 1,
}


@dataclass
class LGBMConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 15
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 30
    seed: int = 0


@dataclass
class MLPConfig:
    d_hidden: int = 64
    dropout: float = 0.3
    lr: float = 1e-4
    weight_decay: float = 0.05
    batch_size: int = 256
    max_epochs: int = 60
    patience: int = 12
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


MODALITIES: tuple[str, ...] = ("mri", "fdg", "amy", "tau")


class Encoder(nn.Module):
    """Linear(d_in, H) -> ReLU -> Dropout(0.3) -> Linear(H, H) -> ReLU."""

    def __init__(self, d_in: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionMLP(nn.Module):
    """One encoder per modality; embeddings concatenated -> linear head to 3 classes."""

    def __init__(self, feat_dims: dict[str, int], d_hidden: int, dropout: float,
                 use_mask: bool = True) -> None:
        super().__init__()
        self.feat_dims = feat_dims
        self.use_mask = use_mask
        self.encoders = nn.ModuleDict({
            k: Encoder(2 * feat_dims[k], d_hidden, dropout) for k in MODALITIES
        })
        # input = sum over modalities of d_latent(=d_hidden) + mask bits
        width = d_hidden * len(MODALITIES) + (len(MODALITIES) if use_mask else 0)
        self.head = nn.Linear(width, 3)

    def forward(self, feats: dict[str, torch.Tensor], mask: torch.Tensor,
                use_mask: bool | None = None) -> torch.Tensor:
        use_mask = self.use_mask if use_mask is None else bool(use_mask)
        parts: list[torch.Tensor] = []
        for i, k in enumerate(MODALITIES):
            x = feats[k]  # [N, 2*d_k]
            emb = self.encoders[k](x)  # [N, d_hidden]
            eff = mask[:, i].unsqueeze(1).to(dtype=torch.float32) if mask is not None else 1.0
            parts.append(emb * eff)
        cat = torch.cat(parts, dim=1)
        if use_mask and mask is not None:
            cat = torch.cat([cat, mask.to(dtype=torch.float32)], dim=1)
        return self.head(cat)


def fit_lgbm(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame,
    cfg: LGBMConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit LightGBM; early stop on validation log-loss; return (val_proba, test_proba, best_iter)."""
    import lightgbm as lgb
    params = dict(LGBM_PARAMS_DEFAULT)
    params.update({
        "learning_rate": cfg.learning_rate,
        "num_leaves": cfg.num_leaves,
        "min_child_samples": cfg.min_child_samples,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "seed": cfg.seed,
        "feature_fraction_seed": cfg.seed,
        "bagging_seed": cfg.seed,
    })
    dtrain = lgb.Dataset(X_train, label=y_train.to_numpy(dtype=int))
    dval = lgb.Dataset(X_val, label=y_val.to_numpy(dtype=int), reference=dtrain)
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=cfg.n_estimators,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    val_proba = booster.predict(X_val)
    test_proba = booster.predict(X_test)
    best_iter = int(booster.best_iteration or booster.current_iteration())
    return val_proba, test_proba, best_iter


def fit_mlp(
    cfg: MLPConfig,
    feat_dims: dict[str, int],
    X_train: dict[str, np.ndarray],
    mask_train: np.ndarray,
    y_train: np.ndarray,
    X_val: dict[str, np.ndarray],
    mask_val: np.ndarray,
    y_val: np.ndarray,
    use_mask: bool = True,
    class_weights: np.ndarray | None = None,
    joint_weights: np.ndarray | None = None,
) -> tuple[LateFusionMLP, dict[str, Any]]:
    """Fit one MLP arm on one fold; return (model, info)."""
    torch.random.manual_seed(cfg.seed)
    model = LateFusionMLP(feat_dims, cfg.d_hidden, cfg.dropout, use_mask=use_mask).to(cfg.device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ce_weight = None
    if class_weights is not None:
        ce_weight = torch.as_tensor(class_weights, dtype=torch.float32, device=cfg.device)

    def make_batch(Xd: dict[str, np.ndarray], mk: np.ndarray, y: np.ndarray,
                   rng: np.random.Generator):
        n = len(y)
        idx = rng.choice(n, size=min(cfg.batch_size, n), replace=False)
        feats_b = {k: torch.as_tensor(Xd[k][idx], dtype=torch.float32, device=cfg.device)
                   for k in MODALITIES}
        mask_b = torch.as_tensor(mk[idx], dtype=torch.bool, device=cfg.device)
        yb = torch.as_tensor(y[idx], dtype=torch.int64, device=cfg.device)
        return feats_b, mask_b, yb

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1: float = -1.0
    bad = 0
    history: list[dict[str, Any]] = []
    n_train = len(y_train)
    total = n_train
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = rng.permutation(n_train)
        total_loss = 0.0
        n_seen = 0
        for start in range(0, n_train, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            feats_b = {k: torch.as_tensor(X_train[k][idx], dtype=torch.float32, device=cfg.device)
                       for k in MODALITIES}
            mask_b = torch.as_tensor(mask_train[idx], dtype=torch.bool, device=cfg.device)
            yb = torch.as_tensor(y_train[idx], dtype=torch.int64, device=cfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(feats_b, mask_b, use_mask=use_mask)
            loss = torch.nn.functional.cross_entropy(logits, yb, weight=ce_weight, reduction="none")
            if joint_weights is not None:
                w = torch.as_tensor(joint_weights[idx], dtype=torch.float32, device=cfg.device)
                loss = (loss * w).mean()
            else:
                loss = loss.mean()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)
            n_seen += len(idx)
        # validation macro-F1
        model.eval()
        with torch.no_grad():
            feats_v = {k: torch.as_tensor(X_val[k], dtype=torch.float32, device=cfg.device)
                       for k in MODALITIES}
            mask_v = torch.as_tensor(mask_val, dtype=torch.bool, device=cfg.device)
            logits_v = model(feats_v, mask_v, use_mask=use_mask)
            proba_v = torch.softmax(logits_v, dim=1).cpu().numpy()
        pred_v = proba_v.argmax(axis=1)
        from sklearn.metrics import f1_score
        val_f1 = float(f1_score(y_val, pred_v, average="macro", labels=[0, 1, 2], zero_division=0))
        history.append({"epoch": epoch, "train_loss": total_loss / max(n_seen, 1),
                        "val_macro_f1": val_f1})
        if val_f1 > best_val_f1 + 1e-12:
            best_val_f1 = val_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    if best_state is None:
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_val_macro_f1": best_val_f1,
                   "epochs_run": len(history), "n_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def predict_mlp(model: LateFusionMLP, Xd: dict[str, np.ndarray], mask: np.ndarray,
                use_mask: bool = True) -> np.ndarray:
    model.eval()
    feats = {k: torch.as_tensor(Xd[k], dtype=torch.float32, device=model.head.in_features and "cpu")
             for k in MODALITIES} if False else \
        {k: torch.as_tensor(Xd[k], dtype=torch.float32)
         for k in MODALITIES}
    m = torch.as_tensor(mask, dtype=torch.bool)
    with torch.no_grad():
        logits = model(feats, m, use_mask=use_mask)
    return torch.softmax(logits, dim=1).numpy()


def fit_m0(X_train: pd.DataFrame, y_train: pd.Series,
           X_val: pd.DataFrame, y_val: pd.Series,
           X_test: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """M0: multinomial logistic regression on the mask bits only."""
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed, multi_class="multinomial")
    clf.fit(X_train, y_train.to_numpy(dtype=int))
    val_proba = clf.predict_proba(X_val)
    test_proba = clf.predict_proba(X_test)
    return val_proba, test_proba
