"""Methods / arms for PIPELINE v6 (section 5).

M1  baseline
M2  class-weighted loss: w_c = N / (3 * n_c) from train labels
M3  joint reweighting: w(c,p) = P(c)P(p)/P(c,p) from train, alpha=0.5,
     clipped to [0.2, 5], renormalised to mean 1
M4  cell-wise logit adjustment (post-hoc, no retraining):
       adjusted_logit_c(x,p) = log P_model(c|x,p) - tau * log Pbar(c|p)
       Pbar(c|p) = (n_{c,p} + kappa*P(c)) / (n_p + kappa)
     tau chosen on VALIDATION by macro cell accuracy; tau=0 == M1.
M4c class-only logit adjustment: Pbar(c) instead of Pbar(c|p)
M5  Group DRO over class x pattern cells (MLP only): per-batch cell losses,
     cell weights q_g updated by exponentiated gradient (eta=0.01);
     cells with <10 train samples share one group.
LGBM supports M1, M2, M3, M4, M4c. MLP supports all. Always run M0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Sequence

CLASSES: Sequence[str] = ("CN", "MCI", "Dementia")
N_CLASSES: int = len(CLASSES)


# ---------------------------------------------------------------------------
# M2 class weights
# ---------------------------------------------------------------------------

def class_weights(y: np.ndarray) -> np.ndarray:
    """w_c = N / (C * n_c) computed on the supplied labels (mean over samples = 1)."""
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=N_CLASSES).astype(float)
    counts = np.where(counts > 0, counts, 1.0)
    w = (len(y) / (N_CLASSES * counts))
    return w[y]


# ---------------------------------------------------------------------------
# M3 joint reweighting
# ---------------------------------------------------------------------------

def joint_reweight_table(y: np.ndarray, pattern: np.ndarray,
                         alpha: float = 0.5, wmin: float = 0.2,
                         wmax: float = 5.0) -> pd.DataFrame:
    """Return w(c,p) = clip(P(c)P(p)/P(c,p)) as DataFrame[class, pattern]."""
    y = np.asarray(y, dtype=int)
    pattern = np.asarray(pattern, dtype=str)
    pats = sorted(set(pattern.tolist()))
    n_cm = np.zeros((N_CLASSES, len(pats)), dtype=float)
    for c, p in zip(y, pattern):
        n_cm[c, pats.index(p)] += 1.0
    n = n_cm + alpha
    P = n / n.sum()
    Pc = P.sum(axis=1, keepdims=True)
    Pm = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(P > 0, P / (Pc * Pm), 1.0)
    w = np.clip(np.nan_to_num(w, nan=1.0, posinf=wmax), wmin, wmax)
    return pd.DataFrame(w, index=list(CLASSES), columns=pats)


def joint_weights(y: np.ndarray, pattern: np.ndarray, alpha: float = 0.5,
                  wmin: float = 0.2, wmax: float = 5.0) -> np.ndarray:
    """Per-sample joint weight, normalised to mean 1 over the supplied samples."""
    table = joint_reweight_table(y, pattern, alpha=alpha, wmin=wmin, wmax=wmax)
    w = np.array([float(table.loc[CLASSES[c], m]) for c, m in zip(y, pattern)])
    return w / w.mean()


# ---------------------------------------------------------------------------
# M4 / M4c cell-wise logit adjustment
# ---------------------------------------------------------------------------

@dataclass
class CellPrior:
    """Pbar(c|p) (M4) or Pbar(c) (M4c) estimated from train counts.

    Pbar(c|p) = (n_{c,p} + kappa * P(c)) / (n_p + kappa)
    where P(c) is the global class prior (over the same train counts).
    """

    classes: tuple[str, ...]
    P_c: np.ndarray                  # global class prior, shape (C,)
    # M4: Pbar(c|p) as [C, n_patterns]
    Pbar_class_pattern: np.ndarray | None = None
    patterns: tuple[str, ...] = ()
    # M4c: Pbar(c) as [C]
    Pbar_class_only: np.ndarray | None = None

    @classmethod
    def from_counts(cls, y: np.ndarray, pattern: np.ndarray,
                    kappa: float = 20.0) -> "CellPrior":
        """Estimate from train labels + pattern."""
        y = np.asarray(y, dtype=int)
        pattern = np.asarray(pattern, dtype=str)
        pats = sorted(set(pattern.tolist()))
        n_cm = np.zeros((N_CLASSES, len(pats)), dtype=float)
        for c, p in zip(y, pattern):
            n_cm[c, pats.index(p)] += 1.0
        P_c = n_cm.sum(axis=1) / n_cm.sum()
        n_p = n_cm.sum(axis=0, keepdims=True)
        Pbar_cp = (n_cm + kappa * P_c[:, None]) / (n_p + kappa)
        cp = cls(classes=tuple(CLASSES), P_c=P_c, Pbar_class_pattern=Pbar_cp,
                 patterns=tuple(pats))
        # M4c control
        Pbar_c = (n_cm.sum(axis=1) + kappa * P_c) / (n_cm.sum() + kappa)
        cp.Pbar_class_only = Pbar_c
        return cp

    @classmethod
    def from_counts_class_only(cls, y: np.ndarray, kappa: float = 20.0) -> "CellPrior":
        y = np.asarray(y, dtype=int)
        n_c = np.bincount(y, minlength=N_CLASSES).astype(float)
        P_c = n_c / n_c.sum()
        Pbar_c = (n_c + kappa * P_c) / (n_c.sum() + kappa)
        return cls(classes=tuple(CLASSES), P_c=P_c, Pbar_class_only=Pbar_c)


def adjust_logits(meta_logit: np.ndarray, prior: CellPrior,
                  tau: float, pattern: np.ndarray | None = None,
                  use_class_only: bool = False) -> np.ndarray:
    """adjusted_logit_c = logit_c - tau * log Pbar(...).

    - M4: use prior.Pbar_class_pattern and pattern.
    - M4c: use prior.Pbar_class_only (ignores pattern).
    tau=0 returns meta_logit unchanged (unit test T3).
    """
    if tau == 0.0:
        return meta_logit.copy()
    adjusted = meta_logit.copy()
    if use_class_only or prior.Pbar_class_pattern is None:
        pbar = prior.Pbar_class_only  # [C]
        if pbar is None:
            raise ValueError("no class-only prior available for M4c")
        # subtract tau * log Pbar(c) from each sample
        log_pbar = np.log(np.clip(pbar, 1e-12, 1.0))
        adjusted = adjusted - tau * log_pbar[None, :]
    else:
        # M4: per-sample pattern
        pats = list(prior.patterns)
        log_pbar = np.log(np.clip(prior.Pbar_class_pattern, 1e-12, 1.0))  # [C, n_patterns]
        pat_idx = np.array([pats.index(str(p)) for p in pattern], dtype=int)
        adjusted = adjusted - tau * log_pbar[:, pat_idx].T
    return adjusted


def choose_tau(meta_logit: np.ndarray, y: np.ndarray, pattern: np.ndarray,
               prior: CellPrior, tau_grid: Sequence[float],
               min_cell_n: int = 30) -> float:
    """Pick tau from tau_grid by macro cell accuracy on validation (within-cell recall)."""
    best_tau = 0.0
    best_score = -1.0
    for tau in tau_grid:
        adj = adjust_logits(meta_logit, prior, tau, pattern,
                            use_class_only=False)
        pred = adj.argmax(axis=1)
        cells = pd.crosstab(pd.Series(y, dtype=int), pd.Series(pattern, dtype=str))
        cells = cells.reindex(index=[0, 1, 2], columns=list(prior.patterns), fill_value=0)
        recalls: list[float] = []
        for c in range(3):
            for p in prior.patterns:
                n = int(cells.loc[c, p])
                if n >= min_cell_n:
                    m = (y == c)
                    recalls.append(float((pred[m] == c).mean()))
        score = float(np.mean(recalls)) if recalls else float("nan")
        if score > best_score + 1e-12:
            best_score = score
            best_tau = float(tau)
    return best_tau


# ---------------------------------------------------------------------------
# M5 Group DRO over class x pattern cells
# ---------------------------------------------------------------------------

def cell_id_for_sample(y: int, pattern: str, patterns: Sequence[str],
                       small_cell_n: int, y_counts_by_cell: dict[tuple[int, str], int]) -> int:
    """Map (y, pattern) -> group index. Small cells (<small_cell_n in train) share one group."""
    key = (y, pattern)
    if key not in y_counts_by_cell:
        return -1  # unknown cell -> small group
    if y_counts_by_cell[key] < small_cell_n:
        return -1
    return hash(key) % 100000  # stable-ish; production code would build a fixed mapping


def build_cell_groups(y: np.ndarray, pattern: np.ndarray, small_cell_n: int = 10) -> tuple[np.ndarray, int]:
    """Return (group_id per sample, n_groups). Small cells share group -1.

    Groups: one per (c,p) cell with >= small_cell_n train samples, plus one shared
    group for all small/rare cells.
    """
    y = np.asarray(y, dtype=int)
    pattern = np.asarray(pattern, dtype=str)
    counts: dict[tuple[int, str], int] = {}
    for c, p in zip(y, pattern):
        counts[(int(c), str(p))] = counts.get((int(c), str(p)), 0) + 1
    group_ids: list[int] = []
    small_group_id = -1
    cell_to_group: dict[tuple[int, str], int] = {}
    next_gid = 0
    for (c, p), n in counts.items():
        if n >= small_cell_n:
            cell_to_group[(c, p)] = next_gid
            next_gid += 1
        else:
            cell_to_group[(c, p)] = small_group_id
    for c, p in zip(y, pattern):
        group_ids.append(cell_to_group.get((int(c), str(p)), small_group_id))
    return np.array(group_ids, dtype=int), next_gid


def group_dro_loss(logits: np.ndarray, y: np.ndarray, group_ids: np.ndarray,
                   q: np.ndarray, n_groups: int) -> tuple[float, np.ndarray]:
    """One-step Group DRO over cells.

    q_g updated by exponentiated gradient with eta; loss = sum_g q_g * mean_loss_g.
    Returns (loss, q_new).
    """
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float)
    group_ids = np.asarray(group_ids, dtype=int)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    ce = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12))
    per_sample_loss = ce

    n_groups = int(n_groups)
    if n_groups == 0:
        return float(per_sample_loss.mean()), q
    group_loss = np.zeros(n_groups + 1)  # +1 for small group id -1
    group_count = np.zeros(n_groups + 1)
    for i, gid in enumerate(group_ids):
        g = gid if gid >= 0 else n_groups  # map -1 to the last slot
        group_loss[g] += per_sample_loss[i]
        group_count[g] += 1
    for g in range(n_groups + 1):
        if group_count[g] > 0:
            group_loss[g] /= group_count[g]
        else:
            group_loss[g] = 0.0

    q_use = np.concatenate([q, np.array([1.0])]) if q.size else np.ones(n_groups + 1)
    loss = float(np.dot(q_use, group_loss))

    # exponentiated gradient update
    eta = 0.01
    q_new = q_use * np.exp(eta * group_loss)
    q_new = q_new / q_new.sum()
    return loss, q_new[:n_groups]
