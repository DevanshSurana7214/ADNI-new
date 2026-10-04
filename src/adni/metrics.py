"""Evaluation metrics (PIPELINE_v2 §8).

Standard natural-distribution metrics (§8.1), the pattern-neutral (PN) reweighting and weighted
balanced accuracy (§8.2), stratified reporting (§8.3) and the subject-cluster bootstrap statistics
(§8.5). No models are trained here.

Conventions
-----------
* ``y`` is an integer array with classes ``0, 1, 2`` = CN, MCI, Dementia.
* ``proba`` is ``[N, n_classes]`` predicted class probabilities (rows sum to ~1).
* ``pattern`` is a string array of availability bits ``m_mri m_fdg m_amy m_tau`` (e.g. ``"1000"``).
* ``groups`` is the subject id (RID) used for the cluster bootstrap.
* ``prior_table`` is a DataFrame of ``P(c | m)`` with index ``CLASSES`` and columns = pattern strings.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from statsmodels.stats.multitest import multipletests

CLASSES: Sequence[str] = ["CN", "MCI", "Dementia"]


# --------------------------------------------------------------------------- §8.1 standard metrics
def expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """ECE over ``n_bins`` equal-width bins on the maximum predicted probability."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(conf, edges[1:-1], right=True)  # 0 .. n_bins-1
    n = len(y)
    ece = 0.0
    for b in range(n_bins):
        m = bin_idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(y: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Multiclass Brier score: mean over samples of sum_c (p_c - 1[y=c])**2."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def natural_metrics(y: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> dict[str, Any]:
    """Natural-distribution metrics (§8.1).

    Returns a dict with keys ``macro_auroc``, ``macro_f1``, ``balanced_accuracy``,
    ``f1_per_class`` (dict class index -> F1), ``brier`` and ``ece``.
    """
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)

    aucs: list[float] = []
    for c in range(n_classes):
        pos = y == c
        if pos.any() and (~pos).any():
            aucs.append(float(roc_auc_score(pos.astype(int), proba[:, c])))
    macro_auroc = float(np.mean(aucs)) if aucs else float("nan")

    labels = list(range(n_classes))
    macro_f1 = float(f1_score(y, pred, average="macro", labels=labels, zero_division=0))
    bacc = float(balanced_accuracy_score(y, pred))
    f1s = f1_score(y, pred, average=None, labels=labels, zero_division=0)
    f1_per_class = {c: float(f1s[c]) for c in range(n_classes)}

    return {
        "macro_auroc": macro_auroc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bacc,
        "f1_per_class": f1_per_class,
        "brier": multiclass_brier(y, proba, n_classes),
        "ece": expected_calibration_error(y, proba),
    }


# --------------------------------------------------------------------------- §8.2 pattern-neutral
def pi_table_from_counts(
    n_cm: np.ndarray,
    patterns: Sequence[str],
    classes: Sequence[str] = CLASSES,
    alpha: float = 0.5,
) -> pd.DataFrame:
    """Build ``P(c | m)`` (DataFrame index=classes, columns=patterns) from a class × pattern count
    table with Dirichlet smoothing ``alpha`` (alpha=0 recovers the empirical conditionals)."""
    from .priors import smoothed_joint

    n_cm = np.asarray(n_cm, dtype=float)
    P_cm, _, Pm, _ = smoothed_joint(n_cm, alpha=alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_c_given_m = P_cm / Pm[None, :]
    P_c_given_m = np.nan_to_num(P_c_given_m, nan=0.0)
    return pd.DataFrame(P_c_given_m, index=list(classes), columns=list(patterns))


def pn_weights(
    y: np.ndarray,
    pattern: np.ndarray,
    prior_table: pd.DataFrame,
    rho: str | Sequence[float] = "uniform",
    wmin: float = 0.2,
    wmax: float = 5.0,
) -> np.ndarray:
    """Pattern-neutral importance weights ``w_i = rho(c_i) / pi_hat(c_i | m_i)``.

    ``pi_hat`` is read from *prior_table* (P(c|m), fitted on the TRAINING split only). ``rho`` is the
    target class marginal: ``"uniform"`` -> ``1/|C|``, or an explicit length-|C| sequence indexed by
    class. Weights are clipped to ``[wmin, wmax]`` and normalised to mean 1 over the supplied samples.
    """
    y = np.asarray(y, dtype=int)
    pattern = np.asarray(pattern, dtype=str)
    classes = list(prior_table.index)

    pi = np.array([float(prior_table.loc[classes[c], m]) for c, m in zip(y, pattern)])

    if isinstance(rho, str):
        if rho == "uniform":
            rho_i = np.full(len(y), 1.0 / len(classes))
        else:
            raise ValueError(f"unknown rho {rho!r}; use 'uniform' or an explicit sequence")
    else:
        rho_arr = np.asarray(rho, dtype=float)
        if rho_arr.shape != (len(classes),):
            raise ValueError(f"rho must have length {len(classes)}, got {rho_arr.shape}")
        rho_i = rho_arr[y]

    w = rho_i / pi
    w = np.clip(w, wmin, wmax)
    return w / w.mean()


def pn_balanced_accuracy(y: np.ndarray, pred: np.ndarray, weights: np.ndarray) -> float:
    """Weighted balanced accuracy: mean over classes of the ``w``-weighted per-class recall."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    weights = np.asarray(weights, dtype=float)
    recalls: list[float] = []
    for c in np.unique(y):
        m = y == c
        denom = weights[m].sum()
        if denom > 0:
            recalls.append(float((weights[m] * (pred[m] == c)).sum() / denom))
    return float(np.mean(recalls)) if recalls else float("nan")


# --------------------------------------------------------------------------- §8.5 bootstrap
class BootstrapCI(NamedTuple):
    point: float
    ci_low: float
    ci_high: float
    n_valid: int


def _group_index(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the row indices of every cluster.

    Returns ``(order, starts)`` such that the rows of cluster ``g`` are
    ``order[starts[g]:starts[g + 1]]``. Doing this once makes each bootstrap resample O(N + G)
    instead of O(N x G).
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")          # ascending row index within each cluster
    counts = np.bincount(inv, minlength=len(uniq))
    starts = np.zeros(len(uniq) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return order, starts


def _resample_indices(
    order: np.ndarray, starts: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Resample clusters with replacement and concatenate their (precomputed) row indices."""
    n_groups = len(starts) - 1
    chosen = rng.integers(0, n_groups, size=n_groups)
    return np.concatenate([order[starts[g]:starts[g + 1]] for g in chosen])


def cluster_bootstrap(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    proba: np.ndarray,
    groups: np.ndarray,
    n: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Subject-cluster bootstrap: resample SUBJECTS (groups) with replacement, return the point
    estimate and the 95 % percentile CI of ``metric_fn(y, proba)`` (a scalar)."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    groups = np.asarray(groups)
    point = float(metric_fn(y, proba))
    rng = np.random.default_rng(seed)
    order, starts = _group_index(groups)
    stats: list[float] = []
    for _ in range(n):
        idx = _resample_indices(order, starts, rng)
        try:
            v = float(metric_fn(y[idx], proba[idx]))
        except Exception:
            v = float("nan")
        if np.isfinite(v):
            stats.append(v)
    if not stats:
        return BootstrapCI(point, float("nan"), float("nan"), 0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return BootstrapCI(point, float(lo), float(hi), len(stats))


def cluster_bootstrap_multi(
    metric_fns: Mapping[str, Callable[[np.ndarray, np.ndarray], float]],
    y: np.ndarray,
    proba: np.ndarray,
    groups: np.ndarray,
    n: int = 2000,
    seed: int = 0,
) -> dict[str, BootstrapCI]:
    """Subject-cluster bootstrap for several scalar metrics sharing the same resamples.

    Equivalent to calling :func:`cluster_bootstrap` once per metric, but the cluster index and the
    resample are built once per draw, which matters when the metric family is expensive. Each metric
    keeps its own count of finite draws (``n_valid``), so a metric that is undefined on some resample
    does not discard the others.
    """
    y = np.asarray(y)
    proba = np.asarray(proba)
    groups = np.asarray(groups)
    points = {k: float(f(y, proba)) for k, f in metric_fns.items()}
    rng = np.random.default_rng(seed)
    order, starts = _group_index(groups)
    stats: dict[str, list[float]] = {k: [] for k in metric_fns}
    for _ in range(n):
        idx = _resample_indices(order, starts, rng)
        for k, f in metric_fns.items():
            try:
                v = float(f(y[idx], proba[idx]))
            except Exception:
                v = float("nan")
            if np.isfinite(v):
                stats[k].append(v)
    out: dict[str, BootstrapCI] = {}
    for k, vals in stats.items():
        if vals:
            lo, hi = np.percentile(vals, [2.5, 97.5])
            out[k] = BootstrapCI(points[k], float(lo), float(hi), len(vals))
        else:
            out[k] = BootstrapCI(points[k], float("nan"), float("nan"), 0)
    return out


def paired_cluster_bootstrap(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    groups: np.ndarray,
    n: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Paired subject-cluster bootstrap CI for ``metric_fn(a) - metric_fn(b)`` (same resamples for
    both arms, so the two arms are compared on the same episodes)."""
    y = np.asarray(y)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    groups = np.asarray(groups)
    point = float(metric_fn(y, proba_a)) - float(metric_fn(y, proba_b))
    rng = np.random.default_rng(seed)
    order, starts = _group_index(groups)
    stats: list[float] = []
    for _ in range(n):
        idx = _resample_indices(order, starts, rng)
        try:
            d = float(metric_fn(y[idx], proba_a[idx])) - float(metric_fn(y[idx], proba_b[idx]))
        except Exception:
            d = float("nan")
        if np.isfinite(d):
            stats.append(d)
    if not stats:
        return BootstrapCI(point, float("nan"), float("nan"), 0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return BootstrapCI(point, float(lo), float(hi), len(stats))


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (statsmodels). Empty input returns an empty array."""
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p
    return multipletests(p, method="holm")[1]


# --------------------------------------------------------------------------- §8.3 stratified
def stratified_report(
    y: np.ndarray,
    proba: np.ndarray,
    strata: np.ndarray,
    min_n: int = 30,
    n_classes: int = 3,
) -> dict[str, dict[str, Any]]:
    """Natural metrics per stratum, pooling strata with ``n < min_n`` into an ``"other"`` bucket."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    strata = np.asarray(strata)
    values, counts = np.unique(strata, return_counts=True)

    out: dict[str, dict[str, Any]] = {}
    for v, c in zip(values, counts):
        if c >= min_n:
            m = strata == v
            out[str(v)] = natural_metrics(y[m], proba[m], n_classes)

    small = values[counts < min_n]
    if small.size:
        m = np.isin(strata, small)
        out["other"] = natural_metrics(y[m], proba[m], n_classes)
    return out
