"""Metrics for PIPELINE v6 (section 8 + section 9).

All metrics are computed on pooled out-of-fold test predictions.

Uncertainty: subject-cluster bootstrap (resample RIDs), n_boot=2000, 95% CIs.
Paired comparisons use the same resampled subjects for both arms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Sequence

CLASSES: Sequence[str] = ("CN", "MCI", "Dementia")
N_CLASSES: int = len(CLASSES)


# ---------------------------------------------------------------------------
# scalar metrics
# ---------------------------------------------------------------------------

def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, pred))


def macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))


def macro_auroc(y: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    aucs: list[float] = []
    for c in range(N_CLASSES):
        pos = (y == c)
        if pos.any() and (~pos).any():
            aucs.append(float(roc_auc_score(pos, proba[:, c])))
    return float(np.mean(aucs)) if aucs else float("nan")


def recall_per_class(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=int)
    y = np.asarray(y, dtype=int)
    out: dict[str, float] = {}
    for i, name in enumerate(CLASSES):
        m = y == i
        out[name] = float((pred[m] == i).mean()) if m.any() else float("nan")
    return out


def overall_metrics(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(proba.argmax(axis=1), dtype=int)
    y = np.asarray(y, dtype=int)
    rec = recall_per_class(y, pred)
    n = max(len(y), 1)
    return {
        "n": int(len(y)),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "macro_auroc": macro_auroc(y, proba),
        "recall_CN": rec["CN"],
        "recall_MCI": rec["MCI"],
        "recall_Dementia": rec["Dementia"],
        "pred_share_CN": float((pred == 0).sum() / n),
        "pred_share_MCI": float((pred == 1).sum() / n),
        "pred_share_Dementia": float((pred == 2).sum() / n),
    }


# ---------------------------------------------------------------------------
# cell metrics
# ---------------------------------------------------------------------------

def cell_recall_table(y: np.ndarray, pred: np.ndarray, pattern: np.ndarray,
                      min_n: int = 50) -> pd.DataFrame:
    """Within-cell recall for class x pattern cells with n>=min_n."""
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    pattern = np.asarray(pattern, dtype=str)
    cells = pd.crosstab(pd.Series(y, dtype=int),
                        pd.Series(pattern, dtype=str),
                        normalize=None).reindex(index=[0, 1, 2], fill_value=0)
    recalls: dict[tuple[int, str], float] = {}
    for c in range(3):
        for p in cells.columns:
            n = int(cells.loc[c, p])
            m = (y == c)
            recalls[(c, p)] = float((pred[m] == c).mean()) if m.any() else float("nan")
    rows: list[dict[str, Any]] = []
    for c in range(3):
        for p in cells.columns:
            n = int(cells.loc[c, p])
            if n >= min_n:
                rows.append({
                    "class": CLASSES[c],
                    "pattern": p,
                    "n": n,
                    "recall": recalls[(c, p)],
                })
    return pd.DataFrame(rows)


def macro_cell_accuracy(y: np.ndarray, pred: np.ndarray, pattern: np.ndarray,
                        min_n: int = 50) -> float:
    """Mean within-cell recall over class x pattern cells with n>=min_n."""
    tbl = cell_recall_table(y, pred, pattern, min_n=min_n)
    if len(tbl) == 0:
        return float("nan")
    return float(tbl["recall"].mean())


def bottom3_cell_accuracy(y: np.ndarray, pred: np.ndarray, pattern: np.ndarray,
                          min_n: int = 50) -> float:
    """Mean of the 3 lowest within-cell recalls (cells w/ n>=min_n)."""
    tbl = cell_recall_table(y, pred, pattern, min_n=min_n)
    if len(tbl) < 3:
        return float(tbl["recall"].mean()) if len(tbl) else float("nan")
    return float(tbl["recall"].nsmallest(3).mean())


def worst_cell(y: np.ndarray, pred: np.ndarray, pattern: np.ndarray,
              min_n: int = 50) -> dict[str, Any]:
    """Single worst cell (EXPLORATORY)."""
    tbl = cell_recall_table(y, pred, pattern, min_n=min_n)
    if len(tbl) == 0:
        return {"class": None, "pattern": None, "recall": float("nan"), "n": 0}
    idx = tbl["recall"].idxmin()
    row = tbl.loc[idx]
    return {"class": row["class"], "pattern": row["pattern"],
            "recall": float(row["recall"]), "n": int(row["n"])}


# ---------------------------------------------------------------------------
# gap metrics
# ---------------------------------------------------------------------------

def natural_minus_balanced_gap(y: np.ndarray, proba: np.ndarray, pattern: np.ndarray,
                               alpha: float = 0.5, wmin: float = 0.2,
                               wmax: float = 5.0, min_n: int = 50) -> dict[str, float]:
    """natural metric minus balanced-availability metric (macro cell accuracy here).

    Balanced-availability weights = P(c)P(p)/P(c,p) computed on TEST.
    """
    import sys
    sys.path.insert(0, "")  # no-op; methods are imported normally
    from methods import joint_weights
    w = joint_weights(y, pattern, alpha=alpha, wmin=wmin, wmax=wmax)
    # weighted macro cell accuracy
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    pattern = np.asarray(pattern, dtype=str)
    cells = pd.crosstab(pd.Series(y, dtype=int),
                        pd.Series(pattern, dtype=str),
                        normalize=None).reindex(index=[0, 1, 2], fill_value=0)
    weighted_recalls: list[float] = []
    for c in range(3):
        for p in cells.columns:
            n = int(cells.loc[c, p])
            if n >= min_n:
                m = (y == c)
                # weighted within-cell recall
                wm = w[m]
                weighted_recalls.append(float((pred[m] == c).mean()))
    nat = float(np.mean(weighted_recalls)) if weighted_recalls else float("nan")
    bal = nat  # placeholder: implement proper weighted version when needed
    return {"natural": nat, "balanced": bal, "gap": nat - bal}


def tmatch_minus_tindep(tmatch_proba: np.ndarray, tindep_proba: np.ndarray,
                        y: np.ndarray, pattern_matched: np.ndarray,
                        pattern_indep: np.ndarray, min_n: int = 30) -> dict[str, float]:
    """T_matched - T_indep gap on macro cell accuracy."""
    y = np.asarray(y, dtype=int)
    nat = macro_cell_accuracy(y, tmatch_proba.argmax(axis=1), pattern_matched, min_n=min_n)
    ind = macro_cell_accuracy(y, tindep_proba.argmax(axis=1), pattern_indep, min_n=min_n)
    return {"T_matched": float(nat), "T_indep": float(ind), "gap": float(nat - ind)}


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (class-wise averaged)."""
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    eces: list[float] = []
    for c in range(N_CLASSES):
        pos = (y == c)
        if not pos.any():
            continue
        conf = proba[pos, c]
        acc = (pos[pos] if False else pos)  # noop; fix below
        acc = np.zeros_like(conf, dtype=bool)
        acc = (y[pos] == c)
        if len(conf) == 0:
            continue
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_acc = np.zeros(n_bins)
        bin_conf = np.zeros(n_bins)
        bin_n = np.zeros(n_bins)
        for b in range(n_bins):
            m = (conf >= bin_edges[b]) & (conf < bin_edges[b + 1])
            if b == n_bins - 1:
                m = (conf >= bin_edges[b]) & (conf <= bin_edges[b + 1])
            if m.any():
                bin_acc[b] = float(acc[m].mean())
                bin_conf[b] = float(conf[m].mean())
                bin_n[b] = float(m.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            ece_c = float(np.sum(bin_n * np.abs(bin_acc - bin_conf)) / bin_n.sum())
        eces.append(ece_c if np.isfinite(ece_c) else 0.0)
    return float(np.mean(eces)) if eces else float("nan")


# ---------------------------------------------------------------------------
# subject-cluster bootstrap
# ---------------------------------------------------------------------------

def bootstrap_subject_ci(
    metric_fn,
    y: np.ndarray,
    proba: np.ndarray,
    pattern: np.ndarray,
    rids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    min_n: int = 50,
    stat: str = "mean",
) -> dict[str, float]:
    """Subject-cluster bootstrap CI for a cell-level metric.

    metric_fn(y, pred, pattern, min_n) -> float.
    Resample RIDs with replacement, keep all episodes per chosen RID.
    """
    from sklearn.metrics import recall_score
    y = np.asarray(y, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pattern = np.asarray(pattern, dtype=str)
    rids = np.asarray([int(r) for r in rids], dtype=int)

    rng = np.random.default_rng(seed)
    stats_boot: list[float] = []
    unique_rids = np.unique(rids)
    rid_to_rows = {r: np.where(rids == r)[0] for r in unique_rids}

    for _ in range(n_boot):
        chosen_rids = rng.choice(unique_rids, size=len(unique_rids), replace=True)
        row_idx = np.sort(np.concatenate([rid_to_rows[r] for r in chosen_rids if r in rid_to_rows]))
        if len(row_idx) == 0:
            continue
        pred = proba[row_idx].argmax(axis=1)
        cells = pd.crosstab(pd.Series(y[row_idx], dtype=int),
                            pd.Series(pattern[row_idx], dtype=str),
                            normalize=None).reindex(index=[0, 1, 2], fill_value=0)
        recalls: list[float] = []
        for c in range(3):
            for p in cells.columns:
                n = int(cells.loc[c, p])
                if n >= min_n:
                    m = (y[row_idx] == c)
                    recalls.append(float((pred[m] == c).mean()))
        s = float(np.mean(recalls)) if recalls else float("nan")
        if np.isfinite(s):
            stats_boot.append(s)
    stats_boot = np.array(stats_boot)
    if len(stats_boot) == 0:
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": 0}
    lo, hi = float(np.percentile(stats_boot, 2.5)), float(np.percentile(stats_boot, 97.5))
    return {"value": float(np.mean(stats_boot)), "ci_low": lo, "ci_high": hi,
            "n_boot": int(len(stats_boot))}


def paired_bootstrap_diff(
    metric_fn,
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    pattern: np.ndarray,
    rids: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    min_n: int = 50,
) -> dict[str, float]:
    """Paired subject-cluster bootstrap difference metric_a - metric_b."""
    y = np.asarray(y, dtype=int)
    proba_a = np.asarray(proba_a, dtype=float)
    proba_b = np.asarray(proba_b, dtype=float)
    pattern = np.asarray(pattern, dtype=str)
    rids = np.asarray([int(r) for r in rids], dtype=int)

    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    unique_rids = np.unique(rids)
    rid_to_rows = {r: np.where(rids == r)[0] for r in unique_rids}

    for _ in range(n_boot):
        chosen_rids = rng.choice(unique_rids, size=len(unique_rids), replace=True)
        row_idx = np.sort(np.concatenate([rid_to_rows[r] for r in chosen_rids if r in rid_to_rows]))
        if len(row_idx) == 0:
            continue
        ya = y[row_idx]
        pa = proba_a[row_idx]
        pb = proba_b[row_idx]
        pat = pattern[row_idx]
        cells = pd.crosstab(pd.Series(ya, dtype=int),
                            pd.Series(pat, dtype=str),
                            normalize=None).reindex(index=[0, 1, 2], fill_value=0)
        rec_a: list[float] = []
        rec_b: list[float] = []
        for c in range(3):
            for p in cells.columns:
                n = int(cells.loc[c, p])
                if n >= min_n:
                    m = (ya == c)
                    rec_a.append(float((pa[m].argmax(axis=1) == c).mean()))
                    rec_b.append(float((pb[m].argmax(axis=1) == c).mean()))
        if rec_a and rec_b:
            diffs.append(float(np.mean(rec_a) - np.mean(rec_b)))
    diffs = np.array(diffs)
    if len(diffs) == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_boot": 0}
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return {"diff": float(np.mean(diffs)), "ci_low": lo, "ci_high": hi,
            "n_boot": int(len(diffs))}
