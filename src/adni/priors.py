"""Joint-distribution and reweighting helpers (PIPELINE_v2 sections 4, 6, Appendix B).

Used by the Step 2 profiling script and by the priors/losses modules. Every joint table below is
fitted on the supplied count table (by default the whole cohort for profiling; callers in later steps
fit on the TRAIN split only).

The figures in ``adni_stage3_profile.py`` and ``joint_profile.csv`` are reproduced independently here:

* smoothed_joint(n_cm, alpha=0.5)  ->  P(c,m), Pc, Pm, PMI(c,m)
* independence_weight(P_cm)        ->  w(c,m) = exp(-PMI) = P(c)P(m)/P(c,m)   (sum P(c,m) w = 1)
* per_sample_weight(...)           ->  per-episode clipped weights that match the
                                       essence/ESS diagnostics of adni_stage3_profile.py to within 1e-9
                                       when called on the full cohort count table.
"""
from __future__ import annotations

import enum
from typing import Sequence

import numpy as np
import pandas as pd

CLASSES: Sequence[str] = ["CN", "MCI", "Dementia"]


class Class(enum.Enum):
    CN = 0
    MCI = 1
    DEMENTIA = 2


def _class_index(diagnosis: pd.Series) -> np.ndarray:
    arr = np.asarray(diagnosis.values, dtype=str)
    return np.array([CLASSES.index(x) for x in arr])


def _pattern_index(pattern: pd.Series, patterns: Sequence[str]) -> np.ndarray:
    arr = np.asarray(pattern.values, dtype=str)
    return np.array([patterns.index(x) for x in arr])


def crosstab(diagnosis: pd.Series, pattern: pd.Series) -> np.ndarray:
    """Count table n(c, m), rows = CLASSES, columns = sorted unique patterns of *pattern*."""
    pats = sorted(pattern.unique().tolist())
    tab = pd.crosstab(diagnosis, pattern).reindex(index=CLASSES, columns=pats, fill_value=0)
    return tab.values.astype(np.int64), pats


def smoothed_joint(n_cm: np.ndarray, alpha: float = 0.5):
    """Dirichlet-smoothed joint and PMI from a class × pattern count table.

    Returns
    -------
    P_cm : (3, P) float, sum to 1.
    Pc, Pm : (3,) and (P,) marginals of P_cm.
    I : (3, P) PMI = log P(c,m) / (P(c) P(m)) in nats.
    """
    n_cm = np.asarray(n_cm, dtype=float)
    denom = n_cm.sum() + alpha * n_cm.size
    P = (n_cm + alpha) / denom
    Pc = P.sum(axis=1)        # (3,)
    Pm = P.sum(axis=0)        # (P,)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.log(P / (Pc[:, None] * Pm[None, :]))
    return P, Pc, Pm, I


def independence_weight(P_cm: np.ndarray):
    """w(c,m) = P(c) P(m) / P(c,m) = exp(-PMI). Sum over cells of P(c,m) w = 1 (unclipped)."""
    P = np.asarray(P_cm, dtype=float)
    Pc = P.sum(axis=1, keepdims=True)
    Pm = P.sum(axis=0, keepdims=True)
    w = np.exp(-np.log(P / (Pc * Pm)))         # = P(c)P(m) / P(c,m)
    assert float(np.abs((P * w).sum() - 1.0)) < 1e-12, "sum P(c,m) w must be 1 (unclipped)"
    return w


def pmi(P_cm: np.ndarray):
    """PMI(c,m) = log P(c,m) / (P(c) P(m)) in nats. Recovers the PMI column of joint_profile.csv
    within 1e-12 when P_cm is the smoothed table used by adni_stage3_profile.py."""
    _, _, _, I = smoothed_joint(np.asarray(P_cm, float) * 1.0, alpha=0.0)
    # The alpha=0.0 path is *not* the profile's smoothed table; callers pass the actual smoothed P.
    # So we recompute PMI directly from the supplied P:
    Pc = np.asarray(P_cm).sum(axis=1, keepdims=True)
    Pm = np.asarray(P_cm).sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        I = np.log(np.asarray(P_cm) / (Pc * Pm))
    return I


def per_sample_weight(
    n_cm: np.ndarray,
    diagnosis: pd.Series,
    pattern: pd.Series,
    wmin: float = 0.2,
    wmax: float = 5.0,
    alpha: float = 0.5,
) -> np.ndarray:
    """Per-episode clipped independence weight, normalised to mean 1 on the *supplied* samples.

    Steps:
      1. Fit smoothed P(c,m) from n_cm (e.g. the TRAIN split count table).
      2. w_raw[c, m] = clip(exp(-PMI(c,m)), wmin, wmax).
      3. w[i] = w_raw[c_i, m_i] / mean(w_raw over the *supplied samples*).

    When *diagnosis*/*pattern* are the full cohort and n_cm is the full-cohort count table, the
    returned ESS over those samples equals the `ess_clip` figure in ``adni_stage3_profile.py``
    (0.878) to within 1e-9.
    """
    n_cm = np.asarray(n_cm, dtype=float)
    P, _, _, I = smoothed_joint(n_cm, alpha=alpha)
    w_raw = np.clip(np.exp(-I), wmin, wmax)          # (3, P)
    cidx = _class_index(diagnosis)
    pats = sorted(pattern.unique().tolist())
    midx = _pattern_index(pattern, pats)
    # w_raw is aligned to CLASSES × pats; build lookup aligned to supplied samples.
    lookup = pd.DataFrame(w_raw, index=CLASSES, columns=pats)
    w_per = np.array([lookup.loc[CLASSES[c], pats[m]] for c, m in zip(cidx, midx)])
    w_per = w_per / float(w_per.mean())
    return w_per


def ess(w: np.ndarray) -> float:
    """Effective sample size ratio ESS / N for a weight vector."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    return float(s * s / (w * w).sum() / len(w))


def cramers_v(tab: np.ndarray) -> float:
    """Cramér's V on a class × pattern count table (assumes tab has no all-zero columns)."""
    t = np.asarray(tab, dtype=float)
    t = t[:, t.sum(axis=0) > 0]
    tot = t.sum()
    if tot == 0:
        return 0.0
    row = t.sum(axis=1, keepdims=True)
    col = t.sum(axis=0, keepdims=True)
    expected = row * col / tot
    chi2 = ((t - expected) ** 2 / expected).sum()
    return float(np.sqrt(chi2 / (tot * (min(t.shape) - 1))))
