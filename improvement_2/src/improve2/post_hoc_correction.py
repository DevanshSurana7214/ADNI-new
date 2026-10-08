"""Post-hoc correction on EXISTING predicted probabilities (EXPLORATORY).

Corrected class probabilities are proportional to

    p'_c  proportional to  p_c * rho(c) / pi(c | m)^tau

renormalised over c. ``pi(c | m)`` is the mask-process prior for the episode's natural pattern
(or the natural-smoothed P(c | m) for the B1 baseline). ``tau = 0`` reproduces the raw
probabilities exactly (unit-tested). This module is pure maths: it does not touch any model,
any logits, or any OOF row semantics beyond what is passed in here.
"""
from __future__ import annotations

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

CLASSES = ("CN", "MCI", "Dementia")


def _build_lookup(prior: pd.DataFrame) -> pd.DataFrame:
    return prior.copy()


def correction_factor(
    proba: np.ndarray, pattern: Sequence[str], tau: float = 1.0,
    pi: pd.DataFrame | None = None, pi_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the per-row correction factor ``1 / pi(c | m)^tau`` for the requested patterns.

    Either *pi* (shape (|C|, |patterns|), index=CLASSES, columns=pattern strings) or
    *pi_lookup* (a length-|pattern| Series of the prior for each requested natural pattern)
    must be provided.
    """
    pat = np.asarray(pattern, dtype=str)
    if pi is not None:
        if pi_lookup is not None:
            raise ValueError("provide pi XOR pi_lookup, not both")
        return _correction_factor_from_pi(pi, pat, tau)
    if pi_lookup is not None:
        v = np.asarray(pi_lookup, dtype=float)
        return pd.DataFrame(1.0 / (v ** tau), index=CLASSES)
    raise ValueError("provide pi or pi_lookup")


def _correction_factor_from_pi(pi: pd.DataFrame, pat: Sequence[str], tau: float) -> pd.DataFrame:
    """Returns a per-row correction factor using the pandas DataFrame *pi* (classes x patterns).

    For each pattern in *pat* we look up the matching column in *pi* and compute
    ``1 / pi(c | m)^tau`` for every class c, for every row. Unmatched patterns yield NaN.
    """
    lut = _build_lookup(pi)
    pats = np.asarray(list(pat), dtype=str)
    cols = list(lut.columns)
    col_lookup = {c: i for i, c in enumerate(cols)}
    out = np.full((len(pats), len(lut.index)), np.nan, dtype=float)
    for j, p in enumerate(pats):
        ci = col_lookup.get(p)
        if ci is None:
            continue
        out[j, :] = lut.to_numpy()[:, ci]
    factor = 1.0 / (out ** tau)
    return factor


def apply_correction(
    proba: np.ndarray,
    pattern: Sequence[str],
    prior: pd.DataFrame,
    tau: float = 1.0,
    rho: np.ndarray | None = None,
    pi_lookup: pd.DataFrame | None = None,
) -> np.ndarray:
    """Apply the post-hoc joint correction to existing probabilities.

    Parameters
    ----------
    proba : (N, 3) array of predicted class probabilities (rows sum to ~1).
    pattern : length-N sequence of natural-pattern strings (the observed mask m).
    prior : P(c | m) table, index=CLASSES, columns=patterns, fitted on the TRAIN split.
    tau : correction temperature; tau=0 returns the input unchanged.
    rho : length-3 target class prior; None -> uniform (1/3 each).
    pi_lookup : optional per-row prior (length N) for the natural pattern; if absent, derived
        from *prior* and *pattern*.

    Returns
    -------
    corrected : (N, 3) array summing to 1 per row.
    """
    p = np.asarray(proba, dtype=float)
    N, K = p.shape
    assert K == 3
    if rho is None:
        rho = np.full(K, 1.0 / K)
    rho = np.asarray(rho, dtype=float)
    assert rho.shape == (K,)

    if pi_lookup is None:
        pat = np.asarray(pattern, dtype=str)
        pats = [str(c) for c in prior.columns]
        lut = _build_lookup(prior)
        col_idx = np.searchsorted(pats, pat)
        col_idx = np.clip(col_idx, 0, len(pats) - 1)
        row_idx = np.repeat(np.arange(len(prior.index)), len(pat))
        pi_vals = lut.to_numpy()[row_idx, np.tile(col_idx, len(prior.index))]
        pi_lookup = pd.Series(pi_vals, index=np.arange(N))
    else:
        pi_lookup = pd.Series(np.asarray(pi_lookup, dtype=float), index=np.arange(N))

    factor = 1.0 / (pi_lookup.to_numpy() ** tau)            # (N,) factor per row
    corrected = p * rho[None, :] * factor[:, None]
    s = corrected.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return corrected / s


def check_row_sums_close(corrected: np.ndarray, proba: np.ndarray, atol: float = 1e-12) -> bool:
    """Return True if both arrays sum to 1 per row to within *atol*."""
    return bool(np.allclose(corrected.sum(axis=1), 1.0, atol=atol)
                and np.allclose(proba.sum(axis=1), 1.0, atol=atol))
