"""Prior-loading helpers for step13a (EXPLORATORY).

Reusable across the correction and evaluation recipes: load the TRAIN-split prior for every
(seed, fold) for S1 and S2, in the exact form each model X needs:

- B1 (tabular): natural-smoothed P(c|m) on the TRAIN split (alpha 0.5).
- Neural arm (A0(R2), A4(R2)): the mask-process prior ``priors.pi_table(P_cm, p_drop=0.25)``
  evaluated at the NATURAL mask of each episode.

No model is trained here; this is just data + prior table construction.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adni.baselines import CLASSES, training_prior_table
from adni.losses import priors_from_counts
from adni.priors import crosstab


def _fold_ids(blob: dict, seed: int, fold: int, split: str):
    node = blob[split][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _fold_ids_s2(blob: dict, split: str):
    node = blob[split]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def load_s1_prior_table(ep: pd.DataFrame, blob: dict, seed: int, fold: int,
                        alpha: float = 0.5) -> pd.DataFrame:
    """TRAIN-split natural-smoothed P(c|m) for one S1 fold (B1-style prior)."""
    tr, _, _ = _fold_ids(blob, seed, fold, "s1")
    return training_prior_table(ep, tr, alpha=alpha)


def load_s1_priors(ep: pd.DataFrame, blob: dict, seeds=(0, 1, 2), folds=(0, 1, 2, 3, 4),
                   p_drop: float = 0.25, alpha: float = 0.5) -> dict[tuple[int, int], Any]:
    """Return, for each (seed, fold), the mask-process priors object for the neural arm.

    The returned object has ``pi`` (DataFrame, index=CLASSES, columns=submasks), ``pc`` (train
    class marginal), and ``patterns`` (natural patterns, for lookup).
    """
    out: dict[tuple[int, int], Any] = {}
    for seed in seeds:
        for fold in folds:
            tr, _, _ = _fold_ids(blob, seed, fold, "s1")
            sub = ep.loc[tr]
            diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
            n_cm, pats = crosstab(diag, sub["pattern"])
            priors = priors_from_counts(n_cm, pats, p_drop=p_drop, alpha=alpha)
            out[(seed, fold)] = priors
    return out


def load_s2_prior_table(ep: pd.DataFrame, blob: dict, alpha: float = 0.5) -> pd.DataFrame:
    """TRAIN-split natural-smoothed P(c|m) for the S2 train side (B1-style prior)."""
    _, _, _ = _fold_ids_s2(blob, "s2")
    tr, _, _ = _fold_ids_s2(blob, "s2")
    return training_prior_table(ep, tr, alpha=alpha)


def load_s2_priors(ep: pd.DataFrame, blob: dict, p_drop: float = 0.25,
                   alpha: float = 0.5) -> Any:
    """Return mask-process priors object for the S2 train side (neural arm)."""
    tr, _, _ = _fold_ids_s2(blob, "s2")
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    return priors_from_counts(n_cm, pats, p_drop=p_drop, alpha=alpha)


def natural_pattern_lookup(ep: pd.DataFrame, ids: np.ndarray) -> pd.Series:
    """Per-row natural pattern string for a set of episode ids (used to index pi at m)."""
    return ep["pattern"].loc[list(np.asarray(ids))].astype(str)


def s1_episode_test_ids(blob: dict, seed: int, fold: int) -> np.ndarray:
    _, _, te = _fold_ids(blob, seed, fold, "s1")
    return te


def s2_episode_test_ids(blob: dict) -> np.ndarray:
    _, _, te = _fold_ids_s2(blob, "s2")
    return te
