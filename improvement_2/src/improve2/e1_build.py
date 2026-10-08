"""E1 build: average of the two corrected posteriors, plus the uncorrected E1.

E1 is the ensemble formed by averaging the corrected probabilities of B1 and A0(R2). The
uncorrected E1 is the same average applied to the raw (uncorrected) probabilities. Both are
computed per (seed, fold, episode) on the test split, seed-averaged exactly as the frozen Step 9
uniform protocol does.

No model is trained here; this only assembles existing OOF rows for the analysis recipe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROBA_COLS = ("p_CN", "p_MCI", "p_Dementia")


def build_e1_corrected(p_b1: pd.DataFrame, p_a0: pd.DataFrame) -> pd.DataFrame:
    """Average of the corrected B1 and A0(R2) probabilities -> E1, corrected.

    Both inputs must share (seed, fold, episode_id) and contain PROBA_COLS. Rows are aligned on
    that key.
    """
    key = ["seed", "fold", "episode_id"]
    m = p_b1[key + list(PROBA_COLS)].merge(
        p_a0[key + list(PROBA_COLS)], on=key, suffixes=("_b1", "_a0"))
    for c in PROBA_COLS:
        m[c] = (m[f"{c}_b1"] + m[f"{c}_a0"]) / 2.0
    return m[key + list(PROBA_COLS)]


def build_e1_uncorrected(p_b1: pd.DataFrame, p_a0: pd.DataFrame) -> pd.DataFrame:
    """Average of the RAW B1 and A0(R2) probabilities -> E1, uncorrected."""
    return build_e1_corrected(p_b1, p_a0)
