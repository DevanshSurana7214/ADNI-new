"""Stage 1 invariants (PIPELINE_v2 sections 2.2, 3.1).

Asserts structural properties of the produced ``episodes.csv``. These do not depend on the exact
reference values; they check the episode construction is internally consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_ROOT / "results" / "step01" / "stage1" / "episodes.csv"

MODS = ["mri", "fdg", "amy", "tau"]
CLASS_MAP = {"CN": 0, "MCI": 1, "Dementia": 2}
WINDOW = 180


@pytest.fixture(scope="module")
def ep() -> pd.DataFrame:
    assert EPISODES.is_file(), f"missing {EPISODES} — run scripts/adni_stage1_build.py first"
    # pattern is an all-digit string; keep it as str (same convention as adni_stage3_profile.py).
    return pd.read_csv(EPISODES, dtype={"pattern": str})


def test_n_mod_and_pattern_consistency(ep: pd.DataFrame) -> None:
    assert (ep["n_mod"] >= 1).all(), "every episode must have at least one modality"
    expected_pattern = ep[[f"m_{k}" for k in MODS]].astype(int).astype(str).agg("".join, axis=1)
    assert (ep["pattern"] == expected_pattern).all(), "pattern must be the concatenation of m_mri,m_fdg,m_amy,m_tau"
    assert (ep["n_mod"] == ep[[f"m_{k}" for k in MODS]].sum(axis=1)).all()
    assert not (ep["pattern"] == "0000").any(), "pattern 0000 must be excluded"


def test_present_modality_has_date_and_gap_within_window(ep: pd.DataFrame) -> None:
    """For m_k == 1 the scan date must exist and |gap| <= window; for m_k == 0 both must be NaN.

    The >50%-missing-feature rule zeroes m_k *after* the merge, so the script explicitly nulls the
    date/gap of any absent modality. Every downstream analysis must gate scan-date/gap use on
    m_k == 1.
    """
    for k in MODS:
        present = ep[ep[f"m_{k}"] == 1]
        assert present[f"{k}_date"].notna().all(), f"{k}: present modality must have a scan date"
        assert present[f"{k}_gap_days"].notna().all(), f"{k}: present modality must have a gap"
        assert (present[f"{k}_gap_days"].abs() <= WINDOW).all(), f"{k}: |gap| must be <= {WINDOW} days"
        absent = ep[ep[f"m_{k}"] == 0]
        assert absent[f"{k}_date"].isna().all(), f"{k}: absent modality must have no scan date"
        assert absent[f"{k}_gap_days"].isna().all(), f"{k}: absent modality must have no gap"


def test_labels_in_range_and_consistent(ep: pd.DataFrame) -> None:
    assert set(ep["y"].unique()).issubset({0, 1, 2})
    assert (ep["y"] == ep["DIAGNOSIS"].map(CLASS_MAP)).all(), "y must match DIAGNOSIS (CN=0, MCI=1, Dementia=2)"


def test_no_duplicate_rid_date(ep: pd.DataFrame) -> None:
    dup = ep.duplicated(["RID", "dx_date"])
    assert not dup.any(), f"{int(dup.sum())} duplicated (RID, dx_date) rows"


def test_episode_id_unique(ep: pd.DataFrame) -> None:
    assert ep["episode_id"].is_unique
