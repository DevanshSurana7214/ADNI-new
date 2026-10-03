"""Stage 3 / Step 2 reproduction test.

Asserts that ``results/step02/joint_profile.csv`` (regenerated from the canonical
``docs/episodes.csv``) matches the provided reference ``../joint_profile.csv`` to within
1e-9 on the shared numeric columns.

PIPELINE_v2 §4 / Step 2.
"""
from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # adni-shortcut/
SUPPLIED = PROJECT_ROOT.parent / "joint_profile.csv"
GENERATED = PROJECT_ROOT / "results" / "step02" / "joint_profile.csv"
METRICS = PROJECT_ROOT / "results" / "step02" / "metrics.json"

NUMERIC_COLUMNS = ["n", "P_cm", "P_c", "P_m", "PMI", "R_sigmoid",
                   "w_indep", "P_c_given_m"]

TOL = 1e-9


@pytest.fixture(scope="module")
def ref() -> pd.DataFrame:
    assert SUPPLIED.is_file(), f"missing supplied reference {SUPPLIED}"
    ref = pd.read_csv(SUPPLIED, dtype={"DIAGNOSIS": str, "pattern": str})
    ref = ref.sort_values(["DIAGNOSIS", "pattern"]).reset_index(drop=True)
    return ref


@pytest.fixture(scope="module")
def gen() -> pd.DataFrame:
    assert GENERATED.is_file(), f"missing generated {GENERATED} — run scripts/adni_stage3_profile.py first"
    gen = pd.read_csv(GENERATED, dtype={"DIAGNOSIS": str, "pattern": str})
    gen = gen.sort_values(["DIAGNOSIS", "pattern"]).reset_index(drop=True)
    return gen


def test_files_exist(ref: pd.DataFrame, gen: pd.DataFrame) -> None:
    assert METRICS.is_file(), f"missing {METRICS}"
    assert list(ref.columns) == list(gen.columns), f"columns differ: ref {list(ref.columns)} vs gen {list(gen.columns)}"
    assert ref.shape == gen.shape, f"shape differs: ref {ref.shape} vs gen {gen.shape}"


def test_identity_columns_like_each_other(ref: pd.DataFrame, gen: pd.DataFrame) -> None:
    """DIAGNOSIS x pattern join keys must be the same set of cells in the same order."""
    assert (ref["DIAGNOSIS"] == gen["DIAGNOSIS"]).all()
    assert (ref["pattern"] == gen["pattern"]).all()


def test_numeric_columns_within_tolerance(ref: pd.DataFrame, gen: pd.DataFrame) -> None:
    """Every shared numeric column must match within TOL (1e-9)."""
    for c in NUMERIC_COLUMNS:
        a = ref[c].astype(float).values
        b = gen[c].astype(float).values
        diff = abs(a - b).max()
        if diff > TOL:
            bad = abs(a - b) > TOL
            detail = pd.DataFrame({c + "_ref": a, c + "_gen": b, "abs_diff": abs(a - b)})
            fail = detail[bad].head(20)
            pytest.fail(f"column {c} max abs diff {diff} > {TOL} on {int(bad.sum())} cells:\n{fail}")
        assert diff <= TOL, f"column {c} max abs diff {diff} > {TOL}"
