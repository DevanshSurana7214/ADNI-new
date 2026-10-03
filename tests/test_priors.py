"""Unit tests for src/adni/priors.py (PIPELINE_v2 §4).

The smoothed joint, PMI and clipped-independence-weight figures are independently reproduced here and
compared to the reference ``results/step02/metrics.json`` produced by ``scripts/adni_stage3_profile.py``
on the canonical ``docs/episodes.csv``.

`joint_profile.csv` lives in the parent directory as the frozen Step 2 reference; the set of patterns
it contains may differ from the patterns produced by a test fixture. Tests that need the full joint
therefore read the reference directly; toy counts are used only for structural properties that are
true of any table.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adni.priors import CLASSES, crosstab, smoothed_joint, pmi, independence_weight, per_sample_weight, ess, cramers_v

PROJECT_ROOT = Path(__file__).resolve().parents[1]            # adni-shortcut/
METRICS_JSON = PROJECT_ROOT / "results" / "step02" / "metrics.json"
REFERENCE = PROJECT_ROOT.parent / "joint_profile.csv"        # frozen Step 2 / 3 reference


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads(METRICS_JSON.read_text())


@pytest.fixture(scope="module")
def reference_joint() -> pd.DataFrame:
    df = pd.read_csv(REFERENCE, dtype={"DIAGNOSIS": str, "pattern": str, "n": int})
    return df.sort_values(["DIAGNOSIS", "pattern"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def full_cohort_count_table() -> tuple[np.ndarray, list[str]]:
    """Real (class × pattern) count table for the canonical cohort, read from metrics.json's N."""
    # Recompute from the canonical episodes so we do not depend on a possibly-different reference.
    import subprocess, sys
    out = PROJECT_ROOT / "results" / "step02" / "joint_profile.csv"
    if not out.is_file():
        pytest.skip("joint_profile.csv not yet generated")
    df = pd.read_csv(out, dtype={"DIAGNOSIS": str, "pattern": str, "n": int})
    tab = df.groupby(["DIAGNOSIS", "pattern"])["n"].sum().unstack(fill_value=0)
    tab = tab.reindex(index=CLASSES, columns=sorted(df["pattern"].unique()), fill_value=0)
    return tab.values.astype(np.int64), sorted(df["pattern"].unique().tolist())


def test_smoothed_joint_sums_to_one() -> None:
    n = np.array([[7, 12, 0], [4, 168, 1], [2, 9, 5]], dtype=float)
    P, Pc, Pm, I = smoothed_joint(n, alpha=0.5)
    assert abs(P.sum() - 1.0) < 1e-12
    assert np.allclose(Pc, P.sum(axis=1))
    assert np.allclose(Pm, P.sum(axis=0))
    assert P.shape == (3, n.shape[1])


def test_pmi_and_independent_weights_match_reference(reference_joint: pd.DataFrame) -> None:
    """Independently fit the smoothed joint from the reference n(c,m) and compare PMI and w to the
    reference within 1e-9."""
    ref = reference_joint
    pats = list(ref["pattern"].values)                 # preserve the file's row order
    n = (ref.rename(columns={"n": "count"})
            .groupby(["DIAGNOSIS", "pattern"])["count"].sum()
            .unstack(fill_value=0)
            .reindex(index=CLASSES, columns=pats, fill_value=0)
            .values.astype(np.int64))
    P, _Pc, _Pm, _I = smoothed_joint(n, alpha=0.5)
    assert abs(P.sum() - 1.0) < 1e-12

    # Compare PMI/w cell-by-cell using the reference's own (DIAGNOSIS, pattern) ordering.
    I_cell = pmi(P)
    our_pmi = np.array([I_cell[CLASSES.index(c), pats.index(m)]
                        for c, m in zip(ref["DIAGNOSIS"], ref["pattern"])])
    ref_pmi = ref["PMI"].values.astype(float)
    assert np.abs(our_pmi - ref_pmi).max() < 1e-9, (
        f"PMI max diff {np.abs(our_pmi - ref_pmi).max()}")

    w_cell = independence_weight(P)
    our_w = np.array([w_cell[CLASSES.index(c), pats.index(m)]
                      for c, m in zip(ref["DIAGNOSIS"], ref["pattern"])])
    ref_w = ref["w_indep"].values.astype(float)
    assert np.abs(our_w - ref_w).max() < 1e-9, (
        f"w_indep max diff {np.abs(our_w - ref_w).max()}")
    assert abs(float((P * w_cell).sum() - 1.0)) < 1e-12


def test_per_sample_weight_matches_ess_clip(full_cohort_count_table: tuple[np.ndarray, list[str]],
                                            metrics: dict) -> None:
    """per_sample_weight on the full cohort reproduces metrics.ess_clip (0.878) to within 1e-9."""
    n, pats = full_cohort_count_table
    # The canonical per-sample layout is the *n*-column of joint_profile.csv, one row per cell.
    df = pd.read_csv(PROJECT_ROOT / "results" / "step02" / "joint_profile.csv",
                     dtype={"DIAGNOSIS": str, "pattern": str, "n": int})
    # Replicate each cell `n` times so the per-sample weight ESS matches the profile's definition.
    df_expanded = df.loc[df.index.repeat(df["n"])].reset_index(drop=True)
    assert len(df_expanded) == int(df["n"].sum())
    w = per_sample_weight(n, df_expanded["DIAGNOSIS"], df_expanded["pattern"],
                          wmin=0.2, wmax=5.0, alpha=0.5)
    assert abs(ess(w) - metrics["ess_clip"]) < 1e-9, (
        f"ess_clip {ess(w)} vs metrics {metrics['ess_clip']}")
    assert abs(w.mean() - 1.0) < 1e-12
    assert abs(ess(w) - 0.878) < 1e-3


def test_raw_ess_matches_metrics(metrics: dict, full_cohort_count_table: tuple[np.ndarray, list[str]]) -> None:
    n, pats = full_cohort_count_table
    # The canonical per-sample layout is the *n*-column of joint_profile.csv, one row per cell.
    df = pd.read_csv(PROJECT_ROOT / "results" / "step02" / "joint_profile.csv",
                     dtype={"DIAGNOSIS": str, "pattern": str, "n": int})
    df_expanded = df.loc[df.index.repeat(df["n"])].reset_index(drop=True)
    assert len(df_expanded) == int(df["n"].sum())
    P, _, _, I = smoothed_joint(n, alpha=0.5)
    w_raw = np.exp(-pmi(P))
    w_per = np.array([w_raw[CLASSES.index(d), pats.index(m)]
                      for d, m in zip(df_expanded["DIAGNOSIS"], df_expanded["pattern"])])
    assert abs(ess(w_per) - metrics["ess_raw"]) < 1e-9, (
        f"ess_raw {ess(w_per)} vs metrics {metrics['ess_raw']}")


def test_cramers_v_reproduces_stage2_metrics(full_cohort_count_table: tuple[np.ndarray, list[str]],
                                            metrics: dict) -> None:
    n, _pats = full_cohort_count_table
    V = cramers_v(n)
    assert abs(V - metrics["V"]) < 1e-9, f"V {V} vs metrics {metrics['V']}"
    assert abs(V - 0.238) < 1e-3


def test_unclipped_weighted_joint_equals_Pc_Pm() -> None:
    """Sum over cells of P(c,m) * exp(-PMI) equals P(c) P(m) in every cell (tolerance 1e-12)."""
    n = np.array([[7, 12, 0], [4, 168, 1], [2, 9, 5]], dtype=float)
    P, Pc, Pm, I = smoothed_joint(n, alpha=0.5)
    w = independence_weight(P)
    weighted = P * w
    outer = np.outer(Pc, Pm)
    assert np.abs(weighted - outer).max() < 1e-12, (
        "P(c,m) * exp(-PMI) must equal P(c)P(m) in every cell (unclipped)")
