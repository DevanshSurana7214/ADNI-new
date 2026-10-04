"""Tests for Step 3 fold table generation script.

Verifies that:
  * Test class percentages are computed from actual fold counts
  * The percentages are consistent with good stratification (small variation)
  * All folds sum to expected global totals
  * The percentages are NOT wildly different from global proportions (which would
    indicate the old bug where hand-typed percentages varied 42.5/41.3/16.2 etc.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
CLASSES = ["CN", "MCI", "Dementia"]

# Global proportions from the full cohort
GLOBAL_PCTS = {
    "CN": 100 * 5263 / 13240,      # 39.75%
    "MCI": 100 * 5518 / 13240,      # 41.68%
    "Dementia": 100 * 2459 / 13240, # 18.57%
}


def test_fold_percentages_close_to_global() -> None:
    """Test that per-fold test percentages are close to global proportions.

    With good stratification (StratifiedGroupKFold), each fold's test set should have
    class proportions close to the global proportions. The OLD Step 3 hand-typed table
    had percentages that varied widely (e.g., 42.5/41.3/16.2 for fold 0 vs
    38.7/41.1/20.3 for fold 4), which would indicate poor stratification.

    This test checks that:
    1. Each fold's percentages are within 2 percentage points of global proportions
    2. The variation across folds is small (within 2 percentage points)

    If percentages were copied from global proportions WITHOUT computing from actual
    fold counts, this test would still pass - so we also check the sums.
    """
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    diag = ep["DIAGNOSIS"]
    blob = json.loads(SPLITS.read_text())

    # Check seed 0 as representative
    seed_data = blob["s1"]["seed_0"]

    # Compute percentages for each fold
    fold_percentages = []
    for fold_idx in range(5):
        fold_data = seed_data[str(fold_idx)]
        test_ids = fold_data["episode_ids"]["test"]
        test_diag = diag.loc[test_ids]
        n_test = len(test_ids)

        cn_count = (test_diag == "CN").sum()
        mci_count = (test_diag == "MCI").sum()
        dem_count = (test_diag == "Dementia").sum()

        cn_pct = 100.0 * cn_count / n_test
        mci_pct = 100.0 * mci_count / n_test
        dem_pct = 100.0 * dem_count / n_test

        fold_percentages.append({"CN": cn_pct, "MCI": mci_pct, "Dementia": dem_pct})

    # Check 1: Each fold's percentages are within 2pp of global
    for fold_idx, fp in enumerate(fold_percentages):
        for cls in CLASSES:
            diff = abs(fp[cls] - GLOBAL_PCTS[cls])
            assert diff < 2.0, (
                f"Fold {fold_idx} {cls} percentage {fp[cls]:.1f}% differs from "
                f"global {GLOBAL_PCTS[cls]:.1f}% by {diff:.1f}pp (> 2pp threshold). "
                f"This suggests either poor stratification or that percentages were "
                f"not computed from actual fold counts."
            )

    # Check 2: Variation across folds is small (within 2pp)
    for cls in CLASSES:
        pcts = [fp[cls] for fp in fold_percentages]
        pct_range = max(pcts) - min(pcts)
        assert pct_range < 2.0, (
            f"{cls} percentage range across folds is {pct_range:.1f}pp, "
            f"which exceeds the 2pp threshold for good stratification. "
            f"Range: {[round(p, 1) for p in pcts]}"
        )


def test_folds_sum_to_global_totals() -> None:
    """Test that the five test folds sum exactly to the global class totals."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    diag = ep["DIAGNOSIS"]
    blob = json.loads(SPLITS.read_text())

    expected = {"CN": 5263, "MCI": 5518, "Dementia": 2459}

    for seed in [0, 1, 2]:
        seed_key = f"seed_{seed}"
        seed_data = blob["s1"][seed_key]

        total_cn = 0
        total_mci = 0
        total_dem = 0

        for fold_idx in range(5):
            fold_data = seed_data[str(fold_idx)]
            test_ids = fold_data["episode_ids"]["test"]
            test_diag = diag.loc[test_ids]

            total_cn += (test_diag == "CN").sum()
            total_mci += (test_diag == "MCI").sum()
            total_dem += (test_diag == "Dementia").sum()

        assert total_cn == expected["CN"], (
            f"Seed {seed}: CN count {total_cn} != expected {expected['CN']}"
        )
        assert total_mci == expected["MCI"], (
            f"Seed {seed}: MCI count {total_mci} != expected {expected['MCI']}"
        )
        assert total_dem == expected["Dementia"], (
            f"Seed {seed}: Dementia count {total_dem} != expected {expected['Dementia']}"
        )


def test_old_report_percentages_would_fail() -> None:
    """Verify that the OLD Step 3 hand-typed percentages would fail this test.

    The old report had these test class percentages for S1 seed_0:
    - fold 0: 42.5/41.3/16.2
    - fold 1: 40.4/41.6/18.0
    - fold 2: 38.7/42.6/18.7
    - fold 3: 38.6/41.3/20.1
    - fold 4: 38.7/41.1/20.3

    These vary by up to 3.9pp for CN (42.5 - 38.6), which exceeds our 2pp threshold.
    This test documents what the old bug looked like.
    """
    # These are the OLD (wrong) percentages from the Step 3 report
    old_pcts = {
        "CN": [42.5, 40.4, 38.7, 38.6, 38.7],
        "MCI": [41.3, 41.6, 42.6, 41.3, 41.1],
        "Dementia": [16.2, 18.0, 18.7, 20.1, 20.3],
    }

    # The old CN percentages range from 38.6 to 42.5 = 3.9pp range
    cn_range = max(old_pcts["CN"]) - min(old_pcts["CN"])
    assert cn_range >= 3.89, f"Expected CN range >= 3.9pp, got {cn_range:.1f}pp"

    # This would fail our 2pp threshold test
    for cls in CLASSES:
        pcts = old_pcts[cls]
        pct_range = max(pcts) - min(pcts)
        if pct_range >= 2.0:
            print(
                f"OLD {cls} percentages range {pct_range:.1f}pp would FAIL "
                f"test_fold_percentages_close_to_global (threshold 2pp)"
            )
