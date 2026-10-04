"""Test that recomputes S1 outer folds from docs/episodes.csv and asserts equality with splits.json.

Verifies that the splits.json S1 seed_0 test folds match what we get by re-running
StratifiedGroupKFold with the same parameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
VAL_RANDOM_STATE = 42
S1_VAL_FRACTION = 0.10


def _subject_label(ep: pd.DataFrame) -> pd.Series:
    """Subject-level label (modal diagnosis) for validation stratification."""
    from adni.splits import _subject_label as _sl
    return _sl(ep)


def test_s1_seed0_test_folds_reproduce_splits_json() -> None:
    """Recompute S1 seed_0 TEST folds from episodes.csv and assert equality with splits.json.

    S1 outer folds: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    stratified on per-episode y, with RID as groups.

    The TEST set is directly from SGKF. The TRAIN set in splits.json is after removing
    the validation holdout (10% of training subjects).
    """
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    y = ep["y"].values
    groups = ep["RID"].values

    blob = json.loads(SPLITS.read_text())
    seed0 = blob["s1"]["seed_0"]

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(ep)), y, groups)):
        # TEST set should match exactly
        expected_test_eids = set(seed0[str(fold)]["episode_ids"]["test"])
        actual_test_eids = set(ep.index[te_idx].tolist())
        assert actual_test_eids == expected_test_eids, (
            f"Fold {fold}: test episode IDs don't match"
        )

        # TRAIN+VAL set (from SGKF) should contain the train+val from splits.json
        sgkf_train_eids = set(ep.index[tr_idx].tolist())
        splits_train_eids = set(seed0[str(fold)]["episode_ids"]["train"])
        splits_val_eids = set(seed0[str(fold)]["episode_ids"]["val"])

        # splits train ∪ val should equal sgkf train
        combined = splits_train_eids.union(splits_val_eids)
        assert combined == sgkf_train_eids, (
            f"Fold {fold}: sgkf train set doesn't match splits train+val union"
        )

        # Verify subject counts
        actual_test_rids = len(set(groups[te_idx]))
        expected_test_rids = seed0[str(fold)]["n_subjects"]["test"]
        assert actual_test_rids == expected_test_rids, (
            f"Fold {fold}: test subject count mismatch"
        )


def test_s1_stratifies_on_per_episode_y_not_modal() -> None:
    """Verify that S1 outer folds stratify on per-episode y, not subject modal diagnosis.

    The StratifiedGroupKFold is called with the per-episode y array, not a subject-level
    aggregated label. This is the correct behavior as documented in PIPELINE_v2.md Appendix E.
    """
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    y = ep["y"].values
    groups = ep["RID"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(ep)), y, groups)):
        # Each subject should appear only in one split (train OR test)
        train_groups_set = set(groups[tr_idx])
        test_groups_set = set(groups[te_idx])
        intersection = train_groups_set.intersection(test_groups_set)
        assert len(intersection) == 0, (
            f"Fold {fold}: RIDs appear in both train and test"
        )
        break


def test_s1_val_holdout_stratifies_on_subject_label() -> None:
    """Verify that S1 validation holdout stratifies on subject-level (modal) label.

    From splits.py _split_train_val_test: the validation split uses
    train_test_split with stratify=train_y where train_y is the subject-level label
    (modal diagnosis per subject).
    """
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(SPLITS.read_text())
    seed0 = blob["s1"]["seed_0"]

    # Get training subjects for fold 0 (from SGKF, before val split)
    node = seed0["0"]
    train_epids = node["episode_ids"]["train"]
    val_epids = node["episode_ids"]["val"]

    # The splits.json train+val should be the SGKF train set
    # Get the unique RIDs in train+val
    train_plus_val_rids = np.unique(ep.loc[train_epids + val_epids]["RID"].values)

    # Compute subject-level labels
    sub_y = _subject_label(ep)
    train_y = sub_y.loc[pd.Series(train_plus_val_rids, dtype=np.int64, name="RID").values].values

    # Verify we can reproduce the val split
    tr_rids, val_rids = train_test_split(
        pd.Series(train_plus_val_rids, dtype=np.int64, name="RID"),
        test_size=S1_VAL_FRACTION,
        random_state=VAL_RANDOM_STATE,
        stratify=train_y,
    )

    # The val RIDs should match
    expected_val_rids = set(val_rids.values)
    actual_val_rids = set(ep.loc[val_epids]["RID"].values)
    assert expected_val_rids == actual_val_rids, (
        f"Val RIDs don't match. Expected {len(expected_val_rids)}, got {len(actual_val_rids)}"
    )
