"""Split-structure tests (PIPELINE_v2 §7).

Loads results/step03/splits.json and asserts:
* S1 (3 seeds × 5 folds): for every (seed, fold) train/val/test are pairwise disjoint on
  episode_id AND on RID; their union is all episodes; every RID is in exactly one of train+val /
  test.
* S2: 1,784 training-side and 1,361 test-side subjects, zero overlap between train-side and test.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
S2_TRAIN_SUBJECTS_EXPECTED = 1784
S2_TEST_SUBJECTS_EXPECTED = 1361


@pytest.fixture(scope="module")
def ep() -> pd.DataFrame:
    assert EPISODES.is_file()
    df = pd.read_csv(EPISODES, dtype={"pattern": str})
    return df.set_index("episode_id")


@pytest.fixture(scope="module")
def splits() -> dict:
    assert SPLITS.is_file()
    return json.loads(SPLITS.read_text())


def _rid_sets(ep: pd.DataFrame, ids: dict) -> dict[str, set]:
    rid_of = ep["RID"]
    return {k: set(rid_of.loc[list(v)].values) for k, v in ids.items()}


def test_s1_disjoint_and_union_all(ep: pd.DataFrame, splits: dict) -> None:
    s1 = splits["s1"]
    all_eids = set(ep.index)
    all_rids = set(ep["RID"].values)
    for sk in sorted(s1):
        for fk, f in s1[sk].items():
            ids = f["episode_ids"]
            sets = {k: set(ids[k]) for k in ("train", "val", "test")}
            assert sets["train"] & sets["val"] == set(), f"{sk}/{fk}: train-val overlap"
            assert sets["train"] & sets["test"] == set(), f"{sk}/{fk}: train-test overlap"
            assert sets["val"] & sets["test"] == set(), f"{sk}/{fk}: val-test overlap"
            union = sets["train"] | sets["val"] | sets["test"]
            assert union == all_eids, f"{sk}/{fk}: union != all episodes (miss {len(all_eids - union)}, extra {len(union - all_eids)})"
            rid_sets = _rid_sets(ep, ids)
            tr_val_rids = rid_sets["train"] | rid_sets["val"]
            assert tr_val_rids & rid_sets["test"] == set(), f"{sk}/{fk}: RID overlap train+val / test"
            assert tr_val_rids | rid_sets["test"] == all_rids, f"{sk}/{fk}: not all RIDs covered"


def test_s1_every_rid_in_exactly_one_test_fold_per_seed(ep: pd.DataFrame, splits: dict) -> None:
    s1 = splits["s1"]
    all_rids = set(ep["RID"].values)
    for sk in sorted(s1):
        # A subject must appear in exactly one test fold for this seed (across the 5 folds).
        test_rids_by_fold: dict[int, set] = {}
        for fk, f in s1[sk].items():
            rid_sets = _rid_sets(ep, f["episode_ids"])
            test_rids_by_fold[int(fk)] = rid_sets["test"]
        covered: set = set()
        for fk, rset in test_rids_by_fold.items():
            overlap = rset & covered
            assert not overlap, f"{sk}: RID in multiple test folds {sorted(overlap)[:5]}"
            covered |= rset
        assert covered == all_rids, f"{sk}: not all RIDs appear in some test fold (miss {len(all_rids - covered)})"


def test_s2_counts_and_disjointness(ep: pd.DataFrame, splits: dict) -> None:
    s2 = splits["s2"]
    assert s2["n_subjects"]["train"] + s2["n_subjects"]["val"] == s2["n_subjects_pre_val_split"]["train_plus_val"], (
        "S2 post-val train+val subjects must equal pre-val train+val subjects")
    assert s2["n_subjects"]["train"] + s2["n_subjects"]["val"] + s2["n_subjects"]["test"] == (
        s2["n_subjects_pre_val_split"]["train_plus_val"] + s2["n_subjects_pre_val_split"]["test"]
    ), "S2 subject counts must add up"
    assert s2["n_subjects_pre_val_split"]["train_plus_val"] == S2_TRAIN_SUBJECTS_EXPECTED, (
        f"S2 training-side subjects {s2['n_subjects_pre_val_split']['train_plus_val']} != {S2_TRAIN_SUBJECTS_EXPECTED}")
    assert s2["n_subjects_pre_val_split"]["test"] == S2_TEST_SUBJECTS_EXPECTED, (
        f"S2 test-side subjects {s2['n_subjects_pre_val_split']['test']} != {S2_TEST_SUBJECTS_EXPECTED}")

    ids = s2["episode_ids"]
    sets = {k: set(ids[k]) for k in ("train", "val", "test")}
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    rid_sets = _rid_sets(ep, ids)
    train_side_rids = rid_sets["train"] | rid_sets["val"]
    assert train_side_rids & rid_sets["test"] == set(), "S2 train-side ∩ test RID overlap"
    assert train_side_rids | rid_sets["test"] == set(ep["RID"].values), "S2 does not cover all RIDs"
    assert len(train_side_rids) == S2_TRAIN_SUBJECTS_EXPECTED, "S2 train-side RID count"
    assert len(rid_sets["test"]) == S2_TEST_SUBJECTS_EXPECTED, "S2 test RID count"


def test_s1_fold_counts_reasonable(ep: pd.DataFrame, splits: dict) -> None:
    s1 = splits["s1"]
    for sk in sorted(s1):
        for fk, f in s1[sk].items():
            ns = f["n_subjects"]
            ne = f["n_episodes"]
            assert ns["train"] > 0 and ns["val"] > 0 and ns["test"] > 0, f"{sk}/{fk}: empty split"
            # val is ~10% of pre-val train+val subjects
            pre = f["n_subjects_pre_val_split"]["train_plus_val"]
            expected_val = round(pre * 0.10)
            assert abs(ns["val"] - expected_val) <= 3, (
                f"{sk}/{fk}: val={ns['val']} not ~10% of pre-val {pre} (expected {expected_val})")
            assert ne["train"] + ne["val"] + ne["test"] == len(ep), f"{sk}/{fk}: episode count mismatch"
