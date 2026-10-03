"""ADNI split builder (PIPELINE_v2 §7).

S1 (primary): StratifiedGroupKFold(n_splits=5, shuffle=True) over subjects (group = RID),
repeated for seeds [0, 1, 2]; within each training fold 10 % of the *training* subjects form the
validation set (grouped on RID, stratified on the subject-level label y, fixed seed 42).

S2 (protocol shift): subjects first enrolled in ADNI1 / ADNI-GO / ADNI2 → train+val side;
subjects first enrolled in ADNI3 / ADNI4 → test side; validation = 10 % of the training subjects
(grouped, stratified on y, seed 42).

All splits are subject-grouped: every RID appears in exactly one of train(val)/test for a given
(seed, fold) — and for S2, in exactly one of train-side / test-side.

Outputs: results/step03/splits.json (one object per (seed, fold) for S1, plus one S2 object),
with episode_ids for train/val/test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

EPISODES_COLUMNS = [
    "episode_id", "RID", "ORIGPROT", "COLPROT", "VISCODE2", "dx_date",
    "DIAGNOSIS", "y", "mri_date", "m_mri", "mri_gap_days", "fdg_date",
    "m_fdg", "fdg_gap_days", "amy_date", "m_amy", "amy_gap_days",
    "tau_date", "m_tau", "tau_gap_days", "pattern", "n_mod", "mri_dropped_lowfeat",
    "fdg_dropped_lowfeat", "amy_dropped_lowfeat", "tau_dropped_lowfeat",
]
VAL_RANDOM_STATE = 42
S1_SPLITS = 5
S1_SEEDS = [0, 1, 2]
S1_TEST_FRACTION = 1 / S1_SPLITS          # 20 % test
S1_VAL_FRACTION = 0.10                     # 10 % of training subjects → val
S2_TRAIN_PROTOCOLS = ["ADNI1", "ADNIGO", "ADNI2"]
S2_TEST_PROTOCOLS = ["ADNI3", "ADNI4"]


def _build_subject_label(ep: pd.DataFrame) -> pd.Series:
    """Subject-level label for stratification.

    A subject can have episodes with different diagnoses across visits (631 of 3,145 subjects have
    > 1 distinct y). We use the subject's *modal* diagnosis (most frequent y); ties are broken by
    the earliest episode's diagnosis. This gives every subject a single label suitable for
    StratifiedGroupKFold and for the validation holdout stratification.
    """
    def _mode(g: pd.DataFrame) -> int:
        vc = g["y"].value_counts()
        top = vc[vc == vc.max()].index
        if len(top) == 1:
            return int(top[0])
        return int(g.sort_values("dx_date")["y"].iloc[0])
    return ep.groupby("RID").apply(_mode, include_groups=False).astype(np.int64)


def _subject_label(ep: pd.DataFrame) -> pd.Series:
    return _build_subject_label(ep)


def _split_train_val_test(
    ep: pd.DataFrame,
    train_rids: np.ndarray,
    test_rids: np.ndarray,
    val_fraction: float,
    seed_val: int,
) -> dict[str, Any]:
    """From the sets of training/test *subject* IDs (unique RIDs, no duplicates),
    build train/val/test *episode* id arrays and the unique subject counts per split.

    Validation is drawn from the training subjects only (10 %, grouped on RID, stratified on the
    subject-level label, fixed seed 42).

    Parameters
    ----------
    train_rids : unique training subject RIDs (post-SGKF, before val split).
    test_rids : unique test subject RIDs.
    """
    sub_y = _subject_label(ep)

    train_rids_u = np.unique(train_rids)
    train_y = sub_y.loc[pd.Series(train_rids_u, dtype=np.int64, name="RID").values].values.astype(np.int64)

    tr, val = train_test_split(
        pd.Series(train_rids_u, dtype=np.int64, name="RID"),
        test_size=val_fraction,
        random_state=seed_val,
        stratify=train_y,
    )
    tr = np.asarray(tr, dtype=np.int64)
    val = np.asarray(val, dtype=np.int64)

    def episode_ids(rids: np.ndarray) -> np.ndarray:
        mask = np.isin(ep["RID"].values, rids)
        return ep.loc[mask, "episode_id"].to_numpy(copy=True)

    return {
        "train": episode_ids(tr),
        "val": episode_ids(val),
        "test": episode_ids(test_rids),
        "train_subjects": int(len(tr)),
        "val_subjects": int(len(val)),
        "test_subjects": int(len(np.unique(test_rids))),
        "train_plus_val_subjects_pre_split": int(len(train_rids_u)),
    }


def build_s1(ep: pd.DataFrame) -> dict[str, Any]:
    """S1: 3 seeds × 5 folds, subject-grouped, stratified on y, with 10 % val holdout."""
    y = ep["y"].values
    groups = ep["RID"].values
    out: dict[str, Any] = {"s1": {}, "s2": None}

    for seed in S1_SEEDS:
        sgkf = StratifiedGroupKFold(n_splits=S1_SPLITS, shuffle=True,
                                     random_state=seed)
        folds: dict[str, Any] = {}
        for fold, (tr_idx, te_idx) in enumerate(sgkf.split(np.zeros(len(ep)), y, groups)):
            train_rids = np.unique(groups[tr_idx])   # unique subject RIDs, before val split
            test_rids = np.unique(groups[te_idx])
            splits = _split_train_val_test(ep, train_rids, test_rids,
                                           S1_VAL_FRACTION, VAL_RANDOM_STATE)
            folds[str(fold)] = {
                "seed": seed,
                "fold": fold,
                "n_subjects": {
                    "train": splits["train_subjects"],
                    "val": splits["val_subjects"],
                    "test": splits["test_subjects"],
                },
                "n_subjects_pre_val_split": {
                    "train_plus_val": splits["train_plus_val_subjects_pre_split"],
                    "test": splits["test_subjects"],
                },
                "n_episodes": {
                    k: int(len(v)) for k, v in splits.items() if k in ("train", "val", "test")
                },
                "episode_ids": {
                    k: v.tolist() for k, v in splits.items() if k in ("train", "val", "test")
                },
            }
        out["s1"][f"seed_{seed}"] = folds
    return out


def build_s2(ep: pd.DataFrame) -> dict[str, Any]:
    """S2: protocol shift. Train+val = first enrolment ORIGPROT in ADNI1/GO/ADNI2,
    test = ADNI3/4. Validation = 10 % of train subjects (grouped, stratified, seed 42)."""
    first_per_sub = (ep.sort_values("dx_date")
                     .groupby("RID")
                     .agg(first_ORIGPROT=("ORIGPROT", "first"))
                     .reset_index())
    sub_y = _subject_label(ep)
    first_per_sub = first_per_sub.set_index("RID")

    train_rids = first_per_sub.index[first_per_sub["first_ORIGPROT"].isin(S2_TRAIN_PROTOCOLS)].values
    test_rids = first_per_sub.index[first_per_sub["first_ORIGPROT"].isin(S2_TEST_PROTOCOLS)].values
    # S2 train_rids[test_rids] are already unique (one row per subject), but ensure it here.
    train_rids = np.unique(train_rids)
    test_rids = np.unique(test_rids)

    splits = _split_train_val_test(ep, train_rids, test_rids,
                                   S1_VAL_FRACTION, VAL_RANDOM_STATE)

    return {
        "s2": {
            "n_subjects": {
                "train": splits["train_subjects"],
                "val": splits["val_subjects"],
                "test": splits["test_subjects"],
            },
            "n_subjects_pre_val_split": {
                "train_plus_val": splits["train_plus_val_subjects_pre_split"],
                "test": splits["test_subjects"],
            },
            "n_episodes": {
                k: int(len(v)) for k, v in splits.items() if k in ("train", "val", "test")
            },
            "episode_ids": {
                k: v.tolist() for k, v in splits.items() if k in ("train", "val", "test")
            },
            "train_protocols": S2_TRAIN_PROTOCOLS,
            "test_protocols": S2_TEST_PROTOCOLS,
        }
    }


def make_splits(episodes_path: Path, out_path: Path) -> dict[str, Any]:
    """Build and write the S1 + S2 splits."""
    ep = pd.read_csv(episodes_path, dtype={"pattern": str})
    for c in EPISODES_COLUMNS:
        if c not in ep.columns:
            raise KeyError(f"episodes.csv missing required column {c!r}")

    blob = build_s1(ep)
    blob.update(build_s2(ep))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2))
    return blob
