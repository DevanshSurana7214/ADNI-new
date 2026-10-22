"""Splits for PIPELINE v6 (section 3.1).

R1: every split is subject-grouped (RID).  R2/R3: preprocessing and all tuning
are fit on TRAIN only; within each training fold we carve a subject-grouped
validation split.

- FULL: reuse the frozen S1 folds from results/step03/splits.json (3 seeds x 5 folds).
- BASELINE / DIAL: StratifiedGroupKFold (5 folds x n_repeats), stratify on y,
  group on RID, different seed per repeat; within each training fold carve a
  subject-grouped val_frac validation split.
- Phase-held-out: train on {ADNI1, ADNIGO, ADNI2}, test on {ADNI3, ADNI4},
  remove any subject who has an episode in TEST, validation = grouped split of train.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RID_COL = "RID"
Y_COL = "y"


# ---------------------------------------------------------------------------
# frozen S1 folds
# ---------------------------------------------------------------------------

@dataclass
class S1Fold:
    seed: int
    fold: int
    train_rids: np.ndarray
    val_rids: np.ndarray
    test_rids: np.ndarray
    train_episode_ids: np.ndarray
    val_episode_ids: np.ndarray
    test_episode_ids: np.ndarray


def load_s1_folds(splits_path: Path, seed_col: str = "s1") -> dict[int, list[S1Fold]]:
    """Return {seed: [S1Fold for fold 0..4]}."""
    raw = json.loads(splits_path.read_text())
    col = raw[seed_col]
    out: dict[int, list[S1Fold]] = {}
    for key, fold_dict in col.items():
        seed = int(fold_dict["seed"])
        folds: list[S1Fold] = []
        for fk, fv in fold_dict.items():
            if not isinstance(fv, dict) or "episode_ids" not in fv:
                continue
            ep_ids = fv["episode_ids"]
            train_ids = np.array([int(x) for x in ep_ids["train"]], dtype=np.int64)
            val_ids = np.array([int(x) for x in ep_ids["val"]], dtype=np.int64)
            test_ids = np.array([int(x) for x in ep_ids["test"]], dtype=np.int64)
            folds.append(S1Fold(
                seed=seed,
                fold=int(fk),
                train_rids=np.array([], dtype=np.int64),  # not stored in s1 file; derived if needed
                val_rids=np.array([], dtype=np.int64),
                test_rids=np.array([], dtype=np.int64),
                train_episode_ids=train_ids,
                val_episode_ids=val_ids,
                test_episode_ids=test_ids,
            ))
        out[seed] = folds
    return out


# ---------------------------------------------------------------------------
# subject-grouped CV folds with val carve
# ---------------------------------------------------------------------------

@dataclass
class CVFold:
    repeat: int
    fold: int
    seed: int
    train_rids: np.ndarray
    val_rids: np.ndarray
    test_rids: np.ndarray
    train_episode_ids: np.ndarray
    val_episode_ids: np.ndarray
    test_episode_ids: np.ndarray


def _ids_for_rids(ep_df: pd.DataFrame, rids: np.ndarray) -> np.ndarray:
    rids = np.asarray(rid for rid in rids if rid in ep_df.index
                 ) if False else np.asarray([int(r) for r in rids])
    mask = ep_df.index.isin(rids)
    return ep_df.index.to_numpy()[mask]


def _val_carve(train_rids: np.ndarray, y_by_rid: Mapping[int, int],
               val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split TRAIN rids into train/val, subject-grouped, stratified on y by rid."""
    rids = np.array([int(r) for r in train_rids], dtype=np.int64)
    y = np.array([y_by_rid[int(r)] for r in rids], dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=max(2, int(round(1.0 / val_frac))), shuffle=True,
                                 random_state=seed)
    # Use the first split as the val carve.
    for train_idx, val_idx in sgkf.split(np.zeros(len(rids)), y, groups=rids):
        return rids[train_idx], rids[val_idx]
    # Fallback: if only one group per class, put everything in train.
    return rids, np.array([], dtype=np.int64)


def build_subject_folds(
    ep_df: pd.DataFrame,
    n_folds: int,
    n_repeats: int,
    val_frac: float,
    base_seed: int = 0,
) -> list[CVFold]:
    """Build n_repeats x n_folds subject-grouped folds with a val carve per training fold."""
    rids = np.array(sorted(ep_df.index.to_numpy()), dtype=np.int64)
    y = ep_df.loc[rids, Y_COL].to_numpy(dtype=int)
    # y by rid: take the label of the subject's earliest episode (by dx_date then VISCODE2).
    y_by_rid: dict[int, int] = {}
    order: list[int] = []
    for rid in rids:
        sub = ep_df.loc[rid].copy()
        sub["_dx"] = pd.to_datetime(sub["dx_date"], errors="coerce")
        sub = sub.sort_values(["_dx", "VISCODE2"], na_position="last")
        y_by_rid[int(rid)] = int(sub.iloc[0][Y_COL])
        order.append(int(rid))
    rids = np.array(order, dtype=np.int64)
    y = np.array([y_by_rid[int(r)] for r in rids], dtype=int)

    folds: list[CVFold] = []
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    for repeat in range(n_repeats):
        sgkf.random_state = base_seed + repeat + 1
        for fold_idx, (train_full_idx, test_idx) in enumerate(sgkf.split(np.zeros(len(rids)), y, groups=rids)):
            train_full_rids = rids[train_full_idx]
            test_rids = rids[test_idx]
            train_rids, val_rids = _val_carve(train_full_rids, y_by_rid, val_frac,
                                               seed=base_seed + repeat * 1000 + fold_idx)
            folds.append(CVFold(
                repeat=repeat,
                fold=fold_idx,
                seed=base_seed + repeat,
                train_rids=train_rids,
                val_rids=val_rids,
                test_rids=test_rids,
                train_episode_ids=_ids_for_rids(ep_df, train_rids),
                val_episode_ids=_ids_for_rids(ep_df, val_rids),
                test_episode_ids=_ids_for_rids(ep_df, test_rids),
            ))
    return folds


# ---------------------------------------------------------------------------
# phase-held-out split (section 8.3)
# ---------------------------------------------------------------------------

@dataclass
class PhaseHeldOutSplit:
    train_rids: np.ndarray
    val_rids: np.ndarray
    test_rids: np.ndarray
    train_episode_ids: np.ndarray
    val_episode_ids: np.ndarray
    test_episode_ids: np.ndarray
    train_phases: set[str]
    test_phases: set[str]


def build_phase_held_out(
    ep_df: pd.DataFrame,
    train_phases: Sequence[str],
    test_phases: Sequence[str],
    val_frac: float,
    seed: int = 0,
) -> PhaseHeldOutSplit:
    """Train on train_phases, test on test_phases. Remove subjects with any test episode from train."""
    test_mask = ep_df["COLPROT"].isin(test_phases).to_numpy(dtype=bool)
    test_rids_all = np.unique(ep_df.index[test_mask].to_series().map(lambda x: int(ep_df.loc[x, "RID"]))
                               if False else ep_df.loc[ep_df.index[test_mask], "RID"].to_numpy())
    test_rids_all = np.unique(ep_df.loc[ep_df.index[test_mask], "RID"]).astype(int)
    # train candidates: train_phase episodes whose RID is NOT in test_rids_all.
    train_mask = ep_df["COLPROT"].isin(train_phases).to_numpy(dtype=bool)
    train_candidates = ep_df.index[train_mask]
    train_candidates_rids = ep_df.loc[train_candidates, "RID"].to_numpy(dtype=int)
    keep = ~np.isin(train_candidates_rids, test_rids_all)
    train_candidates = train_candidates[keep]
    if len(train_candidates) == 0:
        raise ValueError("no train episodes remain after removing subjects with test episodes")

    # Build subject-level labels for the train candidates.
    rids = np.unique(train_candidates_rids[keep]) if False else np.unique(
        ep_df.loc[train_candidates, "RID"].to_numpy(dtype=int)
    )
    # Re-derive cleanly:
    train_candidate_df = ep_df.loc[train_candidates]
    rids = np.unique(train_candidate_df["RID"]).astype(int)
    y_by_rid: dict[int, int] = {}
    for rid in rids:
        sub = train_candidate_df[train_candidate_df["RID"] == rid].copy()
        sub["_dx"] = pd.to_datetime(sub["dx_date"], errors="coerce")
        sub = sub.sort_values(["_dx", "VISCODE2"], na_position="last")
        y_by_rid[int(rid)] = int(sub.iloc[0][Y_COL])
    rids_sorted = np.array(sorted(rids), dtype=np.int64)
    y = np.array([y_by_rid[int(r)] for r in rids_sorted], dtype=int)

    # One stratified subject split into train/val.
    sgkf = StratifiedGroupKFold(n_splits=max(2, int(round(1.0 / val_frac))), shuffle=True,
                                 random_state=seed)
    for train_idx, val_idx in sgkf.split(np.zeros(len(rids_sorted)), y, groups=rids_sorted):
        train_rids = rids_sorted[train_idx]
        val_rids = rids_sorted[val_idx]
        break
    else:
        train_rids = rids_sorted
        val_rids = np.array([], dtype=np.int64)

    def ids_for(rids_arr: np.ndarray) -> np.ndarray:
        rid_set = set(int(r) for r in rids_arr)
        return ep_df.index[ep_df["RID"].isin(rid_set)].to_numpy()

    return PhaseHeldOutSplit(
        train_rids=train_rids,
        val_rids=val_rids,
        test_rids=test_rids_all,
        train_episode_ids=ids_for(train_rids),
        val_episode_ids=ids_for(val_rids),
        test_episode_ids=ids_for(test_rids_all),
        train_phases=set(train_phases),
        test_phases=set(test_phases),
    )


def subject_disjoint(train_rids: np.ndarray, val_rids: np.ndarray,
                     test_rids: np.ndarray) -> bool:
    s: set[int] = set(int(x) for x in train_rids)
    return (not s.intersection(int(x) for x in val_rids)
            and not s.intersection(int(x) for x in test_rids)
            and not set(int(x) for x in val_rids).intersection(int(x) for x in test_rids))
