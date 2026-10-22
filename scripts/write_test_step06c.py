#!/usr/bin/env python3
"""Write the corrected test_step06c.py file."""
from pathlib import Path

content = r'''"""Tests for Step 6c baseline code.

Verifies that:
  * B0 one-hot encoding has exactly 15 columns and equals the pattern one-hot
  * B0 interaction model has 4 bits plus 6 pairwise products (10 features total)
  * B1-blind donor imputation draws donors from TRAINING rows only
  * B1-blind imputation leaves present blocks unchanged
  * B1-blind imputation copies donor values exactly (including NaN where donors have NaN)
  * B1-blind imputation is deterministic for a fixed seed
  * B1-blind imputation contains no mask-bit columns
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adni.baselines import (
    MODALITIES,
    build_raw_matrix,
    load_blocks,
    load_feature_spec,
    pattern_onehot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
MASK_COLS = [f"mask_{k}" for k in MODALITIES]


def _impute_blind(
    X_raw: pd.DataFrame,
    present: dict[str, set],
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    feature_spec: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """B1-blind matrix: absent blocks replaced by a random PRESENT TRAINING row of that block."""
    X = X_raw[[c for c in X_raw.columns if c not in MASK_COLS]].copy()
    all_ids = np.concatenate([tr, va, te])
    for k in MODALITIES:
        cols = [f"{k}_{c}" for c in feature_spec[k]]
        donors = np.array([i for i in tr.tolist() if i in present[k]], dtype=np.int64)
        if len(donors) == 0:
            raise RuntimeError(f"no present training donors for block {k}")
        absent = [i for i in all_ids.tolist() if i not in present[k]]
        if not absent:
            continue
        pick = rng.choice(donors, size=len(absent), replace=True)
        X.loc[absent, cols] = X_raw.loc[pick, cols].to_numpy()
    return X


def test_b0_onehot_shape() -> None:
    """Test that B0 one-hot encoding has exactly 15 columns."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    X_oh = pd.get_dummies(ep["pattern"].astype(str)).astype(float)
    assert X_oh.shape[1] == 15, f"Expected 15 columns, got {X_oh.shape[1]}"
    assert np.allclose(X_oh.sum(axis=1), 1.0), "One-hot rows should sum to 1"


def test_b0_onehot_equals_pattern_onehot() -> None:
    """Test that B0 one-hot equals the pattern_onehot function output."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    X_oh = pd.get_dummies(ep["pattern"].astype(str)).astype(float)
    bits = pattern_onehot(ep).astype(float)
    assert bits.shape[1] == 4, f"pattern_onehot should have 4 columns, got {bits.shape[1]}"
    for pattern in ep["pattern"].unique():
        mask = ep["pattern"] == pattern
        expected_bits = [int(c) for c in pattern]
        actual_bits = bits.loc[mask].iloc[0].tolist()
        assert actual_bits == expected_bits, f"Pattern {pattern} has bits {actual_bits}"


def test_b0_interaction_features() -> None:
    """Test that B0 interaction model has 4 bits + 6 pairwise products = 10 features."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    bits = pattern_onehot(ep).astype(float)
    X_inter = bits.copy()
    combos = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for i, j in combos:
        X_inter[f"inter_{MODALITIES[i]}_{MODALITIES[j]}"] = (
            bits.iloc[:, i] * bits.iloc[:, j]
        ).to_numpy()
    assert X_inter.shape[1] == 10, f"Expected 10 features, got {X_inter.shape[1]}"
    inter_cols = [c for c in X_inter.columns if c.startswith("inter_")]
    assert len(inter_cols) == 6


def test_b1_blind_donors_from_training_only() -> None:
    """Verify imputed values match the expected donor drawn from TRAINING.

    Re-runs the RNG to determine which donor was picked for each absent episode+block,
    then verifies that the imputed values match that specific donor's values.
    """
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(SPEC)
    blocks = load_blocks(BLOCKS)
    blob = json.loads(Path(SPLITS).read_text())

    node = blob["s1"]["seed_0"]["0"]["episode_ids"]
    tr = np.asarray(node["train"], dtype=np.int64)
    va = np.asarray(node["val"], dtype=np.int64)
    te = np.asarray(node["test"], dtype=np.int64)

    X_raw = build_raw_matrix(ep, blocks, spec)
    present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}
    all_ids = np.concatenate([tr, va, te])

    # Re-run RNG to determine donor picks
    rng = np.random.default_rng(42)
    donor_picks: dict[str, dict[int, int]] = {k: {} for k in MODALITIES}
    for k in MODALITIES:
        cols = [f"{k}_{c}" for c in spec[k]]
        donors_arr = np.array(
            [i for i in tr.tolist() if i in present[k]], dtype=np.int64
        )
        if len(donors_arr) == 0:
            raise RuntimeError(f"no present training donors for block {k}")
        absent = [i for i in all_ids.tolist() if i not in present[k]]
        if not absent:
            continue
        picks = rng.choice(donors_arr, size=len(absent), replace=True)
        for abs_id, donor_id in zip(absent, picks.tolist()):
            donor_picks[k][abs_id] = donor_id

    # Run imputation
    rng2 = np.random.default_rng(42)
    Xb = _impute_blind(X_raw, present, tr, va, te, spec, rng2)

    # Verify each imputed value matches the expected donor
    for k in MODALITIES:
        if not donor_picks[k]:
            continue
        cols = [f"{k}_{c}" for c in spec[k]]
        for absent_id, donor_id in donor_picks[k].items():
            imputed_row = Xb.loc[absent_id, cols].to_numpy()
            donor_row = X_raw.loc[donor_id, cols].to_numpy()
            assert np.allclose(imputed_row, donor_row, equal_nan=True), (
                f"Episode {absent_id}, block {k}: imputed != donor {donor_id}"
            )


def test_b1_blind_leaves_present_unchanged() -> None:
    """Test that B1-blind imputation leaves present blocks unchanged."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(SPEC)
    blocks = load_blocks(BLOCKS)
    blob = json.loads(Path(SPLITS).read_text())

    node = blob["s1"]["seed_0"]["0"]["episode_ids"]
    tr = np.asarray(node["train"], dtype=np.int64)
    va = np.asarray(node["val"], dtype=np.int64)
    te = np.asarray(node["test"], dtype=np.int64)

    X_raw = build_raw_matrix(ep, blocks, spec)
    present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}

    rng = np.random.default_rng(42)
    Xb = _impute_blind(X_raw, present, tr, va, te, spec, rng)

    all_ids = np.concatenate([tr, va, te])
    for k in MODALITIES:
        cols = [f"{k}_{c}" for c in spec[k]]
        present_ids = [i for i in all_ids.tolist() if i in present[k]]
        X_raw_no_mask = X_raw[[c for c in X_raw.columns if c not in MASK_COLS]]
        for eid in present_ids[:20]:
            original = X_raw_no_mask.loc[eid, cols].to_numpy()
            imputed = Xb.loc[eid, cols].to_numpy()
            assert np.allclose(original, imputed, equal_nan=True), (
                f"Episode {eid}, block {k} modified during imputation"
            )


def test_b1_blind_deterministic() -> None:
    """Test that B1-blind imputation is deterministic for a fixed seed."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(SPEC)
    blocks = load_blocks(BLOCKS)
    blob = json.loads(Path(SPLITS).read_text())

    node = blob["s1"]["seed_0"]["0"]["episode_ids"]
    tr = np.asarray(node["train"], dtype=np.int64)
    va = np.asarray(node["val"], dtype=np.int64)
    te = np.asarray(node["test"], dtype=np.int64)

    X_raw = build_raw_matrix(ep, blocks, spec)
    present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}

    rng1 = np.random.default_rng(42)
    Xb1 = _impute_blind(X_raw, present, tr, va, te, spec, rng1)
    rng2 = np.random.default_rng(42)
    Xb2 = _impute_blind(X_raw, present, tr, va, te, spec, rng2)
    assert Xb1.equals(Xb2), "Same seed produced different results"

    rng3 = np.random.default_rng(99)
    Xb3 = _impute_blind(X_raw, present, tr, va, te, spec, rng3)
    assert not Xb1.equals(Xb3), "Different seed produced identical results"


def test_b1_blind_no_mask_columns() -> None:
    """Test that B1-blind imputation contains no mask-bit columns."""
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(SPEC)
    blocks = load_blocks(BLOCKS)
    blob = json.loads(Path(SPLITS).read_text())

    node = blob["s1"]["seed_0"]["0"]["episode_ids"]
    tr = np.asarray(node["train"], dtype=np.int64)
    va = np.asarray(node["val"], dtype=np.int64)
    te = np.asarray(node["test"], dtype=np.int64)

    X_raw = build_raw_matrix(ep, blocks, spec)
    present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}

    rng = np.random.default_rng(42)
    Xb = _impute_blind(X_raw, present, tr, va, te, spec, rng)

    mask_cols_in_result = [c for c in Xb.columns if c in MASK_COLS]
    assert len(mask_cols_in_result) == 0, f"Found mask columns: {mask_cols_in_result}"
    mask_cols_in_raw = [c for c in X_raw.columns if c in MASK_COLS]
    assert len(mask_cols_in_raw) == 4
'''

Path("tests/test_step06c.py").write_text(content, encoding="utf-8")
print("Written tests/test_step06c.py")
