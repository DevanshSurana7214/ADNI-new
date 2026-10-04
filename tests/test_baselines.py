"""Baseline module tests (playbook Step 6).

These are fast, self-contained checks of the plumbing rather than full runs:
  * the fixed LightGBM config loads and matches the playbook values;
  * the pattern one-hot reproduces the four availability bits;
  * ``build_raw_matrix`` is block-prefixed, reindexes to the episode index, leaves absent modalities
    NaN and never leaks the label (no ``y`` column);
  * labels are looked up by ``episode_id`` (never by position);
  * B0 probabilities are a valid simplex in class order;
  * B2 predictions are returned only for present episodes;
  * B1/B2 fit and predict with rows summing to 1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adni.baselines import (
    B1_PARAMS,
    MODALITIES,
    build_raw_matrix,
    fit_predict_b0,
    fit_predict_b1,
    fit_predict_b2,
    load_feature_spec,
    pattern_onehot,
    training_prior_table,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_ep(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pats = rng.choice(["1000", "0100", "0010", "0001", "1111"], size=n)
    ep = pd.DataFrame({
        "pattern": pats,
        "y": rng.integers(0, 3, size=n),
        "RID": rng.integers(1, 40, size=n),
    }, index=range(n))
    for i, k in enumerate(MODALITIES):
        ep[f"m_{k}"] = [int(p[i]) for p in pats]
    return ep


def _tiny_X(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"a_{j}": rng.normal(size=n) for j in range(5)}, index=range(n))
    X["b_x"] = rng.normal(size=n)
    X["mask_a"] = rng.integers(0, 2, size=n)
    return X


# --------------------------------------------------------------------------- config
def test_config_matches_playbook():
    import yaml

    cfg = yaml.safe_load((PROJECT_ROOT / "configs" / "b1.yaml").read_text())
    for k, v in B1_PARAMS.items():
        assert cfg[k] == v, (k, cfg[k], v)
    assert cfg["num_leaves"] == 31 and cfg["learning_rate"] == 0.05
    assert cfg["feature_fraction"] == 0.5 and cfg["bagging_fraction"] == 0.8
    assert cfg["class_weight"] is None and cfg["objective"] == "multiclass"
    assert cfg["num_rounds"] == 1000 and cfg["early_stopping"] == 50


# --------------------------------------------------------------------------- features
def test_pattern_onehot_matches_bits():
    ep = _tiny_ep(20)
    oh = pattern_onehot(ep)
    assert list(oh.columns) == [f"mask_{k}" for k in MODALITIES]
    assert (oh.to_numpy() == ep[[f"m_{k}" for k in MODALITIES]].to_numpy()).all()


def test_build_raw_matrix_prefix_and_nan():
    ep = _tiny_ep(30)
    mri_idx = [i for i in range(30) if ep.pattern.iloc[i][0] == "1"]
    blocks = {
        "mri": pd.DataFrame({"f1": np.arange(len(mri_idx), dtype=float),
                             "f2": np.arange(len(mri_idx), dtype=float)}, index=mri_idx),
        "fdg": pd.DataFrame({"g1": np.ones(5)}, index=[0, 1, 2, 3, 4]),
        "amy": pd.DataFrame({"CENTILOIDS": np.ones(7)}, index=list(range(7))),
        "tau": pd.DataFrame({"SUVR": np.ones(3)}, index=list(range(3))),
    }
    spec = {"mri": ["f1", "f2"], "fdg": ["g1"], "amy": ["CENTILOIDS"], "tau": ["SUVR"]}
    X = build_raw_matrix(ep, blocks, spec)
    assert X.shape == (30, 4 + 2 + 1 + 1 + 1)
    assert "y" not in X.columns and "RID" not in X.columns
    assert set(X.columns) == {"mri_f1", "mri_f2", "fdg_g1", "amy_CENTILOIDS", "tau_SUVR",
                              *[f"mask_{k}" for k in MODALITIES]}
    # an absent MRI row must be NaN in the MRI columns and 0 in its mask
    absent = next(i for i in range(30) if ep.pattern.iloc[i][0] == "0")
    assert np.isnan(X.loc[absent, "mri_f1"])
    assert X.loc[absent, "mask_mri"] == 0


# --------------------------------------------------------------------------- B0
def test_b0_simplex_and_class_order():
    ep = _tiny_ep(120)
    tr = ep.index.to_numpy()[:90]
    te = ep.index.to_numpy()[90:]
    p = fit_predict_b0(ep, tr, te, seed=0)
    assert p.shape == (30, 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()
    # B0 sees only the pattern: identical patterns give identical probabilities.
    oh = pattern_onehot(ep).loc[te].to_numpy()
    dup = {}
    for i, row in enumerate(oh):
        key = tuple(row.tolist())
        if key in dup:
            assert np.allclose(p[i], p[dup[key]])
        else:
            dup[key] = i


# --------------------------------------------------------------------------- B1 / B2
def test_b1_fits_and_predicts_simplex():
    ep = _tiny_ep(150)
    X = _tiny_X(150)
    tr = ep.index.to_numpy()[:90]
    va = ep.index.to_numpy()[90:120]
    te = ep.index.to_numpy()[120:]
    p, bst = fit_predict_b1(X, ep, tr, va, te, seed=0,
                            params_override={"num_leaves": 4}, num_rounds=20, early_stopping=5)
    assert p.shape == (len(te), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert bst.num_trees() <= 20


def test_b2_returns_only_present_test_episodes():
    ep = _tiny_ep(150)
    X = _tiny_X(150)
    tr = ep.index.to_numpy()[:90]
    va = ep.index.to_numpy()[90:120]
    te = ep.index.to_numpy()[120:]
    present = ep.index[ep["m_mri"].to_numpy() == 1].to_numpy()
    p, te2, _ = fit_predict_b2(X, ep, present, tr, va, te, seed=0,
                               params_override={"num_leaves": 4}, num_rounds=20, early_stopping=5)
    expected = np.array([i for i in te if i in set(present.tolist())])
    assert np.array_equal(te2, expected)
    assert p.shape == (len(expected), 3)
    assert np.allclose(p.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- priors
def test_training_prior_table_rows_sum_to_one():
    ep = _tiny_ep(200, seed=3)
    tab = training_prior_table(ep, ep.index.to_numpy()[:150])
    assert list(tab.index) == ["CN", "MCI", "Dementia"]
    assert np.allclose(tab.to_numpy().sum(axis=0), 1.0, atol=1e-9)


def test_feature_spec_loader_reads_columns():
    spec = load_feature_spec(PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json")
    assert set(spec) == {"mri", "fdg", "amy", "tau"}
    assert len(spec["mri"]) == 256 and len(spec["fdg"]) == 6


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
