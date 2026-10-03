"""Stage 2 preprocessing tests (PIPELINE_v2 §3.2; playbook Step 4).

The five required checks:
 1. no NaN/inf in any output;
 2. fit uses the training ids only (val/test rows cannot influence the statistics);
 3. absent modalities give all-zero rows and mask 0;
 4. the missing-indicator equals the original NaN pattern on kept features;
 5. train z-scores have mean ~0 and SD ~1 (tol 1e-3) where nothing was clipped.

Plus one extra check that the >20 % missing rule actually drops high-missing features.

All tests run on a small synthetic cohort (no dependence on the on-disk ADNI data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from adni.preprocess import MODALITIES, Preprocessor

SPEC = {
    "mri": ["a", "b", "c"],
    "fdg": ["x", "y"],
    "amy": ["p", "q"],
    "tau": ["r"],
}
N = 40


def _make_cohort(seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = np.arange(N, dtype=np.int64)
    # Presence masks: every block present for ~half the episodes.
    masks = {k: (rng.random(N) < 0.6).astype(np.uint8) for k in MODALITIES}
    masks["mri"][:] = 1          # ensure a fully-present block to exercise all paths
    masks["tau"][:5] = 0         # some definitely-absent rows

    ep = pd.DataFrame({"episode_id": ids})
    for k in MODALITIES:
        ep[f"m_{k}"] = masks[k]
    ep = ep.set_index("episode_id")

    blocks = {}
    for k in MODALITIES:
        vals = rng.normal(10.0, 2.0, size=(N, len(SPEC[k])))
        # Inject NaN only on present rows (absent rows are not part of the block table in practice).
        nan_prob = 0.1
        nan = rng.random(vals.shape) < nan_prob
        vals[nan] = np.nan
        blocks[k] = pd.DataFrame(vals, index=ids, columns=SPEC[k])
    # Feature "c" of MRI is > 20 % missing among present rows -> must be dropped.
    blocks["mri"]["c"] = np.nan
    blocks["mri"].loc[ids[: int(N * 0.1)], "c"] = rng.normal(5.0, 1.0, size=int(N * 0.1))
    return ep, blocks


@pytest.fixture(scope="module")
def cohort():
    return _make_cohort()


def _split_ids(ep: pd.DataFrame):
    ids = ep.index.to_numpy()
    rng = np.random.default_rng(123)
    perm = rng.permutation(ids)
    return perm[:24], perm[24:32], perm[32:]


def test_no_nan_or_inf(cohort):
    ep, blocks = cohort
    tr, va, te = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    for ids in (tr, va, te):
        out = pp.transform(ids)
        for k in MODALITIES:
            X, m = out[k]
            assert X.dtype == np.float32
            assert m.dtype == np.uint8
            assert np.isfinite(X).all(), f"{k}: non-finite values in transform output"
            assert X.shape[0] == len(np.unique(ids))
            assert X.shape[1] == 2 * len(pp.kept[k])


def test_fit_uses_training_ids_only(cohort):
    ep, blocks = cohort
    tr, va, te = _split_ids(ep)
    pp1 = Preprocessor(ep, blocks, SPEC).fit(tr)

    # Corrupt every validation/test row of every block with noise and refit.
    noisy = {k: v.copy() for k, v in blocks.items()}
    rng = np.random.default_rng(999)
    holdout = np.concatenate([va, te])
    for k in MODALITIES:
        noisy[k].loc[holdout, :] = rng.normal(1000.0, 5.0, size=(len(holdout), len(SPEC[k])))
    pp2 = Preprocessor(ep, noisy, SPEC).fit(tr)

    for k in MODALITIES:
        assert pp1.kept[k] == pp2.kept[k], f"{k}: kept list changed with val/test noise"
        assert pp1.dropped[k] == pp2.dropped[k], f"{k}: dropped list changed with val/test noise"
        np.testing.assert_array_equal(pp1.mean_[k], pp2.mean_[k])
        np.testing.assert_array_equal(pp1.sd_[k], pp2.sd_[k])


def test_absent_modalities_all_zero_and_mask_zero(cohort):
    ep, blocks = cohort
    tr, va, te = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    out = pp.transform(ep.index.to_numpy())
    for k in MODALITIES:
        X, m = out[k]
        absent = m == 0
        assert np.all(X[absent] == 0.0), f"{k}: absent rows are not all-zero"
    # Non-MRI blocks have some absent rows; tau[:5] were explicitly absent.
    for k in ("fdg", "amy", "tau"):
        assert (out[k][1] == 0).any(), f"{k}: test cohort has no absent rows"
    assert np.all(out["tau"][0][:5] == 0)


def test_missing_indicator_matches_nan_pattern(cohort):
    ep, blocks = cohort
    tr, va, te = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    ids = ep.index.to_numpy()
    out = pp.transform(ids)
    for k in MODALITIES:
        X, m = out[k]
        kept = pp.kept[k]
        d = len(kept)
        present = m == 1
        raw = blocks[k].loc[ids[present], kept].to_numpy()
        expected = np.isnan(raw).astype(np.float32)
        np.testing.assert_array_equal(X[present, d:], expected, err_msg=f"{k}: missing-indicator mismatch")
        # Absent rows: indicator also zero.
        assert np.all(X[~present, d:] == 0.0)


def test_train_zscores_mean0_sd1_where_unclipped(cohort):
    ep, blocks = cohort
    tr, _, _ = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    out = pp.transform(tr)
    for k in MODALITIES:
        X, m = out[k]
        kept = pp.kept[k]
        d = len(kept)
        z = X[:, :d]
        ind = X[:, d:]
        for j, _feat in enumerate(kept):
            col = z[:, j]
            # Only present rows contribute to the fitted statistics.
            present = m == 1
            # Skip features that were clipped anywhere (winsorisation breaks exact mean/SD).
            if np.any(np.abs(col[present]) == pp.clip):
                continue
            non_missing = present & (ind[:, j] == 0)
            vals = col[non_missing]
            if len(vals) < 2:
                continue
            assert abs(float(vals.mean())) < 1e-3, f"{k}/{kept[j]}: train z mean {vals.mean():.4g}"
            assert abs(float(vals.std()) - 1.0) < 1e-3, f"{k}/{kept[j]}: train z SD {vals.std():.4g}"


def test_high_missing_features_are_dropped(cohort):
    ep, blocks = cohort
    tr, _, _ = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    # MRI "c" is missing in 90 % of training present rows -> dropped.
    assert "c" not in pp.kept["mri"]
    assert "c" in pp.dropped["mri"]
    # Other blocks have only ~10 % missing -> nothing dropped.
    for k in ("fdg", "amy", "tau"):
        assert pp.dropped[k] == []
        assert pp.kept[k] == SPEC[k]


def test_clip_fraction_matches_manual(cohort):
    ep, blocks = cohort
    tr, _, _ = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    ids = pp._resolve_ids(tr)
    for k in MODALITIES:
        z, m = pp.raw_z(k, tr)
        present = m == 1
        finite = np.isfinite(z[present])
        expected = int((np.abs(z[present][finite]) > pp.clip).sum())
        n_clip, n_finite = pp.clip_fraction(k, tr)
        assert n_finite == int(finite.sum())
        assert n_clip == expected


def test_fit_is_deterministic(cohort):
    ep, blocks = cohort
    tr, _, _ = _split_ids(ep)
    pp1 = Preprocessor(ep, blocks, SPEC).fit(tr)
    pp2 = Preprocessor(ep, blocks, SPEC).fit(tr)
    for k in MODALITIES:
        assert pp1.kept[k] == pp2.kept[k]
        np.testing.assert_array_equal(pp1.mean_[k], pp2.mean_[k])
        np.testing.assert_array_equal(pp1.sd_[k], pp2.sd_[k])


def test_transform_before_fit_raises(cohort):
    ep, blocks = cohort
    with pytest.raises(RuntimeError):
        Preprocessor(ep, blocks, SPEC).transform(ep.index.to_numpy())


def test_joblib_roundtrip(cohort, tmp_path):
    import joblib

    ep, blocks = cohort
    tr, _, _ = _split_ids(ep)
    pp = Preprocessor(ep, blocks, SPEC).fit(tr)
    path = tmp_path / "pp.joblib"
    joblib.dump(pp, path)
    pp2 = joblib.load(path)
    a = pp.transform(tr)
    b = pp2.transform(tr)
    for k in MODALITIES:
        np.testing.assert_array_equal(a[k][0], b[k][0])
        np.testing.assert_array_equal(a[k][1], b[k][1])
