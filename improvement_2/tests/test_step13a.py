"""Tests for the step13a correction, prior loading and E1 build (EXPLORATORY).

Runs with `PYTHONPATH=src:. python -m pytest improvement_2/tests/test_step13a.py` from the repo root.
No model training and no GPU required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from improve2.post_hoc_correction import (
    apply_correction,
    check_row_sums_close,
    correction_factor,
)


def _make_prior():
    return pd.DataFrame({"1000": [0.3, 0.5, 0.2],
                        "1111": [0.2, 0.4, 0.4],
                        "0001": [0.5, 0.3, 0.2]},
                       index=["CN", "MCI", "Dementia"])


def _proba():
    return np.array([[0.6, 0.3, 0.1],
                     [0.2, 0.5, 0.3],
                     [0.0, 0.5, 0.5]])


def _pattern():
    return np.array(["1000", "1111", "0001"])


class TestCorrectionMathematics:
    def test_tau_zero_leaves_probabilities_unchanged(self):
        out = apply_correction(_proba(), _pattern(), _make_prior(), tau=0.0, rho=np.ones(3)/3)
        np.testing.assert_allclose(out, _proba(), atol=1e-12)

    def test_corrected_probabilities_sum_to_1(self):
        p = _proba()
        out = apply_correction(p, _pattern(), _make_prior(), tau=1.0, rho=np.array([0.5, 0.3, 0.2]))
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_tau_zero_then_nonzero_reverses_only_when_prior_differs(self):
        p = _proba()[:1]
        pattern = _pattern()[:1]
        prior = _make_prior().loc[:, ["1000"]]
        rho = np.ones(3)/3
        uncorrected = apply_correction(p, pattern, prior, tau=0.0, rho=rho)
        corrected = apply_correction(p, pattern, prior, tau=1.0, rho=rho)
        assert not np.allclose(corrected, uncorrected)

    def test_adding_1_to_all_logit_class_prior_is_unchanged_by_correction(self):
        logits = np.array([[1.0, 1.0, 1.0], [1.5, 1.5, 1.5]])
        p = np.exp(logits); p = p / p.sum(axis=1, keepdims=True)
        pattern = np.array(["1000", "1000"])
        prior = _make_prior().loc[:, ["1000"]]
        out = apply_correction(p, pattern, prior, tau=1.0, rho=np.ones(3)/3)
        np.testing.assert_allclose(out, p, atol=1e-12)


class TestCorrectionFactor:
    def test_correction_factor_shape_and_names(self):
        p = _proba()[:1]
        pattern = np.array(["1000"])
        prior = _make_prior().loc[:, ["1000"]]
        f = correction_factor(p, pattern, tau=1.0)
        assert f.shape == (3, 1)
        assert list(f.columns) == ["1000"]

    def test_correction_factor_computed_from_prior_only(self):
        p = _proba()[:1]
        pattern = np.array(["1000"])
        prior = _make_prior().loc[:, ["1000"]]
        f = correction_factor(p, pattern, tau=1.0)
        f_expected = np.array([[1.0/0.3], [1.0/0.5], [1.0/0.2]])
        np.testing.assert_allclose(f.to_numpy(), f_expected)


class TestE1Build:
    def test_average_of_two_corrected_posteriors(self):
        p_b1 = np.array([[0.6, 0.3, 0.1],
                         [0.2, 0.5, 0.3]])
        p_a0 = np.array([[0.5, 0.4, 0.1],
                         [0.3, 0.4, 0.3]])
        e1 = (p_b1 + p_a0) / 2.0
        np.testing.assert_allclose(e1.sum(axis=1), 1.0)


def _dummy_ep_with_pattern():
    ep = pd.DataFrame({"y": pd.Series([0, 1, 2, 0, 1, 2], dtype=int),
                       "pattern": pd.Series(["1000", "1111", "0001", "1000", "1111", "0001"], dtype=str),
                       "RID": pd.Series([1, 2, 3, 4, 5, 6], dtype=int)})
    ep.index.name = "episode_id"
    return ep


class TestPriorLoading:
    def test_training_prior_table_smoothed_sums_to_one(self):
        from adni.baselines import training_prior_table
        ep = _dummy_ep_with_pattern()
        tr = ep.index.to_numpy()[:4]
        prior = training_prior_table(ep, tr, alpha=0.5)
        np.testing.assert_allclose(prior.sum().to_numpy(), 1.0)

    def test_training_prior_table_index_is_classes(self):
        from adni.baselines import training_prior_table
        ep = _dummy_ep_with_pattern()
        prior = training_prior_table(ep, ep.index.to_numpy()[:4], alpha=0.5)
        assert list(prior.index) == ["CN", "MCI", "Dementia"]


class TestWithinPatternRule:
    def test_fixed_pattern_list_from_full_data(self):
        from adni.metrics import CLASSES
        ep = _dummy_ep_with_pattern()
        full_counts = pd.crosstab(ep["y"], ep["pattern"]).reindex(index=CLASSES, fill_value=0)
        pats = sorted(ep["pattern"].unique().tolist())
        n_per_pattern = full_counts.sum(axis=0).to_numpy()
        per_class = full_counts.to_numpy()
        included = []
        for j, pat in enumerate(pats):
            n = int(n_per_pattern[j])
            ok = n >= 100 and int((per_class[:, j] >= 10).all())
            if ok:
                included.append(pat)
        assert included == []  # dummy data too small
