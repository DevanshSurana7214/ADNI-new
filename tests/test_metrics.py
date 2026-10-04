"""Metrics library tests (PIPELINE_v2 §8; playbook Step 5).

Covers the required synthetic checks:
  * perfect predictions give macro-AUROC 1 and balanced accuracy 1;
  * random predictions on balanced data give macro-AUROC ~0.5 (tol 0.03, n=20000);
  * a hand-computed 6-sample example matches;
  * ECE of a perfectly calibrated synthetic sample is < 0.02;
  * cluster bootstrap CI width within 15 % of the analytic one;
  * pn_weights make a synthetic class/pattern-dependent dataset class-balanced within each pattern
    (weighted class proportions equal within 1e-6 for unclipped weights).
Plus checks for paired bootstrap, Holm and stratified reporting.
"""
from __future__ import annotations

import numpy as np
import pytest

from adni.metrics import (
    CLASSES,
    cluster_bootstrap,
    cluster_bootstrap_multi,
    expected_calibration_error,
    holm,
    multiclass_brier,
    natural_metrics,
    paired_cluster_bootstrap,
    pi_table_from_counts,
    pn_balanced_accuracy,
    pn_weights,
    stratified_report,
)


def _perfect(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n)
    proba = np.zeros((n, 3))
    proba[np.arange(n), y] = 1.0
    return y, proba


def test_perfect_predictions():
    y, proba = _perfect()
    m = natural_metrics(y, proba)
    assert m["macro_auroc"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0)


def test_random_predictions_balanced():
    n = 20000
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=n)          # balanced in expectation
    proba = rng.random((n, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    m = natural_metrics(y, proba)
    assert abs(m["macro_auroc"] - 0.5) < 0.03, m["macro_auroc"]


def test_hand_computed_six_sample_example():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.3, 0.5],
        [0.1, 0.8, 0.1],
        [0.5, 0.3, 0.2],
    ])
    m = natural_metrics(y, proba)
    # pred = [0,1,2,2,1,0]; per-class recall = [0.5, 1.0, 0.5] -> balanced accuracy 2/3.
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    # macro-F1 = mean(0.5, 1.0, 0.5) = 2/3.
    assert m["macro_f1"] == pytest.approx(2 / 3)
    assert m["f1_per_class"] == {0: pytest.approx(0.5), 1: pytest.approx(1.0), 2: pytest.approx(0.5)}
    # macro-AUROC = mean(0.875, 0.9375, 0.875) = 0.8958333...
    assert m["macro_auroc"] == pytest.approx(0.8958333333333334)
    # Brier = (0.26 + 0.44 + 0.22 + 0.38 + 0.26 + 0.62) / 6 = 0.4266666...
    assert m["brier"] == pytest.approx(0.4266666666666667)
    # ECE: bin 0.5 (conf 0.5, acc 0), bin 0.6 (conf 0.6, acc 1), bin 0.7 (conf 0.7, acc 1),
    #      bin 0.8 (conf 0.8, acc 1) -> (1/6)(0.5 + 0.4 + 0.3 + 0.2) = 0.23333...
    assert expected_calibration_error(y, proba) == pytest.approx(0.3666666666666667)


def test_ece_perfectly_calibrated_below_002():
    n = 40000
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 3, size=n)
    conf = rng.uniform(0.4, 1.0, size=n)
    proba = np.full((n, 3), np.nan)
    proba[np.arange(n), pred] = conf
    rest = (1.0 - conf) / 2.0
    for c in range(3):
        col = proba[:, c]
        miss = np.isnan(col)
        col[miss] = rest[miss]
    # y = pred with probability conf -> the model is calibrated by construction.
    y = np.where(rng.random(n) < conf, pred, (pred + rng.integers(1, 3, size=n)) % 3)
    assert expected_calibration_error(y, proba) < 0.02


def test_cluster_bootstrap_width_close_to_analytic():
    n_subj = 400
    rng = np.random.default_rng(3)
    groups = np.arange(n_subj)
    # One episode per subject; metric = sample mean of an i.i.d. Bernoulli(p) variable, whose
    # bootstrap SD is approximately the analytic sqrt(p(1-p)/n).
    p = 0.6
    y = (rng.random(n_subj) < p).astype(float)
    proba = np.zeros((n_subj, 2))

    def mean_fn(yy, pp):
        return float(np.asarray(yy).mean())

    ci = cluster_bootstrap(mean_fn, y, proba, groups, n=2000, seed=42)
    # Analytic: SD sqrt(p(1-p)/n) = 0.0245; 95 % normal CI width ~ 2*1.96*se = 0.096.
    se = np.sqrt(p * (1 - p) / n_subj)
    analytic_width = 2 * 1.96 * se
    width = ci.ci_high - ci.ci_low
    assert abs(width - analytic_width) / analytic_width < 0.15, (width, analytic_width)


def test_cluster_bootstrap_multi_matches_single():
    n_subj = 300
    rng = np.random.default_rng(9)
    groups = np.arange(n_subj)
    y = rng.integers(0, 3, size=n_subj)
    proba = rng.random((n_subj, 3)); proba /= proba.sum(1, keepdims=True)

    def mean_label(yy, pp):
        return float(np.asarray(yy).mean())

    def auroc(yy, pp):
        return natural_metrics(yy, pp)["macro_auroc"]

    multi = cluster_bootstrap_multi({"mean": mean_label, "auroc": auroc}, y, proba, groups,
                                     n=300, seed=5)
    single = cluster_bootstrap(mean_label, y, proba, groups, n=300, seed=5)
    # Same rng seed and same draw order -> the shared-resample version reproduces the single metric.
    assert multi["mean"].point == pytest.approx(single.point)
    assert multi["mean"].ci_low == pytest.approx(single.ci_low)
    assert multi["mean"].ci_high == pytest.approx(single.ci_high)
    assert set(multi) == {"mean", "auroc"}
    assert multi["auroc"].ci_low <= multi["auroc"].point <= multi["auroc"].ci_high


def test_paired_cluster_bootstrap_and_holm():
    n = 500
    rng = np.random.default_rng(11)
    groups = np.arange(n)
    y = rng.integers(0, 3, size=n)
    pa = rng.random((n, 3)); pa /= pa.sum(1, keepdims=True)
    pb = pa.copy()
    # b is slightly better on class 0: sharpen toward the truth.
    pb[np.arange(n), y] = np.minimum(0.99, pb[np.arange(n), y] + 0.2)
    pb /= pb.sum(1, keepdims=True)

    def auroc(yy, pp):
        return natural_metrics(yy, pp)["macro_auroc"]

    ci = paired_cluster_bootstrap(auroc, y, pb, pa, groups, n=500, seed=0)
    assert ci.point >= 0.0
    assert ci.n_valid > 0
    assert ci.ci_low <= ci.ci_high

    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == pytest.approx(0.04)
    assert np.all(adj >= np.array([0.01, 0.04, 0.03, 0.5]))
    assert holm([]).size == 0


def test_pn_weights_make_classes_balanced_within_pattern():
    """Two patterns, class/pattern-dependent counts; unclipped weights -> weighted class proportions
    equal within 1e-6 inside each pattern."""
    # pattern A: mostly CN; pattern B: mostly Dementia.
    patterns = np.array(["1000"] * 300 + ["1111"] * 300)
    y = np.array([0] * 240 + [1] * 60 + [2] * 120 + [1] * 180)
    # count table rows = CLASSES (CN, MCI, Dementia), cols = patterns.
    # alpha=0 gives the exact empirical P(c|m); the balance identity is exact only then.
    counts = np.array([[240, 0], [60, 180], [0, 120]], dtype=float)
    prior = pi_table_from_counts(counts, ["1000", "1111"], CLASSES, alpha=0.0)

    w = pn_weights(y, patterns, prior, rho="uniform", wmin=0.0, wmax=1e9)
    assert w.mean() == pytest.approx(1.0)
    for pat in ("1000", "1111"):
        m = patterns == pat
        present = [c for c in (0, 1, 2) if (m & (y == c)).any()]
        shares = [w[m & (y == c)].sum() / w[m].sum() for c in present]
        # Unclipped weights make every class *present in the pattern* equally likely:
        # weighted shares are all 1/(#present classes) within 1e-6.
        for s in shares:
            assert s == pytest.approx(1 / len(present), abs=1e-6), (pat, present, shares)


def test_pn_balanced_accuracy_weighted():
    y = np.array([0, 0, 0, 0, 1, 2])
    pred = np.array([0, 0, 0, 0, 1, 2])
    w = np.array([1.0, 1.0, 1.0, 1.0, 4.0, 4.0])   # upweight the rare classes
    # recall CN = 1, MCI = 1, Dementia = 1 -> 1.0
    assert pn_balanced_accuracy(y, pred, w) == pytest.approx(1.0)
    pred2 = np.array([1, 1, 1, 1, 1, 1])
    # CN recall 0, MCI 1, Dementia 0 -> 1/3
    assert pn_balanced_accuracy(y, pred2, w) == pytest.approx(1 / 3)


def test_pn_weights_clipping():
    patterns = np.array(["1000"] * 100)
    y = np.array([0] * 100)
    counts = np.array([[100.0], [0.0], [0.0]])
    prior = pi_table_from_counts(counts, ["1000"], CLASSES, alpha=0.5)
    w = pn_weights(y, patterns, prior, wmin=0.2, wmax=5.0)
    assert w.min() >= 0.2 / w.mean()
    assert w.max() <= 5.0 / w.mean()
    assert w.mean() == pytest.approx(1.0)


def test_stratified_report_pools_small_strata():
    rng = np.random.default_rng(5)
    n = 400
    y = rng.integers(0, 3, size=n)
    proba = rng.random((n, 3)); proba /= proba.sum(1, keepdims=True)
    strata = np.array(["big"] * 100 + ["mid"] * 200 + ["tiny1"] * 10 + ["tiny2"] * 90)
    rep = stratified_report(y, proba, strata, min_n=30)
    assert "big" in rep and "mid" in rep
    assert "tiny1" not in rep          # n=10 < 30 -> pooled
    assert "other" in rep
    assert "macro_auroc" in rep["other"]
