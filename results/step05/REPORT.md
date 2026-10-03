STEP 5 REPORT
Metrics library (PIPELINE_v2 §8). No models yet.

Command(s) run:
  PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q
  (synthetic-check printout reproduced below via an ad-hoc python run)

Git commit: (pending — step05 not yet committed)

Tests: 44 passed in 20.40s   (test_metrics.py 10 new, plus the prior 34)

Environment: Python 3.12.10, numpy 2.5.3, scikit-learn 1.9.1, statsmodels 0.15.0. Windows 11.

What I did
  - Implemented src/adni/metrics.py:
      * §8.1 natural_metrics(y, proba): macro-AUROC (one-vs-rest), macro-F1, balanced accuracy,
        per-class F1, multiclass Brier, ECE (15 equal-width bins on the max probability).
      * §8.2 pn_weights(y, pattern, prior_table, rho='uniform', wmin=0.2, wmax=5.0):
        w_i = rho(c_i) / pi_hat(c_i | m_i), pi_hat read from the TRAINING-split P(c|m) table,
        clipped to [wmin, wmax] and normalised to mean 1. Plus pn_balanced_accuracy(y, pred, w)
        (mean over classes of the w-weighted per-class recall) and pi_table_from_counts().
      * §8.5 cluster_bootstrap (resamples SUBJECTS with replacement) and paired_cluster_bootstrap
        (paired by episode), both returning point + 95 % percentile CI; holm(pvalues) via statsmodels.
      * §8.3 stratified_report(y, proba, strata, min_n=30): per-stratum metrics, strata with
        n < min_n pooled into an "other" bucket.
  - Wrote tests/test_metrics.py (10 tests) covering all six required synthetic checks.

Synthetic checks (actual output)

  Check                                              | Expected                | Actual                    | Pass
  ---------------------------------------------------|-------------------------|---------------------------|-----
  Perfect predictions macro-AUROC / balanced acc     | 1.0 / 1.0               | 1.0 / 1.0                 | yes
  Random predictions on balanced data (n=20000)      | AUROC ~0.5 (tol 0.03)   | 0.4963                    | yes
  Hand-computed 6-sample example                     | see below               | matches                   | yes
  ECE of perfectly calibrated synthetic (n=40000)    | < 0.02                  | < 0.02 (test passes)      | yes
  Cluster-bootstrap CI width vs analytic             | within 15 %             | 0.0975 vs 0.0960 (1.5 %)  | yes
  PN weights balance classes within each pattern     | equal within 1e-6       | equal to 1e-6             | yes
  Holm([0.01, 0.04, 0.03, 0.5])                      | [0.04, 0.09, 0.09, 0.5] | [0.04, 0.09, 0.09, 0.5]   | yes

  Hand-computed 6-sample example (y = [0,1,2,0,1,2]):
    pred = [0,1,2,2,1,0]; per-class recall = [0.5, 1.0, 0.5] -> balanced accuracy = 2/3;
    macro-F1 = 2/3; per-class F1 = {CN 0.5, MCI 1.0, Dementia 0.5}; macro-AUROC = 0.8958333...;
    Brier = 0.4266667...; ECE (bins 0.5/0.6/0.7/0.8) = (1/6)(0.5+0.4+0.3+0.2) = 0.3666667.

  PN balance check: two patterns with different class mixes; with the exact empirical P(c|m)
  (alpha=0) and unclipped weights, the weighted share of every class present in a pattern equals
  1/(#present classes) to within 1e-6 (pattern 1000 -> CN/MCI 0.5/0.5; pattern 1111 -> MCI/Dementia
  0.5/0.5), and mean(w) = 1.0 to 9 dp.

Deviations from the spec (if any, with reason):
  - §8.2 defines w_i = 1/(|C|·pi_hat(c_i|m_i)) (rho uniform). I generalised this to
    w_i = rho(c_i)/pi_hat(c_i|m_i) with rho='uniform' as the default, because Step 8's A4 requires the
    rho = P(c) variant as well. The uniform path reproduces the spec's formula exactly.
  - The "pn_weights balance" test uses alpha=0 (exact empirical conditionals). The balance identity
    w = rho(c)/pi(c|m) holds only for the exact conditional; the smoothed pi (alpha=0.5, used in the
    pipeline) is a deliberate, documented choice and is not asserted to balance exactly.

Things I am unsure about:
  - pn_balanced_accuracy averages over classes present in the *evaluation* set (skipping classes with
    no episodes). On the natural test distribution all three classes are always present, so this is
    equivalent to averaging over all three. Documented in the docstring.
  - macro-AUROC skips classes that are absent or all-positive in a bootstrap resample; if every class
    were degenerate the function returns NaN. cluster_bootstrap drops non-finite resamples and records
    how many were valid (n_valid).

Files written:
  src/adni/metrics.py
  tests/test_metrics.py
  results/step05/REPORT.md
