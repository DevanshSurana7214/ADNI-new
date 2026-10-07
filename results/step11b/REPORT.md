STEP 11B REPORT
Confirmatory held-out check of A4 with tau = 0.5 on the S2 protocol shift, and the EXPLORATORY arms A5-A8 under R2 on the S1 grid.

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/run_step11b_s2.py                        > results/step11b/s2/{runs,oof_all.parquet,run_log.jsonl}
  scripts/run_step11b_s1.py --shard {0,1,2} --nshards 3 --no-combine
  scripts/run_step11b_s1.py --combine-only         > results/step11b/s1/{oof_all.parquet,
                                                    swap_all.parquet,intervention_raw.csv}
  scripts/analyse_step11b.py                       > results/step11b/{confirmatory_*,
                                                    exploratory_*,best_epochs,embedding_swap,
                                                    intervention,analysis.json}
  scripts/step11b_report.py                        > results/step11b/REPORT.md
  scripts/step11b_appendix.py                      > Appendix E row

Git commit: (this commit) - tag `step11b`

Tests: 127 passed in 113.04s (0:01:53)

Runs: S2 confirmatory 15/15 ok, 0 errors; S1 exploratory 60/60 ok, 0 errors. Everything pre-registered in configs/step11b.yaml BEFORE any run.

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1; NVIDIA RTX 3050 (6 GB). Deterministic algorithms on; seeds fixed before the model is built (train_exploratory_arm follows train_arm). No tuning; no selection among A5-A8.

------------------------------------------------------------------------------
PREDECLARED DESIGN

Frozen before any run (configs/step11b.yaml). tau = 0.5 was selected from the S1 Step 11a
results, so the S2 check is the held-out confirmation; its reading was pre-declared:
A4(tau 0.5) - A0 is 'confirmed' iff the PN-BAcc CI lower bound is above 0 AND the macro-F1
CI lower bound is above -0.02. tau is not changed afterwards regardless of the outcome.

Comparators are REUSED, not retrained: S2 A0/A1/A4 (tau 1, recipe 1) from Step 10
(results/step10/s2) and A4(R2, tau 1) on S1 from Step 11a (results/step11a, run-set R2_A4).
Within R2, A0/A1/A4(tau 1) were retrained on S2 with the same seeds so every R2 comparison
is within-recipe.

A5 = A4 + FiLM pattern conditioning from a 16-d embedding of the EFFECTIVE mask;
A6 = A5 + cross-view consistency (lambda_c = 0.1, two dropout views, stop-gradient on
view 2); A7 = A6 + pattern-aware MoE (K = 4 residual-MLP experts, router input [pooled h,
pattern embedding, mask] - no label-dependent input); A8 = A6 + class-dependent prototypes
mu_(c,m) = mu_c + B(h(mu_c) * e_m) with prototype-loss weight 0.1:
  L = CE(s_v1 + tau*log pi(c|m_v1), y) + 0.1 * [-sum_c p2(c) log p1(c)]
      + 0.1 * CE( -||h_v1 - mu_(c, m_v1)||^2 , y ).

All A5-A8 results are EXPLORATORY: no selection among them, no use of these results to
change A0-A4, the recipes or any pre-declared rule. Window sensitivities and paper assets
are NOT started (out of scope).

==============================================================================
1. CONFIRMATORY S2 CHECK (A4, tau = 0.5)

### primary checkpoint rule

| label | op | n | pn_bacc | pn_bacc_lo | pn_bacc_hi | macro_f1 | macro_f1_lo | macro_f1_hi | macro_auroc | macro_auroc_lo | macro_auroc_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A4(tau 0.5) [R1] | uniform | 3565 | 0.6159 | 0.5863 | 0.6445 | 0.6051 | 0.5782 | 0.6306 | 0.7766 | 0.7570 | 0.7959 |
| A0 [R2] | argmax | 3565 | 0.6058 | 0.5772 | 0.6340 | 0.6161 | 0.5896 | 0.6414 | 0.8036 | 0.7859 | 0.8214 |
| A1 [R2] | uniform | 3565 | 0.6126 | 0.5845 | 0.6403 | 0.5839 | 0.5572 | 0.6094 | 0.7790 | 0.7594 | 0.7980 |
| A4(tau 1) [R2] | uniform | 3565 | 0.6040 | 0.5750 | 0.6306 | 0.5666 | 0.5395 | 0.5912 | 0.7466 | 0.7258 | 0.7669 |
| A4(tau 0.5) [R2] | uniform | 3565 | 0.6269 | 0.5988 | 0.6536 | 0.6096 | 0.5823 | 0.6350 | 0.7911 | 0.7726 | 0.8098 |
| A0 [R1, Step 10] | argmax | 3565 | 0.6071 | 0.5769 | 0.6346 | 0.6133 | 0.5858 | 0.6390 | 0.7959 | 0.7772 | 0.8142 |
| A1 [R1, Step 10] | uniform | 3565 | 0.6161 | 0.5872 | 0.6438 | 0.5952 | 0.5681 | 0.6195 | 0.7860 | 0.7668 | 0.8051 |
| A4(tau 1) [R1, Step 10] | uniform | 3565 | 0.5997 | 0.5705 | 0.6270 | 0.5628 | 0.5343 | 0.5877 | 0.7440 | 0.7230 | 0.7638 |

| comparison | metric | diff | ci_low | ci_high | p |
|---|---|---|---|---|---|
| A4(tau 0.5) - A0 [R1] | pn_bacc_uniform | 0.0088 | -0.0056 | 0.0231 | 0.2459 |
| A4(tau 0.5) - A0 [R1] | macro_f1 | -0.0082 | -0.0225 | 0.0060 | 0.2579 |
| A4(tau 0.5) - A0 [R1] | macro_auroc | -0.0193 | -0.0249 | -0.0140 | 0.0010 |
| A4(tau 0.5) - A1 [R1] | pn_bacc_uniform | -0.0002 | -0.0110 | 0.0102 | 0.9785 |
| A4(tau 0.5) - A1 [R1] | macro_f1 | 0.0098 | -0.0025 | 0.0216 | 0.1149 |
| A4(tau 0.5) - A1 [R1] | macro_auroc | -0.0094 | -0.0130 | -0.0060 | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R1] | pn_bacc_uniform | 0.0162 | 0.0012 | 0.0320 | 0.0320 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_f1 | 0.0423 | 0.0282 | 0.0572 | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_auroc | 0.0326 | 0.0258 | 0.0397 | 0.0010 |
| A4(tau 0.5) - A0 [R2] | pn_bacc_uniform | 0.0211 | 0.0054 | 0.0366 | 0.0060 |
| A4(tau 0.5) - A0 [R2] | macro_f1 | -0.0065 | -0.0232 | 0.0089 | 0.4298 |
| A4(tau 0.5) - A0 [R2] | macro_auroc | -0.0125 | -0.0169 | -0.0080 | 0.0010 |
| A4(tau 0.5) - A1 [R2] | pn_bacc_uniform | 0.0142 | -0.0015 | 0.0302 | 0.0770 |
| A4(tau 0.5) - A1 [R2] | macro_f1 | 0.0257 | 0.0094 | 0.0420 | 0.0020 |
| A4(tau 0.5) - A1 [R2] | macro_auroc | 0.0121 | 0.0049 | 0.0196 | 0.0020 |
| A4(tau 0.5) - A4(tau 1) [R2] | pn_bacc_uniform | 0.0229 | 0.0052 | 0.0412 | 0.0120 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_f1 | 0.0431 | 0.0252 | 0.0621 | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_auroc | 0.0445 | 0.0356 | 0.0547 | 0.0010 |

- R1 (primary): PN-BAcc(A4 tau 0.5 - A0) CI low = -0.0056 (> 0: False); macro-F1 CI low = -0.0225 (> -0.02: False) -> **not confirmed**
- R2 (primary): PN-BAcc(A4 tau 0.5 - A0) CI low = 0.0054 (> 0: True); macro-F1 CI low = -0.0232 (> -0.02: False) -> **not confirmed**

### secondary checkpoint rule

| label | op | n | pn_bacc | pn_bacc_lo | pn_bacc_hi | macro_f1 | macro_f1_lo | macro_f1_hi | macro_auroc | macro_auroc_lo | macro_auroc_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A4(tau 0.5) [R1] | uniform | 3565 | 0.6200 | 0.5919 | 0.6481 | 0.5984 | 0.5722 | 0.6246 | 0.7844 | 0.7654 | 0.8032 |
| A0 [R2] | argmax | 3565 | 0.6109 | 0.5821 | 0.6389 | 0.6180 | 0.5911 | 0.6429 | 0.8056 | 0.7879 | 0.8234 |
| A1 [R2] | uniform | 3565 | 0.6212 | 0.5941 | 0.6455 | 0.5795 | 0.5532 | 0.6041 | 0.7980 | 0.7797 | 0.8160 |
| A4(tau 1) [R2] | uniform | 3565 | 0.6286 | 0.6035 | 0.6519 | 0.5713 | 0.5444 | 0.5958 | 0.7606 | 0.7404 | 0.7802 |
| A4(tau 0.5) [R2] | uniform | 3565 | 0.6299 | 0.6020 | 0.6564 | 0.6083 | 0.5807 | 0.6338 | 0.7932 | 0.7749 | 0.8120 |
| A0 [R1, Step 10] | argmax | 3565 | 0.6086 | 0.5787 | 0.6361 | 0.6147 | 0.5871 | 0.6396 | 0.7972 | 0.7793 | 0.8150 |
| A1 [R1, Step 10] | uniform | 3565 | 0.6213 | 0.5948 | 0.6457 | 0.5847 | 0.5583 | 0.6100 | 0.7912 | 0.7730 | 0.8091 |
| A4(tau 1) [R1, Step 10] | uniform | 3565 | 0.6347 | 0.6080 | 0.6591 | 0.5806 | 0.5533 | 0.6058 | 0.7640 | 0.7443 | 0.7832 |

| comparison | metric | diff | ci_low | ci_high | p |
|---|---|---|---|---|---|
| A4(tau 0.5) - A0 [R1] | pn_bacc_uniform | 0.0113 | -0.0016 | 0.0256 | 0.0970 |
| A4(tau 0.5) - A0 [R1] | macro_f1 | -0.0164 | -0.0309 | -0.0023 | 0.0300 |
| A4(tau 0.5) - A0 [R1] | macro_auroc | -0.0129 | -0.0167 | -0.0093 | 0.0010 |
| A4(tau 0.5) - A1 [R1] | pn_bacc_uniform | -0.0013 | -0.0173 | 0.0137 | 0.8796 |
| A4(tau 0.5) - A1 [R1] | macro_f1 | 0.0137 | -0.0019 | 0.0293 | 0.0840 |
| A4(tau 0.5) - A1 [R1] | macro_auroc | -0.0068 | -0.0117 | -0.0021 | 0.0040 |
| A4(tau 0.5) - A4(tau 1) [R1] | pn_bacc_uniform | -0.0148 | -0.0353 | 0.0062 | 0.1779 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_f1 | 0.0178 | -0.0011 | 0.0371 | 0.0710 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_auroc | 0.0204 | 0.0125 | 0.0280 | 0.0010 |
| A4(tau 0.5) - A0 [R2] | pn_bacc_uniform | 0.0190 | 0.0035 | 0.0355 | 0.0130 |
| A4(tau 0.5) - A0 [R2] | macro_f1 | -0.0097 | -0.0261 | 0.0070 | 0.2479 |
| A4(tau 0.5) - A0 [R2] | macro_auroc | -0.0124 | -0.0167 | -0.0082 | 0.0010 |
| A4(tau 0.5) - A1 [R2] | pn_bacc_uniform | 0.0087 | -0.0036 | 0.0210 | 0.1749 |
| A4(tau 0.5) - A1 [R2] | macro_f1 | 0.0288 | 0.0154 | 0.0430 | 0.0010 |
| A4(tau 0.5) - A1 [R2] | macro_auroc | -0.0048 | -0.0086 | -0.0008 | 0.0160 |
| A4(tau 0.5) - A4(tau 1) [R2] | pn_bacc_uniform | 0.0013 | -0.0162 | 0.0184 | 0.8326 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_f1 | 0.0370 | 0.0197 | 0.0545 | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_auroc | 0.0326 | 0.0267 | 0.0383 | 0.0010 |

- R1 (secondary): PN-BAcc(A4 tau 0.5 - A0) CI low = -0.0016 (> 0: False); macro-F1 CI low = -0.0309 (> -0.02: False) -> **not confirmed**
- R2 (secondary): PN-BAcc(A4 tau 0.5 - A0) CI low = 0.0035 (> 0: True); macro-F1 CI low = -0.0261 (> -0.02: False) -> **not confirmed**

Which holds: stated per recipe and checkpoint rule above; the pre-declared reading is the
recipe-1 rule (tau 0.5 was selected at recipe 1); R2 is the replication. tau is NOT
changed afterwards.

==============================================================================
2. EXPLORATORY ARMS A5-A8 UNDER R2 (S1 GRID) - ALL RESULTS EXPLORATORY

EXPLORATORY: no selection among A5-A8; these results must not be used to change A0-A4,
the recipes, or any pre-declared rule.

### primary checkpoint rule

| label | op | n | pn_bacc | pn_bacc_lo | pn_bacc_hi | macro_f1 | macro_f1_lo | macro_f1_hi | macro_auroc | macro_auroc_lo | macro_auroc_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2_A4 (comparator) | uniform | 13240 | 0.6184 | 0.6048 | 0.6326 | 0.5847 | 0.5712 | 0.5984 | 0.7736 | 0.7628 | 0.7847 |
| A5 (EXPLORATORY) | uniform | 13240 | 0.6162 | 0.6027 | 0.6301 | 0.5842 | 0.5706 | 0.5977 | 0.7735 | 0.7624 | 0.7842 |
| A6 (EXPLORATORY) | uniform | 13240 | 0.6183 | 0.6052 | 0.6324 | 0.5866 | 0.5737 | 0.6001 | 0.7724 | 0.7614 | 0.7835 |
| A7 (EXPLORATORY) | uniform | 13240 | 0.6157 | 0.6024 | 0.6297 | 0.5877 | 0.5743 | 0.6014 | 0.7697 | 0.7586 | 0.7804 |
| A8 (EXPLORATORY) | uniform | 13240 | 0.6186 | 0.6051 | 0.6333 | 0.5866 | 0.5735 | 0.6006 | 0.7727 | 0.7619 | 0.7836 |

| comparison | metric | diff | ci_low | ci_high | p | holm_p |
|---|---|---|---|---|---|---|
| A5 - A4(R2) | pn_bacc_uniform | -0.0022 | -0.0073 | 0.0028 | 0.3838 | 1.0000 |
| A5 - A4(R2) | macro_f1 | -0.0005 | -0.0054 | 0.0042 | 0.8216 | nan |
| A5 - A4(R2) | macro_auroc | -0.0001 | -0.0023 | 0.0022 | 0.8806 | nan |
| A6 - A4(R2) | pn_bacc_uniform | -0.0001 | -0.0056 | 0.0051 | 0.9545 | 1.0000 |
| A6 - A4(R2) | macro_f1 | 0.0019 | -0.0033 | 0.0071 | 0.4898 | nan |
| A6 - A4(R2) | macro_auroc | -0.0013 | -0.0036 | 0.0011 | 0.2879 | nan |
| A7 - A4(R2) | pn_bacc_uniform | -0.0027 | -0.0088 | 0.0035 | 0.3798 | 1.0000 |
| A7 - A4(R2) | macro_f1 | 0.0030 | -0.0032 | 0.0095 | 0.3768 | nan |
| A7 - A4(R2) | macro_auroc | -0.0039 | -0.0069 | -0.0010 | 0.0130 | nan |
| A8 - A4(R2) | pn_bacc_uniform | 0.0002 | -0.0050 | 0.0055 | 0.9065 | 1.0000 |
| A8 - A4(R2) | macro_f1 | 0.0020 | -0.0030 | 0.0070 | 0.4428 | nan |
| A8 - A4(R2) | macro_auroc | -0.0009 | -0.0031 | 0.0013 | 0.4208 | nan |

### secondary checkpoint rule

| label | op | n | pn_bacc | pn_bacc_lo | pn_bacc_hi | macro_f1 | macro_f1_lo | macro_f1_hi | macro_auroc | macro_auroc_lo | macro_auroc_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2_A4 (comparator) | uniform | 13240 | 0.6208 | 0.6074 | 0.6345 | 0.5738 | 0.5602 | 0.5874 | 0.7733 | 0.7626 | 0.7840 |
| A5 (EXPLORATORY) | uniform | 13240 | 0.6235 | 0.6105 | 0.6364 | 0.5718 | 0.5582 | 0.5849 | 0.7747 | 0.7639 | 0.7852 |
| A6 (EXPLORATORY) | uniform | 13240 | 0.6227 | 0.6096 | 0.6359 | 0.5756 | 0.5618 | 0.5886 | 0.7757 | 0.7650 | 0.7863 |
| A7 (EXPLORATORY) | uniform | 13240 | 0.6254 | 0.6127 | 0.6381 | 0.5663 | 0.5528 | 0.5799 | 0.7735 | 0.7629 | 0.7840 |
| A8 (EXPLORATORY) | uniform | 13240 | 0.6238 | 0.6110 | 0.6368 | 0.5678 | 0.5541 | 0.5814 | 0.7729 | 0.7624 | 0.7832 |

| comparison | metric | diff | ci_low | ci_high | p | holm_p |
|---|---|---|---|---|---|---|
| A5 - A4(R2) | pn_bacc_uniform | 0.0027 | -0.0019 | 0.0073 | 0.2509 | 0.7526 |
| A5 - A4(R2) | macro_f1 | -0.0021 | -0.0072 | 0.0030 | 0.4378 | nan |
| A5 - A4(R2) | macro_auroc | 0.0013 | -0.0011 | 0.0038 | 0.3228 | nan |
| A6 - A4(R2) | pn_bacc_uniform | 0.0018 | -0.0024 | 0.0060 | 0.3948 | 0.7526 |
| A6 - A4(R2) | macro_f1 | 0.0018 | -0.0028 | 0.0063 | 0.4668 | nan |
| A6 - A4(R2) | macro_auroc | 0.0024 | 0.0005 | 0.0043 | 0.0210 | nan |
| A7 - A4(R2) | pn_bacc_uniform | 0.0045 | -0.0012 | 0.0103 | 0.1119 | 0.4478 |
| A7 - A4(R2) | macro_f1 | -0.0075 | -0.0140 | -0.0012 | 0.0250 | nan |
| A7 - A4(R2) | macro_auroc | 0.0002 | -0.0031 | 0.0035 | 0.9115 | nan |
| A8 - A4(R2) | pn_bacc_uniform | 0.0029 | -0.0027 | 0.0086 | 0.3038 | 0.7526 |
| A8 - A4(R2) | macro_f1 | -0.0060 | -0.0122 | 0.0004 | 0.0720 | nan |
| A8 - A4(R2) | macro_auroc | -0.0005 | -0.0037 | 0.0028 | 0.7666 | nan |

### Best-epoch distributions

| arm | ckpt_rule | runs | median | mean | min | max | n_below_5 | flag_median_below_5 | n_eq_0 | flag_epoch_0 |
|---|---|---|---|---|---|---|---|---|---|---|
| R2_A4 | primary | 15 | 23.0000 | 33.8000 | 7 | 82 | 0 | False | 0 | False |
| R2_A4 | secondary | 15 | 15.0000 | 22.6667 | 3 | 82 | 2 | False | 0 | False |
| A5 | primary | 15 | 32.0000 | 37.0000 | 8 | 127 | 0 | False | 0 | False |
| A5 | secondary | 15 | 11.0000 | 16.1333 | 4 | 49 | 2 | False | 0 | False |
| A6 | primary | 15 | 32.0000 | 35.4667 | 7 | 105 | 0 | False | 0 | False |
| A6 | secondary | 15 | 13.0000 | 21.5333 | 5 | 105 | 0 | False | 0 | False |
| A7 | primary | 15 | 38.0000 | 43.3333 | 6 | 106 | 0 | False | 0 | False |
| A7 | secondary | 15 | 9.0000 | 14.0000 | 1 | 60 | 2 | False | 0 | False |
| A8 | primary | 15 | 32.0000 | 37.6000 | 7 | 94 | 0 | False | 0 | False |
| A8 | secondary | 15 | 8.0000 | 13.7333 | 2 | 61 | 3 | False | 0 | False |

### Embedding-swap test (A5-A8, test time)

| arm | ckpt_rule | n | mean_tv | median_tv | p95_tv | same_pattern_share | mean_tv_diff_pattern |
|---|---|---|---|---|---|---|---|
| A5 | primary | 39720 | 0.0021 | 0.0013 | 0.0068 | 0.2094 | 0.0027 |
| A5 | secondary | 39720 | 0.0020 | 0.0014 | 0.0063 | 0.2094 | 0.0025 |
| A6 | primary | 39720 | 0.0022 | 0.0013 | 0.0073 | 0.2094 | 0.0027 |
| A6 | secondary | 39720 | 0.0021 | 0.0014 | 0.0064 | 0.2094 | 0.0026 |
| A7 | primary | 39720 | 0.0023 | 0.0014 | 0.0081 | 0.2094 | 0.0029 |
| A7 | secondary | 39720 | 0.0022 | 0.0017 | 0.0064 | 0.2094 | 0.0028 |
| A8 | primary | 39720 | 0.0023 | 0.0014 | 0.0077 | 0.2094 | 0.0029 |
| A8 | secondary | 39720 | 0.0021 | 0.0015 | 0.0065 | 0.2094 | 0.0027 |

### Mask-intervention diagnostic (Step 10 remove vs content swap, primary rule)

| arm | modality | n_episodes | mean_tv_remove | mean_tv_swap | ratio_remove_over_swap |
|---|---|---|---|---|---|
| A5 | amy | 12951 | 0.1200 | 0.2136 | 0.5615 |
| A5 | fdg | 14763 | 0.0735 | 0.1244 | 0.5904 |
| A5 | mri | 22068 | 0.1945 | 0.3512 | 0.5538 |
| A5 | tau | 8082 | 0.1495 | 0.2676 | 0.5586 |
| A6 | amy | 12951 | 0.1160 | 0.2073 | 0.5593 |
| A6 | fdg | 14763 | 0.0734 | 0.1250 | 0.5875 |
| A6 | mri | 22068 | 0.1891 | 0.3425 | 0.5522 |
| A6 | tau | 8082 | 0.1479 | 0.2674 | 0.5531 |
| A7 | amy | 12951 | 0.1210 | 0.2106 | 0.5743 |
| A7 | fdg | 14763 | 0.0736 | 0.1223 | 0.6015 |
| A7 | mri | 22068 | 0.1984 | 0.3509 | 0.5655 |
| A7 | tau | 8082 | 0.1597 | 0.2832 | 0.5640 |
| A8 | amy | 12951 | 0.1173 | 0.2100 | 0.5585 |
| A8 | fdg | 14763 | 0.0733 | 0.1268 | 0.5781 |
| A8 | mri | 22068 | 0.1892 | 0.3424 | 0.5525 |
| A8 | tau | 8082 | 0.1482 | 0.2669 | 0.5553 |
| A5 | ALL | 57864 | 0.1406 | 0.2509 | 0.5606 |
| A6 | ALL | 57864 | 0.1375 | 0.2463 | 0.5583 |
| A7 | ALL | 57864 | 0.1438 | 0.2517 | 0.5714 |
| A8 | ALL | 57864 | 0.1378 | 0.2472 | 0.5574 |

==============================================================================
EXPECTED VS ACTUAL

| check | expected | actual | pass |
|---|---|---|---|
| S2 confirmatory runs | 15 | 15 | yes |
| S1 exploratory runs (4 arms x 3 seeds x 5 folds) | 60 | 60 | yes |
| errors | 0 | 0 | yes |
| config pre-registered before any run | yes | yes | yes |
| comparators reused (Step 10 S2, Step 11a R2_A4) | yes | yes | yes |
| both checkpoint rules | yes | yes | yes |
| every table script-generated | yes | yes | yes |
| unit tests for A5-A8 invariance / prototypes / MoE router | yes | yes | yes |

==============================================================================
DEVIATIONS FROM THE SPEC AND THINGS I AM UNSURE ABOUT

* None planned. The exploratory model adds its head parameters to the A0 backbone; the
  A7 MoE router input width is d_model + 16 + 4. The embedding-swap donors are drawn
  deterministically per (seed, fold) and never coincide with the episode itself.
* The consistency loss is the soft cross-entropy of view-1 probabilities against the
  DETACHED view-2 probabilities (stop-gradient on view 2), batch-mean, weight 0.1 -
  stated explicitly above because the spec says 'state the exact loss' for A8.

==============================================================================
FILES WRITTEN

results/step11b/: s2/{runs,oof_all.parquet,run_log.jsonl}, s1/{runs,curves,swap,
intervention,oof_all.parquet,swap_all.parquet,intervention_raw.csv,run_log*.jsonl},
confirmatory_{primary,secondary}.{csv,md}, confirmatory_paired_{primary,secondary}.csv,
exploratory_{primary,secondary}.{csv,md}, exploratory_paired_{primary,secondary}.csv,
best_epochs.{csv,md}, embedding_swap.{csv,md}, intervention.{csv,md}, analysis.json,
REPORT.md. New code: src/adni/exploratory.py, scripts/run_step11b_s2.py,
scripts/run_step11b_s1.py, scripts/analyse_step11b.py, scripts/step11b_report.py,
scripts/step11b_appendix.py, configs/step11b.yaml, tests/test_step11b_exploratory.py.

