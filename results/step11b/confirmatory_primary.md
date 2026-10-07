# Step 11b (A) - confirmatory S2 check, recipe 1 (tau 0.5 pre-declared rule applied under R1; R2 replication below) (primary checkpoint rule)

Train = S2 train side (ADNI1/GO/2) with the frozen S2 val holdout; test = ADNI3/4 subjects. Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap. PN-BAcc weights from the S2 TRAIN-split pi_hat (rho uniform). Step 10 comparators reused as-is; tau = 0.5 was chosen from S1 Step 11a results, so this is the held-out check.

| run-set | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|
| A4(tau 0.5) [R1] | uniform | 3565 | 0.6159 [0.5863, 0.6445] | 0.6051 [0.5782, 0.6306] | 0.7766 [0.7570, 0.7959] |
| A0 [R2] | argmax | 3565 | 0.6058 [0.5772, 0.6340] | 0.6161 [0.5896, 0.6414] | 0.8036 [0.7859, 0.8214] |
| A1 [R2] | uniform | 3565 | 0.6126 [0.5845, 0.6403] | 0.5839 [0.5572, 0.6094] | 0.7790 [0.7594, 0.7980] |
| A4(tau 1) [R2] | uniform | 3565 | 0.6040 [0.5750, 0.6306] | 0.5666 [0.5395, 0.5912] | 0.7466 [0.7258, 0.7669] |
| A4(tau 0.5) [R2] | uniform | 3565 | 0.6269 [0.5988, 0.6536] | 0.6096 [0.5823, 0.6350] | 0.7911 [0.7726, 0.8098] |
| A0 [R1, Step 10] | argmax | 3565 | 0.6071 [0.5769, 0.6346] | 0.6133 [0.5858, 0.6390] | 0.7959 [0.7772, 0.8142] |
| A1 [R1, Step 10] | uniform | 3565 | 0.6161 [0.5872, 0.6438] | 0.5952 [0.5681, 0.6195] | 0.7860 [0.7668, 0.8051] |
| A4(tau 1) [R1, Step 10] | uniform | 3565 | 0.5997 [0.5705, 0.6270] | 0.5628 [0.5343, 0.5877] | 0.7440 [0.7230, 0.7638] |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4(tau 0.5) - A0 [R1] | pn_bacc_uniform | 0.0088 [-0.0056, 0.0231] | 0.2459 |
| A4(tau 0.5) - A0 [R1] | macro_f1 | -0.0082 [-0.0225, 0.0060] | 0.2579 |
| A4(tau 0.5) - A0 [R1] | macro_auroc | -0.0193 [-0.0249, -0.0140] | 0.0010 |
| A4(tau 0.5) - A1 [R1] | pn_bacc_uniform | -0.0002 [-0.0110, 0.0102] | 0.9785 |
| A4(tau 0.5) - A1 [R1] | macro_f1 | 0.0098 [-0.0025, 0.0216] | 0.1149 |
| A4(tau 0.5) - A1 [R1] | macro_auroc | -0.0094 [-0.0130, -0.0060] | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R1] | pn_bacc_uniform | 0.0162 [0.0012, 0.0320] | 0.0320 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_f1 | 0.0423 [0.0282, 0.0572] | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_auroc | 0.0326 [0.0258, 0.0397] | 0.0010 |
| A4(tau 0.5) - A0 [R2] | pn_bacc_uniform | 0.0211 [0.0054, 0.0366] | 0.0060 |
| A4(tau 0.5) - A0 [R2] | macro_f1 | -0.0065 [-0.0232, 0.0089] | 0.4298 |
| A4(tau 0.5) - A0 [R2] | macro_auroc | -0.0125 [-0.0169, -0.0080] | 0.0010 |
| A4(tau 0.5) - A1 [R2] | pn_bacc_uniform | 0.0142 [-0.0015, 0.0302] | 0.0770 |
| A4(tau 0.5) - A1 [R2] | macro_f1 | 0.0257 [0.0094, 0.0420] | 0.0020 |
| A4(tau 0.5) - A1 [R2] | macro_auroc | 0.0121 [0.0049, 0.0196] | 0.0020 |
| A4(tau 0.5) - A4(tau 1) [R2] | pn_bacc_uniform | 0.0229 [0.0052, 0.0412] | 0.0120 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_f1 | 0.0431 [0.0252, 0.0621] | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_auroc | 0.0445 [0.0356, 0.0547] | 0.0010 |

**Pre-declared reading (primary):** PN-BAcc(A4 tau 0.5 - A0) CI low = -0.0056 (> 0: False); macro-F1 CI low = -0.0225 (> -0.02: False) -> **not confirmed**.

**R2 replication (primary):** PN-BAcc(A4 tau 0.5 [R2] - A0 [R2]) CI low = 0.0054 (> 0: True); macro-F1 CI low = -0.0232 (> -0.02: False) -> **not confirmed**.
