# Step 11b (A) - confirmatory S2 check, recipe 1 (tau 0.5 pre-declared rule applied under R1; R2 replication below) (secondary checkpoint rule)

Train = S2 train side (ADNI1/GO/2) with the frozen S2 val holdout; test = ADNI3/4 subjects. Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap. PN-BAcc weights from the S2 TRAIN-split pi_hat (rho uniform). Step 10 comparators reused as-is; tau = 0.5 was chosen from S1 Step 11a results, so this is the held-out check.

| run-set | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|
| A4(tau 0.5) [R1] | uniform | 3565 | 0.6200 [0.5919, 0.6481] | 0.5984 [0.5722, 0.6246] | 0.7844 [0.7654, 0.8032] |
| A0 [R2] | argmax | 3565 | 0.6109 [0.5821, 0.6389] | 0.6180 [0.5911, 0.6429] | 0.8056 [0.7879, 0.8234] |
| A1 [R2] | uniform | 3565 | 0.6212 [0.5941, 0.6455] | 0.5795 [0.5532, 0.6041] | 0.7980 [0.7797, 0.8160] |
| A4(tau 1) [R2] | uniform | 3565 | 0.6286 [0.6035, 0.6519] | 0.5713 [0.5444, 0.5958] | 0.7606 [0.7404, 0.7802] |
| A4(tau 0.5) [R2] | uniform | 3565 | 0.6299 [0.6020, 0.6564] | 0.6083 [0.5807, 0.6338] | 0.7932 [0.7749, 0.8120] |
| A0 [R1, Step 10] | argmax | 3565 | 0.6086 [0.5787, 0.6361] | 0.6147 [0.5871, 0.6396] | 0.7972 [0.7793, 0.8150] |
| A1 [R1, Step 10] | uniform | 3565 | 0.6213 [0.5948, 0.6457] | 0.5847 [0.5583, 0.6100] | 0.7912 [0.7730, 0.8091] |
| A4(tau 1) [R1, Step 10] | uniform | 3565 | 0.6347 [0.6080, 0.6591] | 0.5806 [0.5533, 0.6058] | 0.7640 [0.7443, 0.7832] |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4(tau 0.5) - A0 [R1] | pn_bacc_uniform | 0.0113 [-0.0016, 0.0256] | 0.0970 |
| A4(tau 0.5) - A0 [R1] | macro_f1 | -0.0164 [-0.0309, -0.0023] | 0.0300 |
| A4(tau 0.5) - A0 [R1] | macro_auroc | -0.0129 [-0.0167, -0.0093] | 0.0010 |
| A4(tau 0.5) - A1 [R1] | pn_bacc_uniform | -0.0013 [-0.0173, 0.0137] | 0.8796 |
| A4(tau 0.5) - A1 [R1] | macro_f1 | 0.0137 [-0.0019, 0.0293] | 0.0840 |
| A4(tau 0.5) - A1 [R1] | macro_auroc | -0.0068 [-0.0117, -0.0021] | 0.0040 |
| A4(tau 0.5) - A4(tau 1) [R1] | pn_bacc_uniform | -0.0148 [-0.0353, 0.0062] | 0.1779 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_f1 | 0.0178 [-0.0011, 0.0371] | 0.0710 |
| A4(tau 0.5) - A4(tau 1) [R1] | macro_auroc | 0.0204 [0.0125, 0.0280] | 0.0010 |
| A4(tau 0.5) - A0 [R2] | pn_bacc_uniform | 0.0190 [0.0035, 0.0355] | 0.0130 |
| A4(tau 0.5) - A0 [R2] | macro_f1 | -0.0097 [-0.0261, 0.0070] | 0.2479 |
| A4(tau 0.5) - A0 [R2] | macro_auroc | -0.0124 [-0.0167, -0.0082] | 0.0010 |
| A4(tau 0.5) - A1 [R2] | pn_bacc_uniform | 0.0087 [-0.0036, 0.0210] | 0.1749 |
| A4(tau 0.5) - A1 [R2] | macro_f1 | 0.0288 [0.0154, 0.0430] | 0.0010 |
| A4(tau 0.5) - A1 [R2] | macro_auroc | -0.0048 [-0.0086, -0.0008] | 0.0160 |
| A4(tau 0.5) - A4(tau 1) [R2] | pn_bacc_uniform | 0.0013 [-0.0162, 0.0184] | 0.8326 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_f1 | 0.0370 [0.0197, 0.0545] | 0.0010 |
| A4(tau 0.5) - A4(tau 1) [R2] | macro_auroc | 0.0326 [0.0267, 0.0383] | 0.0010 |

**Pre-declared reading (secondary):** PN-BAcc(A4 tau 0.5 - A0) CI low = -0.0016 (> 0: False); macro-F1 CI low = -0.0309 (> -0.02: False) -> **not confirmed**.

**R2 replication (secondary):** PN-BAcc(A4 tau 0.5 [R2] - A0 [R2]) CI low = 0.0035 (> 0: True); macro-F1 CI low = -0.0261 (> -0.02: False) -> **not confirmed**.
