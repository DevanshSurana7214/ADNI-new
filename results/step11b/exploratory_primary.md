# Step 11b (B) - exploratory arms under R2, S1 grid (primary checkpoint rule)

EXPLORATORY: no selection among A5-A8; these results must not be used to change A0-A4, the recipes, or any pre-declared rule.

Seed-averaged OOF, pooled, one 2,000-resample subject-cluster bootstrap. A5-A8 are trained with the R2 recipe on the S1 grid; the comparator is A4(R2, tau 1) from Step 11a (reused, not retrained). Holm over the four PN-BAcc comparisons.

| arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|
| R2_A4 (comparator) | uniform | 13240 | 0.6184 [0.6048, 0.6326] | 0.5847 [0.5712, 0.5984] | 0.7736 [0.7628, 0.7847] |
| A5 (EXPLORATORY) | uniform | 13240 | 0.6162 [0.6027, 0.6301] | 0.5842 [0.5706, 0.5977] | 0.7735 [0.7624, 0.7842] |
| A6 (EXPLORATORY) | uniform | 13240 | 0.6183 [0.6052, 0.6324] | 0.5866 [0.5737, 0.6001] | 0.7724 [0.7614, 0.7835] |
| A7 (EXPLORATORY) | uniform | 13240 | 0.6157 [0.6024, 0.6297] | 0.5877 [0.5743, 0.6014] | 0.7697 [0.7586, 0.7804] |
| A8 (EXPLORATORY) | uniform | 13240 | 0.6186 [0.6051, 0.6333] | 0.5866 [0.5735, 0.6006] | 0.7727 [0.7619, 0.7836] |

| comparison | metric | diff | 95% CI | p | holm p |
|---|---|---|---|---|---|
| A5 - A4(R2) | pn_bacc_uniform | -0.0022 [-0.0073, 0.0028] | 0.3838 | 1.0000 |
| A5 - A4(R2) | macro_f1 | -0.0005 [-0.0054, 0.0042] | 0.8216 | - |
| A5 - A4(R2) | macro_auroc | -0.0001 [-0.0023, 0.0022] | 0.8806 | - |
| A6 - A4(R2) | pn_bacc_uniform | -0.0001 [-0.0056, 0.0051] | 0.9545 | 1.0000 |
| A6 - A4(R2) | macro_f1 | 0.0019 [-0.0033, 0.0071] | 0.4898 | - |
| A6 - A4(R2) | macro_auroc | -0.0013 [-0.0036, 0.0011] | 0.2879 | - |
| A7 - A4(R2) | pn_bacc_uniform | -0.0027 [-0.0088, 0.0035] | 0.3798 | 1.0000 |
| A7 - A4(R2) | macro_f1 | 0.0030 [-0.0032, 0.0095] | 0.3768 | - |
| A7 - A4(R2) | macro_auroc | -0.0039 [-0.0069, -0.0010] | 0.0130 | - |
| A8 - A4(R2) | pn_bacc_uniform | 0.0002 [-0.0050, 0.0055] | 0.9065 | 1.0000 |
| A8 - A4(R2) | macro_f1 | 0.0020 [-0.0030, 0.0070] | 0.4428 | - |
| A8 - A4(R2) | macro_auroc | -0.0009 [-0.0031, 0.0013] | 0.4208 | - |
