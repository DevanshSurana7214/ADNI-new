# Step 11b (B) - exploratory arms under R2, S1 grid (secondary checkpoint rule)

EXPLORATORY: no selection among A5-A8; these results must not be used to change A0-A4, the recipes, or any pre-declared rule.

Seed-averaged OOF, pooled, one 2,000-resample subject-cluster bootstrap. A5-A8 are trained with the R2 recipe on the S1 grid; the comparator is A4(R2, tau 1) from Step 11a (reused, not retrained). Holm over the four PN-BAcc comparisons.

| arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|
| R2_A4 (comparator) | uniform | 13240 | 0.6208 [0.6074, 0.6345] | 0.5738 [0.5602, 0.5874] | 0.7733 [0.7626, 0.7840] |
| A5 (EXPLORATORY) | uniform | 13240 | 0.6235 [0.6105, 0.6364] | 0.5718 [0.5582, 0.5849] | 0.7747 [0.7639, 0.7852] |
| A6 (EXPLORATORY) | uniform | 13240 | 0.6227 [0.6096, 0.6359] | 0.5756 [0.5618, 0.5886] | 0.7757 [0.7650, 0.7863] |
| A7 (EXPLORATORY) | uniform | 13240 | 0.6254 [0.6127, 0.6381] | 0.5663 [0.5528, 0.5799] | 0.7735 [0.7629, 0.7840] |
| A8 (EXPLORATORY) | uniform | 13240 | 0.6238 [0.6110, 0.6368] | 0.5678 [0.5541, 0.5814] | 0.7729 [0.7624, 0.7832] |

| comparison | metric | diff | 95% CI | p | holm p |
|---|---|---|---|---|---|
| A5 - A4(R2) | pn_bacc_uniform | 0.0027 [-0.0019, 0.0073] | 0.2509 | 0.7526 |
| A5 - A4(R2) | macro_f1 | -0.0021 [-0.0072, 0.0030] | 0.4378 | - |
| A5 - A4(R2) | macro_auroc | 0.0013 [-0.0011, 0.0038] | 0.3228 | - |
| A6 - A4(R2) | pn_bacc_uniform | 0.0018 [-0.0024, 0.0060] | 0.3948 | 0.7526 |
| A6 - A4(R2) | macro_f1 | 0.0018 [-0.0028, 0.0063] | 0.4668 | - |
| A6 - A4(R2) | macro_auroc | 0.0024 [0.0005, 0.0043] | 0.0210 | - |
| A7 - A4(R2) | pn_bacc_uniform | 0.0045 [-0.0012, 0.0103] | 0.1119 | 0.4478 |
| A7 - A4(R2) | macro_f1 | -0.0075 [-0.0140, -0.0012] | 0.0250 | - |
| A7 - A4(R2) | macro_auroc | 0.0002 [-0.0031, 0.0035] | 0.9115 | - |
| A8 - A4(R2) | pn_bacc_uniform | 0.0029 [-0.0027, 0.0086] | 0.3038 | 0.7526 |
| A8 - A4(R2) | macro_f1 | -0.0060 [-0.0122, 0.0004] | 0.0720 | - |
| A8 - A4(R2) | macro_auroc | -0.0005 [-0.0037, 0.0028] | 0.7666 | - |
