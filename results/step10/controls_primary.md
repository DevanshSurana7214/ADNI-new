# Step 10 - placebo and weight-shuffle controls (primary checkpoint rule)

Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap (Step 6b protocol). point [95% CI]. C1 = A4 with pi(c|m~) from labels permuted within ORIGPROT (train split, true labels still used by the loss); C-shuf = A3 with per-episode weights shuffled within each batch.

| arm | op | n | macro-AUROC | macro-F1 | PN-BAcc(u) |
|---|---|---|---|---|---|
| A3 | argmax | 13240 | 0.7724 [0.7617, 0.7834] | 0.5937 [0.5795, 0.6073] | 0.6143 [0.6000, 0.6290] |
| Cshuf | argmax | 13240 | 0.7941 [0.7832, 0.8046] | 0.6162 [0.6028, 0.6295] | 0.5931 [0.5779, 0.6078] |
| A4 | uniform | 13240 | 0.7735 [0.7627, 0.7848] | 0.5902 [0.5764, 0.6037] | 0.6163 [0.6023, 0.6304] |
| C1 | uniform | 13240 | 0.7785 [0.7676, 0.7894] | 0.5942 [0.5800, 0.6075] | 0.6088 [0.5942, 0.6236] |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4 - C1 | pn_bacc_uniform | 0.0075 | [-0.0005, 0.0152] | 0.0700 |
| A4 - C1 | macro_f1 | -0.0041 | [-0.0119, 0.0038] | 0.3188 |
| A4 - C1 | macro_auroc | -0.0050 | [-0.0093, -0.0008] | 0.0220 |
| A3 - Cshuf | pn_bacc_uniform | 0.0211 | [0.0115, 0.0319] | 0.0010 |
| A3 - Cshuf | macro_f1 | -0.0225 | [-0.0319, -0.0123] | 0.0010 |
| A3 - Cshuf | macro_auroc | -0.0216 | [-0.0275, -0.0155] | 0.0010 |

Pre-declared reading: the joint prior is supported only if the CI lower bound of A4 - C1 on PN-BAcc is above 0. Here the lower bound is -0.0005 -> **NOT supported**.
