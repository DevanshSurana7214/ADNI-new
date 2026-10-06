# Step 10 - placebo and weight-shuffle controls (secondary checkpoint rule)

Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap (Step 6b protocol). point [95% CI]. C1 = A4 with pi(c|m~) from labels permuted within ORIGPROT (train split, true labels still used by the loss); C-shuf = A3 with per-episode weights shuffled within each batch.

| arm | op | n | macro-AUROC | macro-F1 | PN-BAcc(u) |
|---|---|---|---|---|---|
| A3 | argmax | 13240 | 0.7710 [0.7605, 0.7818] | 0.5770 [0.5633, 0.5908] | 0.6198 [0.6060, 0.6338] |
| Cshuf | argmax | 13240 | 0.7932 [0.7827, 0.8036] | 0.6132 [0.5998, 0.6265] | 0.6012 [0.5864, 0.6155] |
| A4 | uniform | 13240 | 0.7711 [0.7608, 0.7816] | 0.5751 [0.5615, 0.5879] | 0.6223 [0.6089, 0.6355] |
| C1 | uniform | 13240 | 0.7826 [0.7722, 0.7931] | 0.5814 [0.5677, 0.5951] | 0.6108 [0.5970, 0.6248] |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4 - C1 | pn_bacc_uniform | 0.0115 | [0.0053, 0.0182] | 0.0010 |
| A4 - C1 | macro_f1 | -0.0063 | [-0.0127, -0.0004] | 0.0360 |
| A4 - C1 | macro_auroc | -0.0114 | [-0.0147, -0.0082] | 0.0010 |
| A3 - Cshuf | pn_bacc_uniform | 0.0186 | [0.0094, 0.0289] | 0.0010 |
| A3 - Cshuf | macro_f1 | -0.0362 | [-0.0450, -0.0264] | 0.0010 |
| A3 - Cshuf | macro_auroc | -0.0221 | [-0.0269, -0.0168] | 0.0010 |

Pre-declared reading: the joint prior is supported only if the CI lower bound of A4 - C1 on PN-BAcc is above 0. Here the lower bound is 0.0053 -> **SUPPORTED**.
