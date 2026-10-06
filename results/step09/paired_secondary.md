# Step 9 - primary paired comparisons (secondary checkpoint rule)

A4(primary/rho-uniform) minus each of A0 (argmax), A1 (rho-uniform), A2 (argmax), A3 (argmax); paired subject-cluster bootstrap (2,000), point [95% CI], two-sided p. Holm correction over the four PN-BAcc(uniform) comparisons. Non-inferiority margin for macro-F1: 0.02.

| comparison | metric | diff | 95% CI | p | holm p |
|---|---|---|---|---|---|
| A4 - A0 | pn_bacc_uniform | 0.0225 | [0.0129, 0.0323] | 0.0010 | 0.0040 |
| A4 - A0 | macro_f1 | -0.0378 | [-0.0472, -0.0283] | 0.0010 | - |
| A4 - A0 | macro_auroc | -0.0202 | [-0.0245, -0.0156] | 0.0010 | - |
| A4 - A1 | pn_bacc_uniform | 0.0087 | [0.0014, 0.0160] | 0.0200 | 0.0400 |
| A4 - A1 | macro_f1 | -0.0137 | [-0.0207, -0.0066] | 0.0010 | - |
| A4 - A1 | macro_auroc | -0.0173 | [-0.0214, -0.0131] | 0.0010 | - |
| A4 - A2 | pn_bacc_uniform | 0.0176 | [0.0076, 0.0276] | 0.0020 | 0.0060 |
| A4 - A2 | macro_f1 | -0.0413 | [-0.0506, -0.0319] | 0.0010 | - |
| A4 - A2 | macro_auroc | -0.0218 | [-0.0262, -0.0171] | 0.0010 | - |
| A4 - A3 | pn_bacc_uniform | 0.0025 | [-0.0035, 0.0080] | 0.3798 | 0.3798 |
| A4 - A3 | macro_f1 | -0.0020 | [-0.0077, 0.0034] | 0.4958 | - |
| A4 - A3 | macro_auroc | 0.0001 | [-0.0030, 0.0030] | 0.9585 | - |
