# Step 9 - primary paired comparisons (primary checkpoint rule)

A4(primary/rho-uniform) minus each of A0 (argmax), A1 (rho-uniform), A2 (argmax), A3 (argmax); paired subject-cluster bootstrap (2,000), point [95% CI], two-sided p. Holm correction over the four PN-BAcc(uniform) comparisons. Non-inferiority margin for macro-F1: 0.02.

| comparison | metric | diff | 95% CI | p | holm p |
|---|---|---|---|---|---|
| A4 - A0 | pn_bacc_uniform | 0.0181 | [0.0088, 0.0275] | 0.0020 | 0.0060 |
| A4 - A0 | macro_f1 | -0.0235 | [-0.0328, -0.0146] | 0.0010 | - |
| A4 - A0 | macro_auroc | -0.0203 | [-0.0251, -0.0156] | 0.0010 | - |
| A4 - A1 | pn_bacc_uniform | 0.0069 | [-0.0006, 0.0141] | 0.0820 | 0.1639 |
| A4 - A1 | macro_f1 | -0.0109 | [-0.0189, -0.0036] | 0.0030 | - |
| A4 - A1 | macro_auroc | -0.0127 | [-0.0171, -0.0084] | 0.0010 | - |
| A4 - A2 | pn_bacc_uniform | 0.0189 | [0.0090, 0.0293] | 0.0010 | 0.0040 |
| A4 - A2 | macro_f1 | -0.0252 | [-0.0348, -0.0154] | 0.0010 | - |
| A4 - A2 | macro_auroc | -0.0204 | [-0.0256, -0.0154] | 0.0010 | - |
| A4 - A3 | pn_bacc_uniform | 0.0020 | [-0.0047, 0.0084] | 0.5657 | 0.5657 |
| A4 - A3 | macro_f1 | -0.0036 | [-0.0101, 0.0027] | 0.2569 | - |
| A4 - A3 | macro_auroc | 0.0011 | [-0.0024, 0.0047] | 0.5407 | - |
