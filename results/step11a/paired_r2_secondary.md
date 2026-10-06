# Step 11a - paired comparisons under R2 (secondary checkpoint rule)

Seed-averaged OOF, paired subject-cluster bootstrap (2,000), point [95% CI]. The two pre-registered PN-BAcc(u) comparisons (A4-A0, A4-A1) carry a Holm adjustment; macro-F1 is judged by the pre-specified non-inferiority margin 0.02.

| comparison | metric | diff | 95% CI | p | holm p |
|---|---|---|---|---|---|
| A4(R2) - R2_A0 | pn_bacc_uniform | 0.0155 | [0.0067, 0.0248] | 0.0010 | 0.0020 |
| A4(R2) - R2_A0 | macro_f1 | -0.0390 | [-0.0478, -0.0301] | 0.0010 | - |
| A4(R2) - R2_A0 | macro_auroc | -0.0191 | [-0.0231, -0.0151] | 0.0010 | - |
| A4(R2) - R2_A1 | pn_bacc_uniform | 0.0073 | [0.0006, 0.0142] | 0.0320 | 0.0320 |
| A4(R2) - R2_A1 | macro_f1 | -0.0158 | [-0.0227, -0.0086] | 0.0010 | - |
| A4(R2) - R2_A1 | macro_auroc | -0.0143 | [-0.0179, -0.0106] | 0.0010 | - |

**H3 verdict under R2 (comparators available at R2: A0, A1)** - wins PN-BAcc(u) on both after Holm = True; macro-F1 non-inferior = False -> **trade-off (A4 wins PN-BAcc but fails macro-F1 non-inferiority)**.

## Carry-over check with the full Step 9 comparator set

The Step 9 H3 rule uses four comparators (A0-A3). Only A0 and A1 are retrained under R2, so the two missing comparators are carried over from recipe 1 (Step 9 A2/A3) and shown here for completeness; this mixes recipes and is a diagnostic, not the pre-registered test.

| comparison | metric | diff | 95% CI | p | holm p (4) |
|---|---|---|---|---|---|
| A4(R2) - R1_A2 | pn_bacc_uniform | 0.0162 | [0.0063, 0.0263] | 0.0020 | 0.0060 |
| A4(R2) - R1_A2 | macro_f1 | -0.0425 | [-0.0519, -0.0328] | 0.0010 | - |
| A4(R2) - R1_A2 | macro_auroc | -0.0196 | [-0.0241, -0.0153] | 0.0010 | - |
| A4(R2) - R1_A3 | pn_bacc_uniform | 0.0011 | [-0.0052, 0.0075] | 0.7726 | 0.7726 |
| A4(R2) - R1_A3 | macro_f1 | -0.0032 | [-0.0098, 0.0033] | 0.3418 | - |
| A4(R2) - R1_A3 | macro_auroc | 0.0023 | [-0.0012, 0.0055] | 0.2049 | - |

Full-rule verdict under R2 (Holm over the four PN-BAcc(u) comparisons) = wins False, non-inferior False -> **negative result**.
