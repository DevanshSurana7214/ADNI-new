# Step 11a - paired comparisons under R2 (primary checkpoint rule)

Seed-averaged OOF, paired subject-cluster bootstrap (2,000), point [95% CI]. The two pre-registered PN-BAcc(u) comparisons (A4-A0, A4-A1) carry a Holm adjustment; macro-F1 is judged by the pre-specified non-inferiority margin 0.02.

| comparison | metric | diff | 95% CI | p | holm p |
|---|---|---|---|---|---|
| A4(R2) - R2_A0 | pn_bacc_uniform | 0.0149 | [0.0054, 0.0243] | 0.0030 | 0.0060 |
| A4(R2) - R2_A0 | macro_f1 | -0.0316 | [-0.0405, -0.0225] | 0.0010 | - |
| A4(R2) - R2_A0 | macro_auroc | -0.0196 | [-0.0246, -0.0147] | 0.0010 | - |
| A4(R2) - R2_A1 | pn_bacc_uniform | 0.0067 | [0.0001, 0.0135] | 0.0470 | 0.0470 |
| A4(R2) - R2_A1 | macro_f1 | -0.0112 | [-0.0172, -0.0047] | 0.0020 | - |
| A4(R2) - R2_A1 | macro_auroc | -0.0127 | [-0.0164, -0.0092] | 0.0010 | - |

**H3 verdict under R2 (comparators available at R2: A0, A1)** - wins PN-BAcc(u) on both after Holm = True; macro-F1 non-inferior = False -> **trade-off (A4 wins PN-BAcc but fails macro-F1 non-inferiority)**.

## Carry-over check with the full Step 9 comparator set

The Step 9 H3 rule uses four comparators (A0-A3). Only A0 and A1 are retrained under R2, so the two missing comparators are carried over from recipe 1 (Step 9 A2/A3) and shown here for completeness; this mixes recipes and is a diagnostic, not the pre-registered test.

| comparison | metric | diff | 95% CI | p | holm p (4) |
|---|---|---|---|---|---|
| A4(R2) - R1_A2 | pn_bacc_uniform | 0.0210 | [0.0111, 0.0310] | 0.0010 | 0.0040 |
| A4(R2) - R1_A2 | macro_f1 | -0.0307 | [-0.0406, -0.0210] | 0.0010 | - |
| A4(R2) - R1_A2 | macro_auroc | -0.0203 | [-0.0255, -0.0152] | 0.0010 | - |
| A4(R2) - R1_A3 | pn_bacc_uniform | 0.0041 | [-0.0033, 0.0117] | 0.2829 | 0.2829 |
| A4(R2) - R1_A3 | macro_f1 | -0.0091 | [-0.0163, -0.0019] | 0.0160 | - |
| A4(R2) - R1_A3 | macro_auroc | 0.0012 | [-0.0023, 0.0048] | 0.5397 | - |

Full-rule verdict under R2 (Holm over the four PN-BAcc(u) comparisons) = wins False, non-inferior False -> **negative result**.
