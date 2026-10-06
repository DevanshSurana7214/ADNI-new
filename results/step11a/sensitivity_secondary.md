# Step 11a - sensitivity paired against recipe 1 (secondary checkpoint rule)

Each sensitivity setting minus its recipe-1 counterpart (Step 9 arm), paired subject-cluster bootstrap (2,000), point [95% CI]. A CI that excludes 0 marks a setting that materially changes that metric.

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4_tau0p5 - R1_A4 | pn_bacc_uniform | -0.0065 | [-0.0144, 0.0011] | 0.0900 |
| A4_tau0p5 - R1_A4 | macro_f1 | 0.0230 | [0.0155, 0.0300] | 0.0010 |
| A4_tau0p5 - R1_A4 | macro_auroc | 0.0150 | [0.0112, 0.0187] | 0.0010 |
| A4_tau1p5 - R1_A4 | pn_bacc_uniform | -0.0146 | [-0.0212, -0.0081] | 0.0010 |
| A4_tau1p5 - R1_A4 | macro_f1 | -0.0266 | [-0.0338, -0.0200] | 0.0010 |
| A4_tau1p5 - R1_A4 | macro_auroc | -0.0141 | [-0.0181, -0.0103] | 0.0010 |
| A3_clip0p5_2 - R1_A3 | pn_bacc_uniform | 0.0027 | [-0.0027, 0.0077] | 0.3218 |
| A3_clip0p5_2 - R1_A3 | macro_f1 | 0.0019 | [-0.0031, 0.0069] | 0.4588 |
| A3_clip0p5_2 - R1_A3 | macro_auroc | -0.0010 | [-0.0034, 0.0013] | 0.3888 |
| A3_clip0p1_10 - R1_A3 | pn_bacc_uniform | 0.0032 | [-0.0009, 0.0072] | 0.1629 |
| A3_clip0p1_10 - R1_A3 | macro_f1 | 0.0014 | [-0.0032, 0.0055] | 0.5787 |
| A3_clip0p1_10 - R1_A3 | macro_auroc | -0.0002 | [-0.0021, 0.0015] | 0.7886 |
| A0_pdrop0 - R1_A0 | pn_bacc_uniform | 0.0007 | [-0.0064, 0.0081] | 0.8596 |
| A0_pdrop0 - R1_A0 | macro_f1 | 0.0036 | [-0.0030, 0.0101] | 0.2839 |
| A0_pdrop0 - R1_A0 | macro_auroc | 0.0019 | [-0.0014, 0.0053] | 0.2609 |
| A0_pdrop0p5 - R1_A0 | pn_bacc_uniform | 0.0001 | [-0.0062, 0.0065] | 0.9775 |
| A0_pdrop0p5 - R1_A0 | macro_f1 | -0.0047 | [-0.0110, 0.0015] | 0.1449 |
| A0_pdrop0p5 - R1_A0 | macro_auroc | -0.0019 | [-0.0050, 0.0013] | 0.2309 |
| A4_pdrop0 - R1_A4 | pn_bacc_uniform | -0.0026 | [-0.0090, 0.0036] | 0.3988 |
| A4_pdrop0 - R1_A4 | macro_f1 | 0.0006 | [-0.0059, 0.0070] | 0.8896 |
| A4_pdrop0 - R1_A4 | macro_auroc | -0.0000 | [-0.0036, 0.0033] | 0.9515 |
| A4_pdrop0p5 - R1_A4 | pn_bacc_uniform | -0.0066 | [-0.0135, -0.0003] | 0.0400 |
| A4_pdrop0p5 - R1_A4 | macro_f1 | -0.0013 | [-0.0083, 0.0056] | 0.7036 |
| A4_pdrop0p5 - R1_A4 | macro_auroc | -0.0011 | [-0.0047, 0.0027] | 0.5597 |

Settings with a CI excluding 0: 6 of 24 rows (['A4_pdrop0p5 - R1_A4', 'A4_tau0p5 - R1_A4', 'A4_tau1p5 - R1_A4']).
