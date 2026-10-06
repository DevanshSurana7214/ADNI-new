# Step 11a - sensitivity paired against recipe 1 (primary checkpoint rule)

Each sensitivity setting minus its recipe-1 counterpart (Step 9 arm), paired subject-cluster bootstrap (2,000), point [95% CI]. A CI that excludes 0 marks a setting that materially changes that metric.

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4_tau0p5 - R1_A4 | pn_bacc_uniform | -0.0007 | [-0.0072, 0.0061] | 0.8536 |
| A4_tau0p5 - R1_A4 | macro_f1 | 0.0165 | [0.0102, 0.0233] | 0.0010 |
| A4_tau0p5 - R1_A4 | macro_auroc | 0.0114 | [0.0081, 0.0147] | 0.0010 |
| A4_tau1p5 - R1_A4 | pn_bacc_uniform | -0.0116 | [-0.0188, -0.0045] | 0.0010 |
| A4_tau1p5 - R1_A4 | macro_f1 | -0.0185 | [-0.0258, -0.0111] | 0.0010 |
| A4_tau1p5 - R1_A4 | macro_auroc | -0.0138 | [-0.0177, -0.0097] | 0.0010 |
| A3_clip0p5_2 - R1_A3 | pn_bacc_uniform | 0.0038 | [-0.0019, 0.0090] | 0.1829 |
| A3_clip0p5_2 - R1_A3 | macro_f1 | 0.0028 | [-0.0024, 0.0078] | 0.2779 |
| A3_clip0p5_2 - R1_A3 | macro_auroc | 0.0010 | [-0.0017, 0.0036] | 0.4808 |
| A3_clip0p1_10 - R1_A3 | pn_bacc_uniform | 0.0017 | [-0.0020, 0.0052] | 0.3668 |
| A3_clip0p1_10 - R1_A3 | macro_f1 | -0.0017 | [-0.0055, 0.0019] | 0.3348 |
| A3_clip0p1_10 - R1_A3 | macro_auroc | -0.0002 | [-0.0022, 0.0016] | 0.7916 |
| A0_pdrop0 - R1_A0 | pn_bacc_uniform | -0.0028 | [-0.0095, 0.0039] | 0.3998 |
| A0_pdrop0 - R1_A0 | macro_f1 | 0.0048 | [-0.0015, 0.0109] | 0.1299 |
| A0_pdrop0 - R1_A0 | macro_auroc | 0.0004 | [-0.0027, 0.0032] | 0.7786 |
| A0_pdrop0p5 - R1_A0 | pn_bacc_uniform | -0.0002 | [-0.0066, 0.0061] | 0.9225 |
| A0_pdrop0p5 - R1_A0 | macro_f1 | 0.0010 | [-0.0053, 0.0071] | 0.7796 |
| A0_pdrop0p5 - R1_A0 | macro_auroc | -0.0042 | [-0.0074, -0.0010] | 0.0090 |
| A4_pdrop0 - R1_A4 | pn_bacc_uniform | -0.0059 | [-0.0131, 0.0017] | 0.1219 |
| A4_pdrop0 - R1_A4 | macro_f1 | 0.0032 | [-0.0040, 0.0106] | 0.3468 |
| A4_pdrop0 - R1_A4 | macro_auroc | -0.0003 | [-0.0042, 0.0036] | 0.8706 |
| A4_pdrop0p5 - R1_A4 | pn_bacc_uniform | -0.0015 | [-0.0084, 0.0057] | 0.6617 |
| A4_pdrop0p5 - R1_A4 | macro_f1 | -0.0022 | [-0.0089, 0.0048] | 0.5157 |
| A4_pdrop0p5 - R1_A4 | macro_auroc | -0.0030 | [-0.0064, 0.0005] | 0.0860 |

Settings with a CI excluding 0: 6 of 24 rows (['A0_pdrop0p5 - R1_A0', 'A4_tau0p5 - R1_A4', 'A4_tau1p5 - R1_A4']).
