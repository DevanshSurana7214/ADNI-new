STEP 11A REPORT
Robustness only: the second training recipe (R2) and the recipe-1 sensitivity settings.

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/run_step11a.py --shard {0,1,2} --nshards 3 --no-combine
  scripts/run_step11a.py --combine-only            > results/step11a/{runs,curves,
                                       oof_all.parquet,run_log.jsonl}
  scripts/analyse_step11a.py                       > results/step11a/{main_table_*,runset_pc_*,
                                       paired_r2_*,best_epochs,sensitivity_*,pn_bacc_vs_macro_f1.png}
  scripts/step11a_report.py                        > results/step11a/REPORT.md
  scripts/step11a_appendix.py                      > Appendix E row

Git commit: (this commit) - tag `step11a`

Tests: 115 passed in 196.67s (0:03:16)

Grid: 165 runs ok, 0 errors (45 R2 + 120 sensitivity), 168.0 min training total across the three shards.

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1, matplotlib; NVIDIA RTX 3050 (6 GB). Deterministic algorithms on; `train_arm` seeds before building the model, so every (run-set, seed, fold) is reproducible (a forced rerun of A0_pdrop0 s0 f0 reproduced the OOF exactly). No tuning.

------------------------------------------------------------------------------
PREDECLARED DESIGN

The grid is frozen in `configs/step11a_robustness.yaml`, written before any run was seen. Recipe 1 is the Step 7b config reused by Steps 8-10; it is NOT retrained here - the A0-A4 recipe-1 baselines are exactly the Step 9 OOF (`results/step09/oof_all.parquet`).

R2 recipe (arms A0, A1, A4): lr 3e-5, weight decay 0.3, batch 128, encoder dropout 0.4, transformer dropout 0.2, max 150 epochs, patience 25; every other value (d_model, heads, layers, FFN, modality dropout 0.25, tau 1, clip [0.2, 5]) is recipe 1.

Sensitivity at recipe 1: A4 tau in {0.5, 1.5}; A3 training weight clip in {[0.5,2], [0.1,10]}; modality dropout p_drop in {0, 0.5} for A0 and A4, with the mask-process prior rebuilt under the matching kernel. The evaluation metric (PN-BAcc, clip [0.2, 5]) and the two checkpoint rules are unchanged, so the sensitivities are training-loss/architecture changes only.

Protocol: the Step 6b uniform aggregation (average probabilities and weights over the 3 seeds per episode, pool OOF, one 2,000-resample subject-cluster bootstrap), separately per checkpoint rule. PRIMARY = best validation macro-F1 (drives early stopping); SECONDARY = best validation PN-BAcc (never stops training). Both best states are evaluated on the test fold.

Not run (per the task): A5-A9, C2, the window (+/-90 / +/-365 d) sensitivities and the summary-only amyloid fallback. These stay deferred to Step 11b/12.

==============================================================================
1. METRICS PER RUN-SET
==============================================================================
### primary checkpoint rule

# Step 11a - metrics per run-set (primary checkpoint rule)

Seed-averaged OOF, pooled, one 2,000-resample subject-cluster bootstrap (Step 6b uniform protocol). point [95% CI]. One row per run-set at its primary operating point (A1/A4 -> rho uniform; A0/A3 -> argmax). R1_* = recipe 1 = Step 9.

| run-set | family | arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|---|---|
| R1_A0 | R1 | A0 | argmax | 13240 | 0.5982 [0.5836, 0.6129] | 0.6137 [0.6004, 0.6268] | 0.7938 [0.7831, 0.8040] |
| R1_A1 | R1 | A1 | uniform | 13240 | 0.6094 [0.5954, 0.6242] | 0.6011 [0.5879, 0.6143] | 0.7862 [0.7751, 0.7968] |
| R1_A2 | R1 | A2 | argmax | 13240 | 0.5974 [0.5824, 0.6125] | 0.6153 [0.6018, 0.6284] | 0.7939 [0.7834, 0.8043] |
| R1_A3 | R1 | A3 | argmax | 13240 | 0.6143 [0.6000, 0.6290] | 0.5937 [0.5795, 0.6073] | 0.7724 [0.7617, 0.7834] |
| R1_A4 | R1 | A4 | uniform | 13240 | 0.6163 [0.6023, 0.6304] | 0.5902 [0.5764, 0.6037] | 0.7735 [0.7627, 0.7848] |
| R2_A0 | R2 | A0 | argmax | 13240 | 0.6035 [0.5884, 0.6178] | 0.6163 [0.6030, 0.6290] | 0.7932 [0.7827, 0.8037] |
| R2_A1 | R2 | A1 | uniform | 13240 | 0.6117 [0.5979, 0.6257] | 0.5959 [0.5827, 0.6097] | 0.7863 [0.7756, 0.7967] |
| R2_A4 | R2 | A4 | uniform | 13240 | 0.6184 [0.6048, 0.6326] | 0.5847 [0.5712, 0.5984] | 0.7736 [0.7628, 0.7847] |
| A4_tau0p5 | sensitivity | A4 | uniform | 13240 | 0.6156 [0.6014, 0.6308] | 0.6067 [0.5940, 0.6199] | 0.7849 [0.7743, 0.7956] |
| A4_tau1p5 | sensitivity | A4 | uniform | 13240 | 0.6047 [0.5906, 0.6191] | 0.5717 [0.5582, 0.5852] | 0.7597 [0.7487, 0.7706] |
| A3_clip0p5_2 | sensitivity | A3 | argmax | 13240 | 0.6181 [0.6039, 0.6324] | 0.5965 [0.5829, 0.6098] | 0.7734 [0.7623, 0.7845] |
| A3_clip0p1_10 | sensitivity | A3 | argmax | 13240 | 0.6159 [0.6017, 0.6308] | 0.5920 [0.5781, 0.6057] | 0.7722 [0.7615, 0.7831] |
| A0_pdrop0 | sensitivity | A0 | argmax | 13240 | 0.5954 [0.5800, 0.6097] | 0.6184 [0.6046, 0.6315] | 0.7942 [0.7837, 0.8045] |
| A0_pdrop0p5 | sensitivity | A0 | argmax | 13240 | 0.5980 [0.5830, 0.6124] | 0.6146 [0.6015, 0.6277] | 0.7896 [0.7790, 0.8003] |
| A4_pdrop0 | sensitivity | A4 | uniform | 13240 | 0.6104 [0.5964, 0.6246] | 0.5934 [0.5803, 0.6064] | 0.7732 [0.7622, 0.7842] |
| A4_pdrop0p5 | sensitivity | A4 | uniform | 13240 | 0.6148 [0.6007, 0.6291] | 0.5879 [0.5742, 0.6016] | 0.7705 [0.7594, 0.7816] |

# Step 11a - A1/A4 run-sets at rho=P(c) (primary checkpoint rule)

Secondary operating point (pattern-neutral at natural class prevalence). Same protocol as the main table.

| run-set | family | arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|---|---|
| R1_A1 | R1 | A1 | P(c) | 13240 | 0.5943 [0.5789, 0.6092] | 0.6138 [0.6004, 0.6271] | 0.7884 [0.7773, 0.7991] |
| R1_A4 | R1 | A4 | P(c) | 13240 | 0.6072 [0.5927, 0.6218] | 0.6082 [0.5945, 0.6219] | 0.7760 [0.7648, 0.7874] |
| R2_A1 | R2 | A1 | P(c) | 13240 | 0.6061 [0.5911, 0.6211] | 0.6168 [0.6036, 0.6299] | 0.7893 [0.7785, 0.8000] |
| R2_A4 | R2 | A4 | P(c) | 13240 | 0.6138 [0.5993, 0.6281] | 0.6046 [0.5915, 0.6178] | 0.7768 [0.7659, 0.7880] |
| A4_tau0p5 | sensitivity | A4 | P(c) | 13240 | 0.5918 [0.5771, 0.6076] | 0.6121 [0.5989, 0.6265] | 0.7860 [0.7753, 0.7971] |
| A4_tau1p5 | sensitivity | A4 | P(c) | 13240 | 0.5984 [0.5840, 0.6133] | 0.5868 [0.5736, 0.6002] | 0.7613 [0.7502, 0.7725] |
| A4_pdrop0 | sensitivity | A4 | P(c) | 13240 | 0.5997 [0.5849, 0.6148] | 0.6060 [0.5923, 0.6201] | 0.7746 [0.7634, 0.7858] |
| A4_pdrop0p5 | sensitivity | A4 | P(c) | 13240 | 0.5988 [0.5840, 0.6131] | 0.6012 [0.5872, 0.6146] | 0.7728 [0.7615, 0.7840] |

### secondary checkpoint rule

# Step 11a - metrics per run-set (secondary checkpoint rule)

Seed-averaged OOF, pooled, one 2,000-resample subject-cluster bootstrap (Step 6b uniform protocol). point [95% CI]. One row per run-set at its primary operating point (A1/A4 -> rho uniform; A0/A3 -> argmax). R1_* = recipe 1 = Step 9.

| run-set | family | arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|---|---|
| R1_A0 | R1 | A0 | argmax | 13240 | 0.5998 [0.5852, 0.6147] | 0.6129 [0.6000, 0.6261] | 0.7913 [0.7805, 0.8018] |
| R1_A1 | R1 | A1 | uniform | 13240 | 0.6135 [0.6003, 0.6273] | 0.5887 [0.5750, 0.6015] | 0.7885 [0.7778, 0.7990] |
| R1_A2 | R1 | A2 | argmax | 13240 | 0.6047 [0.5899, 0.6191] | 0.6163 [0.6034, 0.6291] | 0.7929 [0.7824, 0.8035] |
| R1_A3 | R1 | A3 | argmax | 13240 | 0.6198 [0.6060, 0.6338] | 0.5770 [0.5633, 0.5908] | 0.7710 [0.7605, 0.7818] |
| R1_A4 | R1 | A4 | uniform | 13240 | 0.6223 [0.6089, 0.6355] | 0.5751 [0.5615, 0.5879] | 0.7711 [0.7608, 0.7816] |
| R2_A0 | R2 | A0 | argmax | 13240 | 0.6053 [0.5904, 0.6200] | 0.6128 [0.5995, 0.6256] | 0.7924 [0.7816, 0.8026] |
| R2_A1 | R2 | A1 | uniform | 13240 | 0.6135 [0.5999, 0.6273] | 0.5896 [0.5754, 0.6041] | 0.7876 [0.7771, 0.7980] |
| R2_A4 | R2 | A4 | uniform | 13240 | 0.6208 [0.6074, 0.6345] | 0.5738 [0.5602, 0.5874] | 0.7733 [0.7626, 0.7840] |
| A4_tau0p5 | sensitivity | A4 | uniform | 13240 | 0.6157 [0.6019, 0.6296] | 0.5981 [0.5845, 0.6109] | 0.7862 [0.7757, 0.7965] |
| A4_tau1p5 | sensitivity | A4 | uniform | 13240 | 0.6077 [0.5943, 0.6209] | 0.5484 [0.5357, 0.5617] | 0.7570 [0.7462, 0.7677] |
| A3_clip0p5_2 | sensitivity | A3 | argmax | 13240 | 0.6225 [0.6090, 0.6357] | 0.5790 [0.5657, 0.5918] | 0.7700 [0.7596, 0.7804] |
| A3_clip0p1_10 | sensitivity | A3 | argmax | 13240 | 0.6230 [0.6093, 0.6361] | 0.5784 [0.5646, 0.5917] | 0.7708 [0.7602, 0.7817] |
| A0_pdrop0 | sensitivity | A0 | argmax | 13240 | 0.6005 [0.5854, 0.6151] | 0.6165 [0.6034, 0.6298] | 0.7932 [0.7825, 0.8037] |
| A0_pdrop0p5 | sensitivity | A0 | argmax | 13240 | 0.5999 [0.5854, 0.6146] | 0.6082 [0.5949, 0.6218] | 0.7895 [0.7788, 0.8002] |
| A4_pdrop0 | sensitivity | A4 | uniform | 13240 | 0.6196 [0.6067, 0.6328] | 0.5756 [0.5626, 0.5888] | 0.7711 [0.7608, 0.7815] |
| A4_pdrop0p5 | sensitivity | A4 | uniform | 13240 | 0.6157 [0.6022, 0.6292] | 0.5738 [0.5600, 0.5870] | 0.7701 [0.7593, 0.7806] |

# Step 11a - A1/A4 run-sets at rho=P(c) (secondary checkpoint rule)

Secondary operating point (pattern-neutral at natural class prevalence). Same protocol as the main table.

| run-set | family | arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |
|---|---|---|---|---|---|---|---|
| R1_A1 | R1 | A1 | P(c) | 13240 | 0.6055 [0.5908, 0.6205] | 0.6158 [0.6023, 0.6291] | 0.7922 [0.7815, 0.8028] |
| R1_A4 | R1 | A4 | P(c) | 13240 | 0.6155 [0.6014, 0.6296] | 0.6039 [0.5914, 0.6167] | 0.7761 [0.7652, 0.7870] |
| R2_A1 | R2 | A1 | P(c) | 13240 | 0.6077 [0.5934, 0.6220] | 0.6139 [0.6007, 0.6273] | 0.7911 [0.7802, 0.8018] |
| R2_A4 | R2 | A4 | P(c) | 13240 | 0.6159 [0.6017, 0.6301] | 0.6001 [0.5868, 0.6130] | 0.7778 [0.7669, 0.7888] |
| A4_tau0p5 | sensitivity | A4 | P(c) | 13240 | 0.5993 [0.5844, 0.6143] | 0.6143 [0.6009, 0.6280] | 0.7882 [0.7774, 0.7990] |
| A4_tau1p5 | sensitivity | A4 | P(c) | 13240 | 0.6096 [0.5959, 0.6243] | 0.5793 [0.5658, 0.5929] | 0.7620 [0.7509, 0.7729] |
| A4_pdrop0 | sensitivity | A4 | P(c) | 13240 | 0.6081 [0.5937, 0.6221] | 0.5992 [0.5859, 0.6125] | 0.7759 [0.7650, 0.7868] |
| A4_pdrop0p5 | sensitivity | A4 | P(c) | 13240 | 0.6089 [0.5938, 0.6230] | 0.6015 [0.5877, 0.6146] | 0.7753 [0.7644, 0.7866] |

==============================================================================
2. PAIRED COMPARISONS UNDER R2 AND THE H3 VERDICT
==============================================================================
### primary checkpoint rule

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

### secondary checkpoint rule

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

==============================================================================
3. BEST-EPOCH DISTRIBUTIONS (flag median < 5 and any epoch-0 selection)
==============================================================================
# Step 11a - best-epoch distribution per run-set and checkpoint rule

`best_epoch` is the epoch selected by each rule. A run-set is flagged when its median best epoch is below 5, or when ANY of its runs selected epoch 0.

| run-set | family | rule | runs | median | mean | min | max | #(<5) | flag median<5 | #(==0) | flag epoch 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1_A0 | R1 | primary | 15 | 3.0 | 4.00 | 0 | 12 | 9 | YES | 1 | YES |
| R1_A1 | R1 | primary | 15 | 7.0 | 9.33 | 0 | 35 | 5 | no | 1 | YES |
| R1_A2 | R1 | primary | 15 | 3.0 | 3.87 | 0 | 16 | 10 | YES | 1 | YES |
| R1_A3 | R1 | primary | 15 | 10.0 | 12.93 | 0 | 56 | 3 | no | 1 | YES |
| R1_A4 | R1 | primary | 15 | 10.0 | 8.87 | 0 | 23 | 5 | no | 1 | YES |
| R2_A0 | R2 | primary | 15 | 13.0 | 14.80 | 2 | 35 | 2 | no | 0 | no |
| R2_A1 | R2 | primary | 15 | 25.0 | 25.67 | 9 | 45 | 0 | no | 0 | no |
| R2_A4 | R2 | primary | 15 | 23.0 | 33.80 | 7 | 82 | 0 | no | 0 | no |
| A4_tau0p5 | sensitivity | primary | 15 | 4.0 | 6.27 | 0 | 19 | 8 | YES | 2 | YES |
| A4_tau1p5 | sensitivity | primary | 15 | 13.0 | 18.87 | 4 | 56 | 2 | no | 0 | no |
| A3_clip0p5_2 | sensitivity | primary | 15 | 10.0 | 8.40 | 0 | 21 | 4 | no | 1 | YES |
| A3_clip0p1_10 | sensitivity | primary | 15 | 10.0 | 10.20 | 0 | 33 | 3 | no | 1 | YES |
| A0_pdrop0 | sensitivity | primary | 15 | 2.0 | 4.33 | 0 | 24 | 11 | YES | 1 | YES |
| A0_pdrop0p5 | sensitivity | primary | 15 | 3.0 | 4.33 | 0 | 10 | 9 | YES | 2 | YES |
| A4_pdrop0 | sensitivity | primary | 15 | 7.0 | 10.40 | 0 | 33 | 5 | no | 2 | YES |
| A4_pdrop0p5 | sensitivity | primary | 15 | 8.0 | 10.87 | 0 | 40 | 3 | no | 1 | YES |
| R1_A0 | R1 | secondary | 15 | 5.0 | 6.00 | 1 | 17 | 6 | no | 0 | no |
| R1_A1 | R1 | secondary | 15 | 3.0 | 5.20 | 0 | 23 | 11 | YES | 2 | YES |
| R1_A2 | R1 | secondary | 15 | 3.0 | 4.60 | 1 | 16 | 8 | YES | 0 | no |
| R1_A3 | R1 | secondary | 15 | 4.0 | 3.87 | 0 | 15 | 9 | YES | 4 | YES |
| R1_A4 | R1 | secondary | 15 | 5.0 | 4.53 | 0 | 13 | 7 | no | 4 | YES |
| R2_A0 | R2 | secondary | 15 | 10.0 | 15.93 | 2 | 39 | 5 | no | 0 | no |
| R2_A1 | R2 | secondary | 15 | 21.0 | 21.47 | 4 | 59 | 1 | no | 0 | no |
| R2_A4 | R2 | secondary | 15 | 15.0 | 22.67 | 3 | 82 | 2 | no | 0 | no |
| A4_tau0p5 | sensitivity | secondary | 15 | 4.0 | 4.07 | 0 | 17 | 11 | YES | 3 | YES |
| A4_tau1p5 | sensitivity | secondary | 15 | 5.0 | 9.47 | 0 | 42 | 7 | no | 1 | YES |
| A3_clip0p5_2 | sensitivity | secondary | 15 | 3.0 | 2.87 | 0 | 7 | 10 | YES | 5 | YES |
| A3_clip0p1_10 | sensitivity | secondary | 15 | 2.0 | 3.93 | 0 | 15 | 8 | YES | 4 | YES |
| A0_pdrop0 | sensitivity | secondary | 15 | 4.0 | 3.80 | 0 | 10 | 11 | YES | 1 | YES |
| A0_pdrop0p5 | sensitivity | secondary | 15 | 3.0 | 4.93 | 0 | 15 | 10 | YES | 2 | YES |
| A4_pdrop0 | sensitivity | secondary | 15 | 2.0 | 2.73 | 0 | 7 | 11 | YES | 3 | YES |
| A4_pdrop0p5 | sensitivity | secondary | 15 | 3.0 | 6.27 | 0 | 40 | 10 | YES | 4 | YES |

==============================================================================
4. SENSITIVITY VERSUS RECIPE 1 (paired per episode)
==============================================================================
### primary checkpoint rule

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

### secondary checkpoint rule

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

==============================================================================
5. FIGURE - PN-BAcc vs macro-F1 FOR EVERY RUN-SET
==============================================================================
`pn_bacc_vs_macro_f1.png`: one point per run-set at its primary operating point, both checkpoint rules side by side, with 95% subject-cluster-bootstrap error bars and the run-set name annotated. Grey = recipe 1 (Step 9), red = R2, blue = sensitivity.

==============================================================================
6. DOES ANY CONCLUSION CHANGE?
==============================================================================

Step 9 concluded a NEGATIVE result under BOTH checkpoint rules (A4 did not beat A1/A3 on PN-BAcc after Holm). Step 11a re-checks that with a second training recipe and by perturbing the correction hyperparameters.

| checkpoint rule | Step 9 verdict | H3 under R2 (A0/A1 comparators) | H3 under R2 (full 4 comparators, A2/A3 carried over) | primary conclusion changes |
|---|---|---|---|---|
| primary | negative result | trade-off (A4 wins PN-BAcc but fails macro-F1 non-inferiority) | negative result | False |
| secondary | negative result | trade-off (A4 wins PN-BAcc but fails macro-F1 non-inferiority) | negative result | False |

A4(R2) - A1(R2) PN-BAcc(u): primary 0.0067 (Holm p 0.0470); secondary 0.0073 (Holm p 0.0320).

Sensitivity settings whose paired CI against recipe 1 excludes 0 (either checkpoint rule): 6 table rows. See section 4 for which ones and on which metric.

**Overall:** the primary conclusion of the pipeline is UNCHANGED by the R2 recipe and the sensitivities.

==============================================================================
EXPECTED VS ACTUAL
==============================================================================

| check | expected | actual | pass |
|---|---|---|---|
| R2 runs (A0/A1/A4 x 3 seeds x 5 folds) | 45 | 45 | yes |
| sensitivity runs (8 settings x 3 x 5) | 120 | 120 | yes |
| total Step 11a runs | 165 | 165 | yes |
| both checkpoint rules tracked for every run | yes | yes | yes |
| rule-1 baselines reused from Step 9 (not retrained) | yes | yes | yes |
| seed fixed before the model is built | yes | yes | yes |
| CIs per run-set (2,000 subject-cluster bootstrap) | yes | yes | yes |
| every table script-generated | yes | yes | yes |
| errors | 0 | 0 | yes |

==============================================================================
DEVIATIONS FROM THE SPEC AND THINGS I AM UNSURE ABOUT
==============================================================================

* The Step 9 rule defines H3 with FOUR comparators (A0-A3). Under R2 only A0 and A1 are retrained, so the pre-registered under-R2 verdict uses the two available comparators (Holm over the two PN-BAcc comparisons). A second, clearly-labelled diagnostic carries A2 and A3 over from recipe 1 to apply the full four-comparator rule; it mixes recipes and is reported for completeness only.
* PIPELINE_v2 section 6/10 says five seeds per fold and a clip grid {3,5,10}. The established protocol uses three seeds, and the pre-registered Step 11a clip grid is {[0.5,2], [0.1,10]} against the recipe-1 default [0.2,5]. This was fixed before the runs.
* The A3 clip and the A4 tau sensitivity change only the TRAINING loss; the evaluation PN-BAcc keeps its fixed clip [0.2, 5] so the four run-sets remain directly comparable.

==============================================================================
FILES WRITTEN
==============================================================================

results/step11a/: runs/ (per-run parquet), curves/ (per-epoch curves), oof_all.parquet, run_log.jsonl (+ run_log_shard{0,1,2}.jsonl, shard{0,1,2}.log), main_table_{primary,secondary}.{csv,md}, runset_pc_{primary,secondary}.{csv,md}, paired_r2_{primary,secondary}.{csv,md}, paired_r2_carryover_{primary,secondary}.csv, best_epochs.{csv,md}, sensitivity_{primary,secondary}.{csv,md}, pn_bacc_vs_macro_f1.png, analysis.json, REPORT.md. New code: scripts/run_step11a.py, scripts/analyse_step11a.py, scripts/step11a_report.py, scripts/step11a_appendix.py, configs/step11a_robustness.yaml, tests/test_step11a_robustness.py; modified src/adni/losses.py and src/adni/train.py (clip override).
