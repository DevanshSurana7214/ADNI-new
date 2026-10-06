STEP 10 REPORT
Diagnostics, controls (C1, C-shuf) and the S2 enrolment-cohort shift.

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/run_step10_s1.py           > results/step10/s1/{runs,curves,weights,run_log.jsonl,
                                       oof_all.parquet}   (A0/A1/A4 retrain + C1 + C-shuf)
  scripts/run_step10_s2.py           > results/step10/s2/{runs,run_log.jsonl,oof_all.parquet}
  scripts/run_step10_intervention.py > results/step10/diagnostics/intervention_*
  scripts/analyse_step10.py          > results/step10/{verification,controls,s2,best_epochs}*
  scripts/step10_report.py           > results/step10/REPORT.md
  scripts/step10_appendix.py         > Appendix E row

Git commit: (this commit) - tag `step10`

Tests: 105 passed in 104.66s (0:01:44)

S1 grid: 75 runs ok, 0 errors, 28.2 min training (A0/A1/A4 retrain + C1 + C-shuf).
S2 grid: 18 runs ok, 0 errors, 4.2 min training (A0-A4 + B1).

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1, LightGBM; NVIDIA RTX 3050 (6 GB). Deterministic algorithms on; `train_arm` seeds before building the model. Frozen config `configs/model.yaml` (Step 7b); no tuning.

------------------------------------------------------------------------------
SCOPE AND PRE-DECLARATION

The frozen config, arms A0-A4, p_drop 0.25, alpha 0.5, tau 1, clip [0.2, 5] and the two checkpoint rules of Step 9 are reused unchanged. The uniform protocol of Step 6b is used throughout (seed-averaged probabilities, pooled OOF, one 2,000-resample subject-cluster bootstrap), separately per checkpoint rule.

Pre-declared reading (C1): the joint prior is supported only if the CI lower bound of A4 - C1 on PN-BAcc is above 0. A4 - C1 and A3 - C-shuf are reported with CIs. The mask intervention uses the rho-uniform operating point (softmax of the raw logits) for every arm.

Deferred to Step 11 per the task: A5-A9, C2, the tau/clip/p_drop sensitivities, the embedding-swap test and a second training recipe.

Verdict (C1): joint prior supported = False (primary rule), True (secondary rule).

==============================================================================
1. RETRAIN VERIFICATION (A0/A1/A4 reproduce the Step 9 OOF)
==============================================================================
Global max probability deviation = 0.000e+00.

# Step 10 - retrain verification (A0/A1/A4 reproduce the Step 9 OOF predictions)

Max absolute deviation of the retrained test-fold probabilities from the Step 9 `oof_all.parquet`, matched on (episode_id, seed, fold, arm, ckpt_rule, operating_point).

| arm | ckpt rule | rows | max |dp| |
|---|---|---|---|
| A0 | primary | 39720 | 0.000e+00 |
| A0 | secondary | 39720 | 0.000e+00 |
| A1 | primary | 79440 | 0.000e+00 |
| A1 | secondary | 79440 | 0.000e+00 |
| A4 | primary | 79440 | 0.000e+00 |
| A4 | secondary | 79440 | 0.000e+00 |

**Global max deviation = 0.000e+00** - training is deterministic and the retrain reproduces the Step 9 predictions exactly.

==============================================================================
2. COUNTERFACTUAL MASK INTERVENTION (remove vs content swap)
==============================================================================
### primary checkpoint rule

# Step 10 - mask intervention (primary checkpoint rule)

Test episodes with >= 2 observed modalities. (a) REMOVE an observed modality (mask -> 0); (b) CONTENT SWAP its values with a random other test episode that has it (mask kept). Predictions use the rho-uniform operating point (softmax of the raw logits). mean TV over episodes; ratio = (a)/(b) > 1 means the model reacts to the mask beyond the lost content.

| arm | ckpt | modality | n | mean TV remove (a) | mean TV swap (b) | ratio (a)/(b) | predicted shares orig (CN/MCI/Dem) | remove | swap |
|---|---|---|---|---|---|---|---|---|---|
| A0 | primary | ALL | 57864 | 0.1278 | 0.2039 | 0.627 | 0.416/0.402/0.183 | 0.421/0.403/0.176 | 0.391/0.451/0.158 |
| A0 | primary | amy | 12951 | 0.1037 | 0.1722 | 0.602 | 0.448/0.398/0.153 | 0.415/0.433/0.152 | 0.429/0.445/0.127 |
| A0 | primary | fdg | 14763 | 0.1197 | 0.1617 | 0.740 | 0.299/0.464/0.236 | 0.367/0.407/0.226 | 0.280/0.512/0.207 |
| A0 | primary | mri | 22068 | 0.1411 | 0.2576 | 0.548 | 0.410/0.402/0.188 | 0.431/0.391/0.178 | 0.375/0.459/0.166 |
| A0 | primary | tau | 8082 | 0.1452 | 0.1853 | 0.784 | 0.590/0.291/0.119 | 0.501/0.384/0.115 | 0.578/0.328/0.093 |
| A1 | primary | ALL | 57864 | 0.1500 | 0.2375 | 0.632 | 0.412/0.356/0.232 | 0.414/0.352/0.234 | 0.385/0.405/0.210 |
| A1 | primary | amy | 12951 | 0.1238 | 0.2009 | 0.616 | 0.448/0.354/0.199 | 0.403/0.398/0.198 | 0.421/0.404/0.174 |
| A1 | primary | fdg | 14763 | 0.1161 | 0.1525 | 0.761 | 0.293/0.413/0.294 | 0.358/0.355/0.286 | 0.276/0.451/0.273 |
| A1 | primary | mri | 22068 | 0.1850 | 0.3230 | 0.573 | 0.407/0.356/0.237 | 0.425/0.326/0.249 | 0.369/0.414/0.217 |
| A1 | primary | tau | 8082 | 0.1583 | 0.2177 | 0.727 | 0.587/0.254/0.158 | 0.502/0.341/0.157 | 0.569/0.297/0.134 |
| A4 | primary | ALL | 57864 | 0.1419 | 0.2423 | 0.586 | 0.417/0.346/0.236 | 0.414/0.349/0.237 | 0.394/0.394/0.213 |
| A4 | primary | amy | 12951 | 0.1133 | 0.1940 | 0.584 | 0.437/0.354/0.208 | 0.436/0.369/0.195 | 0.411/0.402/0.187 |
| A4 | primary | fdg | 14763 | 0.0848 | 0.1425 | 0.595 | 0.348/0.361/0.291 | 0.342/0.364/0.294 | 0.333/0.401/0.266 |
| A4 | primary | mri | 22068 | 0.1959 | 0.3351 | 0.584 | 0.416/0.345/0.240 | 0.418/0.330/0.252 | 0.388/0.395/0.217 |
| A4 | primary | tau | 8082 | 0.1450 | 0.2489 | 0.583 | 0.517/0.310/0.173 | 0.502/0.340/0.159 | 0.494/0.360/0.146 |

### secondary checkpoint rule

# Step 10 - mask intervention (secondary checkpoint rule)

Test episodes with >= 2 observed modalities. (a) REMOVE an observed modality (mask -> 0); (b) CONTENT SWAP its values with a random other test episode that has it (mask kept). Predictions use the rho-uniform operating point (softmax of the raw logits). mean TV over episodes; ratio = (a)/(b) > 1 means the model reacts to the mask beyond the lost content.

| arm | ckpt | modality | n | mean TV remove (a) | mean TV swap (b) | ratio (a)/(b) | predicted shares orig (CN/MCI/Dem) | remove | swap |
|---|---|---|---|---|---|---|---|---|---|
| A0 | secondary | ALL | 57864 | 0.1365 | 0.2189 | 0.624 | 0.421/0.394/0.186 | 0.426/0.393/0.182 | 0.393/0.448/0.159 |
| A0 | secondary | amy | 12951 | 0.1109 | 0.1824 | 0.608 | 0.455/0.390/0.156 | 0.418/0.427/0.155 | 0.431/0.440/0.128 |
| A0 | secondary | fdg | 14763 | 0.1185 | 0.1585 | 0.748 | 0.302/0.459/0.239 | 0.370/0.398/0.232 | 0.282/0.508/0.210 |
| A0 | secondary | mri | 22068 | 0.1601 | 0.2882 | 0.556 | 0.415/0.394/0.191 | 0.435/0.378/0.188 | 0.376/0.458/0.166 |
| A0 | secondary | tau | 8082 | 0.1460 | 0.1984 | 0.736 | 0.598/0.281/0.121 | 0.515/0.368/0.117 | 0.582/0.323/0.095 |
| A1 | secondary | ALL | 57864 | 0.1345 | 0.2196 | 0.612 | 0.422/0.330/0.248 | 0.423/0.331/0.246 | 0.396/0.376/0.229 |
| A1 | secondary | amy | 12951 | 0.1142 | 0.1849 | 0.618 | 0.458/0.329/0.213 | 0.418/0.367/0.215 | 0.438/0.373/0.189 |
| A1 | secondary | fdg | 14763 | 0.1240 | 0.1805 | 0.687 | 0.306/0.381/0.314 | 0.369/0.333/0.297 | 0.284/0.423/0.294 |
| A1 | secondary | mri | 22068 | 0.1501 | 0.2758 | 0.544 | 0.416/0.331/0.253 | 0.431/0.311/0.258 | 0.379/0.383/0.238 |
| A1 | secondary | tau | 8082 | 0.1435 | 0.1929 | 0.744 | 0.592/0.237/0.171 | 0.508/0.320/0.171 | 0.578/0.275/0.147 |
| A4 | secondary | ALL | 57864 | 0.1265 | 0.2233 | 0.567 | 0.415/0.329/0.256 | 0.414/0.334/0.252 | 0.393/0.372/0.235 |
| A4 | secondary | amy | 12951 | 0.1102 | 0.1907 | 0.578 | 0.435/0.336/0.229 | 0.435/0.354/0.212 | 0.412/0.379/0.209 |
| A4 | secondary | fdg | 14763 | 0.0973 | 0.1645 | 0.591 | 0.355/0.334/0.311 | 0.343/0.345/0.312 | 0.339/0.374/0.287 |
| A4 | secondary | mri | 22068 | 0.1580 | 0.2846 | 0.555 | 0.414/0.328/0.259 | 0.419/0.318/0.263 | 0.388/0.371/0.241 |
| A4 | secondary | tau | 8082 | 0.1203 | 0.2160 | 0.557 | 0.495/0.313/0.191 | 0.493/0.331/0.176 | 0.477/0.355/0.168 |

==============================================================================
3. CONTROLS - placebo C1 and weight-shuffle C-shuf
==============================================================================
### primary checkpoint rule

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

### secondary checkpoint rule

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

==============================================================================
4. S2 ENROLMENT-COHORT SHIFT (test side = ADNI3/4)
==============================================================================
### primary checkpoint rule

# Step 10 - S2 enrolment-cohort shift, test side = ADNI3/4 (primary checkpoint rule)

Train = S2 train side (ADNI1/GO/2) with the frozen S2 val holdout; test = ADNI3/4 subjects. Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap. PN-BAcc uses rho-uniform weights from the S2 TRAIN-split pi_hat.

| arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC | F1 CN | F1 MCI | F1 Dem |
|---|---|---|---|---|---|---|---|---|
| A0 | argmax | 3565 | 0.6071 [0.5769, 0.6346] | 0.6133 [0.5858, 0.6390] | 0.7959 [0.7772, 0.8142] | 0.7699 | 0.4715 | 0.5984 |
| A1 | uniform | 3565 | 0.6161 [0.5872, 0.6438] | 0.5952 [0.5681, 0.6195] | 0.7860 [0.7668, 0.8051] | 0.7648 | 0.4118 | 0.6090 |
| A2 | argmax | 3565 | 0.6001 [0.5694, 0.6310] | 0.6086 [0.5810, 0.6350] | 0.7943 [0.7751, 0.8131] | 0.7677 | 0.4686 | 0.5895 |
| A3 | argmax | 3565 | 0.6057 [0.5758, 0.6328] | 0.5764 [0.5481, 0.6009] | 0.7446 [0.7235, 0.7644] | 0.7151 | 0.4329 | 0.5812 |
| A4 | uniform | 3565 | 0.5997 [0.5705, 0.6270] | 0.5628 [0.5343, 0.5877] | 0.7440 [0.7230, 0.7638] | 0.6971 | 0.4086 | 0.5827 |
| B1 | proba | 3565 | 0.5747 [0.5442, 0.6037] | 0.6204 [0.5927, 0.6464] | 0.8074 [0.7899, 0.8247] | 0.7631 | 0.5359 | 0.5622 |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4 - A0 | pn_bacc_uniform | -0.0074 | [-0.0278, 0.0126] | 0.4938 |
| A4 - A0 | macro_f1 | -0.0505 | [-0.0707, -0.0303] | 0.0010 |
| A4 - A0 | macro_auroc | -0.0519 | [-0.0626, -0.0415] | 0.0010 |
| A4 - A1 | pn_bacc_uniform | -0.0164 | [-0.0328, -0.0010] | 0.0360 |
| A4 - A1 | macro_f1 | -0.0324 | [-0.0491, -0.0165] | 0.0010 |
| A4 - A1 | macro_auroc | -0.0420 | [-0.0506, -0.0339] | 0.0010 |

### secondary checkpoint rule

# Step 10 - S2 enrolment-cohort shift, test side = ADNI3/4 (secondary checkpoint rule)

Train = S2 train side (ADNI1/GO/2) with the frozen S2 val holdout; test = ADNI3/4 subjects. Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap. PN-BAcc uses rho-uniform weights from the S2 TRAIN-split pi_hat.

| arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC | F1 CN | F1 MCI | F1 Dem |
|---|---|---|---|---|---|---|---|---|
| A0 | argmax | 3565 | 0.6086 [0.5787, 0.6361] | 0.6147 [0.5871, 0.6396] | 0.7972 [0.7793, 0.8150] | 0.7743 | 0.4646 | 0.6053 |
| A1 | uniform | 3565 | 0.6213 [0.5948, 0.6457] | 0.5847 [0.5583, 0.6100] | 0.7912 [0.7730, 0.8091] | 0.7716 | 0.3719 | 0.6105 |
| A2 | argmax | 3565 | 0.5908 [0.5613, 0.6193] | 0.5995 [0.5712, 0.6253] | 0.7960 [0.7776, 0.8139] | 0.7682 | 0.4544 | 0.5758 |
| A3 | argmax | 3565 | 0.6411 [0.6152, 0.6653] | 0.5839 [0.5569, 0.6089] | 0.7697 [0.7500, 0.7881] | 0.7537 | 0.4092 | 0.5888 |
| A4 | uniform | 3565 | 0.6347 [0.6080, 0.6591] | 0.5806 [0.5533, 0.6058] | 0.7640 [0.7443, 0.7832] | 0.7401 | 0.4157 | 0.5860 |
| B1 | proba | 3565 | 0.5747 [0.5442, 0.6037] | 0.6204 [0.5927, 0.6464] | 0.8074 [0.7899, 0.8247] | 0.7631 | 0.5359 | 0.5622 |

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| A4 - A0 | pn_bacc_uniform | 0.0261 | [0.0025, 0.0494] | 0.0270 |
| A4 - A0 | macro_f1 | -0.0341 | [-0.0572, -0.0111] | 0.0060 |
| A4 - A0 | macro_auroc | -0.0332 | [-0.0435, -0.0229] | 0.0010 |
| A4 - A1 | pn_bacc_uniform | 0.0135 | [-0.0036, 0.0308] | 0.1299 |
| A4 - A1 | macro_f1 | -0.0041 | [-0.0210, 0.0120] | 0.6147 |
| A4 - A1 | macro_auroc | -0.0271 | [-0.0342, -0.0201] | 0.0010 |

### P(c|m) training side vs test side

# Step 10 - S2 class distribution given pattern, P(c|m) (Dirichlet alpha 0.5)

Only patterns with >= 30 episodes on BOTH sides.

| pattern | n train | n test | class | P(c|m) train | P(c|m) test | delta |
|---|---|---|---|---|---|---|
| 0001 | 41 | 91 | CN | 0.3412 | 0.6865 | +0.3453 |
| 0001 | 41 | 91 | MCI | 0.5059 | 0.2000 | -0.3059 |
| 0001 | 41 | 91 | Dementia | 0.1529 | 0.1135 | -0.0394 |
| 0011 | 99 | 75 | CN | 0.6716 | 0.6340 | -0.0377 |
| 0011 | 99 | 75 | MCI | 0.2438 | 0.2810 | +0.0373 |
| 0011 | 99 | 75 | Dementia | 0.0846 | 0.0850 | +0.0004 |
| 1000 | 3517 | 1080 | CN | 0.3222 | 0.4998 | +0.1776 |
| 1000 | 3517 | 1080 | MCI | 0.4387 | 0.4027 | -0.0360 |
| 1000 | 3517 | 1080 | Dementia | 0.2392 | 0.0975 | -0.1416 |
| 1001 | 85 | 893 | CN | 0.5838 | 0.6970 | +0.1132 |
| 1001 | 85 | 893 | MCI | 0.2832 | 0.2163 | -0.0669 |
| 1001 | 85 | 893 | Dementia | 0.1329 | 0.0866 | -0.0463 |
| 1010 | 381 | 196 | CN | 0.4301 | 0.6000 | +0.1699 |
| 1010 | 381 | 196 | MCI | 0.4275 | 0.3367 | -0.0907 |
| 1010 | 381 | 196 | Dementia | 0.1425 | 0.0633 | -0.0792 |
| 1011 | 282 | 682 | CN | 0.7284 | 0.7557 | +0.0273 |
| 1011 | 282 | 682 | MCI | 0.1993 | 0.1778 | -0.0215 |
| 1011 | 282 | 682 | Dementia | 0.0723 | 0.0666 | -0.0057 |
| 1100 | 1854 | 42 | CN | 0.2751 | 0.0575 | -0.2177 |
| 1100 | 1854 | 42 | MCI | 0.4762 | 0.8621 | +0.3859 |
| 1100 | 1854 | 42 | Dementia | 0.2487 | 0.0805 | -0.1683 |
| 1111 | 73 | 156 | CN | 0.1007 | 0.0603 | -0.0404 |
| 1111 | 73 | 156 | MCI | 0.6644 | 0.7333 | +0.0689 |
| 1111 | 73 | 156 | Dementia | 0.2349 | 0.2063 | -0.0286 |

==============================================================================
5. BEST-EPOCH DISTRIBUTION (every new run)
==============================================================================
# Step 10 - best-epoch distribution for every new run

Neural runs only (B1 has no epochs). `best_epoch` is the epoch selected by each checkpoint rule; a run is flagged when any of its epochs is 0.

| source | arm | ckpt rule | runs | median | mean | min | max | #(==0) | flag epoch 0 |
|---|---|---|---|---|---|---|---|---|---|
| S1 | A0 | primary | 15 | 3.0 | 4.00 | 0 | 12 | 1 | YES |
| S1 | A1 | primary | 15 | 7.0 | 9.33 | 0 | 35 | 1 | YES |
| S1 | A4 | primary | 15 | 10.0 | 8.87 | 0 | 23 | 1 | YES |
| S1 | C1 | primary | 15 | 9.0 | 11.47 | 0 | 38 | 1 | YES |
| S1 | Cshuf | primary | 15 | 3.0 | 3.67 | 0 | 12 | 1 | YES |
| S2 | A0 | primary | 3 | 3.0 | 4.33 | 1 | 9 | 0 | no |
| S2 | A1 | primary | 3 | 4.0 | 4.00 | 1 | 7 | 0 | no |
| S2 | A2 | primary | 3 | 4.0 | 5.00 | 2 | 9 | 0 | no |
| S2 | A3 | primary | 3 | 7.0 | 10.33 | 7 | 17 | 0 | no |
| S2 | A4 | primary | 3 | 7.0 | 6.00 | 4 | 7 | 0 | no |
| S1 | A0 | secondary | 15 | 5.0 | 6.00 | 1 | 17 | 0 | no |
| S1 | A1 | secondary | 15 | 3.0 | 5.20 | 0 | 23 | 2 | YES |
| S1 | A4 | secondary | 15 | 5.0 | 4.53 | 0 | 13 | 4 | YES |
| S1 | C1 | secondary | 15 | 4.0 | 6.13 | 0 | 31 | 3 | YES |
| S1 | Cshuf | secondary | 15 | 4.0 | 4.73 | 0 | 12 | 1 | YES |
| S2 | A0 | secondary | 3 | 3.0 | 4.00 | 1 | 8 | 0 | no |
| S2 | A1 | secondary | 3 | 3.0 | 2.67 | 1 | 4 | 0 | no |
| S2 | A2 | secondary | 3 | 3.0 | 4.33 | 2 | 8 | 0 | no |
| S2 | A3 | secondary | 3 | 3.0 | 2.67 | 1 | 4 | 0 | no |
| S2 | A4 | secondary | 3 | 1.0 | 2.00 | 1 | 4 | 0 | no |

==============================================================================
EXPECTED VS ACTUAL
==============================================================================

| check | expected | actual | pass |
|---|---|---|---|
| retrain reproduces Step 9 (A0/A1/A4) | max dev ~0 | 0.00e+00 | yes |
| S1 grid complete | 75 | 75 | yes |
| S2 grid complete | 18 | 18 | yes |
| both checkpoint rules tracked | yes | yes | yes |
| intervention reported per modality and arm | yes | yes | yes |
| every table script-generated | yes | yes | yes |
| no tuning after seeing results | yes | yes | yes |

Files written under results/step10/: s1/ (runs, curves, weights [gitignored .pt], run_log, oof_all.parquet), s2/ (runs, run_log, oof_all.parquet), diagnostics/ (intervention_*.csv/md, intervention.json, intervention_raw.csv), verification.{csv,md}, controls_{rule}.{csv,md}, controls_paired_{rule}.csv, s2_{rule}.{csv,md}, s2_paired_{rule}.csv, s2_pc_given_m.{csv,md}, best_epochs.{csv,md}, analysis.json, REPORT.md. New code: scripts/run_step10_s1.py, scripts/run_step10_s2.py, scripts/run_step10_intervention.py, scripts/analyse_step10.py, scripts/step10_report.py, scripts/step10_appendix.py, src/adni/diagnostics.py, tests/test_step10_controls.py; modified src/adni/losses.py and src/adni/train.py (shuffle_a3 control).
