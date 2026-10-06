STEP 09 REPORT
Main experiment: arms A0-A4 x S1 seeds [0,1,2] x folds [0..4] = 75 runs.

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/run_step09.py      > results/step09/{runs/*.parquet, curves/*.csv, run_log.jsonl,
                                oof_all.parquet}
  scripts/analyse_step09.py  > results/step09/{main_table_*, paired_*, breakdown_*,
                                stability_per_seed_*, stability_per_fold_*, best_epochs_*,
                                runtime}.{csv,md}, analysis.json
  scripts/step09_report.py   > results/step09/REPORT.md
  scripts/step09_appendix.py > Appendix E row in docs/PIPELINE_v2.md and ../PIPELINE_v2.md
  python -m pytest -q

Git commit: (this commit) - tag `step09`

Tests: 94 passed in 103.51s (0:01:43)

Grid: 75 runs completed, 0 run errors; total training wall-clock 29.6 min (see the runtime table).

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1. GPU: NVIDIA RTX 3050 (6 GB). Training used `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Seeds are fixed per run; `train_arm` seeds before the model is built, so each run is reproducible.

------------------------------------------------------------------------------
PRE-DECLARATION (written before any grid result was seen)

PRIMARY checkpoint rule: best validation macro-F1, exactly as in Steps 7-8. Early stopping and the patience-15 counter use this rule only. SECONDARY checkpoint rule: best validation PN balanced accuracy (rho uniform, weights from the fold's TRAIN-split pi_hat via `metrics.pn_weights`), selected among the epochs the run actually trained. Both are tracked per epoch in the SAME run, both best states are kept, and the test fold is evaluated with BOTH. Both rules apply identically to every arm.

Which scores validation macro-F1 uses for A1 and A4: it uses the argmax of the RAW model logits `s_c`, for every arm A0-A4. For A1 (trained with `log P(c)` added) and A4 (trained with `tau*log pi(c|m_tilde)` added) the training-time score transform is NOT applied at validation: `argmax_c s_c` is identical to `argmax_c (s_c + log rho_uniform(c))`, because a uniform deployment prior rho adds the constant `log(1/3)` to every class. A0, A2 and A3 also use `argmax_c s_c`. A0-A4 are therefore treated consistently, and no arm-specific score adjustment enters the validation macro-F1. The same raw logits are the `uniform` operating point used for A1/A4 in every reported table.

Operating points and the paired comparison: each arm's PRIMARY operating point is used for the primary comparison (A0/A2/A3 argmax; A1/A4 rho uniform); A4 at rho=P(c) is reported as a supplementary operating point in the main table. All analysis is done separately for the primary and the secondary checkpoint rule.

Deviation from PIPELINE_v2 section 6: the spec text states early stopping on the validation PN-BAcc and 'five seeds per fold'. Step 9 follows the pre-declaration above and the established Steps 7-8 protocol (3 seeds, early stopping on validation macro-F1), because the pipeline text was written before the checkpoint-rule decision; both rules are nevertheless tracked and both verdicts are reported. This is recorded in Appendix E.

Bug found and fixed during Step 9 (before any table was trusted): adding the dual-checkpoint tracking to `src/adni/train.py` initially put the raw per-epoch validation PN-BAcc into the per-epoch history dict. When it is not computable (no `val_weights`) that value is NaN, and because `NaN != NaN` two byte-identical runs then compared UNEQUAL. This failed the two pre-existing determinism tests (`tests/test_models.py::test_train_a0_reproducible_without_external_seeding` and `tests/test_priors_losses.py::test_train_arm_deterministic_through_real_entry_point`) - not because training was non-deterministic, but because the NaN sentinel is not self-equal. Fix: the per-epoch history now stores `None` ("not computed") when PN-BAcc is non-finite, while the internal metric stays NaN so the finite guard for the SECONDARY selection is unchanged. No test was weakened or skipped. Impact on the grid: none - every real run supplies `val_weights` (the fold's TRAIN-split `pi_hat`), and all 1785 per-epoch rows across the 75 curve files have a finite `val_pn_bacc`, so loss, epoch counts, checkpoint selection and all reported numbers are unchanged; the grid was therefore NOT re-run. The full suite is green after the fix (see the Tests line above).

Verdicts (H3: A4 beats A0-A3 on PN-BAcc(uniform) after Holm and is non-inferior on natural macro-F1 within 0.02):

| checkpoint rule | A4 wins all 4 PN-BAcc (Holm) | macro-F1 non-inferior | verdict |
|---|---|---|---|
| primary | no | no | negative result |
| secondary | no | no | negative result |

The PRIMARY checkpoint rule is the validation-macro-F1 rule; the secondary rule is reported alongside. Both verdicts are given above and below.

==============================================================================
MAIN TABLE - PRIMARY CHECKPOINT RULE
==============================================================================
# Step 9 - main table (primary checkpoint rule)

Seed-averaged OOF predictions, one 2,000-resample subject-cluster bootstrap (Step 6b uniform protocol). point [95% CI].

| arm | op | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) | ECE | Brier | F1 CN | F1 MCI | F1 Dem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | argmax | 13240 | 0.7938 [0.7831, 0.8040] | 0.6137 [0.6004, 0.6268] | 0.6198 [0.6062, 0.6335] | 0.5982 [0.5836, 0.6129] | 0.5980 [0.5833, 0.6127] | 0.0589 [0.0464, 0.0729] | 0.4891 [0.4767, 0.5019] | 0.6718 | 0.5158 | 0.6535 |
| A1 (uniform) | uniform | 13240 | 0.7862 [0.7751, 0.7968] | 0.6011 [0.5879, 0.6143] | 0.6288 [0.6156, 0.6426] | 0.6094 [0.5954, 0.6242] | 0.6092 [0.5952, 0.6239] | 0.1061 [0.0941, 0.1207] | 0.5219 [0.5070, 0.5372] | 0.6681 | 0.4777 | 0.6575 |
| A1 (P(c)) | P(c) | 13240 | 0.7884 [0.7773, 0.7991] | 0.6138 [0.6004, 0.6271] | 0.6148 [0.6007, 0.6288] | 0.5943 [0.5789, 0.6092] | 0.5940 [0.5786, 0.6090] | 0.0925 [0.0793, 0.1061] | 0.5064 [0.4914, 0.5211] | 0.6637 | 0.5307 | 0.6470 |
| A2 | argmax | 13240 | 0.7939 [0.7834, 0.8043] | 0.6153 [0.6018, 0.6284] | 0.6208 [0.6073, 0.6342] | 0.5974 [0.5824, 0.6125] | 0.5971 [0.5821, 0.6122] | 0.0565 [0.0440, 0.0702] | 0.4892 [0.4763, 0.5023] | 0.6707 | 0.5228 | 0.6524 |
| A3 | argmax | 13240 | 0.7724 [0.7617, 0.7834] | 0.5937 [0.5795, 0.6073] | 0.6200 [0.6066, 0.6340] | 0.6143 [0.6000, 0.6290] | 0.6140 [0.5996, 0.6286] | 0.1112 [0.0974, 0.1254] | 0.5337 [0.5186, 0.5490] | 0.6615 | 0.4636 | 0.6561 |
| A4 (uniform) | uniform | 13240 | 0.7735 [0.7627, 0.7848] | 0.5902 [0.5764, 0.6037] | 0.6227 [0.6095, 0.6360] | 0.6163 [0.6023, 0.6304] | 0.6161 [0.6022, 0.6303] | 0.1090 [0.0952, 0.1229] | 0.5340 [0.5190, 0.5491] | 0.6674 | 0.4506 | 0.6525 |
| A4 (P(c)) | P(c) | 13240 | 0.7760 [0.7648, 0.7874] | 0.6082 [0.5945, 0.6219] | 0.6133 [0.5993, 0.6273] | 0.6072 [0.5927, 0.6218] | 0.6070 [0.5925, 0.6215] | 0.0893 [0.0763, 0.1032] | 0.5173 [0.5028, 0.5320] | 0.6645 | 0.5078 | 0.6521 |
| B0 | - | 13240 | 0.6001 [0.5883, 0.6123] | 0.3544 [0.3452, 0.3643] | 0.4026 [0.3939, 0.4115] | 0.3340 [0.3252, 0.3433] | 0.3334 [0.3247, 0.3425] | 0.0509 [0.0402, 0.0626] | 0.6076 [0.6004, 0.6147] | 0.4771 | 0.5861 | 0.0000 |
| B1 | - | 13240 | 0.8101 [0.7999, 0.8204] | 0.6392 [0.6259, 0.6535] | 0.6323 [0.6182, 0.6471] | 0.5938 [0.5790, 0.6096] | 0.5945 [0.5796, 0.6103] | 0.0179 [0.0133, 0.0314] | 0.4672 [0.4556, 0.4786] | 0.6945 | 0.5798 | 0.6434 |
| B1-blind | - | 13240 | 0.7620 [0.7509, 0.7736] | 0.5826 [0.5689, 0.5970] | 0.5755 [0.5615, 0.5903] | 0.5628 [0.5478, 0.5778] | 0.5625 [0.5475, 0.5774] | 0.0130 [0.0080, 0.0260] | 0.5143 [0.5042, 0.5239] | 0.6454 | 0.5062 | 0.5962 |
| B2_mri | - | 12310 | 0.7661 [0.7548, 0.7772] | 0.5854 [0.5712, 0.6004] | 0.5808 [0.5663, 0.5966] | 0.5702 [0.5549, 0.5862] | 0.5697 [0.5544, 0.5858] | 0.0278 [0.0186, 0.0433] | 0.5117 [0.5006, 0.5223] | 0.6483 | 0.5049 | 0.6030 |

==============================================================================
MAIN TABLE - SECONDARY CHECKPOINT RULE
==============================================================================
# Step 9 - main table (secondary checkpoint rule)

Seed-averaged OOF predictions, one 2,000-resample subject-cluster bootstrap (Step 6b uniform protocol). point [95% CI].

| arm | op | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) | ECE | Brier | F1 CN | F1 MCI | F1 Dem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | argmax | 13240 | 0.7913 [0.7805, 0.8018] | 0.6129 [0.6000, 0.6261] | 0.6207 [0.6074, 0.6344] | 0.5998 [0.5852, 0.6147] | 0.5996 [0.5851, 0.6145] | 0.0808 [0.0674, 0.0943] | 0.4987 [0.4847, 0.5130] | 0.6677 | 0.5118 | 0.6593 |
| A1 (uniform) | uniform | 13240 | 0.7885 [0.7778, 0.7990] | 0.5887 [0.5750, 0.6015] | 0.6321 [0.6196, 0.6448] | 0.6135 [0.6003, 0.6273] | 0.6133 [0.6002, 0.6271] | 0.0903 [0.0783, 0.1050] | 0.5165 [0.5029, 0.5307] | 0.6854 | 0.4335 | 0.6474 |
| A1 (P(c)) | P(c) | 13240 | 0.7922 [0.7815, 0.8028] | 0.6158 [0.6023, 0.6291] | 0.6264 [0.6126, 0.6399] | 0.6055 [0.5908, 0.6205] | 0.6052 [0.5906, 0.6204] | 0.0647 [0.0523, 0.0782] | 0.4931 [0.4799, 0.5062] | 0.6829 | 0.5081 | 0.6563 |
| A2 | argmax | 13240 | 0.7929 [0.7824, 0.8035] | 0.6163 [0.6034, 0.6291] | 0.6263 [0.6127, 0.6398] | 0.6047 [0.5899, 0.6191] | 0.6044 [0.5897, 0.6189] | 0.0686 [0.0551, 0.0821] | 0.4936 [0.4802, 0.5070] | 0.6775 | 0.5146 | 0.6569 |
| A3 | argmax | 13240 | 0.7710 [0.7605, 0.7818] | 0.5770 [0.5633, 0.5908] | 0.6228 [0.6098, 0.6353] | 0.6198 [0.6060, 0.6338] | 0.6197 [0.6059, 0.6337] | 0.0737 [0.0609, 0.0881] | 0.5253 [0.5123, 0.5381] | 0.6752 | 0.4135 | 0.6424 |
| A4 (uniform) | uniform | 13240 | 0.7711 [0.7608, 0.7816] | 0.5751 [0.5615, 0.5879] | 0.6249 [0.6122, 0.6370] | 0.6223 [0.6089, 0.6355] | 0.6221 [0.6087, 0.6353] | 0.0916 [0.0784, 0.1051] | 0.5315 [0.5176, 0.5449] | 0.6799 | 0.4023 | 0.6429 |
| A4 (P(c)) | P(c) | 13240 | 0.7761 [0.7652, 0.7870] | 0.6039 [0.5914, 0.6167] | 0.6178 [0.6042, 0.6309] | 0.6155 [0.6014, 0.6296] | 0.6154 [0.6013, 0.6295] | 0.0589 [0.0473, 0.0739] | 0.5069 [0.4940, 0.5198] | 0.6739 | 0.4809 | 0.6570 |
| B0 | - | 13240 | 0.6001 [0.5883, 0.6123] | 0.3544 [0.3452, 0.3643] | 0.4026 [0.3939, 0.4115] | 0.3340 [0.3252, 0.3433] | 0.3334 [0.3247, 0.3425] | 0.0509 [0.0402, 0.0626] | 0.6076 [0.6004, 0.6147] | 0.4771 | 0.5861 | 0.0000 |
| B1 | - | 13240 | 0.8101 [0.7999, 0.8204] | 0.6392 [0.6259, 0.6535] | 0.6323 [0.6182, 0.6471] | 0.5938 [0.5790, 0.6096] | 0.5945 [0.5796, 0.6103] | 0.0179 [0.0133, 0.0314] | 0.4672 [0.4556, 0.4786] | 0.6945 | 0.5798 | 0.6434 |
| B1-blind | - | 13240 | 0.7620 [0.7509, 0.7736] | 0.5826 [0.5689, 0.5970] | 0.5755 [0.5615, 0.5903] | 0.5628 [0.5478, 0.5778] | 0.5625 [0.5475, 0.5774] | 0.0130 [0.0080, 0.0260] | 0.5143 [0.5042, 0.5239] | 0.6454 | 0.5062 | 0.5962 |
| B2_mri | - | 12310 | 0.7661 [0.7548, 0.7772] | 0.5854 [0.5712, 0.6004] | 0.5808 [0.5663, 0.5966] | 0.5702 [0.5549, 0.5862] | 0.5697 [0.5544, 0.5858] | 0.0278 [0.0186, 0.0433] | 0.5117 [0.5006, 0.5223] | 0.6483 | 0.5049 | 0.6030 |

==============================================================================
PRIMARY COMPARISONS - PRIMARY CHECKPOINT RULE
==============================================================================
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

Verdict (primary): **negative result**.

==============================================================================
PRIMARY COMPARISONS - SECONDARY CHECKPOINT RULE
==============================================================================
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

Verdict (secondary): **negative result**.

==============================================================================
BREAKDOWNS (A0 / A1 / A4 at their primary operating point)
==============================================================================
### Breakdowns - primary checkpoint rule

# Step 9 - breakdowns (primary checkpoint rule; A0/A1/A4 at their primary operating point)

Point estimates on the seed-averaged OOF predictions.

| arm | op | breakdown | stratum | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | argmax | pattern | 0001 | 133 | 0.7871 | 0.6302 | 0.6197 | 0.6232 | 0.6232 |
| A0 | argmax | pattern | 0010 | 304 | 0.7129 | 0.5182 | 0.5289 | 0.5296 | 0.5297 |
| A0 | argmax | pattern | 0011 | 187 | 0.7514 | 0.5180 | 0.5457 | 0.5426 | 0.5428 |
| A0 | argmax | pattern | 0100 | 142 | 0.7658 | 0.5955 | 0.5845 | 0.5793 | 0.5795 |
| A0 | argmax | pattern | 0110 | 137 | 0.7246 | 0.5523 | 0.5601 | 0.5620 | 0.5621 |
| A0 | argmax | pattern | 1000 | 4954 | 0.7640 | 0.5833 | 0.5915 | 0.5912 | 0.5913 |
| A0 | argmax | pattern | 1001 | 983 | 0.8197 | 0.6439 | 0.6336 | 0.6340 | 0.6340 |
| A0 | argmax | pattern | 1010 | 630 | 0.7655 | 0.5914 | 0.5965 | 0.5968 | 0.5969 |
| A0 | argmax | pattern | 1011 | 986 | 0.8012 | 0.6481 | 0.6460 | 0.6464 | 0.6469 |
| A0 | argmax | pattern | 1100 | 2123 | 0.7763 | 0.5877 | 0.5955 | 0.5951 | 0.5951 |
| A0 | argmax | pattern | 1101 | 277 | 0.7896 | 0.4207 | 0.6855 | 0.6849 | 0.6849 |
| A0 | argmax | pattern | 1110 | 2123 | 0.7781 | 0.5984 | 0.5927 | 0.5926 | 0.5926 |
| A0 | argmax | pattern | 1111 | 234 | 0.7917 | 0.5078 | 0.6678 | 0.6701 | 0.6698 |
| A0 | argmax | pattern | other (<30) | 27 | 0.6362 | 0.4931 | 0.5440 | 0.5365 | 0.5421 |
| A0 | argmax | n_mod | n_mod=1 | 5533 | 0.7640 | 0.5848 | 0.5921 | 0.5884 | 0.5885 |
| A0 | argmax | n_mod | n_mod=2 | 4067 | 0.8119 | 0.6330 | 0.6449 | 0.6009 | 0.6009 |
| A0 | argmax | n_mod | n_mod=3 | 3406 | 0.8182 | 0.6397 | 0.6370 | 0.6055 | 0.6056 |
| A0 | argmax | n_mod | n_mod=4 | 234 | 0.7917 | 0.5078 | 0.6678 | 0.6701 | 0.6698 |
| A0 | argmax | cell_count | cell_count low (<= 491 / <= 754) | 4421 | 0.7671 | 0.5942 | 0.5867 | 0.5574 | 0.5566 |
| A0 | argmax | cell_count | cell_count mid (<= 491 / <= 754) | 4438 | 0.8533 | 0.6554 | 0.6583 | 0.6347 | 0.6347 |
| A0 | argmax | cell_count | cell_count high (<= 491 / <= 754) | 4381 | 0.7504 | 0.5279 | 0.6092 | 0.6086 | 0.6086 |
| A0 | argmax | colprot | ADNI1 | 3664 | 0.7744 | 0.5940 | 0.5997 | 0.5939 | 0.5939 |
| A0 | argmax | colprot | ADNI2 | 4541 | 0.7621 | 0.5845 | 0.5821 | 0.5732 | 0.5732 |
| A0 | argmax | colprot | ADNI3 | 3010 | 0.8225 | 0.6276 | 0.6254 | 0.6181 | 0.6176 |
| A0 | argmax | colprot | ADNI4 | 1578 | 0.7825 | 0.5832 | 0.5868 | 0.6030 | 0.6040 |
| A0 | argmax | colprot | ADNIGO | 447 | 0.7519 | 0.5721 | 0.6162 | 0.6128 | 0.6127 |
| A0 | argmax | gap | max|gap| low (<= 11 / <= 35 d) | 4566 | 0.7738 | 0.6012 | 0.6078 | 0.5954 | 0.5953 |
| A0 | argmax | gap | max|gap| mid (<= 11 / <= 35 d) | 4272 | 0.7993 | 0.6119 | 0.6209 | 0.5979 | 0.5980 |
| A0 | argmax | gap | max|gap| high (<= 11 / <= 35 d) | 4402 | 0.8065 | 0.6248 | 0.6259 | 0.5993 | 0.5987 |
| A1 | uniform | pattern | 0001 | 133 | 0.7771 | 0.5627 | 0.5856 | 0.5878 | 0.5878 |
| A1 | uniform | pattern | 0010 | 304 | 0.7006 | 0.5166 | 0.5924 | 0.5917 | 0.5917 |
| A1 | uniform | pattern | 0011 | 187 | 0.7293 | 0.4760 | 0.5337 | 0.5315 | 0.5318 |
| A1 | uniform | pattern | 0100 | 142 | 0.7735 | 0.6205 | 0.6218 | 0.6160 | 0.6163 |
| A1 | uniform | pattern | 0110 | 137 | 0.7345 | 0.5613 | 0.5885 | 0.5908 | 0.5909 |
| A1 | uniform | pattern | 1000 | 4954 | 0.7615 | 0.5767 | 0.6026 | 0.6022 | 0.6022 |
| A1 | uniform | pattern | 1001 | 983 | 0.8026 | 0.6406 | 0.6593 | 0.6594 | 0.6594 |
| A1 | uniform | pattern | 1010 | 630 | 0.7560 | 0.5779 | 0.6114 | 0.6117 | 0.6118 |
| A1 | uniform | pattern | 1011 | 986 | 0.7939 | 0.6158 | 0.6457 | 0.6458 | 0.6467 |
| A1 | uniform | pattern | 1100 | 2123 | 0.7612 | 0.5600 | 0.5869 | 0.5866 | 0.5866 |
| A1 | uniform | pattern | 1101 | 277 | 0.7866 | 0.4263 | 0.7198 | 0.7189 | 0.7188 |
| A1 | uniform | pattern | 1110 | 2123 | 0.7706 | 0.5893 | 0.6082 | 0.6082 | 0.6083 |
| A1 | uniform | pattern | 1111 | 234 | 0.8064 | 0.5311 | 0.7421 | 0.7442 | 0.7441 |
| A1 | uniform | pattern | other (<30) | 27 | 0.6102 | 0.4690 | 0.5231 | 0.5069 | 0.5162 |
| A1 | uniform | n_mod | n_mod=1 | 5533 | 0.7611 | 0.5777 | 0.6045 | 0.6018 | 0.6018 |
| A1 | uniform | n_mod | n_mod=2 | 4067 | 0.7977 | 0.6081 | 0.6403 | 0.6047 | 0.6048 |
| A1 | uniform | n_mod | n_mod=3 | 3406 | 0.8117 | 0.6300 | 0.6528 | 0.6179 | 0.6180 |
| A1 | uniform | n_mod | n_mod=4 | 234 | 0.8064 | 0.5311 | 0.7421 | 0.7442 | 0.7441 |
| A1 | uniform | cell_count | cell_count low (<= 491 / <= 754) | 4421 | 0.7679 | 0.6011 | 0.6049 | 0.5790 | 0.5781 |
| A1 | uniform | cell_count | cell_count mid (<= 491 / <= 754) | 4438 | 0.8359 | 0.6283 | 0.6525 | 0.6297 | 0.6298 |
| A1 | uniform | cell_count | cell_count high (<= 491 / <= 754) | 4381 | 0.7428 | 0.5039 | 0.6180 | 0.6172 | 0.6173 |
| A1 | uniform | colprot | ADNI1 | 3664 | 0.7664 | 0.5681 | 0.5901 | 0.5854 | 0.5854 |
| A1 | uniform | colprot | ADNI2 | 4541 | 0.7586 | 0.5808 | 0.6015 | 0.5929 | 0.5930 |
| A1 | uniform | colprot | ADNI3 | 3010 | 0.8069 | 0.6159 | 0.6411 | 0.6292 | 0.6284 |
| A1 | uniform | colprot | ADNI4 | 1578 | 0.7739 | 0.5894 | 0.6219 | 0.6399 | 0.6410 |
| A1 | uniform | colprot | ADNIGO | 447 | 0.7569 | 0.5401 | 0.6241 | 0.6183 | 0.6183 |
| A1 | uniform | gap | max|gap| low (<= 11 / <= 35 d) | 4566 | 0.7654 | 0.5779 | 0.6040 | 0.5931 | 0.5931 |
| A1 | uniform | gap | max|gap| mid (<= 11 / <= 35 d) | 4272 | 0.7927 | 0.6076 | 0.6367 | 0.6159 | 0.6160 |
| A1 | uniform | gap | max|gap| high (<= 11 / <= 35 d) | 4402 | 0.7994 | 0.6158 | 0.6431 | 0.6189 | 0.6182 |
| A4 | uniform | pattern | 0001 | 133 | 0.7942 | 0.5687 | 0.6009 | 0.6034 | 0.6035 |
| A4 | uniform | pattern | 0010 | 304 | 0.6995 | 0.5176 | 0.6030 | 0.5999 | 0.5997 |
| A4 | uniform | pattern | 0011 | 187 | 0.7001 | 0.4976 | 0.5617 | 0.5594 | 0.5596 |
| A4 | uniform | pattern | 0100 | 142 | 0.7451 | 0.5573 | 0.5850 | 0.5804 | 0.5809 |
| A4 | uniform | pattern | 0110 | 137 | 0.7239 | 0.5473 | 0.5861 | 0.5896 | 0.5896 |
| A4 | uniform | pattern | 1000 | 4954 | 0.7620 | 0.5775 | 0.6069 | 0.6067 | 0.6067 |
| A4 | uniform | pattern | 1001 | 983 | 0.7960 | 0.6397 | 0.6695 | 0.6692 | 0.6692 |
| A4 | uniform | pattern | 1010 | 630 | 0.7469 | 0.5764 | 0.6143 | 0.6142 | 0.6143 |
| A4 | uniform | pattern | 1011 | 986 | 0.7812 | 0.5820 | 0.6290 | 0.6288 | 0.6298 |
| A4 | uniform | pattern | 1100 | 2123 | 0.7705 | 0.5641 | 0.6112 | 0.6107 | 0.6107 |
| A4 | uniform | pattern | 1101 | 277 | 0.7742 | 0.4004 | 0.6967 | 0.6958 | 0.6958 |
| A4 | uniform | pattern | 1110 | 2123 | 0.7752 | 0.5918 | 0.6207 | 0.6206 | 0.6207 |
| A4 | uniform | pattern | 1111 | 234 | 0.8171 | 0.4858 | 0.7137 | 0.7154 | 0.7153 |
| A4 | uniform | pattern | other (<30) | 27 | 0.6045 | 0.4199 | 0.4815 | 0.4777 | 0.4833 |
| A4 | uniform | n_mod | n_mod=1 | 5533 | 0.7592 | 0.5756 | 0.6055 | 0.6055 | 0.6055 |
| A4 | uniform | n_mod | n_mod=2 | 4067 | 0.7803 | 0.5956 | 0.6331 | 0.6217 | 0.6218 |
| A4 | uniform | n_mod | n_mod=3 | 3406 | 0.7870 | 0.6080 | 0.6359 | 0.6203 | 0.6204 |
| A4 | uniform | n_mod | n_mod=4 | 234 | 0.8171 | 0.4858 | 0.7137 | 0.7154 | 0.7153 |
| A4 | uniform | cell_count | cell_count low (<= 491 / <= 754) | 4421 | 0.7836 | 0.6077 | 0.6135 | 0.6104 | 0.6096 |
| A4 | uniform | cell_count | cell_count mid (<= 491 / <= 754) | 4438 | 0.7915 | 0.6007 | 0.6320 | 0.6211 | 0.6212 |
| A4 | uniform | cell_count | cell_count high (<= 491 / <= 754) | 4381 | 0.7371 | 0.4965 | 0.6137 | 0.6135 | 0.6134 |
| A4 | uniform | colprot | ADNI1 | 3664 | 0.7707 | 0.5694 | 0.6016 | 0.6001 | 0.6001 |
| A4 | uniform | colprot | ADNI2 | 4541 | 0.7606 | 0.5803 | 0.6085 | 0.6077 | 0.6078 |
| A4 | uniform | colprot | ADNI3 | 3010 | 0.7656 | 0.5858 | 0.6176 | 0.6171 | 0.6160 |
| A4 | uniform | colprot | ADNI4 | 1578 | 0.7614 | 0.5904 | 0.6230 | 0.6474 | 0.6485 |
| A4 | uniform | colprot | ADNIGO | 447 | 0.7493 | 0.5281 | 0.6309 | 0.6273 | 0.6273 |
| A4 | uniform | gap | max|gap| low (<= 11 / <= 35 d) | 4566 | 0.7619 | 0.5780 | 0.6071 | 0.6030 | 0.6030 |
| A4 | uniform | gap | max|gap| mid (<= 11 / <= 35 d) | 4272 | 0.7789 | 0.5889 | 0.6239 | 0.6187 | 0.6189 |
| A4 | uniform | gap | max|gap| high (<= 11 / <= 35 d) | 4402 | 0.7787 | 0.6003 | 0.6340 | 0.6267 | 0.6260 |

### Breakdowns - secondary checkpoint rule

# Step 9 - breakdowns (secondary checkpoint rule; A0/A1/A4 at their primary operating point)

Point estimates on the seed-averaged OOF predictions.

| arm | op | breakdown | stratum | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) |
|---|---|---|---|---|---|---|---|---|---|
| A0 | argmax | pattern | 0001 | 133 | 0.7886 | 0.6498 | 0.6364 | 0.6396 | 0.6396 |
| A0 | argmax | pattern | 0010 | 304 | 0.7078 | 0.5500 | 0.5751 | 0.5738 | 0.5737 |
| A0 | argmax | pattern | 0011 | 187 | 0.7499 | 0.5126 | 0.5429 | 0.5386 | 0.5390 |
| A0 | argmax | pattern | 0100 | 142 | 0.7770 | 0.6168 | 0.6058 | 0.6006 | 0.6009 |
| A0 | argmax | pattern | 0110 | 137 | 0.7164 | 0.5582 | 0.5694 | 0.5709 | 0.5710 |
| A0 | argmax | pattern | 1000 | 4954 | 0.7648 | 0.5863 | 0.5970 | 0.5967 | 0.5967 |
| A0 | argmax | pattern | 1001 | 983 | 0.8110 | 0.6340 | 0.6226 | 0.6230 | 0.6230 |
| A0 | argmax | pattern | 1010 | 630 | 0.7618 | 0.5855 | 0.5898 | 0.5901 | 0.5902 |
| A0 | argmax | pattern | 1011 | 986 | 0.8009 | 0.6405 | 0.6392 | 0.6398 | 0.6406 |
| A0 | argmax | pattern | 1100 | 2123 | 0.7706 | 0.5803 | 0.5887 | 0.5882 | 0.5883 |
| A0 | argmax | pattern | 1101 | 277 | 0.7602 | 0.3972 | 0.6684 | 0.6676 | 0.6676 |
| A0 | argmax | pattern | 1110 | 2123 | 0.7738 | 0.6032 | 0.6000 | 0.6000 | 0.6001 |
| A0 | argmax | pattern | 1111 | 234 | 0.7879 | 0.4909 | 0.6694 | 0.6706 | 0.6705 |
| A0 | argmax | pattern | other (<30) | 27 | 0.6332 | 0.4931 | 0.5440 | 0.5365 | 0.5421 |
| A0 | argmax | n_mod | n_mod=1 | 5533 | 0.7647 | 0.5895 | 0.5995 | 0.5967 | 0.5967 |
| A0 | argmax | n_mod | n_mod=2 | 4067 | 0.8072 | 0.6273 | 0.6385 | 0.5938 | 0.5939 |
| A0 | argmax | n_mod | n_mod=3 | 3406 | 0.8141 | 0.6376 | 0.6372 | 0.6068 | 0.6069 |
| A0 | argmax | n_mod | n_mod=4 | 234 | 0.7879 | 0.4909 | 0.6694 | 0.6706 | 0.6705 |
| A0 | argmax | cell_count | cell_count low (<= 491 / <= 754) | 4421 | 0.7638 | 0.5914 | 0.5848 | 0.5579 | 0.5569 |
| A0 | argmax | cell_count | cell_count mid (<= 491 / <= 754) | 4438 | 0.8495 | 0.6571 | 0.6619 | 0.6391 | 0.6391 |
| A0 | argmax | cell_count | cell_count high (<= 491 / <= 754) | 4381 | 0.7501 | 0.5261 | 0.6092 | 0.6085 | 0.6085 |
| A0 | argmax | colprot | ADNI1 | 3664 | 0.7748 | 0.5885 | 0.5957 | 0.5900 | 0.5901 |
| A0 | argmax | colprot | ADNI2 | 4541 | 0.7592 | 0.5931 | 0.5932 | 0.5850 | 0.5851 |
| A0 | argmax | colprot | ADNI3 | 3010 | 0.8154 | 0.6100 | 0.6096 | 0.6053 | 0.6047 |
| A0 | argmax | colprot | ADNI4 | 1578 | 0.7809 | 0.5856 | 0.5909 | 0.6073 | 0.6085 |
| A0 | argmax | colprot | ADNIGO | 447 | 0.7444 | 0.5758 | 0.6174 | 0.6133 | 0.6133 |
| A0 | argmax | gap | max|gap| low (<= 11 / <= 35 d) | 4566 | 0.7701 | 0.5947 | 0.6030 | 0.5904 | 0.5905 |
| A0 | argmax | gap | max|gap| mid (<= 11 / <= 35 d) | 4272 | 0.7966 | 0.6104 | 0.6205 | 0.5972 | 0.5974 |
| A0 | argmax | gap | max|gap| high (<= 11 / <= 35 d) | 4402 | 0.8058 | 0.6317 | 0.6350 | 0.6102 | 0.6095 |
| A1 | uniform | pattern | 0001 | 133 | 0.7728 | 0.5344 | 0.5862 | 0.5876 | 0.5878 |
| A1 | uniform | pattern | 0010 | 304 | 0.6974 | 0.4935 | 0.5968 | 0.5947 | 0.5946 |
| A1 | uniform | pattern | 0011 | 187 | 0.7279 | 0.4724 | 0.5431 | 0.5395 | 0.5397 |
| A1 | uniform | pattern | 0100 | 142 | 0.7699 | 0.5754 | 0.5847 | 0.5816 | 0.5816 |
| A1 | uniform | pattern | 0110 | 137 | 0.7129 | 0.5056 | 0.5548 | 0.5561 | 0.5561 |
| A1 | uniform | pattern | 1000 | 4954 | 0.7637 | 0.5642 | 0.6059 | 0.6057 | 0.6057 |
| A1 | uniform | pattern | 1001 | 983 | 0.8109 | 0.6266 | 0.6626 | 0.6627 | 0.6627 |
| A1 | uniform | pattern | 1010 | 630 | 0.7564 | 0.5715 | 0.6189 | 0.6195 | 0.6196 |
| A1 | uniform | pattern | 1011 | 986 | 0.7883 | 0.5893 | 0.6446 | 0.6445 | 0.6455 |
| A1 | uniform | pattern | 1100 | 2123 | 0.7663 | 0.5563 | 0.5946 | 0.5938 | 0.5938 |
| A1 | uniform | pattern | 1101 | 277 | 0.8139 | 0.4072 | 0.7103 | 0.7089 | 0.7089 |
| A1 | uniform | pattern | 1110 | 2123 | 0.7692 | 0.5853 | 0.6195 | 0.6197 | 0.6197 |
| A1 | uniform | pattern | 1111 | 234 | 0.8244 | 0.4804 | 0.7217 | 0.7230 | 0.7229 |
| A1 | uniform | pattern | other (<30) | 27 | 0.6142 | 0.4554 | 0.5185 | 0.5111 | 0.5177 |
| A1 | uniform | n_mod | n_mod=1 | 5533 | 0.7631 | 0.5644 | 0.6073 | 0.6042 | 0.6042 |
| A1 | uniform | n_mod | n_mod=2 | 4067 | 0.8019 | 0.5997 | 0.6450 | 0.6102 | 0.6102 |
| A1 | uniform | n_mod | n_mod=3 | 3406 | 0.8109 | 0.6175 | 0.6555 | 0.6253 | 0.6254 |
| A1 | uniform | n_mod | n_mod=4 | 234 | 0.8244 | 0.4804 | 0.7217 | 0.7230 | 0.7229 |
| A1 | uniform | cell_count | cell_count low (<= 491 / <= 754) | 4421 | 0.7686 | 0.5889 | 0.6032 | 0.5782 | 0.5774 |
| A1 | uniform | cell_count | cell_count mid (<= 491 / <= 754) | 4438 | 0.8404 | 0.6222 | 0.6619 | 0.6429 | 0.6429 |
| A1 | uniform | cell_count | cell_count high (<= 491 / <= 754) | 4381 | 0.7420 | 0.4849 | 0.6194 | 0.6186 | 0.6186 |
| A1 | uniform | colprot | ADNI1 | 3664 | 0.7700 | 0.5671 | 0.6003 | 0.5962 | 0.5962 |
| A1 | uniform | colprot | ADNI2 | 4541 | 0.7557 | 0.5689 | 0.6036 | 0.5966 | 0.5967 |
| A1 | uniform | colprot | ADNI3 | 3010 | 0.8153 | 0.6002 | 0.6461 | 0.6352 | 0.6347 |
| A1 | uniform | colprot | ADNI4 | 1578 | 0.7791 | 0.5532 | 0.6095 | 0.6274 | 0.6285 |
| A1 | uniform | colprot | ADNIGO | 447 | 0.7571 | 0.5155 | 0.6256 | 0.6170 | 0.6169 |
| A1 | uniform | gap | max|gap| low (<= 11 / <= 35 d) | 4566 | 0.7679 | 0.5712 | 0.6102 | 0.5980 | 0.5979 |
| A1 | uniform | gap | max|gap| mid (<= 11 / <= 35 d) | 4272 | 0.7926 | 0.5889 | 0.6360 | 0.6170 | 0.6171 |
| A1 | uniform | gap | max|gap| high (<= 11 / <= 35 d) | 4402 | 0.8039 | 0.6034 | 0.6475 | 0.6255 | 0.6249 |
| A4 | uniform | pattern | 0001 | 133 | 0.7626 | 0.5229 | 0.5923 | 0.5933 | 0.5937 |
| A4 | uniform | pattern | 0010 | 304 | 0.6969 | 0.4943 | 0.5901 | 0.5879 | 0.5878 |
| A4 | uniform | pattern | 0011 | 187 | 0.7014 | 0.4913 | 0.5621 | 0.5599 | 0.5601 |
| A4 | uniform | pattern | 0100 | 142 | 0.7727 | 0.5532 | 0.5968 | 0.5929 | 0.5933 |
| A4 | uniform | pattern | 0110 | 137 | 0.6945 | 0.5435 | 0.5953 | 0.5978 | 0.5978 |
| A4 | uniform | pattern | 1000 | 4954 | 0.7597 | 0.5651 | 0.6093 | 0.6091 | 0.6091 |
| A4 | uniform | pattern | 1001 | 983 | 0.7993 | 0.6239 | 0.6668 | 0.6667 | 0.6667 |
| A4 | uniform | pattern | 1010 | 630 | 0.7489 | 0.5699 | 0.6345 | 0.6348 | 0.6348 |
| A4 | uniform | pattern | 1011 | 986 | 0.7815 | 0.5776 | 0.6490 | 0.6487 | 0.6496 |
| A4 | uniform | pattern | 1100 | 2123 | 0.7717 | 0.5484 | 0.6155 | 0.6149 | 0.6149 |
| A4 | uniform | pattern | 1101 | 277 | 0.8074 | 0.4105 | 0.7186 | 0.7174 | 0.7174 |
| A4 | uniform | pattern | 1110 | 2123 | 0.7740 | 0.5727 | 0.6304 | 0.6303 | 0.6303 |
| A4 | uniform | pattern | 1111 | 234 | 0.8168 | 0.4589 | 0.7044 | 0.7055 | 0.7054 |
| A4 | uniform | pattern | other (<30) | 27 | 0.6339 | 0.4554 | 0.5185 | 0.5079 | 0.5143 |
| A4 | uniform | n_mod | n_mod=1 | 5533 | 0.7567 | 0.5624 | 0.6075 | 0.6071 | 0.6072 |
| A4 | uniform | n_mod | n_mod=2 | 4067 | 0.7754 | 0.5788 | 0.6313 | 0.6271 | 0.6271 |
| A4 | uniform | n_mod | n_mod=3 | 3406 | 0.7854 | 0.5923 | 0.6436 | 0.6358 | 0.6359 |
| A4 | uniform | n_mod | n_mod=4 | 234 | 0.8168 | 0.4589 | 0.7044 | 0.7055 | 0.7054 |
| A4 | uniform | cell_count | cell_count low (<= 491 / <= 754) | 4421 | 0.7918 | 0.6099 | 0.6237 | 0.6246 | 0.6239 |
| A4 | uniform | cell_count | cell_count mid (<= 491 / <= 754) | 4438 | 0.7814 | 0.5767 | 0.6293 | 0.6238 | 0.6238 |
| A4 | uniform | cell_count | cell_count high (<= 491 / <= 754) | 4381 | 0.7312 | 0.4748 | 0.6120 | 0.6118 | 0.6118 |
| A4 | uniform | colprot | ADNI1 | 3664 | 0.7674 | 0.5570 | 0.6048 | 0.6046 | 0.6046 |
| A4 | uniform | colprot | ADNI2 | 4541 | 0.7596 | 0.5688 | 0.6153 | 0.6163 | 0.6164 |
| A4 | uniform | colprot | ADNI3 | 3010 | 0.7654 | 0.5731 | 0.6243 | 0.6289 | 0.6280 |
| A4 | uniform | colprot | ADNI4 | 1578 | 0.7601 | 0.5681 | 0.6239 | 0.6462 | 0.6471 |
| A4 | uniform | colprot | ADNIGO | 447 | 0.7630 | 0.4867 | 0.6247 | 0.6217 | 0.6217 |
| A4 | uniform | gap | max|gap| low (<= 11 / <= 35 d) | 4566 | 0.7612 | 0.5700 | 0.6120 | 0.6105 | 0.6105 |
| A4 | uniform | gap | max|gap| mid (<= 11 / <= 35 d) | 4272 | 0.7765 | 0.5744 | 0.6293 | 0.6295 | 0.6296 |
| A4 | uniform | gap | max|gap| high (<= 11 / <= 35 d) | 4402 | 0.7747 | 0.5776 | 0.6322 | 0.6275 | 0.6270 |

==============================================================================
STABILITY
==============================================================================
### Per-seed metrics (mean +/- SD) - primary checkpoint rule

# Step 9 - per-seed stability (primary checkpoint rule)

Pooled OOF metrics per seed, then mean +/- SD over the 3 seeds.

| arm | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) |
|---|---|---|---|---|
| A0 | 0.7821 +/- 0.0057 | 0.6044 +/- 0.0087 | 0.6104 +/- 0.0121 | 0.5883 +/- 0.0117 |
| A1 (uniform) | 0.7699 +/- 0.0070 | 0.5889 +/- 0.0072 | 0.6169 +/- 0.0072 | 0.5983 +/- 0.0073 |
| A1 (P(c)) | 0.7714 +/- 0.0071 | 0.6014 +/- 0.0050 | 0.6036 +/- 0.0044 | 0.5834 +/- 0.0038 |
| A2 | 0.7816 +/- 0.0058 | 0.6063 +/- 0.0077 | 0.6112 +/- 0.0097 | 0.5884 +/- 0.0100 |
| A3 | 0.7545 +/- 0.0039 | 0.5785 +/- 0.0073 | 0.6060 +/- 0.0047 | 0.6009 +/- 0.0044 |
| A4 (uniform) | 0.7586 +/- 0.0075 | 0.5776 +/- 0.0059 | 0.6102 +/- 0.0058 | 0.6053 +/- 0.0068 |
| A4 (P(c)) | 0.7605 +/- 0.0072 | 0.5938 +/- 0.0038 | 0.5993 +/- 0.0035 | 0.5949 +/- 0.0043 |

### Per-fold PN-BAcc - primary checkpoint rule

# Step 9 - per-fold pn_bacc_uniform (primary checkpoint rule)

| arm | seed | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| A0 | 0 | 0.6129 | 0.5883 | 0.6079 | 0.6044 | 0.6065 |
| A0 | 1 | 0.5263 | 0.5985 | 0.5689 | 0.5970 | 0.5881 |
| A0 | 2 | 0.5759 | 0.5586 | 0.6008 | 0.6044 | 0.5872 |
| A1 | 0 | 0.6315 | 0.5698 | 0.6118 | 0.6284 | 0.6002 |
| A1 | 1 | 0.5525 | 0.5999 | 0.6005 | 0.6053 | 0.6024 |
| A1 | 2 | 0.6036 | 0.5783 | 0.5931 | 0.5999 | 0.5991 |
| A2 | 0 | 0.6103 | 0.5845 | 0.6048 | 0.6121 | 0.6010 |
| A2 | 1 | 0.5283 | 0.6072 | 0.5729 | 0.6060 | 0.5928 |
| A2 | 2 | 0.5833 | 0.5564 | 0.5901 | 0.5916 | 0.5831 |
| A3 | 0 | 0.6046 | 0.6089 | 0.6149 | 0.5926 | 0.6056 |
| A3 | 1 | 0.5886 | 0.6013 | 0.5990 | 0.6116 | 0.6110 |
| A3 | 2 | 0.6040 | 0.5648 | 0.5765 | 0.6230 | 0.6064 |
| A4 | 0 | 0.5987 | 0.6071 | 0.6356 | 0.6240 | 0.6087 |
| A4 | 1 | 0.5811 | 0.6066 | 0.5807 | 0.6160 | 0.6182 |
| A4 | 2 | 0.5999 | 0.5844 | 0.6017 | 0.6152 | 0.6011 |

### Per-fold macro-F1 - primary checkpoint rule

# Step 9 - per-fold macro_f1 (primary checkpoint rule)

| arm | seed | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| A0 | 0 | 0.6215 | 0.6102 | 0.6172 | 0.6157 | 0.6101 |
| A0 | 1 | 0.5560 | 0.6165 | 0.5814 | 0.6118 | 0.6041 |
| A0 | 2 | 0.5827 | 0.5942 | 0.6099 | 0.6083 | 0.6148 |
| A1 | 0 | 0.6084 | 0.5864 | 0.6106 | 0.6013 | 0.5847 |
| A1 | 1 | 0.5323 | 0.6021 | 0.5888 | 0.5977 | 0.5955 |
| A1 | 2 | 0.5893 | 0.5808 | 0.5774 | 0.5728 | 0.5935 |
| A2 | 0 | 0.6179 | 0.6079 | 0.6315 | 0.6196 | 0.6049 |
| A2 | 1 | 0.5585 | 0.6106 | 0.5974 | 0.6250 | 0.6062 |
| A2 | 2 | 0.5902 | 0.5907 | 0.6003 | 0.6130 | 0.6117 |
| A3 | 0 | 0.5955 | 0.5788 | 0.6032 | 0.5741 | 0.5846 |
| A3 | 1 | 0.5422 | 0.5896 | 0.5716 | 0.5811 | 0.5583 |
| A3 | 2 | 0.5725 | 0.5651 | 0.5719 | 0.5906 | 0.5863 |
| A4 | 0 | 0.5873 | 0.5820 | 0.6025 | 0.5868 | 0.5570 |
| A4 | 1 | 0.5409 | 0.5974 | 0.5625 | 0.5821 | 0.5593 |
| A4 | 2 | 0.5832 | 0.5725 | 0.5709 | 0.5827 | 0.5851 |

### Best-epoch distribution - primary checkpoint rule

# Step 9 - best-epoch distribution (primary checkpoint rule)

| arm | runs | median | mean | min | max | #(<5) | flag (median < 5) |
|---|---|---|---|---|---|---|---|
| A0 | 15 | 3.0 | 4.00 | 0 | 12 | 9 | YES |
| A1 | 15 | 7.0 | 9.33 | 0 | 35 | 5 | no |
| A2 | 15 | 3.0 | 3.87 | 0 | 16 | 10 | YES |
| A3 | 15 | 10.0 | 12.93 | 0 | 56 | 3 | no |
| A4 | 15 | 10.0 | 8.87 | 0 | 23 | 5 | no |

### Per-seed metrics (mean +/- SD) - secondary checkpoint rule

# Step 9 - per-seed stability (secondary checkpoint rule)

Pooled OOF metrics per seed, then mean +/- SD over the 3 seeds.

| arm | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) |
|---|---|---|---|---|
| A0 | 0.7789 +/- 0.0069 | 0.6030 +/- 0.0091 | 0.6120 +/- 0.0103 | 0.5909 +/- 0.0091 |
| A1 (uniform) | 0.7752 +/- 0.0059 | 0.5808 +/- 0.0098 | 0.6235 +/- 0.0085 | 0.6045 +/- 0.0092 |
| A1 (P(c)) | 0.7780 +/- 0.0062 | 0.6040 +/- 0.0074 | 0.6145 +/- 0.0065 | 0.5951 +/- 0.0064 |
| A2 | 0.7807 +/- 0.0051 | 0.6047 +/- 0.0067 | 0.6139 +/- 0.0082 | 0.5928 +/- 0.0083 |
| A3 | 0.7588 +/- 0.0034 | 0.5664 +/- 0.0053 | 0.6107 +/- 0.0060 | 0.6081 +/- 0.0081 |
| A4 (uniform) | 0.7591 +/- 0.0052 | 0.5655 +/- 0.0071 | 0.6119 +/- 0.0085 | 0.6091 +/- 0.0105 |
| A4 (P(c)) | 0.7628 +/- 0.0054 | 0.5887 +/- 0.0064 | 0.6018 +/- 0.0065 | 0.5992 +/- 0.0078 |

### Per-fold PN-BAcc - secondary checkpoint rule

# Step 9 - per-fold pn_bacc_uniform (secondary checkpoint rule)

| arm | seed | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| A0 | 0 | 0.6129 | 0.5883 | 0.6079 | 0.6044 | 0.6065 |
| A0 | 1 | 0.5648 | 0.6032 | 0.5689 | 0.6057 | 0.5881 |
| A0 | 2 | 0.5759 | 0.5581 | 0.5776 | 0.6044 | 0.5998 |
| A1 | 0 | 0.6315 | 0.6043 | 0.6230 | 0.6284 | 0.5987 |
| A1 | 1 | 0.5755 | 0.5999 | 0.6005 | 0.6136 | 0.6143 |
| A1 | 2 | 0.5969 | 0.5950 | 0.5939 | 0.5999 | 0.5946 |
| A2 | 0 | 0.6103 | 0.5845 | 0.6079 | 0.6121 | 0.6010 |
| A2 | 1 | 0.5749 | 0.6072 | 0.5765 | 0.6139 | 0.5928 |
| A2 | 2 | 0.5833 | 0.5564 | 0.5901 | 0.5916 | 0.5920 |
| A3 | 0 | 0.6067 | 0.6059 | 0.6352 | 0.6393 | 0.6083 |
| A3 | 1 | 0.5635 | 0.6283 | 0.5990 | 0.6167 | 0.6110 |
| A3 | 2 | 0.6040 | 0.5797 | 0.6177 | 0.6113 | 0.5927 |
| A4 | 0 | 0.6250 | 0.6035 | 0.6356 | 0.6439 | 0.6108 |
| A4 | 1 | 0.5673 | 0.6283 | 0.5904 | 0.6160 | 0.6182 |
| A4 | 2 | 0.6074 | 0.5804 | 0.6017 | 0.6168 | 0.5922 |

### Per-fold macro-F1 - secondary checkpoint rule

# Step 9 - per-fold macro_f1 (secondary checkpoint rule)

| arm | seed | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|---|
| A0 | 0 | 0.6215 | 0.6102 | 0.6172 | 0.6157 | 0.6101 |
| A0 | 1 | 0.5817 | 0.6111 | 0.5814 | 0.6119 | 0.6041 |
| A0 | 2 | 0.5827 | 0.5857 | 0.5796 | 0.6083 | 0.6142 |
| A1 | 0 | 0.6084 | 0.5934 | 0.6131 | 0.6013 | 0.5451 |
| A1 | 1 | 0.5485 | 0.6021 | 0.5888 | 0.5772 | 0.5772 |
| A1 | 2 | 0.5707 | 0.5598 | 0.5781 | 0.5728 | 0.5615 |
| A2 | 0 | 0.6179 | 0.6079 | 0.6159 | 0.6196 | 0.6049 |
| A2 | 1 | 0.5848 | 0.6106 | 0.5894 | 0.6147 | 0.6062 |
| A2 | 2 | 0.5902 | 0.5907 | 0.6003 | 0.6130 | 0.5938 |
| A3 | 0 | 0.5835 | 0.5595 | 0.5976 | 0.5699 | 0.5369 |
| A3 | 1 | 0.5231 | 0.6047 | 0.5716 | 0.5882 | 0.5583 |
| A3 | 2 | 0.5725 | 0.5707 | 0.5608 | 0.5551 | 0.5303 |
| A4 | 0 | 0.5813 | 0.5591 | 0.6025 | 0.5894 | 0.5344 |
| A4 | 1 | 0.5258 | 0.5966 | 0.5638 | 0.5821 | 0.5593 |
| A4 | 2 | 0.5628 | 0.5541 | 0.5709 | 0.5591 | 0.5343 |

### Best-epoch distribution - secondary checkpoint rule

# Step 9 - best-epoch distribution (secondary checkpoint rule)

| arm | runs | median | mean | min | max | #(<5) | flag (median < 5) |
|---|---|---|---|---|---|---|---|
| A0 | 15 | 5.0 | 6.00 | 1 | 17 | 6 | no |
| A1 | 15 | 3.0 | 5.20 | 0 | 23 | 11 | YES |
| A2 | 15 | 3.0 | 4.60 | 1 | 16 | 8 | YES |
| A3 | 15 | 4.0 | 3.87 | 0 | 15 | 9 | YES |
| A4 | 15 | 5.0 | 4.53 | 0 | 13 | 7 | no |

==============================================================================
RUNTIME AND ANOMALIES
==============================================================================
# Step 9 - runtime and anomalies

| arm | runs | total s | mean s | errors | NaN-loss runs |
|---|---|---|---|---|---|
| A0 | 15 | 249.9 | 16.7 | 0 | 0 |
| A1 | 15 | 330.5 | 22.0 | 0 | 0 |
| A2 | 15 | 271.6 | 18.1 | 0 | 0 |
| A3 | 15 | 499.8 | 33.3 | 0 | 0 |
| A4 | 15 | 423.1 | 28.2 | 0 | 0 |
| TOTAL | 75 | 1774.9 | 23.7 | 0 | 0 |

Errors: 0

==============================================================================
EXPECTED VS ACTUAL
==============================================================================

| check | expected | actual | pass |
|---|---|---|---|
| 75 runs completed | 75 | 75 | yes |
| both checkpoint rules tracked per run | yes | yes (history has val_macro_f1 and val_pn_bacc) | yes |
| validation macro-F1 uses raw logits for A1/A4 | yes | yes | yes |
| runner skips completed runs on restart | yes | yes | yes |
| every table script-generated | yes | yes | yes |
| no tuning after seeing results | yes | yes | yes |
| NaN-sentinel bug fixed without weakening tests | yes | yes (history stores None; suite green) | yes |

Files written under results/step09/: runs/<arm>_s<seed>_f<fold>.parquet (75), curves/ (75), run_log.jsonl, oof_all.parquet, main_table_primary/secondary.{csv,md}, paired_primary/secondary.{csv,md}, breakdown_primary/secondary.{csv,md}, stability_per_seed_{rule}.{csv,md}, stability_per_fold_{rule}.{csv,md} and stability_per_fold_{rule}_{metric}.md, best_epochs_{rule}.{csv,md}, runtime.{csv,md}, analysis.json, REPORT.md. New code: scripts/run_step09.py, scripts/analyse_step09.py, scripts/step09_report.py, scripts/step09_appendix.py, tests/test_step09_checkpoints.py; modified src/adni/train.py (dual checkpoint tracking, optional val_weights).
