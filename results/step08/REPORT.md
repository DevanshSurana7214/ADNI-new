STEP 08 REPORT
Joint priors and the loss arms A1-A4 (PIPELINE_v2 section 6, Appendix B).

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/run_step08_pilot.py   > results/step08/{comparison,per_pattern,by_modality,
                                  mask_sensitivity}.{csv,md}, curves_A*.csv, pilot_s0f0.json
  scripts/step08_report.py      > results/step08/REPORT.md
  python -m pytest -q

Git commit: (this commit) - tag `step08`

Tests: 88 passed in 178.18s (0:02:58)

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1. GPU: NVIDIA RTX 3050 (6 GB). Training used `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`; the pilot re-run reproduced every metric exactly.

------------------------------------------------------------------------------
WHAT WAS IMPLEMENTED

`src/adni/priors.py` (section 6 mask-process priors):

  * `submask_kernel(m, p_drop)` - exact K(m_tilde | m) over non-empty sub-masks; each observed
    modality kept independently w.p. 1-p_drop, empty draws repeated until non-empty.
  * `pi_table(P_cm, p_drop)` - pi(c | m_tilde) proportional to sum_m P(c,m) K(m_tilde | m).
  * `pattern_marginal(P_cm, p_drop)` - P(m_tilde) under the same kernel; `class_prior(P_cm)`.

`src/adni/losses.py` (arms as one config switch, same model/trainer/optimiser/early stop):

  * A0 cross-entropy; A1 logit-adjusted CE with log P(c); A2 CE reweighted by 1/P(m_tilde);
    A3 CE reweighted by rho(c)/pi(c|m_tilde) (rho uniform); A4 logit-adjusted CE with
    tau*log pi(c|m_tilde), tau=1. A2/A3 weights clipped to [0.2, 5] and mean-1 per batch.
  * Priors always use the EFFECTIVE mask after modality dropout. Inference: A0/A2/A3 use
    argmax s_c; A1/A4 report argmax(s_c + log rho(c)) with rho uniform (primary) and rho=P(c)
    (secondary). `train_arm` seeds before building the model (same fix as train_a0).

Priors for every arm: Dirichlet alpha 0.5 on the S1 seed-0 fold-0 TRAIN split only; p_drop = 0.25 (the frozen modality dropout); 15 sub-masks occur.
Frozen config (Step 7b) reused for all arms: lr = 1.0e-04, weight_decay = 0.3, batch 256, max_epochs 100, patience 15. No per-arm tuning.

------------------------------------------------------------------------------
PILOT - ARMS A0-A4 ON S1 SEED 0 FOLD 0 (with B1 and B1-blind on the same test episodes)

Script-generated from results/step08/comparison.csv. A1/A4 appear twice: primary (rho=uniform) and secondary (rho=P(c)).

| arm | operating point | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) | F1 CN | F1 MCI | F1 Dem | best epoch | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | rho=uniform (primary) | 2647 | 0.7844 | 0.6215 | 0.6327 | 0.6129 | 0.6128 | 0.6881 | 0.5083 | 0.6680 | 5 | 21 |
| A1 | rho=uniform (primary) | 2647 | 0.7855 | 0.6084 | 0.6479 | 0.6315 | 0.6314 | 0.6909 | 0.4589 | 0.6752 | 4 | 20 |
| A1 | rho=P(c) (secondary) | 2647 | 0.7894 | 0.6199 | 0.6320 | 0.6122 | 0.6121 | 0.6913 | 0.5075 | 0.6607 | 4 | 20 |
| A2 | rho=uniform (primary) | 2647 | 0.7846 | 0.6179 | 0.6286 | 0.6103 | 0.6102 | 0.6912 | 0.5026 | 0.6599 | 5 | 21 |
| A3 | rho=uniform (primary) | 2647 | 0.7578 | 0.5955 | 0.6081 | 0.6046 | 0.6045 | 0.6409 | 0.4948 | 0.6508 | 12 | 28 |
| A4 | rho=uniform (primary) | 2647 | 0.7553 | 0.5873 | 0.6036 | 0.5987 | 0.5986 | 0.6412 | 0.4735 | 0.6472 | 12 | 28 |
| A4 | rho=P(c) (secondary) | 2647 | 0.7558 | 0.5928 | 0.5900 | 0.5884 | 0.5883 | 0.6401 | 0.5079 | 0.6303 | 12 | 28 |
| B1 | - | 2647 | 0.7991 | 0.6311 | 0.6247 | 0.5851 | 0.5853 | 0.6716 | 0.5785 | 0.6430 | - | - |
| B1-blind | - | 2647 | 0.7413 | 0.5751 | 0.5687 | 0.5552 | 0.5551 | 0.6286 | 0.5154 | 0.5812 | - | - |

PER-PATTERN TEST METRICS (patterns with >= 30 test episodes; primary operating point)

| arm | pattern | n | macro-AUROC | macro-F1 | PN-BAcc(u) |
|---|---|---|---|---|---|
| A0 | 0010 | 69 | 0.7054 | 0.4569 | 0.5183 |
| A0 | 0011 | 43 | 0.6838 | 0.4670 | 0.5111 |
| A0 | 0100 | 42 | 0.7940 | 0.5352 | 0.5238 |
| A0 | 0110 | 34 | 0.7843 | 0.6828 | 0.6931 |
| A0 | 1000 | 1004 | 0.7632 | 0.5865 | 0.6085 |
| A0 | 1001 | 200 | 0.7753 | 0.6719 | 0.6717 |
| A0 | 1010 | 128 | 0.7784 | 0.6769 | 0.6715 |
| A0 | 1011 | 180 | 0.6798 | 0.5976 | 0.5871 |
| A0 | 1100 | 451 | 0.7678 | 0.6093 | 0.6109 |
| A0 | 1101 | 43 | 0.7167 | 0.4342 | 0.5640 |
| A0 | 1110 | 398 | 0.7795 | 0.6234 | 0.6434 |
| A0 | 1111 | 35 | 0.6979 | 0.2718 | 0.5160 |
| A1 | 0010 | 69 | 0.6827 | 0.4704 | 0.6162 |
| A1 | 0011 | 43 | 0.6891 | 0.5100 | 0.5889 |
| A1 | 0100 | 42 | 0.7824 | 0.5486 | 0.5536 |
| A1 | 0110 | 34 | 0.7770 | 0.6830 | 0.6931 |
| A1 | 1000 | 1004 | 0.7644 | 0.5719 | 0.6236 |
| A1 | 1001 | 200 | 0.7699 | 0.6708 | 0.7050 |
| A1 | 1010 | 128 | 0.7934 | 0.6480 | 0.6881 |
| A1 | 1011 | 180 | 0.6608 | 0.5451 | 0.5693 |
| A1 | 1100 | 451 | 0.7692 | 0.5925 | 0.6044 |
| A1 | 1101 | 43 | 0.8030 | 0.3248 | 0.4606 |
| A1 | 1110 | 398 | 0.7854 | 0.6120 | 0.6757 |
| A1 | 1111 | 35 | 0.7603 | 0.4159 | 0.6987 |
| A2 | 0010 | 69 | 0.7036 | 0.4569 | 0.5183 |
| A2 | 0011 | 43 | 0.6874 | 0.4670 | 0.5111 |
| A2 | 0100 | 42 | 0.7919 | 0.5333 | 0.5238 |
| A2 | 0110 | 34 | 0.7971 | 0.6828 | 0.6931 |
| A2 | 1000 | 1004 | 0.7657 | 0.5945 | 0.6183 |
| A2 | 1001 | 200 | 0.7773 | 0.6457 | 0.6572 |
| A2 | 1010 | 128 | 0.7720 | 0.6483 | 0.6427 |
| A2 | 1011 | 180 | 0.6820 | 0.5619 | 0.5455 |
| A2 | 1100 | 451 | 0.7727 | 0.6147 | 0.6149 |
| A2 | 1101 | 43 | 0.7167 | 0.4342 | 0.5640 |
| A2 | 1110 | 398 | 0.7785 | 0.6040 | 0.6270 |
| A2 | 1111 | 35 | 0.7051 | 0.2860 | 0.5288 |
| A3 | 0010 | 69 | 0.6712 | 0.4879 | 0.6131 |
| A3 | 0011 | 43 | 0.6561 | 0.4421 | 0.4519 |
| A3 | 0100 | 42 | 0.8019 | 0.5333 | 0.5169 |
| A3 | 0110 | 34 | 0.7468 | 0.7038 | 0.7116 |
| A3 | 1000 | 1004 | 0.7547 | 0.5854 | 0.6060 |
| A3 | 1001 | 200 | 0.7480 | 0.6023 | 0.6376 |
| A3 | 1010 | 128 | 0.7804 | 0.6914 | 0.7026 |
| A3 | 1011 | 180 | 0.6414 | 0.4635 | 0.4711 |
| A3 | 1100 | 451 | 0.7667 | 0.5917 | 0.6012 |
| A3 | 1101 | 43 | 0.6700 | 0.2880 | 0.3892 |
| A3 | 1110 | 398 | 0.7648 | 0.6161 | 0.6394 |
| A3 | 1111 | 35 | 0.7163 | 0.3459 | 0.5865 |
| A4 | 0010 | 69 | 0.6964 | 0.4739 | 0.6016 |
| A4 | 0011 | 43 | 0.6693 | 0.4332 | 0.4519 |
| A4 | 0100 | 42 | 0.7992 | 0.5679 | 0.5585 |
| A4 | 0110 | 34 | 0.7583 | 0.6439 | 0.6561 |
| A4 | 1000 | 1004 | 0.7540 | 0.5717 | 0.5970 |
| A4 | 1001 | 200 | 0.7563 | 0.6181 | 0.6484 |
| A4 | 1010 | 128 | 0.7755 | 0.6828 | 0.6875 |
| A4 | 1011 | 180 | 0.6376 | 0.4713 | 0.4783 |
| A4 | 1100 | 451 | 0.7583 | 0.5817 | 0.5974 |
| A4 | 1101 | 43 | 0.6749 | 0.3292 | 0.4236 |
| A4 | 1110 | 398 | 0.7608 | 0.5931 | 0.6234 |
| A4 | 1111 | 35 | 0.7369 | 0.3679 | 0.5994 |

METRICS BY NUMBER OF OBSERVED MODALITIES (primary operating point)

| arm | n_mod | n | macro-AUROC | macro-F1 | PN-BAcc(u) |
|---|---|---|---|---|---|
| A0 | 1 | 1130 | 0.7613 | 0.5835 | 0.6007 |
| A0 | 2 | 858 | 0.8010 | 0.6585 | 0.6205 |
| A0 | 3 | 624 | 0.8057 | 0.6522 | 0.6336 |
| A0 | 4 | 35 | 0.6979 | 0.2718 | 0.5160 |
| A1 | 1 | 1130 | 0.7618 | 0.5708 | 0.6196 |
| A1 | 2 | 858 | 0.8021 | 0.6485 | 0.6386 |
| A1 | 3 | 624 | 0.8061 | 0.6259 | 0.6474 |
| A1 | 4 | 35 | 0.7603 | 0.4159 | 0.6987 |
| A2 | 1 | 1130 | 0.7635 | 0.5902 | 0.6093 |
| A2 | 2 | 858 | 0.8005 | 0.6507 | 0.6146 |
| A2 | 3 | 624 | 0.8043 | 0.6339 | 0.6138 |
| A2 | 4 | 35 | 0.7051 | 0.2860 | 0.5288 |
| A3 | 1 | 1130 | 0.7508 | 0.5817 | 0.6027 |
| A3 | 2 | 858 | 0.7709 | 0.6197 | 0.6191 |
| A3 | 3 | 624 | 0.7519 | 0.5957 | 0.5887 |
| A3 | 4 | 35 | 0.7163 | 0.3459 | 0.5865 |
| A4 | 1 | 1130 | 0.7508 | 0.5713 | 0.5968 |
| A4 | 2 | 858 | 0.7681 | 0.6133 | 0.6143 |
| A4 | 3 | 624 | 0.7462 | 0.5850 | 0.5817 |
| A4 | 4 | 35 | 0.7369 | 0.3679 | 0.5994 |

MASK SENSITIVITY - MEAN |CHANGE IN PREDICTED CLASS PROBABILITY| WHEN ONE OBSERVED MODALITY
IS REMOVED AT TEST TIME (episodes with >= 2 observed modalities; primary probabilities)

| arm | mean abs delta prob |
|---|---|
| A0 | 0.0859 |
| A1 | 0.0851 |
| A2 | 0.0885 |
| A3 | 0.0975 |
| A4 | 0.0999 |

------------------------------------------------------------------------------
RED FLAGS (per arm; the pilot is a single fold)

| arm | best epoch | val macro-AUROC @ best | test macro-AUROC | val-test gap | NaN loss | best_epoch_below_5 | gap_above_0.05 |
|---|---|---|---|---|---|---|---|
| A0 | 5 | 0.8227 | 0.7844 | 0.0384 | no | no | no |
| A1 | 4 | 0.8198 | 0.7855 | 0.0343 | no | YES | no |
| A2 | 5 | 0.8201 | 0.7846 | 0.0355 | no | no | no |
| A3 | 12 | 0.8044 | 0.7578 | 0.0467 | no | no | no |
| A4 | 12 | 0.8032 | 0.7553 | 0.0479 | no | no | no |

No NaN/Inf loss in any arm. No validation-test gap exceeds 0.05 (largest 0.0479 for A4). One red flag: A1's best epoch is 4 (< 5), i.e. it early-stops very early on this fold.

READING

* A0-A2 reproduce each other closely (test macro-AUROC 0.7844-0.7855, macro-F1 0.6084-0.6215, PN-BAcc(u) 0.6103-0.6315). Class-only logit adjustment (A1) buys the largest PN-BAcc(u) here (0.6315) at a small macro-F1 cost.
* A3 and A4 (the pattern-aware arms) lose about 0.03 macro-AUROC and 0.03 macro-F1 against A0 and do not improve PN-BAcc(u) on this single fold (0.6046 / 0.5987). A4's secondary rho=P(c) operating point recovers MCI F1 (0.5079 vs 0.4735) but not the macro metrics.
* B1 remains the strongest natural-distribution arm (macro-AUROC 0.7991, macro-F1 0.6311); all neural arms sit between B1 and B1-blind (0.7413 / 0.5751).
* Mask sensitivity (mean |delta p| when one observed modality is removed) is 0.085-0.100 across arms, increasing from A0/A1 (0.086) to A3/A4 (0.098-0.100): the pattern-aware arms react slightly more to mask changes, as expected.

------------------------------------------------------------------------------
EXPECTED VS ACTUAL

| check | expected | actual | pass |
|---|---|---|---|
| submask_kernel sums to 1 (all 15 patterns); point mass at p_drop=0 | yes | yes | yes |
| pi_table at p_drop=0 equals smoothed P(c|m) (Step 2) | < 1e-12 | < 1e-12 | yes |
| kernel equals actual modality_dropout (200k draws/pattern) | max diff < 0.005 | < 0.005 | yes |
| Monte-Carlo pi_table vs actual dropout on the training set | max diff < 0.01 (>=500 draws) | < 0.01 (see deviation) | yes* |
| A4(tau=0) reproduces A0 loss on a fixed batch | exactly | exactly | yes |
| logits = -log pi(c|m_tilde)+const -> A4 adjusted logits constant, loss = log 3 | yes | yes | yes |
| no val/test label influences the priors | identical table | identical table | yes |
| synthetic shortcut: A0 follows the pattern prior; A4 (rho uniform) TV<0.05 | yes | yes | yes |
| determinism through train_arm (CPU, same seed) | bit-identical | bit-identical | yes |
| arms A0-A4 trained on S1 seed 0 fold 0 | 5 | 5 | yes |
| B1 and B1-blind in the same table | yes | yes | yes |
| no per-arm tuning | yes | yes (frozen config) | yes |
| full test suite | all pass | 88 passed | yes |

* see the deviation on test 4 below.

------------------------------------------------------------------------------
DEVIATIONS FROM THE SPEC (with reason)

1. Unit test 4 (Monte-Carlo pi_table). The literal spec is not achievable on this data, for two independent reasons, so the test was made statistically valid rather than loosened:
   (a) It compares a Monte-Carlo estimate of the *empirical* conditional with the *alpha=0.5-smoothed* `pi_table`. For the two rarest training patterns (0101: 5 training cells; 0111: 16) the smoothing alone moves pi(c|m_tilde) by 0.0116 and 0.0105 - above the 0.01 tolerance - so the test now validates the mask process against `pi_table` from the unsmoothed joint (alpha=0). Smoothing consistency is covered separately by test 2.
   (b) At 200,000 draws the rarest sub-masks receive only ~557-892 samples, where the Monte-Carlo standard error reaches ~0.021; every seed fails the 0.01 tolerance (verified: max diff 0.017-0.040). The test therefore draws 20,000,000 training-set dropout views and keeps the 0.01 tolerance, which is now a genuine check of the kernel (max diff 0.0023 at seed 0). This is a stricter, not a weaker, test of the stated property.
2. `src/adni/train.py` gained the arm switch the spec calls for: `TrainConfig.arm`/`tau`, `Trainer(..., priors)`, and a `train_arm` entry point. `train_a0` now delegates to `train_arm('A0', ...)` and is behaviourally unchanged.
3. The per-epoch `val_loss` recorded in the curves is plain cross-entropy for every arm (only the training loss changes with the arm). Early stopping is on validation macro-F1 for all arms, identical to Step 7, as the spec requires; the two operating points of A1/A4 differ only in inference, not in training.
4. The mask-sensitivity diagnostic removes one observed modality and reports the mean absolute probability change; it is the first look the playbook asks for, not the full content-swap counterfactual of PIPELINE_v2 section 8.4 (that is a later step).

THINGS I AM UNSURE ABOUT

* A1's best epoch is 4 on this fold, which trips the `best_epoch_below_5` red flag. On one fold this may be early-stopping noise; the full S1 grid in Step 9 will show whether it is systematic.
* A3/A4 trade ~0.03 macro-AUROC for no PN-BAcc(u) gain on this fold. Whether that reverses with more seeds/folds is exactly what Step 9 is for; no tuning was done here.
* PN-BAcc in this pilot uses the natural-pattern P(c|m) estimator for the metric, matching the Step 7 pilot. The mask-process pi(c|m_tilde) is used inside the A3/A4 losses and for the A4 operating points, as specified.

FILES WRITTEN

  src/adni/priors.py
  src/adni/losses.py
  src/adni/train.py
  tests/test_priors_losses.py
  scripts/run_step08_pilot.py
  scripts/step08_report.py
  scripts/step08_appendix.py
  docs/PIPELINE_v2.md
  results/step08/comparison.csv
  results/step08/comparison.md
  results/step08/per_pattern.csv
  results/step08/per_pattern.md
  results/step08/by_modality.csv
  results/step08/by_modality.md
  results/step08/mask_sensitivity.csv
  results/step08/mask_sensitivity.md
  results/step08/curves_A0.csv
  results/step08/curves_A1.csv
  results/step08/curves_A2.csv
  results/step08/curves_A3.csv
  results/step08/curves_A4.csv
  results/step08/pilot_s0f0.json
  results/step08/REPORT.md

