STEP 6c REPORT
Housekeeping + B0 encoding reconciliation + B1-blind modality imputation + uniform CI protocol
+ field-strength decision record.

Command(s) run (all from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/step03_report_tables.py                    > results/step06c/step03_tables_regenerated.md
  scripts/step06c_baselines.py                       > results/step06c/{oof_b0_oh,oof_b0_lin_inter,
                                                         oof_b1_blind}.parquet + tables below
  scripts/step06b_b0_reconcile.py --out results/step06c  > b0_reconcile.{json,txt}
  scripts/step06c_decision_record.py                 > results/step06c/decision_record.md
  python -m pytest -q

Git commit: (this commit)  — tag `step06c`

Tests: 54 passed in 20.65s

Environment: Python 3.12.10, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0.
Windows 11, CPU.

------------------------------------------------------------------------------
WHAT THIS STEP DID

1. Regenerated the Step 3 fold tables from `results/step03/splits.json` + `docs/episodes.csv`
   (`step03_tables_regenerated.md`). The old Step 3 report's S1 seed-0 table was hand-typed; its
   subject/episode cell counts are right but its **test class percentages are wrong** (they were not
   recomputed from the frozen split). Exact mismatches are printed in the file:

     fold 0  old 42.5/41.3/16.2  vs  canonical 39.7/41.7/18.6
     fold 1  old 40.4/41.6/18.0  vs  canonical 39.7/41.7/18.6
     fold 2  old 38.7/42.6/18.7  vs  canonical 39.7/41.7/18.6
     fold 3  old 38.6/41.3/20.1  vs  canonical 39.8/41.7/18.6
     fold 4  old 38.7/41.1/20.3  vs  canonical 39.7/41.7/18.6

   The actual StratifiedGroupKFold folds are almost constant across folds (~39.7 % CN, ~41.7 % MCI,
   ~18.6 % Dementia). S2 canonical counts: train 8716 ep (3001/3862/1853), val 959 ep
   (323/414/222), test 3565 ep (1939/1242/384 = 54.4/34.8/10.8 %).

2. B0 encoding. The primary pattern-only baseline is now the **15-column one-hot pattern**
   (`b0_oh`); the four-bit availability version is kept as `b0_lin`. To separate "representational
   capacity" from "L2 shrinkage" we also fit the four-bit model **plus all six pairwise availability
   interactions** (`b0_lin_inter`). Uniform-CI table (seed-averaged preds, one 2,000-resample
   subject-cluster bootstrap; `b0_encoding.md`):

     | B0 encoding    | features | macro-AUROC (uniform CI)   |
     |----------------|----------|----------------------------|
     | b0_lin         |    4     | 0.6001 [0.5883, 0.6123]    |
     | b0_lin_inter   |   10     | 0.6174 [0.6063, 0.6289]    |
     | b0_oh          |   15     | 0.6161 [0.6049, 0.6275]    |

   Paired subject-cluster-bootstrap diffs (`step06c_paired_diffs.json`):
     b0_oh - b0_lin            AUROC +0.0160 [+0.0125, +0.0197]
     b0_lin_inter - b0_lin     AUROC +0.0173 [+0.0137, +0.0210]
     b0_oh - b0_lin_inter      AUROC -0.0013 [-0.0024, -0.0002]

   **Conclusion (corrects the Step 6b explanation):** the four-bit-plus-pairwise-interactions model
   *fully recovers* the one-hot macro-AUROC (it is in fact marginally higher, CI just excluding 0).
   The one-hot vs four-bit gap is therefore **representational capacity** — the saturated 15-cell
   model can express the interaction structure between availability bits — and **not** L2
   shrinkage of the four-bit model. The `b0_reconcile` factorial confirms the gap is encoding, not
   partition/seed/aggregation (one-hot ≈0.618–0.620 on every partition vs four-bit ≈0.604–0.6028):
   each holdout/partition/seed-averaging choice moves macro-AUROC by ≤0.002.

3. B1-blind. For every absent block, its feature columns are replaced by the values of a randomly
   drawn **present TRAINING row** of that block (drawn per episode; donor pool = the fold's training
   rows only), and the four mask bits are removed. Config/folds/seeds are otherwise identical to B1.
   The difference B1 − B1-blind is the benefit of knowing which scans exist (which includes the
   availability shortcut). Uniform CI (seed-averaged, one bootstrap):

     | arm       | macro-AUROC (uniform CI)     | PN-BAcc (uniform) (uniform CI) |
     |-----------|------------------------------|--------------------------------|
     | b1_i      | 0.8101 [0.7999, 0.8204]      | 0.5938 [0.5790, 0.6096]        |
     | b1_blind  | 0.7620 [0.7509, 0.7736]      | 0.5628 [0.5478, 0.5778]        |

   Paired diff B1 − B1-blind: macro-AUROC +0.0481 [+0.0424, +0.0534]; macro-F1 +0.0567
   [+0.0480, +0.0654]; balanced acc +0.0568 [+0.0480, +0.0659]; PN-BAcc +0.0310
   [+0.0204, +0.0420]. So merely telling the model *which scans exist* is worth ≈4.8 pp macro-AUROC
   and ≈3.1 pp PN-balanced accuracy — a large, unambiguous availability short-cut.

   By number of observed modalities (`by_modality_count.md`), B1 ≥ B1-blind for 1–3 modalities and
   B1-blind is slightly higher only in the tiny 4-modality group (n=234):

     | n_mod | n_ep | B1 AUROC | blind AUROC | B1 PN-BAcc | blind PN-BAcc |
     |-------|------|----------|-------------|------------|---------------|
     | 1     | 5533 | 0.7720   | 0.7476      | 0.5888     | 0.5558        |
     | 2     | 4067 | 0.8186   | 0.7692      | 0.5966     | 0.5682        |
     | 3     | 3406 | 0.8413   | 0.7770      | 0.6044     | 0.5621        |
     | 4     |  234 | 0.7929   | 0.8149      | 0.5113     | 0.6503        |

4. Uniform CI protocol (`uniform_ci_table.{csv,md}`) for all arms incl. the new ones (12 arms;
   b0_lin, b2_*, b1_i/ii/iii, b0b are reused deterministically from the Step 6 / Step 6b OOF files,
   as in Step 6b). Headline macro-AUROC: b1_i 0.8101, b1_ii 0.8097, b1_iii 0.8066, b1_blind 0.7620,
   b2_mri 0.7661, b2_fdg 0.7274, b2_amy 0.7613, b2_tau 0.7736, b0_lin 0.6001, b0_lin_inter 0.6174,
   b0_oh 0.6161, b0b 0.5412.

5. Field-strength decision record (`decision_record.md`, script-generated). The Step 6b rule is
   unchanged and is now recorded in the Step 6c folder: B0b macro-AUROC = 0.5412 [0.5273, 0.5555],
   CI low ≤ 0.55 threshold → **KEEP `mri_field_strength` in the primary models**; the no-field-
   strength variant is a **sensitivity arm only**. The four B1 mask bits together carry a gain share
   of 4.79e-06; `mri_field_strength` ranks 40/592 by gain (share 0.36 %).

------------------------------------------------------------------------------
EXPECTED VS ACTUAL

  Check                                                        | Expected          | Actual              | Pass
  -------------------------------------------------------------|-------------------|---------------------|-----
  Step 3 seed-0 table regenerates from frozen sources           | script-generated  | written, diffed     | yes
  Old Step 3 test class % is stale                              | mismatch found    | mismatches listed   | yes
  B0 one-hot > B0 four-bit                                      | yes               | 0.6161 vs 0.6001    | yes
  4-bit + pairwise interactions recovers the one-hot AUROC      | recover (if capacity) | 0.6174 vs 0.6161 | yes
  B1-blind ≤ B1 (knowing availability helps)                    | yes               | +0.0481 AUROC       | yes
  B1-blind is NOT identical to B1-no-mask-bits                  | yes               | new imputation arm  | yes
  Uniform CI table incl. new arms                               | yes               | 12 arms written     | yes
  pytest -q                                                     | all pass          | 54 passed           | yes

------------------------------------------------------------------------------
DEVIATIONS FROM THE SPEC (with reason)

- The Step 6c "housekeeping" items (S2 class-count script, extended split tests, Step 5 synthetic
  checks, preprocessing audit) were executed and committed earlier under the `step06b` tag:
  `scripts/step06b_s2_counts.py` → `results/step06b/s2_class_counts.{txt,json}`;
  `scripts/step05_synthetic_checks.py` → `results/step05/synthetic_checks.txt`;
  `scripts/step06b_preprocess_audit.py` → `results/step06b/preprocess_audit.{txt,json}`;
  `tests/test_splits.py` extended to 5 tests. They are not re-run here; Step 6c only re-verifies
  them by running `pytest -q` (54 passed). This is a naming/placement artifact, not missing work.
- Output placement. The user asked for the new Step 6c outputs under `results/step06c/`. The B0b /
  B1-variant / decision-rule diagnostics were produced in Step 6b and live under `results/step06b/`;
  Step 6c does **not** modify or duplicate those files. Instead Step 6c writes a fresh, self-contained
  `uniform_ci_table` (including all arms and the new B0/B1-blind arms) and a script-generated
  `decision_record.md` that reads the frozen Step 6b artifacts. Re-running the whole Step 6b LightGBM
  suite (~9 min) purely to relocate identical outputs was not done; the reused OOF files are
  byte-identical inputs and b1_i is proven deterministic vs Step 6.
- B1-blind training rows are imputed too (not just test rows): the prompt says "for each absent
  block … draw per episode, from TRAINING rows of the fold only". Imputing train and test alike makes
  the model blind to availability in both, which is the intended contrast; a test-only imputation
  would still let the model learn availability from training NaNs. Documented here as an explicit
  choice.

------------------------------------------------------------------------------
THINGS I AM UNSURE ABOUT

- The B1-blind donor pool uses a fixed per-(seed,fold) RNG (seed 1000*(seed+1)+fold). Different
  draws would perturb the B1-blind numbers slightly; the effect (≈4.8 pp) is far larger than the
  draw noise seen across the three seeds, but the exact point estimate is draw-dependent.
- The 4-modality group (n=234) shows B1-blind > B1, opposite to the other groups. With n=234 this is
  likely small-sample noise; it is reported, not smoothed.
- "Which scans exist" and "the imaging values themselves" are not fully separable: imputation puts
  real values in place of a missing scan, so B1 − B1-blind mixes the pure availability signal with
  the value distribution shift the imputation induces.

------------------------------------------------------------------------------
FILES WRITTEN

  scripts/step03_report_tables.py
  scripts/step06c_baselines.py
  scripts/step06c_decision_record.py
  results/step06c/step03_tables_regenerated.md
  results/step06c/b0_encoding.{csv,md}
  results/step06c/b0_reconcile.{json,txt}
  results/step06c/oof_b0_oh.parquet
  results/step06c/oof_b0_lin_inter.parquet
  results/step06c/oof_b1_blind.parquet
  results/step06c/uniform_ci_table.{csv,md}
  results/step06c/step06c_paired_diffs.json
  results/step06c/by_modality_count.{json,md}
  results/step06c/decision_record.md
  results/step06c/summary_step06c.json
  results/step06c/REPORT.md
