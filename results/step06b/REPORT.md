STEP 6b REPORT
Combined housekeeping + field-strength diagnostics + mask-reliance + B0 reconciliation + uniform CI protocol.
Nothing in the primary pipeline or the Stage 3 profile was modified; all new outputs are under results/step06b/.

Command(s) run
  git tag step02 cb71908 ; git tag step03 50542d3 ; git tag step04 ac781c3 ; git tag step05 637c9c0 ; git tag step06 0db9d89
  PYTHONPATH=src ./.venv/Scripts/python.exe scripts/step06b_s2_counts.py
  PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest tests/test_splits.py -q
  PYTHONPATH=src ./.venv/Scripts/python.exe scripts/step05_synthetic_checks.py
  PYTHONPATH=src ./.venv/Scripts/python.exe scripts/step06b_preprocess_audit.py
  PYTHONPATH=src ./.venv/Scripts/python.exe scripts/step06b_b0_reconcile.py
  PYTHONPATH=src ./.venv/Scripts/python.exe -u scripts/step06b_baselines.py
  PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q

Git commit: step06b (this report; tagged `step06b`)

Tests: 54 passed in 31.12s   (tests/test_splits.py 5, +1 new S2-count test vs the prior 53)

Environment: Python 3.12.10, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1,
lightgbm 4.7.0, pyyaml 6.0.3. Windows 11.

================================================================================
A. HOUSEKEEPING
================================================================================

A.1 Commit + tag steps 02-06
  The premise "steps 02-06 were never committed" does not hold in this repository: each step is
  already an isolated commit and the working tree was clean at the start of this step. I did NOT
  rewrite history and made NO catch-up commit. The existing commits are:

    step02  cb71908  pyproject.toml, results/step02/*, src/adni/__init__.py, src/adni/priors.py, tests/test_priors.py, tests/test_stage3_repro.py
    step03  50542d3  results/step03/*, scripts/make_splits.py, src/adni/splits.py, tests/test_splits.py
    step04  ac781c3  configs/preprocess.yaml, results/step04/*, scripts/run_preprocess.py, src/adni/preprocess.py, tests/test_preprocess.py
    step05  637c9c0  results/step05/REPORT.md, src/adni/metrics.py, tests/test_metrics.py
    step06  0db9d89  configs/b1.yaml, results/step06/*, scripts/run_baselines.py, src/adni/baselines.py, src/adni/metrics.py, tests/test_baselines.py, tests/test_metrics.py

  Each step's files were separable, so no "steps 02-06 (catch-up)" commit was needed. Tags created:
  step02, step03, step04, step05, step06 (verified with `git tag`). Step 6 was already committed
  separately (`0db9d89`) and is tagged. From now on each step is committed at the end.

A.2 S2 class counts (train / val / test) and the Step 3 discrepancy
  Independently recomputed from docs/episodes.csv + results/step03/splits.json (no use of
  src/adni/splits.py). Full output in results/step06b/s2_class_counts.txt.

    split   episodes     CN    MCI  Dementia      CN%   MCI%   Dem%     Step 3 report %
    train       8716   3001   3862      1853    34.43  44.31  21.26       34.2/44.4/21.3
    val          959    323    414       222    33.68  43.17  23.15       38.2/39.4/22.4
    test        3565   1939   1242       384    54.39  34.84  10.77       54.1/35.0/10.9

  The test side is CN 1,939 / MCI 1,242 / Dementia 384 (3,565 episodes) = 54.4 / 34.8 / 10.8 %, i.e.
  it matches the independent recomputation exactly and does NOT match the 54.1/35.0/10.9 % printed in
  results/step03/REPORT.md. Explanation: the frozen splits.json is authoritative (it reproduces
  1,939/1,242/384 exactly), and the Step 3 report's percentage table is stale/inconsistent with it —
  the S2 *val* row is off by more than 4 pp (38.2/39.4/22.4 vs the actual 33.7/43.2/23.1), which is far
  beyond rounding, so this is a documentation error in that report, not a change in the data. The
  episode counts in the report (8,716 / 959 / 3,565) do match. Nothing was changed in either pipeline.
  Added tests/test_splits.py::test_s2_class_counts_from_episodes, which recomputes these counts from
  docs/episodes.csv and asserts the exact values.

A.3 Split-structure tests
  tests/test_splits.py now asserts, for all 15 S1 (seed, fold) splits and for S2:
    * train/val/test pairwise disjoint on episode_id (all 3 pairs);
    * train/val/test pairwise disjoint on RID (all 3 pairs; previously only train+val vs test);
    * S1 union(train,val,test) = all episodes; S2 union = all episodes (added);
    * every RID appears in exactly one test fold per seed, and all RIDs are covered;
    * S2 subject counts (1,784 / 1,361) and the class counts above.
  The checks that had only been run ad hoc in Step 3 (RID-level disjointness, union, one-test-fold) are
  now inside the test file. `pytest tests/test_splits.py -q` -> 5 passed.

A.4 Step 5 synthetic checks
  scripts/step05_synthetic_checks.py prints every Step 5 synthetic check with its actual value and
  writes results/step05/synthetic_checks.txt. Highlights:
    * perfect predictions: macro-AUROC 1.000000, macro-F1 1.000000, BAcc 1.000000, Brier 0.000000;
    * random balanced (n=20000): macro-AUROC 0.496257 (< 0.03 from chance);
    * hand-computed 6-sample example (y = [0,1,2,0,1,2], proba rows
      [0.6,0.3,0.1],[0.2,0.7,0.1],[0.1,0.2,0.7],[0.2,0.3,0.5],[0.1,0.8,0.1],[0.5,0.3,0.2]):
      macro-AUROC 0.8958333, macro-F1 0.6666667, BAcc 0.6666667, per-class F1 CN 0.5 / MCI 1.0 /
      Dementia 0.5, Brier 0.4266667, ECE 0.3666667;
    * perfectly calibrated synthetic (n=40000): ECE = 0.005561 (< 0.02);
    * cluster-bootstrap CI width 0.097500 vs analytic 0.096020 (1.54 % relative);
    * PN weights balance each pattern's present classes to 1e-9; holm([0.01,0.04,0.03,0.5]) = [0.04,0.09,0.09,0.5];
      cluster_bootstrap_multi reproduces cluster_bootstrap point and CI exactly.

A.5 Preprocessing over all splits
  scripts/step06b_preprocess_audit.py fits the Preprocessor on each of the 15 S1 training folds and on
  S2 (16 fits) and applies it to train/val/test. Results (results/step06b/preprocess_audit.txt):

    block   kept-feature count            min   max
    mri     254 (256 - ST68SV, ST8SV)     254   254
    fdg       6                             6     6
    amy     162                           162   162
    tau     160 (drops VENTRICLE_5TH_SUVR, NON_WM_HYPOINTENSITIES_SUVR)  160  160

  Kept counts are identical across all 16 fits ("at min: 16 fits, at max: 16 fits" for every block),
  and there is no NaN/inf in any transform output (train/val/test x 16 fits x 4 blocks).

================================================================================
B. FIELD-STRENGTH DIAGNOSTICS (decision rule fixed before any result was seen)
================================================================================

B.6 B0b — a classifier using ONLY mri_field_strength
  Multinomial logistic regression on the single binary mri_field_strength feature, restricted to the
  12,310 episodes with MRI present, same S1 folds/config as Step 6. Under the uniform protocol
  (seed-averaged predictions, one 2,000-resample subject-cluster bootstrap):

    macro-AUROC = 0.5412  [0.5273, 0.5555]
    balanced accuracy = 0.3320  [0.3203, 0.3428]
    (per-seed-then-average stability table: macro-AUROC 0.5463 ± 0.0017, BAcc 0.3384 ± 0.0024)

  Cross-tabulation mri_field_strength x COLPROT x class (MRI-present episodes, counts;
  results/step06b/b0b_field_strength_colprot_class.csv):

    fs=0 (n=3,868): ADNI1 CN 976 / MCI 1407 / Dem 935; ADNI2 158/125/107; ADNIGO 67/49/44; ADNI3 & ADNI4 = 0
    fs=1 (n=8,442): ADNI1 83/96/80; ADNI2 1249/1747/688; ADNI3 1460/1016/328; ADNI4 807/498/129; ADNIGO 9/250/2

  mri_field_strength is almost perfectly aligned with the enrolment era: 0 is essentially ADNI1/GO
  (plus a little ADNI2) and never ADNI3/4, while 1 covers ADNI2/3/4. It is therefore a protocol/era
  proxy, which matters for interpreting the B1 variants below.

B.7 Decision rule (fixed at 0.55, not changed)
  Rule: if the B0b macro-AUROC CI lower bound is above 0.55, EXCLUDE mri_field_strength from the
  primary neural models (sensitivity arm only); otherwise KEEP it.
  B0b macro-AUROC CI lower bound = 0.5273, which is ≤ 0.55.
  => BRANCH: **KEEP mri_field_strength in the primary models** (0.5273 ≤ 0.55). See
     results/step06b/decision_rule.json.

================================================================================
C. MASK-RELIANCE AND SHORTCUT CHECKS FOR B1
================================================================================

C.8 B1 variants (same configs/b1.yaml, same folds), uniform CI protocol

    arm      features                                   macro-AUROC            macro-F1   BAcc    PN-BAcc
    b1_i     all blocks + 4 mask bits (Step 6)          0.8101 [0.7999,0.8204] 0.6392     0.6323  0.5938
    b1_ii    all blocks, NO mask bits                   0.8097 [0.7994,0.8204] 0.6384     0.6314  0.5928
    b1_iii   all blocks + masks, NO mri_field_strength  0.8066 [0.7963,0.8171] 0.6307     0.6235  0.5825

  Sanity check: refitting (i) reproduces the Step 6 oof_b1 predictions exactly
  (39,720 rows, max abs probability deviation = 0.000e+00).

  Paired subject-cluster-bootstrap differences (2,000 resamples, seed-averaged predictions):
    (i) - (ii)  [mask bits removed]:
        macro-AUROC +0.0004 [-0.0013, +0.0021]   macro-F1 +0.0009 [-0.0043, +0.0062]
        BAcc        +0.0009 [-0.0043, +0.0064]   PN-BAcc  +0.0010 [-0.0047, +0.0071]
    (i) - (iii) [mri_field_strength removed]:
        macro-AUROC +0.0035 [+0.0015, +0.0054]   macro-F1 +0.0086 [+0.0034, +0.0139]
        BAcc        +0.0088 [+0.0035, +0.0141]   PN-BAcc  +0.0113 [+0.0054, +0.0172]

  Reading: the four mask bits change nothing measurable (all CIs straddle 0), while removing
  mri_field_strength *lowers* B1 by a small but consistent amount (all four CIs exclude 0). Given the
  B.6 crosstab, that gain is most plausibly the enrolment-era/protocol information field strength
  carries, not a diagnosis-specific MRI signal.

C.9 B1(i) importance
    total gain (sum over 15 fits)            1,817,398
    gain share of the four mask bits          4.79e-6  (mask_mri/mask_amy/mask_tau = 0.0, mask_fdg = 8.70)
    rank of mri_field_strength                40 of 592 features, gain share 0.00362 (0.36 %)
    top 5 by gain: mri_ST24TA 0.0572, mri_ST29SV 0.0566, fdg_MetaROI_MEAN 0.0465, mri_ST12SV 0.0246,
                   fdg_MetaROI_MAX 0.0225
  The four mask bits contribute ~5 ppm of total gain (effectively zero): LightGBM can already infer
  availability from the native NaN pattern of the RAW features, so the explicit mask bits are redundant.
  Full ranking in results/step06b/b1_importance.csv.

================================================================================
D. RECONCILING THE B0 NUMBER
================================================================================

D.10 Stage 3's shortcut benchmark (pattern only) under four conditions. "Own partition" = Stage 3's
  StratifiedGroupKFold(5, seed 0) stratified on the per-episode y; "frozen" = the Step 3 S1 seed-0
  splits. The "val holdout" removes 10 % of the training *subjects* (subject-grouped, stratified on the
  subject modal label, seed 42). Metric: macro-OvR AUROC on pooled out-of-fold predictions.

    run                                                   macro-AUROC   BAcc
    (a1) Stage 3 own partition, full train (unmodified)       0.6182     0.4121
    (a2) Stage 3 own partition, -10 % val holdout             0.6200     0.4124
    (b1) frozen Step 3 splits, train only (Step 6)            0.6200     0.4124
    (b2) frozen Step 3 splits, train+val (no holdout)         0.6182     0.4121
    Step 6 B0 (3-seed mean, frozen splits)                    0.6024

  Encoding isolation (Step 6's exact 4-bit B0 definition):
    (c1) frozen splits, 4-bit, train only, seed 0             0.6040     0.4026
    (c2) frozen splits, 4-bit, train+val, seed 0              0.6028     0.4026
    (c3) frozen splits, 4-bit, 3-seed mean                    0.6024     0.4026
    (c4) own partition, 4-bit, seed 0                         0.6028     0.4026

  Which factor explains the gap? The feature ENCODING.
    * Validation holdout: negligible. a1->a2 = 0.6182->0.6200; b1->b2 = 0.6200->0.6182;
      c1->c2 = 0.6040->0.6028. Every delta is <= 0.002 and the sign is not consistent.
    * Partition (own SGKF vs frozen): negligible. With the Stage 3 encoding, own-partition-with-holdout
      (0.6200) equals frozen-train-only (0.6200); with the 4-bit encoding, own partition (0.6028) equals
      frozen train-only seed 0 (0.6040) to ~0.001.
    * Seed averaging: explains only 0.0016 (0.6040 seed 0 vs 0.6024 three-seed mean).
    * Encoding: Stage 3 fits a 15-column one-hot of the pattern string (`pd.get_dummies(pattern)`) with
      L2 (C=1.0); Step 6's B0 fits the 4 binary availability bits. On the SAME frozen folds and the same
      train-only subset the two encodings give 0.6200 (one-hot) vs 0.6040 (4-bit) -- i.e. ~0.016, which
      is the entire 0.602-vs-0.618 gap. The one-hot is an over-parameterised representation of the same
      pattern partition, so its L2 shrinkage (and hence its probabilities/AUROC) differs from the 4-bit
      model's.

  Conclusion: the difference is an encoding/regularisation artefact of the two implementations, not the
  validation holdout and not the fold partition. Nothing was changed in either pipeline.

================================================================================
E. UNIFORM CI PROTOCOL (used for every arm from now on)
================================================================================

E.11 Primary table — average the predicted probabilities over the 3 seeds per episode, pool the
  out-of-fold predictions, and run ONE subject-cluster bootstrap (2,000 resamples).
  Files: results/step06b/uniform_ci_table.csv and uniform_ci_table.md.

    arm      n_ep   macro-AUROC                 macro-F1                BAcc                    PN-BAcc (uniform)       PN-BAcc (P(c))
    b0      13240  0.6001 [0.5883,0.6123]      0.3544 [0.3452,0.3643]   0.4026 [0.3939,0.4115]   0.3340 [0.3252,0.3433]   0.3334
    b1_i    13240  0.8101 [0.7999,0.8204]      0.6392 [0.6259,0.6535]   0.6323 [0.6182,0.6471]   0.5938 [0.5790,0.6096]   0.5945
    b1_ii   13240  0.8097 [0.7994,0.8204]      0.6384 [0.6249,0.6523]   0.6314 [0.6174,0.6454]   0.5928 [0.5785,0.6081]   0.5935
    b1_iii  13240  0.8066 [0.7963,0.8171]      0.6307 [0.6170,0.6444]   0.6235 [0.6089,0.6378]   0.5825 [0.5674,0.5985]   0.5832
    b2_mri  12310  0.7661 [0.7548,0.7772]      0.5854 [0.5712,0.6004]   0.5808 [0.5663,0.5966]   0.5702 [0.5549,0.5862]   0.5697
    b2_fdg   5063  0.7274 [0.7108,0.7437]      0.5166 [0.4947,0.5379]   0.5059 [0.4861,0.5259]   0.5118 [0.4912,0.5327]   0.5111
    b2_amy   4621  0.7613 [0.7450,0.7785]      0.5661 [0.5428,0.5890]   0.5571 [0.5347,0.5800]   0.5459 [0.5221,0.5698]   0.5458
    b2_tau   2827  0.7736 [0.7520,0.7955]      0.5548 [0.5233,0.5860]   0.5398 [0.5106,0.5689]   0.5548 [0.5245,0.5855]   0.5550
    b0b     12310  0.5412 [0.5273,0.5555]      0.2775 [0.2649,0.2894]   0.3320 [0.3203,0.3428]   0.3208 [0.3089,0.3322]   0.3208

    Per-class F1 (natural): b0 0.477/0.586/0.000; b1_i 0.695/0.580/0.643; b1_ii 0.694/0.577/0.643;
    b1_iii 0.684/0.569/0.640; b2_mri 0.648/0.505/0.603; b2_fdg 0.333/0.636/0.581;
    b2_amy 0.712/0.481/0.505; b2_tau 0.803/0.376/0.485; b0b 0.295/0.538/0.000 (CN/MCI/Dementia).

  Supplementary stability table (per-seed metric, then average over seeds; the b0/b1/b2 rows are the
  Step 6 per-seed-then-average values, b1_i is identical to the Step 6 B1 refit). Files:
  results/step06b/stability_table.csv and stability_table.md.

    arm      macro-AUROC (mean ± sd)     macro-F1            BAcc                PN-BAcc
    b0       0.6024 ± 0.0013             0.3544 ± 0.0000     0.4026 ± 0.0000     0.3340 ± 0.0004
    b1_i     0.8041 ± 0.0020             0.6309 ± 0.0024     0.6241 ± 0.0025     0.5851 ± 0.0023
    b1_ii    0.8036 ± 0.0022             0.6295 ± 0.0041     0.6223 ± 0.0046     0.5833 ± 0.0054
    b1_iii   0.8007 ± 0.0011             0.6265 ± 0.0031     0.6190 ± 0.0029     0.5796 ± 0.0038
    b2_mri   0.7599 ± 0.0025             0.5791 ± 0.0029     0.5739 ± 0.0025     0.5632 ± 0.0021
    b2_fdg   0.7230 ± 0.0023             0.5206 ± 0.0046     0.5090 ± 0.0052     0.5123 ± 0.0051
    b2_amy   0.7526 ± 0.0020             0.5642 ± 0.0031     0.5541 ± 0.0032     0.5440 ± 0.0045
    b2_tau   0.7643 ± 0.0010             0.5457 ± 0.0048     0.5317 ± 0.0042     0.5425 ± 0.0050
    b0b      0.5463 ± 0.0017             0.2962 ± 0.0068     0.3384 ± 0.0024     0.3218 ± 0.0016

  The uniform-protocol point estimates differ slightly from the stability table (e.g. b0 0.6001 vs
  0.6024) because averaging probabilities before ranking is not the same as averaging per-seed AUROCs;
  the uniform table is the primary protocol from here on.

Expected vs actual
  Check                                                             | Expected              | Actual               | Pass
  ------------------------------------------------------------------|-----------------------|----------------------|-----
  Steps 02-06 committed and separable                               | 5 commits             | 5 commits, tagged    | yes
  A.2 S2 test class counts from docs/episodes.csv                    | 1939/1242/384         | 1939/1242/384        | yes
  A.2 Step 3 report test % matches the frozen split                  | should match          | 54.1/35.0/10.9 vs 54.4/34.8/10.8 | no (report is stale)
  A.3 split disjointness/union, all 15 S1 + S2                       | all hold              | all hold             | yes
  A.4 Step 5 checks printed with actual values + ECE                 | ECE < 0.02            | ECE 0.005561         | yes
  A.5 kept features per block, 16 fits                               | stable, finite        | 254/6/162/160 stable | yes
  A.5 no NaN/inf in any transform                                    | none                  | none                 | yes
  B.6 B0b macro-AUROC (field strength only)                          | reported              | 0.5412 [0.5273,0.5555] | yes
  B.7 decision rule applied (threshold 0.55)                         | branch stated         | KEEP (0.5273 ≤ 0.55) | yes
  C.8 B1(i) refit == Step 6 B1                                       | identical             | max prob dev 0.000e+00 | yes
  C.8 paired diffs (i)-(ii), (i)-(iii)                               | reported with CIs     | reported             | yes
  C.9 mask-bit total-gain share                                      | reported              | 4.79e-6              | yes
  C.9 mri_field_strength rank                                        | reported              | 40 / 592             | yes
  D.10 four-condition reconcile                                      | factor identified     | encoding             | yes
  E.11 uniform table + stability table, CSV + markdown               | 4 files               | 4 files              | yes
  pytest -q                                                          | all green             | 54 passed            | yes

Deviations from the spec (if any, with reason):
  - A.1: the requested catch-up commits were unnecessary because steps 02-06 were already committed in
    their own commits; I tagged them instead of rewriting history (which was explicitly forbidden). No
    "steps 02-06 (catch-up)" commit was made.
  - B0b is fitted as a multinomial logistic regression on the single raw mri_field_strength column (the
    feature is binary {0,1} with no NaNs on MRI-present episodes), matching B0's classifier family and
    Step 6's folds/metrics — the step did not specify the classifier, only "a classifier using ONLY
    mri_field_strength".
  - E.11: for arms whose OOF already exists in results/step06 (B0 and the four B2 arms) the stored OOF
    predictions are reused rather than refitted; they are deterministic under the same config/folds and
    the B1(i) refit reproduces Step 6 exactly (max dev 0.000e+00), so this is equivalent. The reused
    files are only read, never modified.
  - The uniform-protocol PN-BAcc averages the per-seed PN weights per episode before the single
    bootstrap; the per-seed weights differ only through the fold's train-split P(c|m).

Things I am unsure about:
  - The B0 reconciliation concludes the gap is the encoding (15-column one-hot vs the 4 availability
    bits). Both represent the same pattern partition, so the 0.016 AUROC difference comes from how the
    L2 penalty shrinks an over-parameterised one-hot versus the 4-bit model at the same C=1.0. The clean
    fix is to report the two numbers as different encodings of the same shortcut baseline rather than as
    a discrepancy, which is what this step now does.
  - mri_field_strength is a near-perfect era/protocol proxy (B.6). The fixed rule keeps it, but the
    (i)-(iii) drop means it carries ~0.0035 AUROC of protocol information; whether that is acceptable
    "leakage" for the neural arms is a modelling decision left to later steps.
  - The decision-rule CI is a 95 % subject-cluster bootstrap over poolfolded OOF episodes; it does not
    include the between-seed variability of the seed-averaging step (the per-seed stability sd is 0.0017).

Files written (all new; Step 6 result files untouched):
  scripts/step06b_s2_counts.py, results/step06b/s2_class_counts.{txt,json}
  scripts/step05_synthetic_checks.py, results/step05/synthetic_checks.txt
  scripts/step06b_preprocess_audit.py, results/step06b/preprocess_audit.{txt,json}
  scripts/step06b_b0_reconcile.py, results/step06b/b0_reconcile.{json,txt}
  scripts/step06b_baselines.py
  results/step06b/oof_{b1_i,b1_ii,b1_iii,b0b}.parquet
  results/step06b/uniform_ci_table.{csv,md}
  results/step06b/stability_table.{csv,md}
  results/step06b/b1_variant_diffs.json, results/step06b/b1_importance.{csv,json}
  results/step06b/b0b_field_strength_colprot_class.csv
  results/step06b/decision_rule.json, results/step06b/summary_baselines.json
  tests/test_splits.py (extended: S2 class counts + RID/eid disjointness/union assertions)
  results/step06b/REPORT.md
