STEP 4 REPORT
Stage 2 preprocessing (fit on train only) — PIPELINE_v2 §3.2.

Command(s) run:
  PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q
  PYTHONPATH=src ./.venv/Scripts/python.exe scripts/run_preprocess.py \
      --split s1 --seed 0 --fold 0 \
      --out-txt results/step04/seed0_fold0_report.txt
  (ad-hoc verification: joblib reload reproduces the transform bit-for-bit; raw |z| counts)

Git commit: (pending — step04 not yet committed)

Tests: 34 passed in 17.26s   (test_preprocess.py 10 new, plus the prior 24)

Environment: Python 3.12.10, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, joblib. Windows 11.

What I did
  - Implemented src/adni/preprocess.py: a fit/transform object `Preprocessor`.
      * Inputs: the four Step 1 block tables (results/step01/stage1/blocks/*.csv), the feature
        list in feature_spec.json, and the m_k mask bits from docs/episodes.csv.
      * fit(train_episode_ids): per block, using ONLY the training episodes where that modality is
        present (m_k == 1) — drop features missing in > 20 % of those rows; compute mean/SD of the
        kept features (NaN ignored), SD floored at 1e-6.
      * transform(episode_ids): per block a float32 array [N, 2·d_k] = [z with NaN -> 0 ; per-feature
        missing-indicator], plus the block-level presence mask m_k (uint8). z is winsorised to
        [-5, 5]. Absent modalities give all-zero rows with mask 0.
      * save_fitted() writes the object with joblib; config in configs/preprocess.yaml.
  - Wrote tests/test_preprocess.py (10 tests) covering the five required checks plus drop-rule,
    determinism, clip-fraction and joblib round-trip checks.
  - Wrote scripts/run_preprocess.py to fit a chosen split (default S1 seed 0 fold 0), save the
    fitted object and report the per-block diagnostics.

Expected vs actual (table)

  Check                                             | Expected                       | Actual                          | Pass
  --------------------------------------------------|--------------------------------|---------------------------------|-----
  Tests                                             | 5 required, all pass           | 10 tests pass                   | yes
  Kept features MRI                                 | ~256 minus a few high-missing  | 254 (dropped ST8SV, ST68SV)     | yes
  Kept features FDG                                 | ~6                             | 6 (none dropped)                | yes
  Kept features amy                                 | ~164 minus a few high-missing  | 162 (dropped VENTRICLE_5TH_SUVR, NON_WM_HYPOINTENSITIES_SUVR) | yes
  Kept features tau                                 | ~162 minus a few high-missing  | 160 (dropped VENTRICLE_5TH_SUVR, NON_WM_HYPOINTENSITIES_SUVR) | yes
  Fit uses training ids only (test 2)               | stats unchanged                | identical mean/SD/kept lists    | yes
  Absent modalities all-zero + mask 0 (test 3)      | yes                            | yes                             | yes
  Missing-indicator == NaN pattern (test 4)         | yes                            | exact match on kept features    | yes
  Train z mean~0 / SD~1 where unclipped (test 5)    | tol 1e-3                       | passes                          | yes
  No NaN/inf in any output (test 1)                 | yes                            | yes (float32)                   | yes

S1 seed 0 fold 0 — fitted on the TRAIN split only (train/val/test = 9566/1027/2647 episodes)

  Block | kept | dropped | dropped feature names                                   | present rows | clipped z fraction
  ------|------|---------|---------------------------------------------------------|--------------|-------------------
  mri   | 254  | 2       | ST68SV, ST8SV                                            | 8918         | 0.000460 (1029 / 2,239,293)
  fdg   | 6    | 0       | (none)                                                   | 3632         | 0.000321 (7 / 21,792)
  amy   | 162  | 2       | VENTRICLE_5TH_SUVR, NON_WM_HYPOINTENSITIES_SUVR          | 3350         | 0.000116 (63 / 542,463)
  tau   | 160  | 2       | VENTRICLE_5TH_SUVR, NON_WM_HYPOINTENSITIES_SUVR          | 2090         | 0.004213 (1408 / 334,243)

  Clipped-z fraction = fraction of FINITE present z-scores whose |z| exceeds 5 (i.e. actually
  winsorised). Max raw |z| before clipping: mri 41.5, fdg 8.9, amy 7.2, tau 14.9.

  Per-row missing-indicator rate (fraction of kept features that are NaN), over present rows:

  Block | min    | median | max
  ------|--------|--------|------
  mri   | 0.0000 | 0.0000 | 0.4764
  fdg   | 0.0000 | 0.0000 | 0.0000
  amy   | 0.0000 | 0.0000 | 0.0247
  tau   | 0.0000 | 0.0000 | 0.0188

  The two dropped MRI features match the Step 1 report exactly (ST8SV NaN 99.0 %, ST68SV 91.3 %);
  the two dropped amy/tau features are the same pair flagged in Step 1 (VENTRICLE_5TH_SUVR 98.9 %,
  NON_WM_HYPOINTENSITIES_SUVR 93.2 %). No other feature exceeds the 20 % threshold.

Deviations from the spec (if any, with reason):
  - None. The fit/transform split, the >20 % drop rule, the ±5 winsorisation, the NaN->0 +
    missing-indicator encoding and the m_k presence mask all follow PIPELINE_v2 §3.2.
  - The fitted objects are saved under results/step04/artifacts/, which .gitignore excludes
    (results/*/artifacts/), so they are not committed. This matches the Step 0 .gitignore policy for
    bulky artifacts; the deterministic fit means they can be regenerated at any time.

Things I am unsure about:
  - feature_spec.json lists 256 MRI columns, including `mri_field_strength`, and the block table has
    exactly those 256 columns. The spec text calls MRI "256 = 255 FreeSurfer + field strength", so
    `mri_field_strength` is a legitimate feature and is kept (not dropped) unless it exceeds the
    missing threshold. It is well below 20 % missing, so it is retained.
  - The spec does not say whether the missing-indicator for a dropped feature should be kept; since
    dropped features are removed entirely (width 2·d_k over kept features only), no indicator exists
    for them. This is the natural reading of "drop features ... store the kept list".

Files written:
  src/adni/preprocess.py
  configs/preprocess.yaml
  scripts/run_preprocess.py
  tests/test_preprocess.py
  results/step04/seed0_fold0_report.txt
  results/step04/REPORT.md
  results/step04/artifacts/s1_seed0_fold0.joblib   (gitignored)
