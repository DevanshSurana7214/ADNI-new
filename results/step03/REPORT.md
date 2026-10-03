STEP 3 REPORT
Build and freeze the data splits (PIPELINE_v2 §7).

Command(s) run:
  PYTHONPATH=src ./.venv/Scripts/python.exe scripts/make_splits.py
      --episodes docs/episodes.csv
      --out   results/step03/splits.json              > (printed summary below)
  (disjointness + per-fold class/pattern verification via ad-hoc python checks)

Git commit: (pending — step03 not yet committed)

Tests: 24 passed in 12.07s   (test_splits.py 4, test_priors.py 6, test_stage3_repro.py 3,
        test_stage1_repro.py 6, test_stage1_invariants.py 5)

Environment: Python 3.12.10, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1. Windows 11.

What I did
  - Implemented src/adni/splits.py (S1 + S2 builders) and scripts/make_splits.py (CLI).
  - S1: StratifiedGroupKFold(n_splits=5, shuffle=True) over subjects (group = RID), stratified on a
    subject-level label (modal diagnosis across the subject's visits, ties broken by first-visit label),
    repeated for seeds [0, 1, 2]. Within each training fold, 10 % of the *training subjects* form the
    validation set (grouped on RID, stratified on subject label, fixed seed 42).
  - S2: protocol shift — subjects first enrolled in ADNI1 / ADNI-GO / ADNI2 → train+val side (1,784
    subjects); first enrolled in ADNI3 / ADNI4 → test side (1,361 subjects); val = 10 % of the
    training subjects (grouped, stratified, seed 42).
  - Frozen to results/step03/splits.json (episode_ids for train/val/test of every (seed, fold), plus S2).

Expected vs actual (table)

  Check                                              | Expected          | Actual            | Pass
  ---------------------------------------------------|-------------------|-------------------|-----
  S1: 3 seeds × 5 folds                             | 15 folds          | 15 folds          | yes
  S1: per fold train/val/test disjoint on eid       | yes               | verified (all 15) | yes
  S1: per fold train/val/test disjoint on RID       | yes               | verified (all 15) | yes
  S1: per fold union = all episodes                 | yes               | verified (all 15) | yes
  S1: every RID in exactly one test fold per seed   | yes               | verified (all 3)  | yes
  S1: val ≈ 10 % of pre-val training subjects       | ~10 %            | ~252 / ~2512 = 10.0 % | yes
  S2: training-side subjects                        | 1,784             | 1,784             | yes
  S2: test-side subjects                            | 1,361             | 1,361             | yes
  S2: train-side ∩ test RID overlap                 | 0                 | 0                 | yes
  S2: S2 train protocols                            | ADNI1/GO/ADNI2    | ADNI1/GO/ADNI2    | yes
  S2: S2 test protocols                             | ADNI3/4           | ADNI3/4           | yes

  Subject-level label used for stratification: modal diagnosis across a subject's episodes (ties → first
  visit). Rationale: 631 of 3,145 subjects have > 1 distinct y across visits, so a naive "first episode y"
  label mis-stratifies when a subject progresses. The modal label recovers the subject's dominant diagnosis.

S1 — seed_0, all 5 folds (subjects / episodes; test class %, patterns with < 5 test episodes)

  Fold | train (sub/ep) | val (sub/ep) | test (sub/ep) | test CN / MCI / Dem | test patterns <5 ep
  -----|----------------|--------------|---------------|---------------------|---------------------
  0    | 2260 / 9566    | 252 / 1027   | 633 / 2647    | 42.5 / 41.3 / 16.2  | 0101 (4), 0111 (3)
  1    | 2260 / 9557    | 252 / 1036   | 633 / 2647    | 40.4 / 41.6 / 18.0  | none
  2    | 2261 / 9548    | 252 / 1045   | 632 / 2647    | 38.7 / 42.6 / 18.7  | 0111 (2)
  3    | 2269 / 9519    | 253 / 1074   | 623 / 2647    | 38.6 / 41.3 / 20.1  | 0101 (1)
  4    | 2268 / 9599    | 253 / 989    | 624 / 2652    | 38.7 / 41.1 / 20.3  | 0111 (4), 0101 (2)

  Test class proportions are stable across folds (~39–42 % CN, ~41–43 % MCI, ~16–20 % Dementia).
  The sparse patterns 0101 (n=7 in cohort) and 0111 (n=20) occasionally land with < 5 test episodes in a
  fold — these are flagged here as required and will be pooled in per-pattern reporting (n >= 30 rule).

S2 (protocol shift): 1,784 train-side / 1,361 test-side subjects; zero overlap

  Split | subjects | episodes | CN % | MCI % | Dem %
  ------|----------|----------|------|-------|------
  train | 1605     | 8716     | 34.2 | 44.4  | 21.3
  val   | 179      | 959      | 38.2 | 39.4  | 22.4
  test  | 1361     | 3565     | 54.1 | 35.0  | 10.9

  The test side (ADNI3/4) is CN-heavier than the training side (ADNI1/GO/2): 54.1 % vs 34.2 % CN. This is
  the protocol-shift stress test the spec asks for.

Deviations from the spec (if any, with reason):
  - Spec §7 says "stratified on y" for S1. We stratify on a subject-level label (modal y), because
    StratifiedGroupKFold stratifies by group label and 631 subjects have episodes in multiple classes. Using
    the first-visit y for those subjects would mis-label them. The modal label is deterministic, uses only the
    episode table, and recovers the subject's dominant class. This is a faithful interpretation, not a deviation
    in intent.
  - Within-training validation is stratified on the same subject-level label (not per-episode y), so the val
    subjects reflect the training-subject class mix. This matches "10 % of the training subjects … grouped by
    RID, stratified on y".
  - Validation seed is fixed at 42 inside make_splits (not re-run per seed); this is a single, documented,
    deterministic choice.

Things I am unsure about:
  - The val seed (42) is fixed inside make_splits rather than per (seed, fold); the spec says "seed-fixed" for
    the 10 % holdout without specifying whether it varies. Keeping it constant is the conservative, reproducible
    choice; if a per-fold val seed is wanted, it can be re-run from configs.

Files written:
  src/adni/splits.py
  scripts/make_splits.py
  results/step03/splits.json
  results/step03/REPORT.md
  tests/test_splits.py
