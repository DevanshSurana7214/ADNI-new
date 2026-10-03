STEP 2 REPORT
Reproduce the Stage 3 joint profiling of the ADNI cohort (PIPELINE_v2 §4).

Command(s) run:
  ./.venv/Scripts/python.exe scripts/adni_stage3_profile.py \\
      --episodes results/step01/stage1/episodes.csv \\
      --out results/step02/joint_profile.csv \\
      --metrics-json results/step02/metrics.json    > results/step02/stdout.txt
  (independently regenerated from the canonical docs/episodes.csv — same result)

Git commit: (pending — step02 not yet committed)

Tests: 9 passed in 0.71s   (test_stage3_repro.py 3 tests, test_priors.py 6 tests)

Environment: Python 3.12.10, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1. Windows 11.

What I did
  - Ran the provided profiling script on the canonical Stage 1 episode table (13,240 episodes,
    3,145 subjects) and wrote results/step02/{joint_profile.csv, metrics.json, stdout.txt}.
  - Wrote tests/test_stage3_repro.py: the regenerated joint_profile.csv matches the provided
    ../joint_profile.csv to within 1e-9 on every shared numeric column.
  - Wrote src/adni/priors.py (smoothed joint P(c,m), PMI, independence weight exp(-PMI) and clipped
    per-sample weights) plus tests/test_priors.py; these reproduce the PMI and w_indep columns of
    joint_profile.csv to 1e-9 and the ESS diagnostics (0.739 raw, 0.878 clipped) independently.

Expected vs actual (table)

  Item                                              | Expected              | Actual            | Pass
  --------------------------------------------------|-----------------------|-------------------|-----
  Cramér's V (class × pattern)                     | 0.238, CI [0.225,0.258] | 0.238, [0.225,0.258] | yes
  Pattern-only classifier macro-AUROC / bal acc    | 0.618 / 0.412        | 0.618 / 0.412     | yes
  I(C;M) surviving ORIGPROT / COLPROT              | 91% / 80%            | 91% / 80%         | yes
  Raw weights: range / ESS-N raw / ESS-N clipped   | 0.53–31.6 / 0.739 / 0.878 | 0.53–31.62 / 0.739 / 0.878 | yes
  V after clipped reweighting                       | 0.055                | 0.055             | yes
  Cells with n < 10 / empty cells                  | 6 / 1                | 6 / 1             | yes
  joint_profile.csv identical to ../joint_profile.csv | within 1e-9         | 1e-9 match         | yes
  src/adni/priors.py PMI + w_indep match reference  | 1e-9                 | 1e-9               | yes
  src/adni/priors.py ess_clip = 0.878               | 0.878                | 0.878              | yes

  Median |gap| note (carried from Step 1b): The provided script's printed tau median|gap| was 25d,
  computed over all rows. After Step 1b's m_k==0 => NaN change, the same script now prints 26d,
  computed over m_tau==1 rows only. That does not affect this Step 2 profile: the class × pattern
  joint tables use no scan dates or gaps at all.

Deviations from the spec (if any, with reason):
  - None. The provided profiling script was run unmodified (Standing Rule 1). The only new code is
    src/adni/priors.py and its unit tests, which are an independent re-implementation of the same
    figures, not a change to the script.

Things I am unsure about:
  - None blocking. The bootstrap CI [0.225, 0.258] is exactly the supplied expectation; the profile's
    500-subject bootstrap with rng seed 0 reproduces it. The playbook says the CI "can differ in the
    third decimal" — here it matches to the reported precision.

Files written:
  results/step02/joint_profile.csv
  results/step02/metrics.json
  results/step02/stdout.txt
  results/step02/REPORT.md
  src/adni/priors.py
  tests/test_stage3_repro.py
  tests/test_priors.py
  pyproject.toml
  src/adni/__init__.py
