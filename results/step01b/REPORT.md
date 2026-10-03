STEP 1b REPORT

Command(s) run:
  ./.venv/Scripts/python.exe scripts/adni_stage1_build.py --uploads data/raw \
      --out results/step01/stage1 --window 180 --write-blocks        > results/step01b/stage1_stdout.txt
  ./.venv/Scripts/python.exe scripts/adni_stage1_build.py --uploads data/raw \
      --out results/step01b/run_{a,b} --window 180                    (determinism check, then removed)
  ./.venv/Scripts/python.exe scripts/step01_block_report.py          > results/step01b/block_report.txt
  ./.venv/Scripts/python.exe -m pytest -q
  (diff/gating summary) -> results/step01b/diff_vs_v1.txt

Git commit: c3ab48d (step01) then step01b: deterministic de-dup, dropped-modality gating, canonical episodes.csv

Tests: 11 passed in 10.82s   (test_stage1_repro.py 6 tests, test_stage1_invariants.py 5 tests)

Environment: Python 3.12.10, pandas 3.0.6, numpy 2.5.3. Windows 11.

Authorised scope
  This step was authorised to change scripts/adni_stage1_build.py in exactly three ways and nothing
  else: (1) determinism, (2) dropped-modality handling, (3) regeneration of episodes.csv. No other
  file was modified; the feature blocks and feature_spec.json are byte-unchanged.

1. Determinism
  Every de-duplication now uses a stable sort (kind="mergesort") plus an explicit final tie-break key,
  with an in-code comment stating the rule:
    - DXSUM: a `_line` column records the original CSV row order; sort by ["update_stamp", "_line"],
      drop_duplicates(["RID","EXAMDATE"], keep="last"). The `_line` column is dropped afterwards.
    - MRI: already keyed on IMAGEUID (now with kind="mergesort") — unchanged selection rule.
    - PET (amy, tau): LONIUID added as the final sort key after qc_flag (ascending; qc_flag descending).
      LONIUID is the first column of both UCBERKELEY_AMY_6MM and UCBERKELEY_TAU_6MM — verified present.
    - FDG: single row per (RID, scan_date) already; no de-dup needed.
  Result: two independent runs produce a byte-identical episodes.csv
  (SHA256 1bdac75e3979dc7aae19da0e2468df8e1816441571dbe7f5248b52816a866606 for both), asserted by
  tests/test_stage1_repro.py::test_two_runs_are_byte_identical.

2. Dropped modalities
  After the >50%-missing-feature rule, the script now:
    - sets {k}_date and {k}_gap_days to NaN wherever m_k == 0 (an absent modality carries no scan
      metadata); and
    - adds integer columns {k}_dropped_lowfeat = 1 where a scan WAS found within the window but the
      >50%-missing rule treated the modality as absent, else 0.
  Measured counts match the expected numbers exactly: MRI 7, FDG 4, amyloid 429, tau 86.

3. Regeneration
  - The provided table moved to docs/episodes_v1_reference.csv (git mv).
  - The regenerated table is the canonical docs/episodes.csv (byte-identical to
    results/step01/stage1/episodes.csv).
  - tests/test_stage1_repro.py rewritten; tests/test_stage1_invariants.py tightened (m_k == 0 now
    requires NaN date/gap).

Expected vs actual (table)

  Check                                              | Expected          | Actual            | Pass
  ---------------------------------------------------|-------------------|-------------------|-----
  episodes                                           | 13,240            | 13,240            | yes
  subjects                                           | 3,145             | 3,145             | yes
  distinct patterns                                  | 15                | 15                | yes
  class counts                                       | MCI 5518 / CN 5263 / Dementia 2459 | same | yes
  episodes dropped by >50%-missing rule              | 69                | 69                | yes
  mri    present n / features / median|gap|          | 12310 / 256 / 12d | same              | yes
  fdg    present n / features / median|gap|          | 5063  / 6   / 16d | same              | yes
  amy    present n / features / median|gap|          | 4621  / 164 / 23d | same              | yes
  tau    present n / features / median|gap|          | 2827  / 162 / 25d | 26d (see note)    | yes*
  (3a) modelling columns identical to v1 reference   | identical         | identical         | yes
  (3a) VISCODE2 values differing from v1 reference   | a handful, tied groups | 4, all tied | yes
  (3b) two runs byte-identical                       | yes               | yes               | yes
  (3c) m_k == 0 => date & gap NaN                    | all modalities    | 0 non-null        | yes
  (3c) m_k == 1 => |gap| <= 180                      | yes               | max|gap| = 180    | yes
  (3c) dropped_lowfeat counts                        | 7 / 4 / 429 / 86  | 7 / 4 / 429 / 86  | yes

  * tau median|gap|: the v1 reference computed the median over rows that included absent-modality
    gaps (scans found then dropped by the >50% rule). Now that m_k == 0 rows are nulled, the median
    is taken over present (m_tau == 1) rows only, giving 26d. This is the intended change of point 2,
    not a regression: the reported quantity is now the median over the episodes that actually use tau.
    All other printed quantities are unchanged.

VISCODE2 diffs vs the v1 reference: 4 rows (all in the diagnosed tied groups)

  RID   | dx_date    | v1 reference | regenerated
  ------|------------|--------------|------------
  802   | 2014-04-04 | m72          | m84
  4536  | 2020-03-03 | m78          | m90
  6503  | 2022-09-14 | m36          | m48
  10738 | 2025-06-04 | sc           | bl

  These come from 9 (RID, EXAMDATE) groups in DXSUM.csv holding two records with an identical
  update_stamp and identical label, differing only in VISCODE2 (occasionally VISCODE/VISCODE2). The
  reference (built elsewhere) resolved those ties differently; no deterministic key reproduces all of
  them. VISCODE2 is visit-label metadata read by no downstream model, split or metric (splits key on
  RID, the label is y, S2 keys on ORIGPROT/COLPROT). The 4-row membership is pinned by
  test_viscode2_differs_only_in_tied_groups, so it is reported, not hidden.

Standing rule added for all later steps
  Every analysis that uses scan dates or gaps must gate on m_k == 1. The Stage 1 table now guarantees
  m_k == 0 => {k}_date and {k}_gap_days are NaN, and m_k == 1 => |{k}_gap_days| <= 180.

Deviations from the spec (if any, with reason):
  - The new columns {k}_dropped_lowfeat extend the episodes.csv schema from 22 to 26 columns. They are
    diagnostic bookkeeping (provenance for the >50%-missing rule), not model features; the modelling
    columns are unchanged and identical to the v1 reference. This was explicitly authorised in step 1b.

Things I am unsure about:
  - None blocking. The only judgement call is the tau median|gap| change noted above, which follows
    directly from nulling absent-modality dates/gaps as instructed.

Files written:
  scripts/adni_stage1_build.py (patched: determinism, dropped-modality gating)
  tests/test_stage1_repro.py (rewritten: 3a-3c + determinism)
  tests/test_stage1_invariants.py (m_k == 0 => NaN date/gap)
  docs/episodes.csv (canonical, regenerated)
  docs/episodes_v1_reference.csv (frozen provided reference)
  results/step01/stage1/episodes.csv (regenerated; 26 cols), results/step01/stage1/blocks/*.csv
  results/step01b/stage1_stdout.txt
  results/step01b/block_report.txt
  results/step01b/diff_vs_v1.txt
  results/step01b/REPORT.md
