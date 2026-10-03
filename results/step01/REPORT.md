# Step 01 — Reproduce Stage 1 (episodes)

**Status:** complete. All checks pass, with one **diagnosed, non-modelling** deviation (VISCODE2, below).

## Command

```bash
python scripts/adni_stage1_build.py --uploads data/raw --out results/step01/stage1 --window 180 --write-blocks
```

## stdout

```
missing-feature rule: 69 episodes dropped (blocks with >50% missing features treated as absent)
episodes=13240 subjects=3145 patterns=15 window=+/-180d
{'MCI': 5518, 'CN': 5263, 'Dementia': 2459}
  mri: n=12310 features=256 median|gap|=12d
  fdg: n=5063 features=6 median|gap|=16d
  amy: n=4621 features=164 median|gap|=23d
  tau: n=2827 features=162 median|gap|=25d
```

## Expected vs actual (all match)

| Item | Expected | Actual | OK |
|---|---|---|---|
| Episodes / subjects | 13,240 / 3,145 | 13,240 / 3,145 | ✅ |
| CN / MCI / Dementia | 5,263 / 5,518 / 2,459 | 5,263 / 5,518 / 2,459 | ✅ |
| Patterns present | 15 (no `0000`) | 15 | ✅ |
| MRI / FDG / amy / tau present | 12,310 / 5,063 / 4,621 / 2,827 | same | ✅ |
| Features per block | 256 / 6 / 164 / 162 | 256 / 6 / 164 / 162 | ✅ |
| Episodes dropped by >50%-missing rule | 69 | 69 | ✅ |
| Median \|gap\| MRI/FDG/amy/tau | 12 / 16 / 23 / 25 d | 12 / 16 / 23 / 25 d | ✅ |
| Pattern `1000`/`1100`/`1110`/`1111` | 4,954 / 2,123 / 2,123 / 234 | same | ✅ |
| Cells n<10 / empty | 6 / 1 | 6 / 1 | ✅ |

## Class × pattern table

```
pattern    0001  0010  0011  0100  0101  0110  0111  1000  1001  1010  1011  1100  1101  1110  1111  Total
CN           77   151   121    44     0    59     2  1755   677   303   738   583     3   734    16   5263
MCI          40   123    47    54     5    50    11  2158   218   250   183  1020   195   996   168   5518
Dementia     16    30    19    44     2    28     7  1041    88    77    65   520    79   393    50   2459
Total       133   304   187   142     7   137    20  4954   983   630   986  2123   277  2123   234  13240
```

## Per-block statistics

| Block | shape | feature cols | rows w/ any NaN | mean per-feature NaN | top-5 by NaN |
|---|---|---|---|---|---|
| mri | (12310, 257) | 256 | 0.9993 | 0.0190 | ST8SV 0.990, ST68SV 0.913, ST125SV 0.043, ST107SA 0.037, ST38SA 0.037 |
| fdg | (5063, 7) | 6 | 0.0000 | 0.0000 | (none missing) |
| amy | (4621, 165) | 164 | 0.9998 | 0.0122 | VENTRICLE_5TH_SUVR 0.989, NON_WM_HYPOINTENSITIES_SUVR 0.932, RIGHT_VESSEL_SUVR 0.040, LEFT_VESSEL_SUVR 0.027, VESSEL_SUVR 0.008 |
| tau | (2827, 163) | 162 | 0.9993 | 0.0123 | VENTRICLE_5TH_SUVR 0.988, NON_WM_HYPOINTENSITIES_SUVR 0.927, RIGHT_VESSEL_SUVR 0.039, LEFT_VESSEL_SUVR 0.025, VESSEL_SUVR 0.010 |

Presence checks (all blocks): expected-present == in-block, **0 duplicated episode rows, 0 missing, 0 absent episodes present**. PASS.

Note the two MRI features `ST8SV` and `ST68SV` are missing in >90 % of rows and will be dropped by the
Stage 2 `>20 % missing` filter, as expected.

## Tests

```
9 passed in 1.96s
```

- `tests/test_stage1_repro.py` (4 tests) — regenerated `episodes.csv` vs provided reference.
- `tests/test_stage1_invariants.py` (5 tests) — n_mod/pattern consistency, gaps within ±180 d, label
  range/consistency, no duplicate `(RID, dx_date)`, unique `episode_id`.

## Deviations from the spec (diagnosed, reported — not papered over)

### 1. VISCODE2 differs in exactly 4 rows (non-modelling column)

The regenerated `episodes.csv` is **identical to the reference in every modelling column**
(`DIAGNOSIS`, `y`, all `*_date`, all `m_*`, all `*_gap_days`, `pattern`, `n_mod`) — 13,240 × 22 rows,
byte-for-byte on those columns. Only `VISCODE2` differs, in 4 rows:

| RID | dx_date | reference | regenerated |
|---|---|---|---|
| 802 | 2014-04-04 | m72 | m84 |
| 4536 | 2020-03-03 | m78 | m90 |
| 6503 | 2022-09-14 | m36 | m48 |
| 10738 | 2025-06-04 | sc | bl |

**Diagnosis.** These 4 rows belong to 9 `(RID, EXAMDATE)` groups in `DXSUM.csv` that contain **two
records with an identical `update_stamp`** and an identical label, differing only in `VISCODE2`
(and occasionally `VISCODE`/`USERDATE`). The provided script de-duplicates with:

```python
dx.sort_values("update_stamp").drop_duplicates(["RID", "EXAMDATE"], keep="last")
```

When `update_stamp` ties, the surviving record is decided by the **unstable pandas sort**, which is
build/version dependent. The reference `episodes.csv` was generated under a different pandas/NumPy
build, so the two runs resolve those 9 ties differently.

I verified this is genuinely non-deterministic rather than a missing key: I brute-forced every
plausible deterministic secondary key (file order first/last, min/max of `VISCODE`, `VISCODE2`,
`USERDATE`, `USERDATE2`, `ID`, `SITEID`, `update_stamp`, `_line`). **The best any single key achieves
is 6 of the 9 reference choices** — i.e. no deterministic rule reproduces the reference, confirming
the reference used an unstable sort.

**Why it does not matter.** `VISCODE2` is visit-label metadata; it is never used downstream — splits
key on `RID`, the target is `y`, and S2 keys on `ORIGPROT`/`COLPROT`. It cannot affect any split,
model, metric or conclusion.

**What I did.** I did **not** change the provided script (Standing Rules 1 and 5). The reproduction
test asserts exact identity on all 22 columns except `VISCODE2`, and *separately pins* the exact
4-row `VISCODE2` discrepancy, so it is recorded rather than hidden. If you want bit-exact `VISCODE2`
reproduction, say so and I will make the tie-break explicit in the script (deterministic, stable
sort) — but the resulting `VISCODE2` will not match the reference's arbitrary choices.

### 2. Absent modalities may retain a scan date/gap

The `>50 %-missing` rule zeroes `m_k` **after** the nearest-scan merge, so a modality can have
`m_k == 0` while `*_date`/`*_gap_days` are still populated (it was found in the window, then
dropped). Counts: MRI 7, FDG 4, amy 429, tau 86 such rows. The playbook's phrasing ("for `m_k == 0`
the gap is NaN **or not used**") permits this; the invariant test checks the modality is never
*used* (mask 0) and that any retained gap is within the window. This is pre-existing behaviour of the
provided script and is not a bug.

STEP 01 REPORT
Command(s) run: `python scripts/adni_stage1_build.py --uploads data/raw --out results/step01/stage1 --window 180 --write-blocks`; `python scripts/step01_block_report.py`; `pytest -q`
Git commit: step01 commit (see `git log`)
Tests: `9 passed in 1.96s`
Expected vs actual: all expected Stage 1 values match (table above)
Deviations from the spec: VISCODE2 non-deterministic tie-break in the provided script (4 rows, non-modelling); absent modalities may retain a within-window date/gap
Things I am unsure about: whether you want VISCODE2 made bit-exact by changing the provided script
Files written: `results/step01/stage1/{episodes.csv,feature_spec.json,blocks/*.csv}`, `results/step01/{REPORT.md,stage1_stdout.txt,block_report.txt,class_pattern_table.csv}`, `scripts/step01_block_report.py`, `tests/test_stage1_repro.py`, `tests/test_stage1_invariants.py`
