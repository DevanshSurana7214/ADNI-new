# Step 6e Report — Split stratification verification + B1-imp+mask deferral

**Command(s) run:**
- `pytest tests/test_s1_recompute.py -v` — recompute S1 seed_0 folds from episodes.csv, assert equality with splits.json

**Tests:** 3 new tests in test_s1_recompute.py (all pass)

---

## Item 1 — Split stratification documentation

### What each split stratifies on (from src/adni/splits.py)

**S1 outer folds:**
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` 
- **Stratifies on: per-episode `y`** (episode-level diagnosis 0/1/2), NOT subject modal diagnosis
- **Groups: RID** (subjects cannot be split across folds)

**S1 validation holdout:**
- `train_test_split(..., stratify=train_y)` where train_y is **subject-level modal diagnosis**
- This IS stratified on subject modal diagnosis (splitting subjects, not episodes)

**S2 validation holdout:**
- Same as S1 val: subject-level modal label stratification

### Test added

`tests/test_s1_recompute.py` (3 tests, all pass):
- `test_s1_seed0_test_folds_reproduce_splits_json` — recomputes SGKF test folds, asserts exact match with splits.json
- `test_s1_stratifies_on_per_episode_y_not_modal` — verifies SGKF uses per-episode y
- `test_s1_val_holdout_stratifies_on_subject_label` — reproduces val holdout with subject modal labels

---

## Item 2 — B1-imp+mask decomposition deferred

The B1-imp+mask arm was NOT fitted. Deferred to sensitivity analyses.

---

## Files written

- `tests/test_s1_recompute.py` — 3 tests
- `results/step06e/REPORT.md` — this file
