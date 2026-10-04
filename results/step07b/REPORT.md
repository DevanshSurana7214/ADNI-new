STEP 7b REPORT
Validation-only regularisation search for the neural arms, the frozen configuration, and the A0 pilot comparison.

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/step07b_search.py         > results/step07b/{search_runs.csv, search_summary.json, search.log}
  scripts/step07b_freeze_config.py  > configs/model.yaml (old -> configs/model_v0.yaml)
  scripts/step07b_appendix.py       > docs/PIPELINE_v2.md Appendix E (and the parent copy)
  scripts/step07b_pilot_compare.py  > results/step07b/{comparison_frozen.{csv,md}, pilot_frozen/*}
  python -m pytest -q

Git commit: (this commit) - tag `step07b`

Tests: 79 passed in 62.53s (0:01:02)

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1, LightGBM 4.7.0. GPU: NVIDIA RTX 3050 (6 GB). Training used `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

------------------------------------------------------------------------------
PRE-REGISTRATION

Exactly 8 configurations (arm A0 only) on S1 seed 0 folds 0 and 1: lr in {3e-4,1e-4} x weight_decay in {0.05,0.3} x dropout in {(encoder 0.2, transformer 0.1), (encoder 0.4, transformer 0.2)}. Everything else as `configs/model.yaml` except `max_epochs=60`, `patience=10`. The test fold is never loaded (`_NoTestIds` raises on "test"). Selection rule, fixed before the runs: highest mean best-epoch validation macro-F1 over the two folds; ties within 0.005 go to the larger weight decay, then the lower lr.

SEARCH GRID

| config | lr | weight_decay | encoder dropout | transformer dropout |
|---|---|---|---|---|
| c0 | 3.0e-04 | 0.05 | 0.2 | 0.1 |
| c1 | 3.0e-04 | 0.05 | 0.4 | 0.2 |
| c2 | 3.0e-04 | 0.3 | 0.2 | 0.1 |
| c3 | 3.0e-04 | 0.3 | 0.4 | 0.2 |
| c4 | 1.0e-04 | 0.05 | 0.2 | 0.1 |
| c5 | 1.0e-04 | 0.05 | 0.4 | 0.2 |
| c6 | 1.0e-04 | 0.3 | 0.2 | 0.1 |
| c7 | 1.0e-04 | 0.3 | 0.4 | 0.2 |

ALL 16 RUNS (best epoch = argmax validation macro-F1)

| config | fold | best epoch | val macro-F1 | val macro-AUROC | val loss @ best | epochs run |
|---|---|---|---|---|---|---|
| c0 | 0 | 2 | 0.6671 | 0.8196 | 0.7207 | 13 |
| c0 | 1 | 0 | 0.6523 | 0.8053 | 0.7380 | 11 |
| c1 | 0 | 4 | 0.6758 | 0.8237 | 0.7285 | 15 |
| c1 | 1 | 3 | 0.6381 | 0.7887 | 0.8071 | 14 |
| c2 | 0 | 2 | 0.6652 | 0.8198 | 0.7198 | 13 |
| c2 | 1 | 0 | 0.6510 | 0.8054 | 0.7379 | 11 |
| c3 | 0 | 4 | 0.6759 | 0.8233 | 0.7283 | 15 |
| c3 | 1 | 3 | 0.6380 | 0.7888 | 0.8056 | 14 |
| c4 | 0 | 5 | 0.6706 | 0.8227 | 0.7309 | 16 |
| c4 | 1 | 1 | 0.6515 | 0.8027 | 0.7367 | 12 |
| c5 | 0 | 6 | 0.6688 | 0.8273 | 0.7050 | 17 |
| c5 | 1 | 1 | 0.6479 | 0.8036 | 0.7396 | 12 |
| c6 | 0 | 5 | 0.6691 | 0.8227 | 0.7303 | 16 |
| c6 | 1 | 1 | 0.6515 | 0.8027 | 0.7365 | 12 |
| c7 | 0 | 6 | 0.6687 | 0.8273 | 0.7048 | 17 |
| c7 | 1 | 1 | 0.6479 | 0.8037 | 0.7394 | 12 |

MEAN OVER FOLDS AND SELECTION

| config | mean best-epoch val macro-F1 | within 0.005 of best |
|---|---|---|
| c0 | 0.6597 | yes |
| c1 | 0.6569 | yes |
| c2 | 0.6581 | yes |
| c3 | 0.6570 | yes |
| c4 | 0.6611 | yes |
| c5 | 0.6583 | yes |
| c6 | 0.6603 | yes |
| c7 | 0.6583 | yes |

Best mean = 0.6611. All 8 configurations are within 0.005 of the best, so the tie-break chain decides: largest weight decay (0.3 -> c2,c3,c6,c7), then lowest lr (1e-4 -> c6,c7). c6 and c7 differ only in dropout; the smaller dropout pair is taken, giving **c6**.

------------------------------------------------------------------------------
FROZEN CONFIGURATION

| key | model_v0.yaml (pre-search) | model.yaml (frozen) |
|---|---|---|
| enc_dropout | 0.2 | 0.2 |
| attn_dropout | 0.1 | 0.1 |
| lr | 0.0003 | 0.0001 |
| weight_decay | 0.05 | 0.3 |
| max_epochs | 100 | 100 |
| patience | 15 | 15 |

Only the selected axes change; `max_epochs` (100) and `patience` (15) stay at their `configs/model.yaml` values because the 60/10 cap was search-time only. Every arm A0-A9 reuses this configuration; there is no per-arm tuning.

------------------------------------------------------------------------------
PILOT COMPARISON (S1 seed 0 fold 0, test fold)

The pilot test fold is used here and nowhere else in Step 7b; nothing is tuned on it.

| arm | n_ep | macro-AUROC | macro-F1 | BAcc | PN-BAcc (uniform) | PN-BAcc (P(c)) |
|---|---|---|---|---|---|---|
| A0 (old config, model_v0) | 2647 | 0.7866 | 0.6211 | 0.6212 | 0.5924 | 0.5923 |
| A0 (frozen config) | 2647 | 0.7844 | 0.6215 | 0.6327 | 0.6129 | 0.6128 |
| B1 | 2647 | 0.7991 | 0.6311 | 0.6247 | 0.5851 | 0.5853 |
| B1-blind | 2647 | 0.7413 | 0.5751 | 0.5687 | 0.5552 | 0.5551 |
| B2_mri | 2439 | 0.7517 | 0.5849 | 0.5772 | 0.5688 | 0.5687 |

Reading: freezing moves A0 from the old-config row to the frozen-config row. Test macro-AUROC is essentially unchanged (0.7866 -> 0.7844) while BAcc and both PN-BAcc variants improve (BAcc 0.6212 -> 0.6327, PN-BAcc uniform 0.5924 -> 0.6129), and the best epoch moves from 2 to 5, clearing the Step 7 `best_epoch_below_5` red flag. A0 remains a close second to B1 on macro-AUROC and above the B1-blind and MRI-only references.

------------------------------------------------------------------------------
EXPECTED VS ACTUAL

| check | expected | actual | pass |
|---|---|---|---|
| configurations searched | 8 | 8 | yes |
| runs (8 configs x 2 folds) | 16 | 16 | yes |
| test fold loaded during search | never | never (`_NoTestIds` guard) | yes |
| selection rule | pre-registered | applied (wd, then lr, then dropout) | yes |
| old config preserved | configs/model_v0.yaml | present | yes |
| Appendix E row | added | added to docs + parent | yes |
| pilot uses frozen config | yes | yes | yes |
| full test suite | all pass | 79 passed | yes |

DEVIATIONS FROM THE SPEC (with reason)

- `docs/PIPELINE_v2.md` (420 lines) has no Appendix E; the decision log existed only in the parent `PIPELINE_v2.md`. The script mirrors the parent log into the docs copy and appends the new rows to both.
- The pre-registered tie-break (larger weight decay, then lower lr) is undefined when the remaining difference is the dropout axis. All three steps still left c6 and c7 tied, so the smaller dropout pair was taken as a deterministic fallback.
- The frozen config keeps `max_epochs=100` / `patience=15`; the 60/10 values belong to the search only.

THINGS I AM UNSURE ABOUT

- The mean validation macro-F1 spread is tiny (0.6569-0.6611), so the choice rests on the pre-registered tie-break rather than a clear performance gap.
- The frozen config improves PN-BAcc on fold 0, but the pilot is a single fold and single seed; the full S1 grid will settle A0 vs the reference arms.

FILES WRITTEN

  scripts/step07b_search.py
  scripts/step07b_freeze_config.py
  scripts/step07b_appendix.py
  scripts/step07b_pilot_compare.py
  scripts/step07b_report.py
  tests/test_step07b_search.py
  configs/model.yaml
  configs/model_v0.yaml
  docs/PIPELINE_v2.md
  results/step07b/search_runs.csv
  results/step07b/search_summary.json
  results/step07b/search.log
  results/step07b/comparison_frozen.csv
  results/step07b/comparison_frozen.md
  results/step07b/REPORT.md

