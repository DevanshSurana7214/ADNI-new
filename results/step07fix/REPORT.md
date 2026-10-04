# Step 7 close-out - report (`step07fix`)

Repairs the repository state after `1e141a7` replaced the verified Step 7 model, restores the `FusionModel` implementation, and makes training reproducible.

## A.1 Repository state

- `git rev-parse step07` -> `6761aa7e73d2defc7656c0bd77fb934c3698b51f` (the verified Step 7).
- HEAD after the repair -> `da18f68`.

| recent commits (newest first) |
|---|
| `da18f68 Revert "Step 7: neural model A0 + trainer + config + tests (pilot pending)"` |
| `1e141a7 Step 7: neural model A0 + trainer + config + tests (pilot pending)` |
| `573b2ca Step 6e: split stratification verification + B1-imp+mask deferral` |
| `6761aa7 step07: masked-attention fusion model A0, deterministic trainer, and pilot` |
| `0519b7c step06c: regenerate Step 3 tables, reconcile B0 encoding, add B1-blind imputation arm` |
| `2cd9f94 step06b: add the preprocessing audit outputs missing from the 6b commit` |

## A.2 Files changed by `1e141a7` and the revert

`1e141a7` changed exactly these four files (status = A/M/D vs its parent `573b2ca`); `git revert 1e141a7` restored the `6761aa7` versions of all four.

| status | file | reverts to 6761aa7 version |
|---|---|---|
| M | `configs/model.yaml` | yes |
| M | `src/adni/models.py` | yes |
| M | `src/adni/train.py` | yes |
| M | `tests/test_models.py` | yes |

The other untracked files in the tree (`results/step06d/`, `tests/test_step03_tables.py`, `tests/test_step06c.py`, `scripts/step06d_verify_tables.py`, `scripts/verify_step03_tables.py`) are independent Step 6d/6e verification artifacts and were kept.

## A.3 Test suite and pilot reproducibility

- `pytest -q`: **76 passed in 105.08s (0:01:45)** (the 6761aa7-tagged set was 61; Step 6e added 3 and this repair added 2 model tests, and the untracked Step 6d tests add 10 more).
- `git diff --stat results/step07`: `(empty)` - the committed Step 7 results were not edited.

Re-running `scripts/run_step07_pilot.py` on the reverted code did **not** reproduce `results/step07`: `train_a0` built the model *before* `Trainer.fit` called `set_seed`, so the initial weights were an unseeded draw. The fix seeds before model construction (`src/adni/train.py`); two runs then agree on every field except wall-clock timing.

| pilot field | committed `results/step07` (unseeded draw) | fixed run (`step07fix/det_A`) |
|---|---|---|
| epochs_run | 18 | 18 |
| best_epoch | 2 | 2 |
| best_val_macro_f1 | 0.672328 | 0.667147 |
| test macro_auroc | 0.785894 | 0.786605 |
| test macro_f1 | 0.615417 | 0.621146 |

- deterministic re-runs: curves CSV identical = `True`, comparison CSV identical = `True`, pilot JSON equal excluding timing = `True`. Wall-clock differs (20.71s vs 20.50s); GPU memory is identical.

## A.4 Large-batch inference (restored model)

All 13240 episodes evaluated in one forward call and in chunks of 1024 (13 chunks), device `cuda`.

| quantity | value |
|---|---|
| single-call succeeded (no segfault) | `True` |
| predictions identical (argmax) | `True` |
| logits bitwise identical | `False` |
| max abs logit diff | 8.643e-07 |
| mean abs logit diff | 1.451e-07 |
| max |logit| | 1.0949 |
| peak GPU memory (MB) | 411.7 |

The single 13,240-row call succeeds, so no chunked-inference utility was needed (the replacement segfaulted here because it built an O(N^2) attention tensor). Bitwise equality across batch sizes is not attainable on GPU - float32 matmul reduction order depends on the batch shape - so the decision-relevant check is that predictions agree exactly, which they do (logits agree to float32 round-off, ~1e-7 relative).

## A.5 Why the replacement was made and why it produced empty masks

The replacement rewrote the verified `FusionModel` as a `MultimodalTransformer` with per-modality encoders, a learnable `[CLS]` token and modality dropout (p=0.25) intended to regularise the fusion. Its dropout mask marked dropped entries (`True` = dropped) but the never-empty guard tested `drop_mask.any(dim=1)` (any dropped) instead of `drop_mask.all(dim=1)` (all dropped), so rows with every modality dropped were never detected; compounding this, the fallback wrote `retry_mask[still_empty, 0] = True`, which dropped modality 0 rather than keeping it, so resampled empty rows stayed empty. The result was a fully-empty mask at a rate near p^4 (measured 0.5745% at p=0.25 over 200k draws; ~0.2% in the small 1000-draw test), after which the model could attend only to `[CLS]`; flipping both conditions reduces the rate to 0%.

