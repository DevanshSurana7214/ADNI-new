STEP 7 REPORT
Neural model A0 (masked-attention fusion), its trainer, unit tests, and the S1 seed-0 fold-0 pilot.

Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):

  scripts/run_step07_pilot.py   > results/step07/{curves_s0f0.csv, curves_s0f0.png,
                                  pilot_s0f0.json, comparison_s0f0.{md,csv}}
  python -m pytest -q

Git commit: (this commit)  — tag `step07`

Tests: 61 passed in 40.25s   (tests/test_models.py adds 7; the rest unchanged)

Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1.
GPU: NVIDIA RTX 3050 (6 GB), CUDA. Training used `torch.use_deterministic_algorithms(True)` with
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; no determinism fallback was needed.

------------------------------------------------------------------------------
WHAT WAS BUILT

`src/adni/models.py`
  - `Encoder`: `LayerNorm -> Linear(2·d_k, 256) -> GELU -> Dropout(0.2) -> Linear(256, 128)
    -> LayerNorm` (PIPELINE_v2 §3.3).
  - `FusionModel`: one encoder per modality; tokens `t_k = h_k + type_k` plus a learnable `[CLS]`;
    absent modalities are OMITTED from attention via a key-padding mask (their token is never used
    as a key, so it cannot affect the output — the exact bound tests 1 and 6 check). Two pre-LN
    Transformer layers (d=128, 4 heads, FFN 256, dropout 0.1), head `Linear(128, 3)` on `[CLS]`;
    returns `(logits, h, effective_mask)`.
  - `modality_dropout` (train only): drops each observed modality independently with p=0.25 and
    resamples any row that would become empty; returns the effective mask.

`src/adni/train.py`
  - `Trainer`: AdamW (lr 3e-4, weight decay 0.05), batch 256, ≤100 epochs, early stopping with
    patience 15 on **validation macro-F1** (one rule for every arm), plain cross-entropy (A0).
    Seeds python/numpy/torch; the fold tensors are moved to the device once (`Tensors`).
  - `make_tensors` builds per-block feature tensors + natural mask + labels from a fitted
    `Preprocessor` (labels aligned by sorted episode_id).

`configs/model.yaml` holds the architecture + trainer hyperparameters.

`tests/test_models.py` (7 tests, all passing):
  1. absent-modality invariance — changing an absent modality's input leaves logits bit-identical;
  2. present-modality sensitivity — changing a present modality changes logits;
  3. all 15 patterns run at batch size 1 and in a mixed batch;
  4. overfit — a fixed batch of 64 episodes reaches cross-entropy < 0.05 within 300 steps;
  5. determinism — two seeded runs are bit-identical on CPU (val loss at epoch 3);
  6. gradients reach every encoder when all present; an entirely absent encoder gets exactly zero grad;
  7. modality dropout never returns an empty mask; empirical keep rate 0.7527 vs 0.75 (q=0.75,
     conditioned on non-empty the exact value is 0.75294) — within 0.02.

------------------------------------------------------------------------------
PILOT (A0, S1 seed 0 fold 0)

Training-curve summary (`results/step07/curves_s0f0.csv`, head/tail):

  epoch  train_loss  val_loss  val_macro_f1  val_macro_auroc
  0      0.8358      0.7297    0.6551        0.8138
  1      0.7296      0.6936    0.6645        0.8325
  2      0.6809      0.7058    0.6723        0.8248   <- best epoch (val macro-F1)
  ...
  16     0.3086      1.1404    0.6278        0.7970
  17     0.3031      1.2370    0.6270        0.8024

  params 699,931 (~0.70 M);  epochs run 18;  best epoch 2;
  seconds/epoch 1.22 (22.0 s total, GPU);  peak GPU memory 170 MB.

Fold-0 test metrics next to the references on the **same** test episodes (`comparison_s0f0.md`):

  | arm      | n_ep | macro-AUROC | macro-F1 | BAcc   | PN-BAcc (uniform) |
  |----------|------|-------------|----------|--------|-------------------|
  | A0       | 2647 | 0.7859      | 0.6154   | 0.6148 | 0.5877            |
  | B1       | 2647 | 0.7991      | 0.6311   | 0.6247 | 0.5851            |
  | B1-blind | 2647 | 0.7413      | 0.5751   | 0.5687 | 0.5552            |
  | B2_mri   | 2439 | 0.7517      | 0.5849   | 0.5772 | 0.5688            |

  The B1 / B1-blind / B2_mri rows are the seed-0 fold-0 out-of-fold predictions from Step 6 / 6c.
  A0 by number of observed modalities (macro-AUROC / PN-BAcc): 1 mod 0.7536 / 0.5834 (n=1130);
  2 mod 0.8053 / 0.5954 (n=858); 3 mod 0.8104 / 0.5711 (n=624); 4 mod 0.8347 / 0.7596 (n=35).

  Reading: A0 is a close second to B1 on this fold (and slightly ahead of B1 on PN-BAcc), clearly
  above the B1-blind and MRI-only references — exactly the expected ordering for a first, untuned
  A0. No tuning was done against these test numbers.

  RED FLAG (reported, not hidden): **best epoch = 2 < 5**. Validation macro-F1 peaks very early and
  then declines (val loss rises monotonically after epoch 1) — the model overfits this single fold
  within a few epochs. The other red flags do not fire: test macro-AUROC 0.786 ≥ 0.70, val–test AUROC
  gap |0.825 − 0.786| = 0.039 ≤ 0.05, and train loss decreases steadily. Early stopping did its job
  (it restored the epoch-2 weights), but the patience-15 budget is far larger than the useful epoch
  range on one fold; the full S1 grid (Step 9) may want a lower learning rate or stronger
  regularisation. This is left as-is for the pilot (no test-guided tuning).

------------------------------------------------------------------------------
EXPECTED VS ACTUAL

  Check                                        | Expected            | Actual            | Pass
  ---------------------------------------------|---------------------|-------------------|-----
  model parameter count                        | ~0.7 M              | 699,931           | yes
  unit tests (playbook Step 7, items 1–7)      | 7 pass              | 7 pass            | yes
  full suite                                   | all pass            | 61 passed         | yes
  pilot returns logits for all 4 modality counts | yes              | yes               | yes
  determinism at epoch 3 (CPU)                 | bit-identical       | bit-identical     | yes
  absent modality is bit-invariant             | yes                 | yes               | yes
  red flags reported                           | as they occur       | best epoch < 5    | yes

------------------------------------------------------------------------------
DEVIATIONS FROM THE SPEC (with reason)

- Test 2 perturbs a present modality with a fresh random draw rather than adding a constant: a
  constant added to every feature of a token is exactly LayerNorm-invariant (LayerNorm subtracts the
  per-token mean), so a "+3.0" perturbation would trivially leave the logits unchanged and test
  nothing. The test still checks the intended property (present-modality sensitivity).
- The trainer keeps the whole fold on the device (per the playbook) and iterates mini-batches with a
  permutation; no `DataLoader` is used.

------------------------------------------------------------------------------
THINGS I AM UNSURE ABOUT

- The early peak (best epoch 2) may be a learning-rate / regularisation artefact specific to fold 0.
  I did not sweep lr or dropout — that would be tuning, which the pilot prompt forbids. It is flagged
  for the full run.
- A0 beats B1-blind on macro-AUROC but is a hair below B1; the pilot is a single fold, so no
  conclusion about A0 vs B1 is drawn here.
- The 4-modality group has only 35 test episodes; its 0.8347 AUROC / 0.7596 PN-BAcc are noisy.

------------------------------------------------------------------------------
FILES WRITTEN

  src/adni/models.py
  src/adni/train.py
  configs/model.yaml
  tests/test_models.py
  scripts/run_step07_pilot.py
  results/step07/curves_s0f0.csv
  results/step07/curves_s0f0.png
  results/step07/pilot_s0f0.json
  results/step07/comparison_s0f0.md
  results/step07/comparison_s0f0.csv
  results/step07/REPORT.md
