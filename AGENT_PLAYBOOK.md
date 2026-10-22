# Agent Playbook — ADNI shortcut-aware multimodal pipeline (PIPELINE_v2)

How to use this file: give your coding agent **the Standing Rules once**, then **one step at a time**.
After each step, paste the "Paste back to Claude" items. **Do not start the next step until the checks for the current one pass.**
Paste text, not screenshots.

Files the agent needs (put them in one folder first): `PIPELINE_v2.md`, `adni_stage1_build.py`, `adni_stage3_profile.py`,
`episodes.csv`, `joint_profile.csv`, `feature_spec.json`, and the six ADNI CSVs
(`DXSUM.csv`, `UCSFFSX7_03Oct2026.csv`, `UCBERKELEY_AMY_6MM_03Oct2026.csv`, `UCBERKELEY_TAU_6MM_03Oct2026.csv`,
`UCBERKELEYFDG_8mm_02_17_23_03Oct2026.csv`, `DATADIC_03Oct2026.csv`).

------------------------------------------------------------------------

## Standing Rules (paste this to the agent first)

```text
You are implementing the research pipeline specified in PIPELINE_v2.md. Follow these rules for the whole project.

1. PIPELINE_v2.md is the specification. If something is ambiguous or seems wrong, STOP and ask; never silently change the design.
2. Work ONE STEP at a time. Do not begin a step until I say so. After each step write results/stepNN/REPORT.md and print its contents.
3. Never modify files in data/raw/. Treat them as read-only (chmod a-w).
4. No leakage, ever:
   - the label (DXSUM.DIAGNOSIS) is never an input feature;
   - normalisation, feature filtering, class/pattern priors, weights and early stopping are fitted on the TRAINING part of each fold only;
   - splits are by subject (RID); the same RID must never appear in two of train/val/test.
5. Do not loosen a test, threshold or tolerance to make it pass. If a check fails, report the failure and your diagnosis, and stop.
6. Determinism: fixed seeds (python, numpy, torch), seeds stored in configs. Log library versions and git commit in every REPORT.md.
7. Report only what was actually run. Never invent, estimate or round up a result. If you did not run it, say "not run".
8. Keep all configuration in configs/*.yaml; no magic numbers in code.
9. Data privacy: ADNI data stays on this machine. Do not upload it to any external service.
10. Commit to git at the end of every step with message "stepNN: <title>".
11. Keep code simple and typed; every module gets unit tests in tests/ and `pytest -q` must pass before a step is declared done.
```

------------------------------------------------------------------------

## Step 0 — Project, environment, hardware

**Prompt to agent**

```text
Step 0. Create the project and environment.

1. Create the directory adni-shortcut/ with: data/raw/, src/adni/, scripts/, tests/, configs/, results/, logs/.
2. Copy the six ADNI CSVs into data/raw/ and make them read-only. Copy PIPELINE_v2.md, adni_stage1_build.py,
   adni_stage3_profile.py, episodes.csv, joint_profile.csv, feature_spec.json into the project (scripts/ for the .py files, docs/ for the rest).
3. Create a virtual environment with Python 3.11 or 3.12:  python -m venv .venv ; activate it.
4. Install: numpy pandas scipy scikit-learn torch lightgbm statsmodels matplotlib pyyaml tqdm joblib pytest pyarrow.
   Use the CPU or CUDA build of torch appropriate to this machine. Then run `pip freeze > requirements.lock.txt`.
5. git init; add .gitignore (.venv, __pycache__, results/*/artifacts, *.pt); first commit.
6. Write scripts/env_report.py that prints: OS, python version, CPU count, RAM, GPU name/VRAM if any, torch version,
   torch.cuda.is_available(), and SHA256 of every file in data/raw/.
7. Run `pytest -q` (there are no tests yet; it should report "no tests ran" without error) and run env_report.py.
Write results/step00/REPORT.md with the env_report output and the first 20 lines of `pip freeze`.
```

**Checks:** venv active; torch imports; raw files read-only; hashes printed.

**Paste back to Claude:** `env_report.py` output, `python --version`, the `pip freeze` head, and whether a GPU is present.

------------------------------------------------------------------------

## Step 1 — Reproduce Stage 1 (episodes)

**Prompt to agent**

```text
Step 1. Reproduce the Stage 1 episode table.

1. Run scripts/adni_stage1_build.py with --uploads data/raw --out results/step01/stage1 --window 180 --write-blocks.
   (If a file name differs from the defaults, report it; do not rename raw files.)
2. Compare results/step01/stage1/episodes.csv with the provided episodes.csv: write a test tests/test_stage1_repro.py that asserts
   the two files are IDENTICAL (same rows, columns, values).
3. Write tests/test_stage1_invariants.py asserting on the produced episodes.csv:
   - every row has n_mod >= 1 and pattern equals the concatenation of m_mri, m_fdg, m_amy, m_tau;
   - for each modality with m_k == 1 the scan date exists and |gap_days| <= 180; for m_k == 0 the gap is NaN or not used;
   - the label column y is in {0,1,2} and consistent with DIAGNOSIS (CN=0, MCI=1, Dementia=2);
   - no duplicated (RID, dx_date).
4. Load each block in results/step01/stage1/blocks/*.csv and report, per block: shape, number of feature columns, fraction of
   rows with any NaN, mean per-feature NaN fraction, and the 5 features with the highest NaN fraction. Verify that every
   episode_id with m_k == 1 appears in block k exactly once and no episode with m_k == 0 appears.
5. Print a class x pattern table and write results/step01/REPORT.md.
```

**Expected values (must match exactly)**

| Item | Expected |
|---|---|
| Episodes / subjects | 13,240 / 3,145 |
| CN / MCI / Dementia | 5,263 / 5,518 / 2,459 |
| Patterns present | 15 (no `0000`) |
| MRI / FDG / amy / tau present | 12,310 / 5,063 / 4,621 / 2,827 |
| Features per block | MRI 256 (255 + field strength), FDG 6, amy 164, tau 162 |
| Episodes dropped by the >50 %-missing rule | 69 |
| Median \|gap\| MRI / FDG / amy / tau | 12 / 16 / 23 / 25 days |
| Pattern `1000` / `1100` / `1110` / `1111` | 4,954 / 2,123 / 2,123 / 234 |

**Paste back to Claude:** REPORT.md, the pytest summary line, the class × pattern table, and any file-name deviations.

------------------------------------------------------------------------

## Step 2 — Reproduce Stage 3 (profiling)

**Prompt to agent**

```text
Step 2. Reproduce the Stage 3 joint profiling on the Step 1 episodes.

1. Run scripts/adni_stage3_profile.py --episodes results/step01/stage1/episodes.csv --out results/step02/joint_profile.csv
   --metrics-json results/step02/metrics.json. Save the full stdout to results/step02/stdout.txt.
2. Write tests/test_stage3_repro.py asserting results/step02/joint_profile.csv matches the provided joint_profile.csv
   (numeric columns equal within 1e-9).
3. Independently, in a NEW module src/adni/priors.py, implement: smoothed_joint(n_cm, alpha), pmi(P), independence_weight(P)
   (= exp(-PMI), clipped to [wmin, wmax] and renormalised to mean 1 over the training samples). Write tests that check:
   - the smoothed joint sums to 1;
   - sum over cells of P(c,m) * exp(-PMI) equals 1 (unclipped);
   - the unclipped weighted joint equals P(c) P(m) in every cell (tolerance 1e-12);
   - priors.py reproduces the PMI column of joint_profile.csv within 1e-9.
4. Write results/step02/REPORT.md.
```

**Expected values**

| Item | Expected |
|---|---|
| Cramér's V (class × pattern) | 0.238, CI [0.225, 0.258] (the CI can differ in the third decimal) |
| Pattern-only classifier macro-AUROC / balanced accuracy | 0.618 / 0.412 |
| Share of `I(C;M)` surviving conditioning on `ORIGPROT` / `COLPROT` | 91 % / 80 % |
| Weights: raw range, ESS/N raw → clipped | 0.53–31.6, 0.739 → 0.878 |
| V after clipped reweighting | 0.055 |
| Cells with n < 10 / empty cells | 6 / 1 |

**Paste back to Claude:** `stdout.txt`, the pytest summary, and REPORT.md.

------------------------------------------------------------------------

## Step 3 — Splits

**Prompt to agent**

```text
Step 3. Build and freeze the data splits (PIPELINE_v2 section 7).

1. Create src/adni/splits.py and scripts/make_splits.py. Produce results/step03/splits.json for:
   S1: StratifiedGroupKFold(n_splits=5, shuffle=True) over subjects (group = RID), stratified on episode label y,
       repeated for seeds [0, 1, 2]. Inside each training fold, hold out 10% of the TRAINING subjects as validation
       (grouped by RID, stratified on y, seed-fixed). Store episode_ids for train/val/test of every (seed, fold).
   S2: protocol shift. Train+val subjects = those whose first enrolment protocol ORIGPROT is ADNI1, ADNIGO or ADNI2;
       test subjects = ORIGPROT in {ADNI3, ADNI4}. Validation = 10% of the training subjects (grouped, stratified).
2. Tests (tests/test_splits.py), for every split: train/val/test are pairwise disjoint on RID AND on episode_id; their union is
   all episodes (S1) ; every RID is in exactly one test fold for each seed (S1); S2 has 1,784 training-side and 1,361 test-side subjects
   and zero overlap.
3. REPORT: for every (seed, fold) print: n subjects and n episodes in train/val/test, class proportions, and the number of test
   episodes per pattern. Highlight any fold where a pattern has < 5 test episodes.
```

**Checks:** all disjointness tests pass; S2 subject counts exactly 1,784 / 1,361.

**Paste back to Claude:** the REPORT tables for seed 0 (all 5 folds) and S2, and the pytest summary.

------------------------------------------------------------------------

## Step 4 — Stage 2 preprocessing (fit on train only)

**Prompt to agent**

```text
Step 4. Implement the Stage 2 preprocessing as a fit/transform object (src/adni/preprocess.py).

Spec (PIPELINE_v2 section 3):
- Inputs: the four block tables from Step 1 aligned to episode_id, and the feature list in feature_spec.json.
- fit(train_episode_ids): for each block, using ONLY the training episodes where that modality is present:
    * drop features with > 20% missing; store the kept list;
    * compute mean and SD of the remaining features (ignoring NaN); SD floor 1e-6; clip z-scores to [-5, 5];
- transform(episode_ids): return for each block a float32 array [N, 2*d_k] = [z with NaN->0 ; missing-indicator]
  plus the block-level presence mask m_k from episodes.csv. Rows of absent modalities must be all-zero and flagged absent.
- Save the fitted object per (seed, fold) with joblib.

Tests (tests/test_preprocess.py):
 1. No NaN/inf in any output.
 2. Fit on the training ids only: refit after replacing all validation/test rows by random noise -> identical fitted statistics.
 3. Absent modalities give all-zero rows and mask 0.
 4. The missing-indicator equals the original NaN pattern on kept features.
 5. Train z-scores have mean ~0 and SD ~1 (tolerance 1e-3) for features with no clipping effects.

Run it for S1 seed 0 fold 0 and report: per block, kept-feature count, number of dropped features and their names,
fraction of z-scores clipped, and the distribution (min / median / max) of the per-row missing-indicator rate.
```

**Checks:** all five tests pass; kept counts close to 256 / 6 / 164 / 162 minus a few high-missing features.

**Paste back to Claude:** REPORT.md (kept/dropped features, clip fraction) and the pytest summary.

------------------------------------------------------------------------

## Step 5 — Metrics library

**Prompt to agent**

```text
Step 5. Implement src/adni/metrics.py (PIPELINE_v2 section 8) with unit tests. No models yet.

Functions:
 - natural_metrics(y, proba): macro-AUROC (one-vs-rest), macro-F1, balanced accuracy, per-class F1, Brier score, ECE (15 equal-width bins on max prob).
 - pn_weights(y, pattern, prior_table, rho='uniform', wmin=0.2, wmax=5): w_i = 1/(|C| * pi_hat(c_i | m_i)) clipped and normalised to mean 1,
   where pi_hat is the smoothed P(c|m) estimated on the TRAINING split only.
 - pn_balanced_accuracy(y, pred, weights): weighted balanced accuracy under w.
 - cluster_bootstrap(metric_fn, y, proba, groups, n=2000, seed): resample SUBJECTS (RID) with replacement, return point estimate and 95% percentile CI.
 - paired_cluster_bootstrap(metric_fn, y, proba_a, proba_b, groups, n=2000): CI for metric(a) - metric(b).
 - holm(pvalues): Holm-Bonferroni adjusted p-values (statsmodels is fine).
 - stratified_report(y, proba, strata, min_n=30): metrics per stratum, pooling strata with n < min_n into 'other'.

Tests (tests/test_metrics.py): perfect predictions give AUROC 1 and balanced accuracy 1; random predictions on balanced data give AUROC ~0.5
(tolerance 0.03 with n=20000); a hand-computed 6-sample example matches; ECE of a perfectly calibrated synthetic sample is < 0.02;
cluster bootstrap on a synthetic dataset with known SD gives a CI width within 15% of the analytic one; pn_weights make a synthetic
class/pattern-dependent dataset class-balanced within each pattern (weighted class proportions equal within 1e-6 for unclipped weights).
```

**Paste back to Claude:** the pytest summary and the output of the synthetic checks.

------------------------------------------------------------------------

## Step 6 — Reference baselines B0, B1, B2

**Prompt to agent**

```text
Step 6. Run the non-neural baselines on S1 (3 seeds x 5 folds) using ONLY the Step 3 splits and the Step 4 preprocessing.

B0: multinomial logistic regression on the one-hot pattern only (no imaging values).
B1: LightGBM on the concatenation of all four blocks (use the RAW features, NaN left as NaN, plus the 4 mask bits; do not z-score).
    Hyperparameters: fixed small config in configs/b1.yaml (num_leaves 31, learning_rate 0.05, n_estimators up to 1000 with early
    stopping on the validation set, feature_fraction 0.5, bagging 0.8, class_weight none, objective multiclass).
B2: LightGBM per single block (MRI-only, FDG-only, amyloid-only, tau-only), trained and evaluated on episodes where that modality is present.

For every (seed, fold) save out-of-fold predictions (episode_id, fold, seed, arm, p_CN, p_MCI, p_Dementia) to results/step06/oof_<arm>.parquet.

Report, pooling OOF predictions across folds within each seed and then averaging over seeds:
 - natural metrics (macro-AUROC, macro-F1, balanced accuracy, per-class F1) with subject-cluster bootstrap CIs;
 - the same, split by number of observed modalities (1/2/3/4) and by pattern (patterns with n >= 30 only);
 - PN balanced accuracy using the pattern-neutral weights from Step 5.
Also report B1 feature importance (top 15 by gain) and wall-clock time.
```

**Sanity bounds (red flags, not targets):** B0 macro-AUROC should be about 0.62. Any model with macro-AUROC > 0.97 suggests leakage; an MRI-only model below 0.65 suggests a bug.

**Paste back to Claude:** the main results table, the by-modality-count table, the importance list, and the pytest summary.

------------------------------------------------------------------------

## Step 7 — Neural model A0 (masked-attention fusion) and its unit tests

**Prompt to agent**

```text
Step 7. Implement the neural model (PIPELINE_v2 sections 3.3, 5) as src/adni/models.py and a trainer in src/adni/train.py.

Model:
 - Encoder f_k per modality: LayerNorm -> Linear(2*d_k, 256) -> GELU -> Dropout(0.2) -> Linear(256, 128) -> LayerNorm.
 - Tokens: t_k = h_k + type_embedding_k; [CLS] learnable token. Absent modalities are OMITTED from attention using a key padding mask
   (do not feed them as zeros or substitute a missing token).
 - 2 pre-LN Transformer layers (d=128, 4 heads, FFN 256, dropout 0.1); head: Linear(128, 3) on the [CLS] output. Return logits and h.
 - Modality dropout in training only: drop each observed modality with p=0.25 independently, resampling when none remain. Return the effective mask.
Trainer: AdamW (lr 3e-4, weight decay 0.05), batch 256, max 100 epochs, early stopping patience 15 on validation macro-F1 (the plain
 metric for now), seed-fixed, deterministic where possible. Loss for this step: plain cross-entropy (this is arm A0).

Unit tests (tests/test_models.py):
 1. Absent-modality invariance: changing the input values of an ABSENT modality (mask 0) leaves the logits bit-identical.
 2. Present-modality sensitivity: changing a present modality's values changes the logits.
 3. Token-order independence of the attention output given identical type embeddings is NOT required; instead check the model works
    for all 15 patterns without error and with batch size 1.
 4. Overfit test: on a fixed batch of 64 episodes the loss falls below 0.05 within 300 steps.
 5. Determinism: two runs with the same seed give identical validation loss at epoch 3 (CPU), or within 1e-5 on GPU.
 6. Gradients reach every encoder when all modalities are present.
 7. Modality dropout never returns an empty mask and changes the mask distribution as expected (empirical keep rate within 0.02 of 0.75
    for a pattern with all four modalities, after conditioning on non-empty).

Then train A0 on S1 seed 0 fold 0 ONLY (pilot). Save per-epoch train loss, val loss, val macro-F1 and val AUROC to a CSV and plot them.
Report parameter count, epochs run, best epoch, wall-clock time per epoch, and the test metrics for that fold (natural metrics + per-modality-count).
```

**Paste back to Claude:** the pytest summary, the training-curve CSV (head and tail), parameter count, and fold-0 test metrics.

------------------------------------------------------------------------

## Step 8 — Joint priors and the loss arms (A1–A4)

**Prompt to agent**

```text
Step 8. Implement the mask-process prior and loss arms (PIPELINE_v2 section 6; derivations in Appendix B).

In src/adni/priors.py add:
 - submask_kernel(m, p_drop): the exact distribution K(m_tilde | m) over non-empty sub-masks of m under independent dropping with probability
   p_drop, conditioned on non-empty. Return dict {m_tilde: prob}; probabilities sum to 1.
 - pi_table(P_cm, p_drop): pi(c | m_tilde) proportional to sum_m P(c,m) K(m_tilde | m), for every m_tilde that can occur. P_cm is the smoothed joint from the TRAINING split.
 - class_prior(P_cm) and pattern_prior(P_cm).

In src/adni/losses.py add, all sharing the same model and trainer, selected by config:
 A0: CE.
 A1: logit-adjusted CE with log P(c) (class only).
 A2: CE with sample weights 1/P(m_tilde)-style pattern balancing (normalised to mean 1; clip [0.2, 5]).
 A3: CE with joint-reweighting weights w = rho(c)/pi(c | m_tilde) (rho uniform), clipped [0.2, 5], normalised to mean 1.
 A4: logit-adjusted CE: softmax(s_c + tau * log pi(c | m_tilde)), tau = 1. Prediction uses argmax_c (s_c + log rho(c)); implement rho=uniform and rho=P(c).

Unit tests (tests/test_priors_losses.py):
 1. submask_kernel sums to 1 for all 15 patterns; with p_drop=0 it is a point mass on m.
 2. pi_table with p_drop=0 equals the smoothed P(c|m) from Step 2 within 1e-12.
 3. Monte-Carlo check: sample 200,000 dropout draws on the training set and compare the empirical P(c | m_tilde) with pi_table; max abs difference < 0.01 for sub-masks with >= 500 samples.
 4. A4 with tau = 0 reproduces A0's loss exactly on a fixed batch.
 5. For synthetic logits that equal -log pi(c|m_tilde) + constant, the A4 adjusted logits are constant across classes (the loss equals log 3 for all samples).
 6. No test or validation episode influences pi_table (refit after perturbing val/test labels -> identical table).

Then train A0..A4 on S1 seed 0 fold 0 only (pilot). Report for each arm: natural metrics, PN balanced accuracy (both rho=uniform and rho=P(c) for A4),
class-wise F1, and a per-pattern table for patterns with >= 30 test episodes. Also report the mean absolute change of predicted-class probabilities
when ONE observed modality is removed at test time, per arm (a first look at mask sensitivity).
```

**Paste back to Claude:** the pytest summary, the pilot comparison table, the per-pattern table, and the mask-sensitivity numbers.

------------------------------------------------------------------------

## Step 9 — Full S1 runs (A0–A4) and statistics

**Prompt to agent**

```text
Step 9. Run the main experiment. Arms A0, A1, A2, A3, A4 on S1: seeds [0,1,2] x folds 0-4 (75 runs). Also re-run B0, B1 with the same folds if not already complete.
Use identical hyperparameters across arms. Early stopping on validation macro-F1 for all arms (the same rule; do not tune per arm).

Save OOF predictions for every run to results/step09/oof_<arm>.parquet (episode_id, seed, fold, arm, logits, probs, effective mask = natural mask).
Write scripts/analyse_step09.py that:
 1. Averages predictions over seeds per episode (report also the single-seed spread).
 2. Computes natural metrics and PN balanced accuracy (rho uniform) per arm with subject-cluster bootstrap CIs (2,000 resamples).
 3. Paired comparisons A4 - {A0, A1, A2, A3} for PN balanced accuracy (primary) and natural macro-F1 (non-inferiority margin 0.02), with
    paired cluster-bootstrap CIs and Holm-adjusted p-values for the four primary comparisons.
 4. Per-pattern, per-modality-count, per-training-cell-count-tertile and per-protocol breakdowns for A0 and A4.
 5. Writes the main results table as markdown and CSV.
Do NOT change any design decision based on the test results. If a bug is found, fix it, document it, and rerun the affected arms from scratch.
```

**Paste back to Claude:** the main table (all arms, with CIs), the paired-comparison table, runtime, and any failed or restarted runs.

------------------------------------------------------------------------

## Step 10 — Diagnostics and controls (tests of H2 and the placebo)

**Prompt to agent**

```text
Step 10. Implement and run the diagnostics in PIPELINE_v2 section 8.4 and the controls C1, C2, plus the protocol-shift split S2.

 1. Counterfactual mask intervention (A0 and A4, S1 OOF models; you must save model checkpoints per fold or re-run inference):
    for test episodes with >= 2 observed modalities, (a) remove each one observed modality in turn and record the mean total-variation change in predicted
    probabilities; (b) as a content control, REPLACE that modality's values with those of a randomly chosen other test subject who has the same
    modality (class-agnostic), keeping the mask, and record the same change. Report the ratio (a)/(b) per modality and arm.
 2. Embedding swap: only for arms with pattern conditioning (A5 later); implement the function and test it, run it in Step 11.
 3. Placebo C1: train A4 where pi is estimated from labels PERMUTED WITHIN ORIGPROT (training split only). Run on S1 seed 0..2 x folds 0-4.
 4. C2: A4 with the effective-number factor on the cell counts (beta = 0.999), same grid.
 5. A9: A4's pipeline with sample weight sigmoid(-I) (v1 rarity) as an A3-style reweighting.
 6. S2 protocol shift: train A0..A4 and B1 on the S2 training side (3 seeds) and test on the ADNI3/4-enrolled subjects.
 7. Report PN balanced accuracy, natural macro-F1, macro-AUROC with cluster-bootstrap CIs for each.
```

**Paste back to Claude:** the intervention table, the placebo/C2/A9 comparison against A0 and A4, and the S2 results table.

------------------------------------------------------------------------

## Step 11 — Exploratory modules and sensitivity

**Prompt to agent**

```text
Step 11. Run the exploratory arms and sensitivity analyses. These do not change the primary conclusions; report them even if null.

 1. A5: A4 + pattern conditioning (FiLM from the 16-d pattern embedding). A6: A5 + cross-view consistency (lambda_c = 0.1; two dropout views per episode, stop-gradient on view 2).
    A7: A6 + pattern-aware MoE (K = 4, router input [h_pool, e_m, m]; no label-dependent input). A8: A6 + class-dependent prototypes (v1 gated form).
    Run on S1 seeds [0,1,2] x folds 0-4. Run the embedding-swap test on A5-A8.
 2. Sensitivity for A4: tau in {0.5, 1.0, 1.5}; weight clip in {3, 5, 10} for A3; p_drop in {0, 0.25, 0.5}; window +/-90 d and +/-365 d (rebuild Stage 1, regenerate the splits for that window, keep all else fixed).
 3. Summary-only amyloid fallback: add a variant where the 325 FBP scans with only summary values are kept using centiloids + summary SUVR only.
 4. Ablation ladder: leave-one-out from the best arm among A4-A8 chosen BY VALIDATION METRIC (not test).
Produce the final ablation table and the sensitivity plots (tau, clip, p_drop) with CIs.
```

**Paste back to Claude:** the ablation table, the embedding-swap results and the three sensitivity tables.

------------------------------------------------------------------------

## Step 12 — Paper assets

**Prompt to agent**

```text
Step 12. Produce the paper assets, using only numbers that exist in results/. Every number in a table or caption must be generated by a script that reads results files (no hand-typed numbers).

Tables: 1 cohort and pattern distribution; 2 shortcut quantification (V, pattern-only AUROC, conditional MI shares); 3 main results (B0-B3, A0-A4; natural and PN metrics);
4 ablation ladder A0-A9, C1, C2; 5 per-pattern results; 6 protocol-shift (S2).
Figures: 1 pipeline diagram; 2 heat-map of I(c,m) annotated with n (cells n<10 hatched); 3 pattern x protocol availability; 4 mask-intervention results; 5 sensitivity to tau, clip, p_drop.
Write scripts/make_paper_assets.py that regenerates all of them, a results/MANIFEST.md listing each asset with its source file, and a README.md explaining how to reproduce the entire project from a clean checkout.
```

**Paste back to Claude:** MANIFEST.md and the tables as text; I will check them against the specification and the earlier steps.

------------------------------------------------------------------------

## Report template (agent should end every step with this)

```text
STEP NN REPORT
Command(s) run:
Git commit:
Tests: <pytest summary line>
Expected vs actual (table):
Deviations from the spec (if any, with reason):
Things I am unsure about:
Files written:
```

## If a step fails

Paste: the failing command, the **full** traceback or test output, the file and line, and what the agent changed since the last passing state. I will diagnose before you let the agent continue.
