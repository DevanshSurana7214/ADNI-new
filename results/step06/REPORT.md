STEP 6 REPORT
Reference baselines B0, B1, B2 on S1 (3 seeds x 5 folds). PIPELINE_v2 §9; playbook Step 6.

Command(s) run:
  PYTHONPATH=src ./.venv/Scripts/python.exe -u scripts/run_baselines.py
  PYTHONPATH=src ./.venv/Scripts/python.exe -m pytest -q

Git commit: (pending — step06 not yet committed)

Tests: 53 passed in 35.44s   (tests/test_baselines.py 8 new + 1 new in test_metrics.py, plus the prior 44)

Environment: Python 3.12.10, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1,
lightgbm 4.7.0, pyarrow 25.0.1, pyyaml 6.0.3. Windows 11. CUDA torch not used here.

What I did
  - src/adni/baselines.py — the three non-neural arms:
      * B0: multinomial logistic regression (max_iter 2000, C=1.0) on the four availability bits,
        i.e. the one-hot of the pattern and nothing else (shortcut lower bound, H1).
      * B1: LightGBM on the concatenation of all four blocks using the RAW features (NaN left as NaN,
        trees split on missing natively) plus the four block mask bits; every column is prefixed with
        its block name because amy and tau share SUVR names.
      * B2: the same LightGBM applied to one block at a time (mri / fdg / amy / tau), trained and
        evaluated only on episodes where that modality is present.
      Each arm is fitted on a Step 3 training split and early-stops on the corresponding validation
      split; test predictions are out-of-fold. Config in configs/b1.yaml (num_leaves 31, lr 0.05,
      feature_fraction 0.5, bagging 0.8/freq 1, class_weight none, objective multiclass,
      up to 1000 rounds with early stopping 50).
  - configs/b1.yaml — the fixed, shared LightGBM config (playbook values).
  - scripts/run_baselines.py — runs all arms over S1 seeds [0,1,2] x folds [0..4], writes one OOF file
    per arm, and produces the metrics/CIs/stratified tables/importance/timing in summary.json.
  - tests/test_baselines.py (8 tests) — config matches the playbook, pattern one-hot, block prefix +
    absent-modality NaN + no label leak, label lookup by episode_id, B0 simplex/class order, B2
    present-only restriction, B1 simplex, prior table.

Out-of-fold predictions written (one file per arm; columns episode_id, fold, seed, arm, p_CN, p_MCI,
p_Dementia):
  results/step06/oof_b0.parquet      39,720 rows (3 seeds x 13,240 episodes)
  results/step06/oof_b1.parquet      39,720 rows
  results/step06/oof_b2_mri.parquet  36,930 rows (3 x 12,310 present)
  results/step06/oof_b2_fdg.parquet  15,189 rows (3 x 5,063)
  results/step06/oof_b2_amy.parquet  13,863 rows (3 x 4,621)
  results/step06/oof_b2_tau.parquet   8,481 rows (3 x 2,827)
  results/step06/summary.json, results/step06/natural_metrics.csv

Main results
  Metric protocol (§8.5): within each seed the out-of-fold predictions of the 5 folds are pooled
  (every episode predicted once); a metric is computed per seed and averaged over the 3 seeds. The
  reported 95 % CI is a subject-cluster (RID) bootstrap with 2,000 resamples run per seed, averaged
  over seeds. All metrics are on the natural test distribution except the PN columns.

  arm      macroAUROC             macroF1                BAcc                  PN-BAcc (uniform)      PN-BAcc (P(c))
  b0       0.6024 [0.5900,0.6144] 0.3544 [0.3449,0.3641] 0.4026 [0.3940,0.4111] 0.3340 [0.3253,0.3430] 0.3334
  b1       0.8041 [0.7936,0.8140] 0.6309 [0.6174,0.6444] 0.6241 [0.6101,0.6381] 0.5851 [0.5701,0.6001] 0.5858
  b2_mri   0.7599 [0.7483,0.7713] 0.5791 [0.5645,0.5936] 0.5739 [0.5590,0.5890] 0.5632 [0.5474,0.5789] 0.5626
  b2_fdg   0.7230 [0.7070,0.7388] 0.5206 [0.4997,0.5419] 0.5090 [0.4899,0.5293] 0.5123 [0.4921,0.5335] 0.5118
  b2_amy   0.7526 [0.7355,0.7694] 0.5642 [0.5408,0.5871] 0.5541 [0.5311,0.5772] 0.5440 [0.5201,0.5679] 0.5437
  b2_tau   0.7643 [0.7417,0.7864] 0.5457 [0.5139,0.5769] 0.5317 [0.5028,0.5614] 0.5425 [0.5113,0.5732] 0.5427

  Per-class F1 (mean over seeds):
    b0      CN 0.4771  MCI 0.5861  Dementia 0.0000   (B0 never predicts Dementia under argmax)
    b1      CN 0.6843  MCI 0.5709  Dementia 0.6374
    b2_mri  CN 0.6370  MCI 0.5015  Dementia 0.5988
    b2_fdg  CN 0.3551  MCI 0.6293  Dementia 0.5772
    b2_amy  CN 0.7015  MCI 0.4799  Dementia 0.5113
    b2_tau  CN 0.7998  MCI 0.3692  Dementia 0.4681

Sanity bounds (playbook): met.
  * B0 macro-AUROC 0.6024 ≈ 0.62 -> the pattern-only shortcut is a genuine but weak signal.
  * MRI-only macro-AUROC 0.7599 > 0.65 -> no obvious MRI bug.
  * No arm approaches 0.97 -> no gross leak detected.

By number of observed modalities (macro-AUROC / macro-F1 / balanced accuracy)
  arm      nmod=1                 nmod=2                 nmod=3                 nmod=4
  b0       0.5134/0.2591/0.3479   0.6295/0.3930/0.4359   0.6146/0.4085/0.4536   0.5001/0.0427/0.3333
  b1       0.7651/0.5902/0.5846   0.8132/0.6466/0.6446   0.8361/0.6653/0.6519   0.7794/0.5186/0.5124
  b2_mri   0.7652/0.5872/0.5867   0.7648/0.5899/0.5884   0.7488/0.5547/0.5372   0.7879/0.4310/0.6043
  b2_fdg   0.7867/0.5525/0.5428   0.7285/0.5137/0.5054   0.7100/0.5188/0.5054   0.7691/0.5750/0.6090
  b2_amy   0.7058/0.5396/0.5428   0.7586/0.5902/0.5778   0.7596/0.5608/0.5489   0.6648/0.4373/0.5626
  b2_tau   0.7731/0.5893/0.5759   0.7767/0.5779/0.5557   0.7578/0.5301/0.5175   0.7123/0.3392/0.5267
  (B0 nmod=4 is the constant-pattern case -> chance AUROC; B2 rows only cover episodes where that
   modality is present, so they are not directly comparable to B1.)

By pattern (macro-AUROC / macro-F1 / balanced accuracy), patterns with n >= 30; smaller patterns pooled
into "other". Pattern string order is mri, fdg, amy, tau (1 = present).
  b0: 0001 0.4653/0.2444/0.3333 | 0010 0.4805/0.2212/0.3333 | 0011 0.4512/0.2619/0.3333 |
      0100 0.4513/0.1837/0.3333 | 0110 0.4720/0.1783/0.3333 | 1000 0.4892/0.2023/0.3333 |
      1001 0.4676/0.2719/0.3333 | 1010 0.4827/0.2165/0.3333 | 1011 0.4576/0.2854/0.3333 |
      1100 0.4675/0.2164/0.3333 | 1101 0.5108/0.2754/0.3333 | 1110 0.4822/0.2129/0.3333 |
      1111 0.5001/0.0427/0.3333 | other 0.5390/0.2055/0.4375
  b1: 0001 0.7559/0.5737/0.5477 | 0010 0.6768/0.4701/0.4547 | 0011 0.7518/0.5206/0.5062 |
      0100 0.7687/0.5976/0.5889 | 0110 0.7370/0.5985/0.6071 | 1000 0.7666/0.5906/0.5858 |
      1001 0.8134/0.6537/0.6306 | 1010 0.7663/0.6097/0.5990 | 1011 0.7786/0.5954/0.5931 |
      1100 0.7721/0.5849/0.5796 | 1101 0.7591/0.4907/0.4807 | 1110 0.7775/0.6090/0.5916 |
      1111 0.7794/0.5186/0.5124 | other 0.6015/0.4676/0.4769
  (The per-pattern breakdown for the four B2 arms is in summary.json under each arm's "by_pattern".)

B1 feature importance (top 15 by total gain, summed over all 15 fits)
  mri_ST24TA                 103,907.1
  mri_ST29SV                 102,807.2
  fdg_MetaROI_MEAN            84,473.9
  mri_ST12SV                  44,641.9
  fdg_MetaROI_MAX             40,852.3
  mri_ST88SV                  38,783.3
  mri_ST83TA                  31,157.8
  mri_ST32TA                  29,684.4
  mri_ST89SV                  22,945.4
  mri_ST30SV                  22,540.0
  mri_ST40TA                  21,073.4
  mri_ST71SV                  17,149.4
  fdg_Top50PonsVermis_MEAN    13,970.0
  mri_ST24CV                  12,828.4
  fdg_Top50PonsVermis_MAX     12,204.9

Wall-clock time (total over 15 (seed, fold) fits per arm, single CPU process)
  b0 1.0 s | b1 166.8 s | b2_mri 61.0 s | b2_fdg 4.7 s | b2_amy 20.9 s | b2_tau 14.4 s
  B1 trees per fit (early-stopped): [390, 375, 252, 423, 339, 186, 300, 492, 459, 426, 294, 201, 297,
  234, 252] (mean 328). The bootstrap/metrics stage dominates the total runtime.

Expected vs actual
  Check                                              | Expected             | Actual              | Pass
  ---------------------------------------------------|----------------------|---------------------|-----
  B0 macro-AUROC (sanity, ~0.62)                     | about 0.62           | 0.6024              | yes
  MRI-only macro-AUROC (sanity, > 0.65)              | > 0.65               | 0.7599              | yes
  No arm macro-AUROC > 0.97 (leakage red flag)       | < 0.97               | max 0.8041 (b1)     | yes
  OOF files for every (seed, fold) and arm           | 6 files              | 6 files             | yes
  All arms beat the pattern-only shortcut (AUROC)    | B1/B2 > B0           | yes                 | yes
  pytest -q                                          | all green            | 53 passed in 35.4 s | yes

Deviations from the spec (if any, with reason):
  - §8.5 and the playbook ask for subject-cluster bootstrap CIs on "pooled OOF predictions ... seeds
    averaged". I pool the 5 folds within each seed, compute the metric and a 2,000-resample subject
    (RID) cluster-bootstrap CI per seed, then average the point estimate and the CI bounds over the 3
    seeds. A single pooled-across-seeds bootstrap would resample the same episode up to 3 times; the
    per-seed-then-average route keeps the cluster structure honest. Documented in the runner.
  - PN-BAcc uses the Step 5 pn_weights with the per-fold TRAIN-only P(c|m) (alpha 0.5) and the default
    clip [0.2, 5.0]; rho='uniform' is the primary column and rho=P(c) (train class marginal) is
    reported alongside, as §8.2/§6 require.
  - Added `cluster_bootstrap_multi` to the Step 5 metrics module (and a test) so the five metrics
    share one resample per draw. This is a purely additive change; the existing single-metric
    `cluster_bootstrap` behaviour is unchanged (a test asserts the multi version reproduces it for the
    same seed). I also replaced the per-draw O(N x G) cluster-index construction in the bootstrap with
    a precomputed group index; results are identical, the runtime is not.
  - configs/b1.yaml is the fixed config required by the playbook and is the single source of the
    LightGBM hyperparameters (num_rounds and early_stopping are read from it too, so the budget is
    reproducible rather than hard-coded).
  - B0's C and max_iter are not specified by the spec; I used sklearn's defaults scaled up
    (C=1.0, max_iter=2000) with no feature scaling issues (binary inputs).

Things I am unsure about:
  - The CIs above are the mean of per-seed intervals, not an interval of the seed-averaged statistic;
    if the paper needs a single CI for the seed-averaged prediction, that is a different (also
    defensible) quantity. Flag before Step 11/12 freezes the tables.
  - B0 macro-F1 and balanced accuracy are exactly seed-invariant (0.3544, 0.4026) because B0's argmax
    class depends only on the pattern and the natural test distribution is the whole cohort for every
    seed; only the AUROC varies (probabilities differ slightly). This is expected, not a bug, but it
    makes B0's F1/BAcc CIs uninformative about seed variance.
  - The per-pattern "other" row pools several small patterns, so it is not a single pattern; the
    playbook's "patterns with n >= 30" rule is implemented that way.
  - Patients/folds: no leakage was introduced deliberately, but I have not yet run the mask-intervention
    / embedding-swap diagnostics — those are later steps (§8.4, Steps 9-10).

Files written:
  src/adni/baselines.py
  configs/b1.yaml
  scripts/run_baselines.py
  tests/test_baselines.py
  src/adni/metrics.py (additive: cluster_bootstrap_multi; faster cluster index)
  tests/test_metrics.py (additive: cluster_bootstrap_multi test)
  results/step06/oof_b0.parquet
  results/step06/oof_b1.parquet
  results/step06/oof_b2_mri.parquet
  results/step06/oof_b2_fdg.parquet
  results/step06/oof_b2_amy.parquet
  results/step06/oof_b2_tau.parquet
  results/step06/summary.json
  results/step06/natural_metrics.csv
  results/step06/REPORT.md
