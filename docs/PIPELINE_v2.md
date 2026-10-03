# Shortcut-Aware Multimodal Diagnosis Prediction under Joint Class–Modality-Pattern Imbalance in ADNI

## Tabular-biomarker pipeline (MRI · FDG-PET · amyloid-PET · tau-PET), CN / MCI / Dementia

**Status.** Revision 2. Stages 1 and 3 are implemented and were run on the uploaded ADNI tables
(`adni_stage1_build.py`, `adni_stage3_profile.py`; outputs `episodes.csv`, `joint_profile.csv`,
`feature_spec.json`). Stages 2 and 4–6 are specified here but **not yet trained**, so this document
contains no model results. Every number below is a cohort statistic computed from the data, and each
can be regenerated with the two scripts.

------------------------------------------------------------------------

## 0. What changed from v1, and why

| # | v1 | Problem | v2 |
|---|---|---|---|
| 1 | Pretrained 3D image encoders produce 768-d frozen embeddings | The available data are derived ROI tables (FreeSurfer 7, Berkeley SUVRs), not images | Per-modality **tabular encoders** on ROI features. The image variant stays as an optional drop-in (Appendix C) |
| 2 | Router input `R_i = R(c_i, m_i)` in §5/§14/§15 | Needs the true label, which does not exist at test time | No router input depends on the label. MoE moved to an exploratory arm and uses only `m` |
| 3 | `L_total` omits `L_align`; `G_i` and the pair set `P` are undefined | The loss as written does not match the ablation ladder | Prototype/alignment/gap terms removed. One loss (§6) and one consistency term with a precise definition |
| 4 | Clinical branch takes "concurrent non-diagnostic clinical scores" | MMSE/CDR/ADAS/FAQ largely *define* the ADNI diagnosis: label leakage | Explicit forbidden-field list (§2.5). Demographics/APOE optional (not in the uploaded files) |
| 5 | `R = sigmoid(−I)` called "rarity" | It measures *association*, not rarity. A tiny cell that matches independence gets R = 0.5, and the sigmoid destroys a useful exact property | `w = exp(−I)` (exact independence reweighting) and **joint logit adjustment** with `log P(c\|m) = log P(c) + I(c,m)`. Absolute rarity is reported separately (§4.4) |
| 6 | Additive prototype `μ_c + A e_m` | The class-shared offset cancels in every class difference (v1's own finding), and the gated form was a measured null | Removed from the core. Kept as an exploratory arm with the class-dependent form |
| 7 | "Longitudinal", but episodes treated as independent; splits not stated as subject-disjoint | Repeated measures leak across splits. In this cohort 15–31 % of episodes share a scan with another episode | Cross-sectional task, **subject-grouped** splits, subject-cluster bootstrap |
| 8 | Class named AD | `DXSUM` gives "Dementia"; the AD flag `DXAD` is filled for only 1,129 of 2,996 dementia visits | Label is **Dementia** throughout |
| 9 | Pattern-availability treated as a nuisance to be balanced | In this cohort it is a measurable shortcut, and it is mostly *not* explained by protocol (§4.1) | Shortcut quantification is the paper's first result; the method targets it |
| 10 | Expert 1 specialised for pattern `1111` | `1111` had no training episodes in v1's split | Complete-pattern behaviour is evaluated, not assumed; MoE is exploratory |

------------------------------------------------------------------------

## 1. Problem statement and hypotheses

Multimodal ADNI cohorts are incomplete by design: which scans exist depends on the study phase and on
clinical selection. If the *availability pattern* `m ∈ {0,1}^4` is statistically dependent on the
diagnosis `c`, a classifier can use *which scans exist* as evidence about the label. Class imbalance and
modality imbalance are then not two separable problems; they are one joint distribution `P(c, m)`.

**Hypotheses (fixed before any model is trained).**

* **H1 (the shortcut exists).** A classifier given only `m` beats chance, with a subject-cluster bootstrap CI excluding chance.
* **H2 (standard models use it).** A masked multimodal model without correction predicts differently when scans are removed *beyond* what the lost measurements explain (§8.4), and gains from knowing `m` on the natural test distribution that vanish on the pattern-neutral distribution (§8.2).
* **H3 (joint correction helps).** Joint logit adjustment (A4) beats class-only (A1), pattern-only (A2) and joint reweighting (A3) on the pre-specified primary metric, and is non-inferior on natural-prevalence macro-F1 (margin 0.02).
* **H4 (v1 components).** Pattern embedding, consistency, MoE and prototypes add nothing detectable beyond A4. This is exploratory; report it as such.

**Primary endpoint.** Pattern-neutral balanced accuracy, PN-BAcc (§8.2), pooled over out-of-fold predictions.

------------------------------------------------------------------------

## 2. Data and cohort (Stage 1 — implemented)

### 2.1 Sources

| File | Content | Use |
|---|---|---|
| `DXSUM.csv` | Clinician diagnosis per visit | **Label** and visit date (anchor) |
| `UCSFFSX7` | FreeSurfer 7 cortical/subcortical volumes, area, thickness | MRI block |
| `UCBERKELEY_AMY_6MM` | Regional amyloid SUVR, centiloids | Amyloid block |
| `UCBERKELEY_TAU_6MM` | Regional tau SUVR | Tau block |
| `UCBERKELEYFDG_8mm` | Two FDG summary ROIs | FDG block |
| `DATADIC` | Data dictionary | Field definitions (e.g. `ST10CV` = ICV) |

### 2.2 Episode definition

One episode = one `DXSUM` diagnosis visit with a non-missing label (duplicate rows for the same
subject-date that disagree on the label are dropped). For each modality the **nearest scan within
±180 days** is attached. No scan in the window means mask bit 0. Episodes with no imaging modality are
excluded (pattern `0000`: nothing to encode). A scan row with more than 50 % of its features missing
counts as an absent modality. Sensitivity windows: ±90 d and ±365 d (about 12.7 k and 14.1 k episodes).

Tracers: **florbetapir (FBP, AV45)** and **flortaucipir (FTP, AV1451)** only, matching v1's specification. The other tracers
(FBB/NAV; MK6240/PI2620) are on different SUVR scales and are excluded from the primary analysis.
Reserved for the optional cross-tracer experiment (§9).

### 2.3 Cohort (±180 d)

| | |
|---|---|
| Episodes / subjects | **13,240 / 3,145** (median 4 episodes per subject, max 17) |
| CN / MCI / Dementia | 5,263 (39.7 %) / 5,518 (41.7 %) / 2,459 (18.6 %) |
| MRI / FDG / amyloid / tau present | 12,310 (93.0 %) / 5,063 (38.2 %) / 4,621 (34.9 %) / 2,827 (21.4 %) |
| Modalities per episode 1 / 2 / 3 / 4 | 5,533 / 4,067 / 3,406 / **234 (1.8 %)** |
| Median \|scan − diagnosis\| gap (MRI / FDG / amy / tau) | 12 / 16 / 23 / 25 days (95th pct 95 / 162 / 163 / 130) |
| Episodes sharing a scan with another episode | MRI 15 %, FDG 27 %, amyloid 31 %, tau 27 % |

The pattern distribution (all 15 non-empty patterns occur; bits are MRI·FDG·AMY·TAU):

| Pattern | CN | MCI | Dementia | Total | % |
|---|---:|---:|---:|---:|---:|
| 1000 | 1755 | 2158 | 1041 | 4954 | 37.4 |
| 1100 | 583 | 1020 | 520 | 2123 | 16.0 |
| 1110 | 734 | 996 | 393 | 2123 | 16.0 |
| 1011 | 738 | 183 | 65 | 986 | 7.4 |
| 1001 | 677 | 218 | 88 | 983 | 7.4 |
| 1010 | 303 | 250 | 77 | 630 | 4.8 |
| 0010 | 151 | 123 | 30 | 304 | 2.3 |
| 1101 | 3 | 195 | 79 | 277 | 2.1 |
| 1111 | 16 | 168 | 50 | 234 | 1.8 |
| 0011 | 121 | 47 | 19 | 187 | 1.4 |
| 0100 | 44 | 54 | 44 | 142 | 1.1 |
| 0110 | 59 | 50 | 28 | 137 | 1.0 |
| 0001 | 77 | 40 | 16 | 133 | 1.0 |
| 0111 | 2 | 11 | 7 | 20 | 0.2 |
| 0101 | 0 | 5 | 2 | 7 | 0.1 |

Of the 45 class × pattern cells, 39 have n ≥ 10, six have n < 10 and one is empty (CN × 0101). Single-modality
episodes (`1000`, `0100`, `0010`, `0001`) are valid data and are kept everywhere. The all-modality
subset is 71.8 % MCI and 6.8 % CN, against marginals of 41.7 % and 39.7 %. **A complete-case analysis
would therefore be a different, heavily MCI-skewed problem.**

### 2.4 Protocol drives availability

| Pattern | Visit protocol composition |
|---|---|
| `1111` | 100 % ADNI3 |
| `1110` | 87 % ADNI2 |
| `1100` | 78 % ADNI1 |
| `1001` | 70 % ADNI3, 30 % ADNI4 |

Tau is present in 0 % of ADNI1/GO visits, 1.8 % of ADNI2, 64 % of ADNI3 and 52 % of ADNI4 visits. Protocol is
therefore a confounder for every pattern-level comparison and is used for stratified diagnostics (§4.1),
the protocol-shift split (§7) and covariate reporting.

### 2.5 Leakage rules (binding)

* **Target:** `DXSUM.DIAGNOSIS` only. Never an input: any `DX*` field, `DXCHANGE`, `PHASES`, `AMYLOID_STATUS*` (a thresholded summary of the amyloid block, excluded to keep the block continuous), and any diagnosis-derived quantity.
* **Forbidden as inputs (they define the diagnosis):** MMSE, MOCA, CDR/CDR-SB, ADAS-Cog, FAQ, logical-memory scores, GDS-based criteria. A separate, clearly labelled *diagnosis-adjacent* analysis may include them; it is never the primary result.
* **Allowed if added:** age, sex, education, APOE ε4 count (not in the uploaded files; `PTDEMOG` and `APOERES` are needed).
* **If a biomarker-defined stage (A/T/N) is ever used as the target,** the blocks that define it must be masked, otherwise the label leaks.

------------------------------------------------------------------------

## 3. Stage 2 — Feature blocks and encoders

### 3.1 Feature blocks

| Block | Features | Preprocessing |
|---|---|---|
| MRI | 256 = 255 FreeSurfer features (cortical volume, subcortical volume, surface area, thickness mean) + field strength | Volumes ÷ ICV (`ST10CV`); thickness SD (`TS`) dropped; QC-`Fail` scans excluded |
| FDG | 6: `MetaROI` and `Top50PonsVermis` × {mean, max, SD} | none beyond z-score |
| Amyloid | 164: regional SUVR + centiloids | Constant reference column dropped (whole cerebellum ≡ 1) |
| Tau | 162: regional SUVR | Constant reference column dropped (inferior cerebellum ≡ 1) |

### 3.2 Normalisation and missing values (all fitted on the training split only)

* Per-feature z-score, winsorised at ±5 SD. Features missing in > 20 % of training rows are dropped.
* Within-block missing values are **mandatory to handle**: more than 99 % of MRI, amyloid and tau rows have at least one missing regional value. Rule: missing → 0 after z-scoring, and a binary missing-indicator is concatenated per feature (input width `2·d_k`).
* Block-level absence is the mask bit `m_k` (§2.2). The 325 FBP scans that carry only summary values (QC flag −1) are treated as absent in the primary analysis. A summary-only variant is a sensitivity analysis.

### 3.3 Encoders

```text
f_k(x_k) = LN( W2 · Dropout_0.2( GELU( W1 · [z_k ; miss_k] ) ) )      W1: 2d_k → 256,  W2: 256 → 128
h_k = f_k(x_k) ∈ R^128          k ∈ {mri, fdg, amy, tau}
```

Trained jointly with fusion. There is no frozen branch, no clinical branch in the core model, and no cached
embeddings.

------------------------------------------------------------------------

## 4. Stage 3 — Joint class–modality profiling (implemented)

### 4.1 Quantifying the shortcut (cohort-level result)

All estimates use the 13,240 episodes. Uncertainty is a bootstrap over **subjects**.

| Quantity | Value |
|---|---|
| Cramér's V (class × pattern) | **0.238**, 95 % CI [0.225, 0.258] |
| Classifier with **no imaging values**: pattern only | macro-AUROC **0.618**, balanced accuracy 0.412 (chance 0.5 / 0.333) |
| … enrolment protocol only (`ORIGPROT`) | 0.604 / 0.377 |
| … pattern + `ORIGPROT` | 0.663 / 0.417 |
| Share of `I(C;M)` surviving conditioning on protocol (permutation-null corrected) | **91 %** (`ORIGPROT`), **80 %** (visit `COLPROT`) |

Reading: pattern alone predicts the diagnosis well above chance, and most of that dependence is **not**
explained by protocol. Selection into scanning within a protocol depends on clinical status. H1 is
supported descriptively. Whether trained models exploit it is H2.

### 4.2 Estimators (training split only)

```text
P(c,m) = (n_cm + α) / (N + α·|C|·|M|),     α = 0.5          # Dirichlet smoothing
P(c) = Σ_m P(c,m),   P(m) = Σ_c P(c,m)
I(c,m) = log [ P(c,m) / (P(c)·P(m)) ]  =  log P(c|m) − log P(c)   # PMI keeps its textbook sign
```

### 4.3 What the PMI is used for (two exact results, proofs in Appendix B)

1. **Independence reweighting.** `w(c,m) = exp(−I(c,m)) = P(c)P(m)/P(c,m)` makes the weighted joint equal `P(c)P(m)`, so class and pattern become independent. This is the Kamiran–Calders "reweighing" form. In this cohort the raw weights span 0.53–31.6 (effective sample size 0.74·N). Clipping to [0.2, 5] and renormalising to mean 1 gives ESS 0.88·N and reduces V from 0.238 to 0.055.
2. **Joint logit adjustment.** `log P(c|m) = log P(c) + I(c,m)`, so the PMI *is* the pattern-specific logit offset. Training a softmax on `s_c + τ·log P(c|m)` and predicting from `s_c` removes the pattern-dependent class prior (§6).

The bounded score `R(c,m) = sigmoid(−I)` from v1 is retained **only** as a control (A9) and for
reporting. The extreme cells in this cohort: CN × `1101` (n = 3, I = −3.45, w = 31.6), CN × `1111` (n = 16, w = 5.7).

### 4.4 Absolute rarity is a separate quantity

Over the 45 cells, Spearman(cell count, `sigmoid(−I)`) = −0.26. The PMI score is not a count proxy
(v1 measured +0.09 on its smaller cohort). For rarity-stratified evaluation use **training cell counts**
(tertiles of `n_cm`), not `R`. An optional absolute-rarity factor is the Cui et al. effective number,
`E_n = (1 − β^n)/(1 − β)`, β = 0.999, tested only as its own arm.

### 4.5 Table to publish

`joint_profile.csv`: one row per (class, pattern) with `n`, `P(c,m)`, `P(c|m)`, `I`, `exp(−I)`, `sigmoid(−I)`. Show as a heat-map of `I(c,m)` annotated with `n`, with cells `n < 10` hatched.

------------------------------------------------------------------------

## 5. Stage 4–5 — Fusion model

### 5.1 Tokens, masking, modality dropout

```text
t_k = h_k + type_k                      (k present)       type_k: learnable modality-type embedding
tokens = [CLS] + { t_k : m̃_k = 1 }      absent modalities are OMITTED from attention (key-padding mask)
```

Absent modalities are removed rather than substituted by a fill-in token. The learnable-missing-token
form from v1 is retained as a baseline (B3).

**Modality dropout (training only).** Each observed modality is dropped independently with p = 0.25, resampling so at least one remains.
The kernel `K(m̃ | m)` is known exactly, which §6 uses.

### 5.2 Encoder over tokens

Two pre-LN Transformer layers (d = 128, 4 heads, FFN 256, dropout 0.1) over the tokens. The output of `[CLS]` is `h ∈ R^128`.
Prediction head: Linear(128, 3) → logits `s_c`.

### 5.3 Optional pattern conditioning (arm A5)

```text
h ← h ⊙ (1 + γ(e_m)) + β(e_m)           e_m = PatternEmbedding(m̃) ∈ R^16,  γ, β linear
```

Whether the embedding helps once the prior is corrected is an empirical question (H4).

### 5.4 Exploratory modules (arms A7, A8; reported separately)

* **Pattern-aware MoE:** K = 4 residual-MLP experts, router input `[h_pool, e_m, m]`. No label-dependent input.
* **Class-dependent prototypes:** `μ_(c,m) = μ_c + B(h(μ_c) ⊙ e_m)`. The additive class-shared form is excluded because it cancels in every class difference.

------------------------------------------------------------------------

## 6. Training objective

For a training view with effective mask `m̃` (after modality dropout):

```text
π(c | m̃) = Σ_m P(c,m) K(m̃|m) / Σ_{c',m} P(c',m) K(m̃|m)         # prior under the TRAINING mask process

L_cls = − log  exp( s_y + τ·log π(y|m̃) ) / Σ_c exp( s_c + τ·log π(c|m̃) )          τ = 1
```

Computed exactly by enumerating the ≤ 15 sub-masks of each pattern. With p_drop = 0 this is `P(c|m)`.
Using `π(c|m̃)` rather than `P(c|m)` matters: dropout makes `m̃` carry information about the natural `m`, and the
prior must be the one the model actually sees.

**Inference.** `ŷ = argmax_c ( s_c + log ρ(c) )`, with deployment prior `ρ`:

* `ρ` uniform → pattern-neutral and class-balanced operating point (primary).
* `ρ = P(c)` → pattern-neutral at natural class prevalence (reported alongside).

**Cross-view consistency (arm A6).** For an episode with ≥ 2 observed modalities, draw two independent dropout views and
set `L_cons = 1 − cos( h(m̃₁), stopgrad[ h(m̃₂) ] )`. The "compatible pairs" are exactly the two views of one episode.

```text
L_total = L_cls + λ_c · L_cons            λ_c = 0.1  (A6 onward; 0 otherwise)
```

**Optimisation.** AdamW, lr 3·10⁻⁴, weight decay 0.05, batch 256, ≤ 100 epochs, early stopping (patience 15) on the
validation PN-BAcc, identical for every arm. Five seeds per fold.

------------------------------------------------------------------------

## 7. Splits (subject-grouped everywhere)

* **S1 (primary):** 5-fold `StratifiedGroupKFold` over subjects (`RID`), repeated with 3 seeds. Within each training fold 10 % of subjects form the validation set. Normalisation, feature filtering, `P(c,m)`, `π`, weights and thresholds are fitted on the training part only.
* **S2 (protocol shift):** train on subjects first enrolled in ADNI1 / ADNI-GO / ADNI2 (1,784 subjects), test on subjects first enrolled in ADNI3 / ADNI4 (1,361). The test split is not built from the weights used in training, so it checks that gains are not an artefact of the metric.
* **Sensitivity:** window ±90 / ±365 d; FBP-only amyloid vs summary-only fallback.

------------------------------------------------------------------------

## 8. Evaluation

### 8.1 Standard metrics
Macro-AUROC (one-vs-rest), macro-F1, balanced accuracy, per-class F1, ECE (15 bins) and Brier score. Everything is reported on the natural test distribution.

### 8.2 Pattern-neutral metrics (primary)
Target distribution `Q(c,m,x) = (1/|C|)·P(m)·P(x|c,m)`: classes balanced and independent of pattern. Importance weights
`w_i = 1 / (|C| · π̂(c_i | m_i))`, with `π̂` from the training split, clipped as in training. **PN-BAcc** = weighted accuracy under `w`.
Because A3/A4 use the same estimator, PN-BAcc alone would favour them. For that reason the natural-distribution metrics, the S2 split and the placebo control (§10) are mandatory companions.

### 8.3 Stratified reporting
Per pattern (only patterns with n ≥ 30 test episodes, others pooled); by number of observed modalities (1 / 2 / 3 / 4); by
training cell-count tertile; by maximum scan gap (tertiles); by visit protocol.

### 8.4 Shortcut diagnostics (tests of H2)
* **Mask-only baseline** (§4.1) as the reference line for every pattern stratum.
* **Counterfactual mask intervention:** for test episodes with ≥ 2 modalities, remove one modality and measure the shift in predicted class probabilities; compare against the shift from replacing that modality's values with a random other subject's values from the same class-agnostic pool. A model reacting to *mask* more than to *content* is using the shortcut.
* **Embedding-swap test (A5 onward):** replace `e_m` by the embedding of a random other pattern at test time. The accuracy drop measures reliance on the explicit pattern signal. A fully mask-blind model is impossible when modalities are absent, so reliance is measured by intervention instead.

### 8.5 Statistics
Subject-cluster bootstrap (2,000 resamples) on pooled out-of-fold predictions, paired by episode. Seeds averaged before comparison.
Primary comparisons (A4 vs A0, A1, A2, A3) use Holm correction. Report effect sizes with CIs, not only p-values. Pre-specified non-inferiority margin for natural macro-F1: 0.02.

------------------------------------------------------------------------

## 9. Baselines and ablations

All neural arms share encoders, fusion, modality dropout, optimiser and selection metric. Only the loss/prior changes.

| ID | Arm | Purpose |
|---|---|---|
| B0 | Mask-only multinomial logistic regression | Shortcut lower bound (H1) |
| B1 | LightGBM / XGBoost on concatenated blocks with native NaN + mask bits | Strong tabular reference (tree models are often hard to beat on tabular data) |
| B2 | MRI-only and each unimodal model | Modality value |
| B3 | Early-fusion MLP, learnable missing tokens (v1 form) | Standard multimodal baseline |
| A0 | Masked-attention fusion, plain cross-entropy | Core model, no correction |
| A1 | A0 + class-only logit adjustment `log P(c)` | Conventional class imbalance |
| A2 | A0 + pattern-only reweighting `1/P(m)` | Modality-aware balancing |
| A3 | A0 + joint reweighting `w = ρ(c)/π(c\|m)`, clipped | Reweighing form of the joint correction |
| **A4** | **A0 + joint logit adjustment (§6)** | **Proposed core** |
| A5 | A4 + pattern conditioning (§5.3) | v1 pattern embedding |
| A6 | A5 + cross-view consistency | v1 consistency, redefined |
| A7 | A6 + pattern-aware MoE | v1 MoE (exploratory) |
| A8 | A6 + class-dependent prototypes | v1 prototypes (exploratory) |
| A9 | A4 using `sigmoid(−I)` as the sample weight | v1's rarity score as a control |
| C1 | **Placebo prior:** A4 with `π` estimated from labels permuted within enrolment protocol | If A4's gain survives, it is not due to the joint prior |
| C2 | A3/A4 with cell-count effective-number factor | Absolute rarity vs association |

Leave-one-out from the strongest arm: − joint prior, − pattern conditioning, − consistency, − modality dropout.

------------------------------------------------------------------------

## 10. Threats to validity (state these in the paper)

1. **Observational shortcut.** The class–pattern dependence arises from study design and clinical selection; removing it from the training signal does not make the test distribution unbiased.
2. **Circularity of PN-BAcc** (§8.2). Mitigated by S2, natural-distribution metrics and the placebo prior C1.
3. **Sparse cells.** Six of 45 cells have n < 10 and one is empty; weights/priors in those cells come from smoothing. Report cell-level uncertainty, never pattern-level headline numbers for n < 30.
4. **Weight variance.** Raw weights reach 31.6; clipping trades bias for variance (ESS 0.74 → 0.88·N). Report τ ∈ {0.5, 1, 1.5} and clip ∈ {3, 5, 10}.
5. **Dependence among episodes.** 4.2 episodes per subject on average; scans are reused. Handled by subject grouping and the subject bootstrap.
6. **Label noise and definition.** `DIAGNOSIS` is a clinical judgement; "Dementia" is not only AD.
7. **Missing covariates.** Age and sex are not in the uploaded tables; unadjusted age differences between classes may interact with availability. Add `PTDEMOG`/`APOERES` and run an age-adjusted sensitivity analysis.
8. **Regional values are not harmonised across tracers.** Primary analysis uses a single tracer per PET block.
9. **Scan–diagnosis gap.** Up to 180 days; stratified and varied (§8.3, §7).

------------------------------------------------------------------------

## 11. Pipeline overview

```text
 DXSUM ──► diagnosis visits (label, date) ──┐
                                            ▼
 MRI / FDG / AMY(FBP) / TAU(FTP) tables ──► nearest scan within ±180 d  ──►  EPISODES + mask m ∈ {0,1}^4   [Stage 1]
                                            │
                       train split only ────┤ z-score, missing-indicators, feature filter
                                            ▼
              f_mri   f_fdg   f_amy   f_tau   (tabular encoders → R^128)                                  [Stage 2]
                                            │
      P(c,m) → I(c,m) → π(c|m̃) , w = exp(−I)   (train split, mask-process aware)                          [Stage 3]
                                            │
        tokens (absent modalities omitted) + modality dropout (+ optional e_m FiLM)                       [Stage 4]
                                            │
                       masked Transformer over tokens → h → logits s_c                                    [Stage 5]
                                            │
        L = CE( s + τ·log π(·|m̃) ) (+ λ·L_cons)                                          [Stage 6: train]
        ŷ = argmax( s + log ρ )  — pattern-neutral prediction                                [Stage 6: infer]
                                            │
     natural + pattern-neutral metrics · per-pattern · mask-intervention · S2 shift · placebo        [Eval]
```

------------------------------------------------------------------------

## 12. Paper skeleton

**Working title.** *Which scans exist tells you the diagnosis: shortcut-aware multimodal learning under joint class–modality-pattern imbalance in ADNI.*

* **Contribution 1 (analysis):** the availability pattern is a measurable shortcut (V = 0.24; a no-imaging classifier reaches AUROC 0.62), largely independent of protocol.
* **Contribution 2 (method):** PMI-based joint prior correction: exact independence reweighting and joint logit adjustment, with its relation to the Bayes-optimal rule under a pattern-invariant prior.
* **Contribution 3 (evaluation protocol):** pattern-neutral metrics, mask-intervention diagnostics, placebo-prior control and protocol-shift split.
* **Honest negative space:** pattern embedding, MoE and prototypes are tested and reported even if null.

**Tables:** 1 cohort and pattern distribution; 2 shortcut quantification; 3 main results (A0–A4, B0–B3; natural and pattern-neutral); 4 ablation ladder; 5 per-pattern results; 6 protocol-shift.
**Figures:** 1 pipeline; 2 heat-map of `I(c,m)` with counts; 3 pattern × protocol availability; 4 mask-intervention drift; 5 sensitivity to τ and clipping.

------------------------------------------------------------------------

## Appendix A — Reproduction

```bash
python adni_stage1_build.py   --uploads <dir> --out stage1 --window 180 --write-blocks
python adni_stage3_profile.py --episodes stage1/episodes.csv --out joint_profile.csv --metrics-json metrics.json
```

Stage 1 writes `episodes.csv` (IDs, dates, label, mask, pattern, scan gaps), `feature_spec.json` and, with
`--write-blocks`, one feature table per modality. Stage 3 prints the cohort statistics quoted in §2–§4. Running Stage 3 on all
episodes is *descriptive profiling*; inside cross-validation it must be refitted per training fold.

## Appendix B — Derivations

**B1. Independence reweighting.** For cells with `P(c,m) > 0` let `w = P(c)P(m)/P(c,m)`. Then
`Σ_{c,m} P(c,m) w = Σ_{c,m} P(c)P(m) = 1` and the weighted joint is `P(c,m) w = P(c)P(m)`. ∎ (Clipping and smoothing make this approximate: V falls from 0.238 to 0.055, not to 0.)

**B2. Logit adjustment conditioned on the pattern.** The Bayes posterior is `P(c|x,m) ∝ p(x|c,m)·P(c|m)`. Suppose the model's logits satisfy
`s_c(x,m) + log P(c|m) = log P(c|x,m) + g(x,m)` at the optimum of the adjusted softmax (well-specified, τ = 1). Then
`s_c = log p(x|c,m) + g'(x,m)`, so `argmax_c s_c` is the Bayes rule under a **uniform** prior and
`argmax_c [s_c + log ρ(c)]` is Bayes under prior `ρ`, for every pattern. This is the argument of Menon et al. (2021) with `P(c)` replaced by `P(c|m)`. Because `log P(c|m) = log P(c) + I(c,m)`, the PMI is the exact pattern-specific correction on top of ordinary class-prior adjustment.

**B3. Mask-process prior.** If training uses a random sub-mask `m̃ ⊆ m` with kernel `K`, the class distribution given `m̃` is
`π(c|m̃) ∝ Σ_m P(c,m) K(m̃|m)`. Using `P(c|m̃_natural)` instead would leave a residual availability signal.

**B4. Why `μ_c + A e_m` cannot change separability.** `μ_(c₁,m) − μ_(c₂,m) = μ_c₁ − μ_c₂` for every `m`, because `A e_m` does not depend on `c` (v1 measured the gap as exactly 0).

## Appendix C — Returning to image encoders

Stage 1 (episodes, labels, mask) is unchanged. Replace Stage 2 by frozen pretrained 3D encoders giving `z_k ∈ R^768`
and `Linear(768,128)` projections. Everything from Stage 3 on needs no change. Report it as a separate variant: image encoders
and ROI-feature encoders answer different questions.

## Appendix D — References to verify before submission

Menon et al., *Long-tail learning via logit adjustment*, ICLR 2021 · Kamiran & Calders, *Data preprocessing techniques for classification without discrimination*, KAIS 2012 · Byrd & Lipton, *What is the effect of importance weighting in deep learning?*, ICML 2019 · Cui et al., *Class-balanced loss based on effective number of samples*, CVPR 2019 · Neverova et al., *ModDrop*, TPAMI 2016 · Sagawa et al., *Distributionally robust neural networks*, ICLR 2020 · Geirhos et al., *Shortcut learning in deep neural networks*, Nat. Mach. Intell. 2020 · Grinsztajn et al., *Why do tree-based models still outperform deep learning on tabular data?*, NeurIPS 2022 · Jack et al., NIA-AA research framework, Alzheimer's & Dementia 2018 · Klunk et al., *The Centiloid Project*, Alzheimer's & Dementia 2015. Check titles, venues and years against the originals; they were written from memory.
