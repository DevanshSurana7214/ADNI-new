# PIPELINE v3: post-hoc joint class-by-pattern correction on naturally trained content models

**Status.** Designed AFTER seeing v2 test results, so the first run is **exploratory**. It becomes confirmatory only if the rule below is re-run unchanged on fresh CV seeds (§7). Stages 0-2 are reused from v2; nothing is retrained in the first run.

## 1. What v2 established (facts, not claims)
- **H1 supported:** class and availability pattern are dependent (Cramér's V 0.238); a classifier with no imaging gets macro-AUROC 0.616.
- **H2 supported:** removing tau shifts A0's predicted CN share by -0.089, a content swap by -0.012; the training-time correction (A4) reduces the removal effect to about -0.015.
- **H3 (v2 rule) not supported:** A4 gained +0.007 to +0.015 PN-BAcc over A1/A0 but lost macro-F1 beyond the 0.02 margin.
- **H4 not supported:** pattern conditioning, consistency, MoE and prototypes added nothing (Holm-adjusted).

## 2. Why the v2 test of H3 was structurally harsh
1. **Operating-point mismatch.** A4 was scored at a uniform class prior and compared with A0's natural-prevalence predictions on natural macro-F1. Uniform priors move predictions away from the large MCI class by design. At A4's natural-prevalence point (rho = P(c)) the gap is already small (Step 9: macro-F1 0.608 vs 0.614).
2. **Pooled macro-AUROC rewards between-pattern separation**, which a pattern-neutral model must give up. The fair content-skill measure is **within-pattern AUROC**.
3. **Training-time correction interacted with unstable neural training** (best epochs 0-3 under recipe 1). A post-hoc correction on a stable model removes that interaction.

## 3. The v3 pipeline
```
Stage 0-1  episodes, masks, subject-grouped splits (unchanged from v2)
Stage 2    content models trained NATURALLY (no correction in training):
             B1  = LightGBM on raw blocks (NaN native)           [Step 6]
             A0  = masked-attention fusion, recipe R2             [Step 11a]
             E1  = average of the two CORRECTED posteriors
Stage 3    post-hoc joint correction (no training, no tuning):
             p'_c  proportional to  p_c * rho(c) / pi(c | m)^tau ,  tau = 1, alpha = 0.5
             pi(c|m): fold's TRAIN split; neural arm uses the mask-process prior (p_drop 0.25), trees use natural P(c|m)
             rho: uniform (class-balanced, pattern-neutral)  or  P(c) (natural prevalence, pattern-neutral)
Stage 4    evaluation: natural + pattern-neutral metrics + WITHIN-PATTERN macro-AUROC
Stage 5    controls: placebo correction (pi from labels permuted within ORIGPROT), S2 cohort shift, mask intervention
Stage 6    confirmation on fresh CV seeds (§7)
```
The correction is the post-hoc form of logit adjustment (Menon et al. 2021) with the joint prior `log pi(c|m) = log P(c) + I(c,m)`.

## 4. Hypotheses (v3), fixed before computing any v3 number
**H3' for model X is supported only if all three hold, under a given checkpoint rule (neural arm) / once (trees):**
1. PN-BAcc at rho uniform: corrected minus uncorrected has a CI lower bound > 0 after Holm over the 3 models.
2. Natural macro-F1 at rho = P(c): corrected minus uncorrected has a CI lower bound > -0.02.
3. Within-pattern macro-AUROC (patterns with n >= 100 and >= 10 episodes of every class, size-weighted): corrected minus uncorrected has a CI lower bound > -0.01.

**H4' (placebo):** the real correction must beat the placebo correction on PN-BAcc (CI lower bound > 0). Otherwise the gain is not specific to the joint prior.

## 5. Possible outcomes and how each is reported
- **H3' and H4' both supported:** the correction fixes the shortcut without losing natural performance; report as the main method (after §7).
- **H3' supported, H4' not:** gains come from generic re-balancing; report as class-prior adjustment.
- **H3' not supported:** report that the pattern-neutral gain costs natural performance; the contribution is H1, H2 and the evaluation protocol.

## 6. Fixed choices (no tuning)
tau 1; alpha 0.5; margins 0.02 and 0.01; patterns for within-pattern AUROC listed from the full data once; no clipping (clipped variants only as a sensitivity). Everything is written to `configs/step13.yaml` and committed before any v3 number is computed.

## 7. Confirmation (Phase 1)
If any model passes H3' in the exploratory run, re-run Stage 2 with CV seeds 3, 4, 5 (new fold partitions of the same subjects), apply the correction with the same fixed rule, and report both. This is not independent data; the paper must say so. A model that fails confirmation is reported as failed.

## 8. Limits to state in the paper
Same 3,145 subjects throughout; the v3 design was informed by v2 test results; pattern cells with n < 10 are smoothed; unadjusted for age, sex, APOE unless `PTDEMOG`/`APOERES` are added (recommended before submission).

## 9. Deliverables for the professor
Figures and tables from the professor pack, plus: before/after correction bars (PN-BAcc, natural macro-F1, within-pattern AUROC), placebo comparison, S2 results, and the mask-intervention comparison for corrected vs uncorrected A0.
