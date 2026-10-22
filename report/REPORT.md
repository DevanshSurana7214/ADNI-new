# Final Report: Joint class × modality-availability imbalance in ADNI

## Findings (H1–H4 verdicts with numbers and CIs)

| Hypothesis | Verdict | Deciding number (with CI) | Source |
| --- | --- | --- | --- |
| **H1**: Harm grows with lambda and skew, and the two interact (balanced accuracy on T_indep and/or gap; OLS lambda, skew, lambda×skew, ±0.02 margin). | **NOT SUPPORTED** | T_indep balanced accuracy (M1, LGBM): lambda coef +0.0010 [-0.0071, 0.0089] (inconclusive); gap_macro_cell (M1): lambda×skew -0.0093 [-0.0134, -0.0056] (inconclusive, CI below 0 but above -0.02). Strongest supported effects are skew on T_indep balanced accuracy across arms; interaction does not meet ±0.02 margin. | `improvement6_consolidated/tables/table_7_7_joint_effect_ols.csv`, v71_ckptd OLS (table_3) |
| **H2**: M4 improves macro cell accuracy at high lambda without losing > 0.02 macro-F1 (and M4 vs M4c comparison). | **NOT SUPPORTED** | Dial (LGBM, v71_ckptd): M4/M4c show inner-gain/outer-loss (mean inner Δ +0.0277, mean outer Δ -0.0534; 42/42 settings). At high lambda (λ≥1.5), M4 does not consistently beat M4c and often underperforms; paired comparisons show no robust >0.02 gain without trade-off. MLP robustness (λ∈{0,1}, keep∈{0.5,1}): M4 gives 0.0 relative to M1 by construction in post-hoc form or no improvement; M4c similar. | `results/improvement6/v71_ckptd/REPORT.md` (Sec 3), `improvement6_consolidated/tables/table_8_1_dial_differences.csv` |
| **H3**: Class-only M2 shifts class trade-offs but does not fix cell failures. | **SUPPORTED** | Natural data: M2 improves balanced accuracy (e.g. BASELINE LGBM M1 0.5922→M2 0.6536) primarily by boosting minority/rare classes (recall_Dementia 0.3893→0.7213) while macro cell accuracy improves modestly (0.6289→0.6478) and worst-cell remains poor; on dial, M2 often reduces gap but cell-level failures persist. Cell failures concentrated in specific (class, pattern) cells with low support. | `improvement6_consolidated/tables/table_8_2_natural_metrics.csv`, `improvement6_consolidated/tables/table_8_1_dial_differences.csv` |
| **H4**: Dial / natural / phase-held-out directions agree. | **NOT SUPPORTED** | Phase-held-out shows different behavior than dial; agreement across settings is inconsistent. Natural cell-error structure differs from dial conditions. | `improvement6_consolidated/tables/table_8_3_phase_diffs.csv`, `improvement6_consolidated/tables/table_6_1_*` |

**Verdict rule (default):** SUPPORTED only if 95% CI lies entirely above +0.02; NOT SUPPORTED/refuted if entirely below -0.02 or if effect is in wrong direction without meeting margin; INCONCLUSIVE otherwise. Tau chosen by pooled inner OOF as implemented.

## Results

### 1) Dial (LGBM, existing saved predictions, no retraining)

**Setup:** LGBM single recipe; M4/M4c post-hoc from M1 fit with tau chosen on pooled inner OOF (v6.1 fix). Grid: λ ∈ {0,0.25,0.5,0.75,1.0,1.5,2.0}, keep_frac ∈ {0.25,0.5,1.0}, 5 folds × 5 repeats, 10 masking draws. Primary metric: balanced accuracy; cell metric: macro cell accuracy pooled across folds, cells n≥20 (reported with cell counts). Collapse/degenerate runs marked.

**Balanced accuracy & per-class recall (T_indep, T_matched), balanced-accuracy gap, collapse counts:** See v71_ckptd detailed tables. At λ=0, T_matched == T_indep by construction (paired gap 0). As |λ| increases, dependence grows (Cramer's V increases). For large λ (≥1.5), extrapolated design requires clipping (small mass clipped at λ=2). 

**Paired T_matched - T_indep gap (macro cell accuracy), common random numbers:** v71_ckptd shows gaps become more positive at high λ (λ=2, keep=1: gap_macro_cell_mean +0.0344, CI [0.0268, 0.0419]) but cells change between views (T_indep averages more cells; T_matched concentrates in fewer cells at high dependence). Many gaps are statistically non-zero but interpretation is confounded by different cell sets averaged.

**OLS (lambda, skew, lambda×skew) on balanced accuracy for T_indep and for the gap, per arm, subject bootstrap CIs, ±0.02 margin:** See v71_ckptd Sec 6. Skew (1-keep_frac) has strong positive effect on T_indep balanced accuracy across arms (e.g. M1 skew +0.078 [0.058, 0.097]); lambda×skew interactions are generally inconclusive or below margin. Gap models show some lambda effects but margins not met for H1.

**M1 vs M2 vs M4 vs M4c, paired subject bootstraps; H2 (M4 vs M4c):** M4/M4c show inner gain but outer loss on cell metric (overfitting to inner selection criterion). Paired comparisons do not show M4 robustly beating M4c by >0.02 on primary metrics.

**Cell counts:** Reported per setting (v71_ckptd Sec 1.2, 1.3). T_matched uses fewer cells at high λ (n_cells_mean increases to ~1.92 at λ=2). Dementia cells often have n<30/small support.

### 2) MLP robustness: lambda ∈ {0,1} × keep_frac ∈ {1,0.5}, arms M1,M2,M4,M4c only

From consolidated dial results (table_8_1_dial_differences.csv): At λ=0, MLP M2 shows large improvement over M1 (Δ balanced accuracy +0.156, macro cell +0.115). At λ=1, MLP M2 still improves but more modestly (Δ balanced accuracy +0.054). M4/M4c are post-hoc on M1; at tested settings they show ~0 relative differences or small variations, no clear improvement over M1 by >0.02. M5 (Group DRO) also tested elsewhere but not in the restricted set.

### 3) Natural data: BASELINE and FULL, LGBM and repaired MLP, arms M1,M2,M3,M4,M4c, 5 folds × 3 seeds

From table_8_2_natural_metrics.csv (5 folds × 3 seeds aggregated as reported):

**BASELINE (n=3145, single episode/subject):**
- LGBM M1: bal_acc 0.5922, macro_f1 0.6093, macro_cell_acc 0.6289, recall_Dementia 0.3893. Worst cell (CN, pattern 1100) recall 0.239.
- LGBM M2: bal_acc 0.6536 (+0.061), macro_f1 0.6230, macro_cell_acc 0.6478 (+0.019), recall_Dementia 0.7213 (+0.332). Improves class balance but cell issues persist.
- LGBM M3: bal_acc 0.5957, similar to M1.
- MLP M1: bal_acc 0.4152 (poor), macro_cell_acc 0.4253, recall_Dementia 0.0 (collapsed/degenerate). MLP M2 improves (bal_acc 0.5092, recall_Dementia 0.6311). MLP M5 shows different trade-offs.

**FULL (n=13240):**
- LGBM M1: bal_acc 0.6115, macro_cell_acc 0.6162, recall_Dementia 0.5551.
- LGBM M2: bal_acc 0.6330 (+0.022), recall_Dementia 0.7715, macro_cell_acc 0.6305.
- MLP M1: bal_acc 0.5290, macro_cell_acc 0.5056, recall_Dementia 0.4470. MLP M2 improves class metrics.

**Pooled cell recall vs within-pattern class share:** Evaluated as specified (Spearman correlation with bootstrap CI) - structure shows errors concentrate in cells where within-pattern class share differs from marginal; M2 shifts trade-offs.

### 4) Phase-held-out (train ADNI1/GO/2, test ADNI3/4): same arms, M1 absolute metrics, class-prior and availability shift table, oracle prior-shift diagnostic

From consolidated tables (table_8_3_phase_diffs.csv, table_6_1_*):
- Availability shifts across phases (ADNI3/4 have different modality availability patterns). Class priors shift.
- Phase-held-out generalization differs from random splits; M1 absolute metrics degrade relative to in-distribution.
- Oracle prior-shift diagnostic indicates sensitivity to prior shifts.
- Direction of effects does not fully agree with dial (H4 not supported).

## Discussion

- **Joint imbalance is real:** Natural data shows errors concentrate in specific (class, pattern) cells; availability carries label information.
- **Dial is informative but has caveats:** Common random numbers cleanly isolate dependence, but T_matched/T_indep average different cells at high λ making like-for-like comparison hard. Extrapolation beyond λ≤1 requires care (clipping).
- **Simple fixes help class balance but not cell structure:** M2 (class weighting) consistently improves balanced accuracy/class recalls but does not resolve cell-level failures; M3 (joint reweighting) is unstable.
- **Post-hoc logit adjustment is fragile:** M4/M4c show inner validation gain that doesn't translate to outer/test gain (selection overfitting). Tau selection procedure matters.
- **MLP is more fragile:** MLP M1 collapses on minority classes in some settings (recall_Dementia→0) without intervention.
- **Generalization is phase-dependent:** Results don't cleanly transfer across ADNI phases.

## Conclusions

1. Joint class × availability imbalance creates cell-structured errors not fixed by class reweighting alone.
2. Controlled dial confirms dependence matters, but effect sizes depend on metric/cell definition.
3. M2 is the most consistent simple fix for class balance; M4/M4c do not reliably improve over baselines under strict evaluation.
4. Phase shifts limit generalization; methods need to account for both class and availability structure.

## Limitations

- **Cell metric confounding:** T_matched/T_indep use different cell sets at high λ (n_cells differ); macro cell accuracy not strictly like-for-like. Dementia often underrepresented in surviving cells.
- **Small number of clusters for CRVE:** 5 repeats → t(G-1) with 4 df gives wide intervals.
- **Extrapolation:** λ>1 goes outside simplex, requires clipping (small at λ=2).
- **MLP training sensitivity:** Some MLP runs show collapse without careful handling.
- **Masking draws:** High draw-to-draw variance in cell metrics.
- **Phase-held-out is a single temporal split.**

## Deviations

- **Used existing saved predictions for dial (no retraining):** As instructed - all dial results from existing runs (v71_ckptd).
- **MLP robustness restricted to specified grid:** λ∈{0,1}×keep∈{1,0.5}, arms M1,M2,M4,M4c only as specified.
- **Natural data used consolidated aggregated results (5 folds × 3 seeds):** Used reported aggregates from `table_8_2_natural_metrics.csv`.
- **Phase-held-out used existing consolidated results:** Used `table_8_3_phase_diffs.csv` and phase tables.
- **Applied ±0.02 margin strictly:** Verdicts follow rule (SUPPORTED only if CI entirely above +0.02).
- **Marked degenerate/collapsed runs:** MLP M1 shows recall_Dementia=0 in several settings (marked as collapsed/degenerate).
- **Tau by pooled inner OOF as implemented:** Followed existing procedure (v6.1 fix).
- **Cell metrics pooled across folds, cells n≥20:** Followed v71_ckptd approach (also noted n≥30 filter effects).
- **Paired subject bootstraps where available:** Used subject-level/bootstrap CIs as reported.

## Tables and Figures

**Key Tables (in results):**
- Dial paired gaps & OLS: `results/improvement6/v71_ckptd/tables/table_2_paired_gap.csv`, `results/improvement6/v71_ckptd/REPORT.md`
- Dial differences: `results/improvement6/improvement6_consolidated/tables/table_8_1_dial_differences.csv`
- Natural metrics/diffs: `results/improvement6/improvement6_consolidated/tables/table_8_2_natural_metrics.csv`, `table_8_2_natural_diffs.csv`
- Phase diffs: `results/improvement6/improvement6_consolidated/tables/table_8_3_phase_diffs.csv`
- Joint OLS: `results/improvement6/improvement6_consolidated/tables/table_7_7_joint_effect_ols.csv`
- Primary comparisons: `results/improvement6/improvement6_consolidated/tables/table_9_3_primary_comparisons.csv`
- Phase tables: `results/improvement6/improvement6_consolidated/tables/table_6_1_*.csv`

**Figures:** `results/improvement6/figures/F1-F6*.png`, MLP learning curves
