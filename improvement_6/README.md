# Improvement workstream 6 — PIPELINE v6

Joint class × modality-availability imbalance in ADNI.

Research question: how do class imbalance and modality-availability imbalance interact,
when does their combination hurt a classifier, and do cell-aware methods help.

Structure: Measure (Part A, natural data) -> Isolate (Part B, imbalance dial) -> Fix
(Part C, methods along the dial, natural + phase-held-out).

Exploratory companion to the frozen step sequence; does not modify v2/v3 artefacts.

Layout:
  configs/v6.yaml        single YAML config driving everything
  requirements.txt       pinned environment
  README.md              this file
  run_all.sh             reproducible commands
  results/
    runs.jsonl           one line per run (config, seed, fold, metrics)
    cohorts/             FULL / BASELINE / DIAL episode_id lists
    tables/              every table in PIPELINE_v6.txt sections 6-8
    figures/             F1-F6 PNGs
    dial/                dial grid raw results
  report/
    REPORT.md            Introduction, Data, Methods, Experiments, Results, Discussion,
                         Limitations, Findings (H1-H4)
  src/
    __init__.py
    data.py              loaders, audits, cohort builders, dial design distributions
    splits.py            S1 reuse, StratifiedGroupKFold for BASELINE/DIAL, val carve,
                         phase-held-out split
    preprocess.py        per-modality drop>50% missing, median impute, z-score, clip[-5,5],
                         missing-modality encoding, hide_mask option
    models.py            LGBM, MLP late fusion, M0 logreg
    methods.py           M1-M5: class weights, joint reweighting, M4 cell logit adj,
                         M4c, M5 group DRO
    metrics.py           overall + macro cell accuracy + bottom-3 cell + T_matched-T_indep
                         gap + natural-vs-balanced gap + ECE + subject-cluster bootstrap
    dial.py              dial grid runner: lambda x keep_frac, train masking,
                         T_matched/T_indep/T_full, pattern-response matrices
    run.py               runner entry point (reads configs/v6.yaml)
