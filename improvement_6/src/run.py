#!/usr/bin/env python3
"""Runner entry point for PIPELINE v6.

Reads configs/v6.yaml.  One line per run is appended to results/runs.jsonl.
Run with:
  PYTHONPATH="src;improvement_6/src" python improvement_6/run.py --part A|B|C [--ctl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "v6.yaml"
_here = Path(__file__).resolve().parent


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    import yaml as _yaml
    with open(path) as f:
        return _yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "C"], required=True)
    ap.add_argument("--ctl", action="store_true", help="checkpoint-only: print audit numbers and exit")
    args = ap.parse_args(argv)

    cfg = load_config()
    root = Path(cfg["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    (root / "dial").mkdir(exist_ok=True)
    (root / "cohorts").mkdir(exist_ok=True)

    sys.path.insert(0, str(_here))
    from data import (Episodes, audit_assertions, imbalance_tables, load_merged,
                      write_cohorts, cohort_summary, estimate_dial_design,
                      cramers_v_for_dial_design, DialDesign)
    from splits import (build_phase_held_out, build_subject_folds, load_s1_folds)

    ep = load_merged(
        Path(cfg["data"]["episodes"]),
        Path(cfg["data"]["blocks"]),
        Path(cfg["data"]["spec"]),
    )
    audit = audit_assertions(ep)
    print("CHECKPOINT 1 — DATA AUDIT")
    print("=" * 60)
    for k, v in audit.items():
        if k != "blocks":
            print(f"{k}: {v}")
    print()
    print("blocks:")
    for b in audit["blocks"]:
        print(f"  {b['modality']}: {b}")

    # cohorts
    cohorts = write_cohorts(ep, root / "cohorts",
                            dial_mods=cfg["cohorts"]["dial_mods"])
    cs = cohort_summary(ep, cohorts)
    print()
    print("COHORTS:")
    for name in ("FULL", "BASELINE", "DIAL"):
        print(name, cs[name])

    # joint imbalance tables for FULL (Part A primary is BASELINE; we compute both)
    for name in ("FULL", "BASELINE"):
        tbls = imbalance_tables(cohorts[name])
        print()
        print(f"--- joint imbalance: {name} ---")
        print("class_counts:", tbls["class_counts"])
        print("cramers_v:", round(tbls["cramers_v"], 4))
        print("mutual_information_nats:", round(tbls["mutual_information_nats"], 4))
        print("n_patterns:", tbls["pattern_distribution"].shape[0])
        pt = tbls["phase_tables"]
        print("per_phase_cramers:")
        print(pt["per_phase_cramers"].to_string(index=False))

    if args.ctl:
        print()
        print("CHECKPOINT 1 complete. Run without --ctl to continue.")
        return

    # S1 folds (FULL reuse)
    s1_folds = load_s1_folds(Path(cfg["data"]["folds"]),
                             seed_col=cfg["splits"]["s1_seed_col"])
    print()
    print("S1 folds loaded:", {k: len(v) for k, v in s1_folds.items()})

    if args.part == "A":
        print()
        print("PART A: MEASURE (natural data)")
        print("- Train M1 (LGBM and MLP) and M0 on FULL + BASELINE")
        print("- Cell analysis, error decomposition, sensitivities")
        # Placeholder: full Part A is wired by a later step.
        print("[PART A not yet wired — implement Part A runner next]")
        return

    if args.part == "B":
        print()
        print("PART B: ISOLATE (imbalance dial)")
        # Validate dial design
        design = estimate_dial_design(ep, base_mods=cfg["cohorts"]["dial_mods"])
        print()
        print("CHECKPOINT 2 — DIAL VALIDITY (Cramér's V vs lambda)")
        print("base_mods:", cfg["cohorts"]["dial_mods"])
        print("patterns (BASE_MODS):", design.patterns)
        print("P_marg:", {k: round(v, 4) for k, v in design.P_marg.items()})
        print()
        print("Cramér's V for train sets at each lambda (n_per_class=2000):")
        rows: list[dict[str, Any]] = []
        for lam in cfg["dial"]["lam_grid"]:
            cv = cramers_v_for_dial_design(design, lam, n_per_class=2000)
            rows.append({"lambda": lam, "cramers_v": round(cv, 4)})
            print(f"  lambda={lam}: Cramér's V = {cv:.4f}")
        print()
        print("Monotonic increase check:", "PASS" if all(
            rows[i]["cramers_v"] <= rows[i + 1]["cramers_v"] for i in range(len(rows) - 1)
        ) else "FAIL")
        # Real Part B runs the grid; placeholder.
        print("[PART B grid not yet run — implement after CHECKPOINT 2 approval]")
        return

    if args.part == "C":
        print()
        print("PART C: FIX AND VALIDATE ON NATURAL DATA")
        # Phase-held-out split
        pho = build_phase_held_out(
            ep.df,
            train_phases=cfg["splits"]["phase_heldout_train"],
            test_phases=cfg["splits"]["phase_heldout_test"],
            val_frac=cfg["splits"]["val_frac"],
            seed=0,
        )
        print("Phase-held-out split:")
        print("  train rids:", len(pho.train_rids), "val rids:", len(pho.val_rids),
              "test rids:", len(pho.test_rids))
        print("  train phases:", pho.train_phases)
        print("  test phases:", pho.test_phases)
        print("[PART C not yet wired — implement after Part A/B]")
        return


if __name__ == "__main__":
    main()
