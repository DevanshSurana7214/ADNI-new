#!/usr/bin/env python3
"""Build and freeze the S1 + S2 splits (PIPELINE_v2 §7), writing results/step03/splits.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from adni.splits import make_splits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ADNI subject-grouped splits (S1 + S2).")
    ap.add_argument("--episodes", type=Path,
                    default=PROJECT_ROOT / "docs" / "episodes.csv")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "results" / "step03" / "splits.json")
    a = ap.parse_args()
    blob = make_splits(a.episodes, a.out)
    n_s1 = sum(len(v) for v in blob["s1"].values())
    s2 = blob["s2"]
    print(f"Wrote {a.out}")
    print(f"S1: {n_s1} folds (3 seeds x 5 folds)")
    print(f"  S2 pre-val: train+val={s2['n_subjects_pre_val_split']['train_plus_val']} subjects, test={s2['n_subjects_pre_val_split']['test']} subjects")
    print(f"  S2 post-val: train={s2['n_subjects']['train']} val={s2['n_subjects']['val']} test={s2['n_subjects']['test']} subjects "
          f"({s2['n_episodes']['train']}/{s2['n_episodes']['val']}/{s2['n_episodes']['test']} episodes)")
    for seed_key in sorted(blob["s1"]):
        for fold_key, fd in blob["s1"][seed_key].items():
            print(f"  {seed_key}/{fold_key}: "
                  f"pre-val tr+val={fd['n_subjects_pre_val_split']['train_plus_val']} te={fd['n_subjects_pre_val_split']['test']} | "
                  f"post-val subjects tr={fd['n_subjects']['train']} val={fd['n_subjects']['val']} te={fd['n_subjects']['test']} | "
                  f"episodes tr={fd['n_episodes']['train']} val={fd['n_episodes']['val']} te={fd['n_episodes']['test']}")


if __name__ == "__main__":
    main()
