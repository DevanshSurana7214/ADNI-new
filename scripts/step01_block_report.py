#!/usr/bin/env python3
"""Step 1 item 4/5 — per-block stats, presence checks and the class x pattern table.

Reads results/step01/stage1/{episodes.csv, blocks/*.csv} and prints:
  * per block: shape, feature columns, fraction of rows with any NaN, mean per-feature NaN fraction,
    and the 5 features with the highest NaN fraction;
  * presence check: every episode with m_k == 1 appears in block k exactly once and no m_k == 0
    episode appears;
  * the class x pattern table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1 = PROJECT_ROOT / "results" / "step01" / "stage1"
MODS = ["mri", "fdg", "amy", "tau"]
CLASSES = ["CN", "MCI", "Dementia"]


def main() -> None:
    ep = pd.read_csv(STAGE1 / "episodes.csv", dtype={"pattern": str})
    print(f"episodes={len(ep)} subjects={ep.RID.nunique()} patterns={ep.pattern.nunique()}")

    print("\n=== BLOCKS ===")
    for k in MODS:
        blk = pd.read_csv(STAGE1 / "blocks" / f"{k}.csv")
        feat = [c for c in blk.columns if c != "episode_id"]
        nan_frac_rows = blk[feat := feat].isna().any(axis=1).mean() if feat else 0.0
        per_feat = blk[feat].isna().mean()
        top5 = per_feat.sort_values(ascending=False).head(5)
        print(f"\n[{k}] shape={blk.shape}  feature_cols={len(feat)}")
        print(f"  fraction of rows with any NaN = {nan_frac_rows:.4f}")
        print(f"  mean per-feature NaN fraction = {per_feat.mean():.4f}")
        print("  top-5 features by NaN fraction:")
        for name, v in top5.items():
            print(f"     {name:40s} {v:.4f}")
        # presence check
        present_ids = set(ep.loc[ep[f"m_{k}"] == 1, "episode_id"])
        absent_ids = set(ep.loc[ep[f"m_{k}"] == 0, "episode_id"])
        blk_ids = blk["episode_id"]
        counts = blk_ids.value_counts()
        dup = int((counts > 1).sum())
        in_block = set(blk_ids)
        missing = present_ids - in_block
        extra_absent = absent_ids & in_block
        extra_unknown = in_block - present_ids - absent_ids
        print(f"  presence: expected present={len(present_ids)} in_block={len(in_block)} "
              f"dup_episode_rows={dup} missing_present={len(missing)} "
              f"absent_present_in_block={len(extra_absent)} unknown={len(extra_unknown)}")
        assert dup == 0 and not missing and not extra_absent and not extra_unknown, f"{k}: presence check FAILED"

    print("\n=== CLASS x PATTERN ===")
    tab = pd.crosstab(ep.DIAGNOSIS, ep.pattern).reindex(index=CLASSES)
    tab = tab.loc[:, tab.sum(0) > 0]
    tab = tab[sorted(tab.columns)]
    tab.loc["Total"] = tab.sum(0)
    tab["Total"] = tab.sum(1)
    print(tab.to_string())
    print(f"\ncells n<10: {int((tab.iloc[:3, :-1] < 10).values.sum())}  "
          f"empty cells: {int((tab.iloc[:3, :-1] == 0).values.sum())}")

    out = PROJECT_ROOT / "results" / "step01" / "class_pattern_table.csv"
    tab.to_csv(out)
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()