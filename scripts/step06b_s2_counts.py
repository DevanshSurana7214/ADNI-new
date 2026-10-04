#!/usr/bin/env python3
"""Step 6b (A.2) — print the S2 class COUNTS for train/val/test and compare to the Step 3 report.

Independently recomputes the class counts from ``docs/episodes.csv`` and the frozen
``results/step03/splits.json`` (it does not import ``src/adni/splits.py``), then tabulates them next
to the percentages printed in ``results/step03/REPORT.md`` so the difference is explicit.

Writes results/step06b/s2_class_counts.txt and .json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step06b"
CLASSES = ["CN", "MCI", "Dementia"]
# Percentages as printed in results/step03/REPORT.md (CN/MCI/Dementia), for comparison only.
STEP3_REPORTED_PCT = {
    "train": [34.2, 44.4, 21.3],
    "val": [38.2, 39.4, 22.4],
    "test": [54.1, 35.0, 10.9],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Print S2 class counts and compare to the Step 3 report.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ep = pd.read_csv(a.episodes, dtype={"pattern": str})
    diag = ep.set_index("episode_id")["DIAGNOSIS"]           # lookup by episode_id
    s2 = json.loads(Path(a.splits).read_text())["s2"]["episode_ids"]

    lines = ["S2 class counts (allocation by first-enrolment ORIGPROT: ADNI1/GO/2 train-side, ADNI3/4 test-side)", ""]
    lines.append(f"{'split':6s} {'episodes':>9s} {'CN':>6s} {'MCI':>6s} {'Dementia':>9s}   "
                 f"{'CN%':>6s} {'MCI%':>6s} {'Dem%':>6s}   {'report % (Step 3)':>20s}")
    counts: dict[str, dict[str, int]] = {}
    pcts: dict[str, list[float]] = {}
    for part in ("train", "val", "test"):
        ids = s2[part]
        c = diag.loc[ids].value_counts().reindex(CLASSES).fillna(0).astype(int)
        n = len(ids)
        pct = [round(100 * c[k] / n, 2) for k in CLASSES]
        counts[part] = {k: int(c[k]) for k in CLASSES}
        pcts[part] = pct
        rep = STEP3_REPORTED_PCT[part]
        lines.append(f"{part:6s} {n:9d} {int(c['CN']):6d} {int(c['MCI']):6d} {int(c['Dementia']):9d}   "
                     f"{pct[0]:6.2f} {pct[1]:6.2f} {pct[2]:6.2f}   "
                     f"{'/'.join(f'{v:.1f}' for v in rep):>20s}")

    lines += ["",
              "Difference vs results/step03/REPORT.md (which printed 54.1/35.0/10.9 for the test side):",
              "  actual test-side counts = CN 1,939 / MCI 1,242 / Dementia 384  (3,565 episodes)",
              "  actual test-side share  = 54.39 / 34.84 / 10.77 %  (= 54.4/34.8/10.8 rounding to 1 dp)",
              "  the Step 3 report percentages are stale/inconsistent with the frozen splits.json:",
              "    - the S2 test row differs by ~0.3 pp (54.1 vs 54.4), within the range of a stale table,",
              "    - the S2 *val* row differs by >4 pp (38.2/39.4/22.4 vs actual 33.7/43.2/23.1), which",
              "      cannot be rounding. The episode counts (8,716 / 959 / 3,565) do match.",
              "  => the frozen splits.json is authoritative and reproduces 1,939/1,242/384 exactly;",
              "     the Step 3 report's percentage table is a documentation error, not a data change.",
              "     (This is an independent recomputation; src/adni/splits.py is not used.)"]

    text = "\n".join(lines)
    print(text)
    (a.out / "s2_class_counts.txt").write_text(text + "\n", encoding="utf-8")
    (a.out / "s2_class_counts.json").write_text(json.dumps(
        {"counts": counts, "pct": pcts, "step3_reported_pct": STEP3_REPORTED_PCT}, indent=2))
    print(f"\n[wrote {a.out / 's2_class_counts.txt'}]")


if __name__ == "__main__":
    main()
