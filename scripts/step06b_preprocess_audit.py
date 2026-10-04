#!/usr/bin/env python3
"""Step 6b (A.5) — fit Stage 2 preprocessing on all 15 S1 splits and on S2.

For every (seed, fold) and for S2 the Preprocessor is fitted on that split's TRAIN episode_ids only,
then applied to train/val/test. We report, per block, the min and max number of kept features across
the 16 fits, and assert that no transform output contains NaN or inf.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adni.preprocess import MODALITIES, Preprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step06b"


def _read_inputs(episodes: Path, blocks_dir: Path, spec_path: Path):
    ep = pd.read_csv(episodes, dtype={"pattern": str})
    blocks = {k: pd.read_csv(blocks_dir / f"{k}.csv").set_index("episode_id") for k in MODALITIES}
    spec = json.loads(Path(spec_path).read_text())
    return ep, blocks, spec


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit preprocessing on all 16 splits and audit transforms.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    ep_df, blocks, spec = _read_inputs(a.episodes, a.blocks, a.feature_spec)
    ep = ep_df.set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())

    fits: list[tuple[str, dict[str, int]]] = []
    nonfinite: list[str] = []

    def run(tag: str, train_ids: list[int], parts: dict[str, list[int]]) -> None:
        pp = Preprocessor(ep, blocks, spec).fit(train_ids)
        fits.append((tag, pp.kept_counts()))
        for part, ids in parts.items():
            out = pp.transform(ids)
            for k in MODALITIES:
                X, m = out[k]
                if not np.isfinite(X).all():
                    nonfinite.append(f"{tag}/{part}/{k}")

    for seed in [0, 1, 2]:
        for fold in range(5):
            node = blob["s1"][f"seed_{seed}"][str(fold)]
            ids = node["episode_ids"]
            run(f"s1_seed{seed}_fold{fold}", ids["train"],
                {"train": ids["train"], "val": ids["val"], "test": ids["test"]})

    s2 = blob["s2"]["episode_ids"]
    run("s2", s2["train"], {"train": s2["train"], "val": s2["val"], "test": s2["test"]})

    summary: dict = {"n_fits": len(fits), "kept": {}, "nonfinite_outputs": nonfinite}
    lines = [f"Stage 2 preprocessing audit - {len(fits)} fits (15 S1 + S2)"]
    lines.append(f"fit tag -> kept features per block ({', '.join(MODALITIES)})")
    for tag, kc in fits:
        lines.append(f"  {tag:20s} " + "  ".join(f"{k}={kc[k]}" for k in MODALITIES))

    for k in MODALITIES:
        vals = [kc[k] for _, kc in fits]
        summary["kept"][k] = {"min": int(min(vals)), "max": int(max(vals)),
                              "n_fits_at_min": int(sum(v == min(vals) for v in vals)),
                              "n_fits_at_max": int(sum(v == max(vals) for v in vals))}
        lines.append(f"  [{k}] kept min={min(vals)} max={max(vals)} "
                     f"(at min: {summary['kept'][k]['n_fits_at_min']} fits, "
                     f"at max: {summary['kept'][k]['n_fits_at_max']} fits)")

    lines.append("")
    if nonfinite:
        lines.append(f"FAIL: non-finite transform outputs: {nonfinite}")
    else:
        lines.append("No NaN/inf in any transform output (train/val/test x 16 fits x 4 blocks).")
    text = "\n".join(lines)
    print(text)
    (a.out / "preprocess_audit.txt").write_text(text + "\n", encoding="utf-8")
    (a.out / "preprocess_audit.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote {a.out / 'preprocess_audit.txt'}]")
    if nonfinite:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
