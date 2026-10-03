#!/usr/bin/env python3
"""Step 4 — fit Stage 2 preprocessing for a given split and report per-block diagnostics.

Default: S1 seed 0 fold 0. Fits on that fold's TRAINING episode_ids only, saves the fitted object
with joblib, and prints per block:
  * kept-feature count and the dropped features (names);
  * the fraction of z-scores clipped to ±clip;
  * min / median / max of the per-row missing-indicator rate (present rows).

Also writes the same summary to --out-txt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from adni.preprocess import MODALITIES, Preprocessor, save_fitted

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"


def _load_ids(splits_path: Path, split: str, seed: int, fold: int):
    blob = json.loads(Path(splits_path).read_text())
    if split == "s1":
        node = blob["s1"][f"seed_{seed}"][str(fold)]
    elif split == "s2":
        node = blob["s2"]
    else:
        raise ValueError(split)
    return node["episode_ids"]["train"], node["episode_ids"]["val"], node["episode_ids"]["test"]


def _row_missing_rate(X: np.ndarray, d: int, present: np.ndarray) -> np.ndarray:
    """Per-row fraction of *kept features* that are missing, over present rows."""
    ind = X[present, d:]
    return ind.mean(axis=1) if ind.size else np.array([])


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit Stage 2 preprocessing for a split and report.")
    ap.add_argument("--split", choices=["s1", "s2"], default="s1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--max-missing", type=float, default=0.20)
    ap.add_argument("--sd-floor", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None,
                    help="joblib path for the fitted object (default results/step04/artifacts/...)")
    ap.add_argument("--out-txt", type=Path, default=None,
                    help="optional text file to tee the report to")
    a = ap.parse_args()

    train_ids, val_ids, test_ids = _load_ids(a.splits, a.split, a.seed, a.fold)
    pp = Preprocessor.from_files(a.episodes, a.blocks, a.feature_spec,
                                 max_missing=a.max_missing, sd_floor=a.sd_floor, clip=a.clip)
    pp.fit(train_ids)

    if a.out is None:
        tag = f"{a.split}_seed{a.seed}_fold{a.fold}" if a.split == "s1" else "s2"
        a.out = PROJECT_ROOT / "results" / "step04" / "artifacts" / f"{tag}.joblib"
    save_fitted(pp, a.out)

    lines: list[str] = []
    lines.append(f"Stage 2 preprocessing — split={a.split} seed={a.seed} fold={a.fold}")
    lines.append(f"max_missing={pp.max_missing}  sd_floor={pp.sd_floor:g}  clip={pp.clip:g}")
    lines.append(f"train/val/test episodes = {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    lines.append("")

    out = pp.transform(train_ids)
    for k in MODALITIES:
        X, m = out[k]
        d = len(pp.kept[k])
        present = m == 1
        n_clip, n_finite = pp.clip_fraction(k, train_ids)
        clipped = (n_clip / n_finite) if n_finite else 0.0
        rate = _row_missing_rate(X, d, present)
        lines.append(f"[{k}] kept={d}  dropped={len(pp.dropped[k])}  present_rows={int(present.sum())}")
        lines.append(f"    dropped features: {pp.dropped[k] if pp.dropped[k] else '(none)'}")
        lines.append(f"    clipped z fraction (|z|>{pp.clip:g} of finite present z): {clipped:.6f} "
                     f"({n_clip}/{n_finite})")
        if rate.size:
            lines.append(f"    per-row missing-indicator rate: min={rate.min():.4f} "
                         f"median={np.median(rate):.4f} max={rate.max():.4f}")
        else:
            lines.append("    per-row missing-indicator rate: (no present rows)")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if a.out_txt is not None:
        a.out_txt.parent.mkdir(parents=True, exist_ok=True)
        a.out_txt.write_text(text + "\n")
        print(f"[wrote {a.out_txt}]")
    print(f"[saved fitted object -> {a.out}]")


if __name__ == "__main__":
    main()
