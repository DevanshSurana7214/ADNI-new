"""Step 7b - validation-only regularisation search for arm A0 (pre-registered).

Searches exactly 8 configurations on S1 seed 0, folds 0 and 1:

    lr          in {3e-4, 1e-4}
    weight_decay in {0.05, 0.3}
    dropout      in {(enc 0.2, transformer 0.1), (enc 0.4, transformer 0.2)}

Everything else follows ``configs/model.yaml`` except ``max_epochs=60`` and ``patience=10``.
The test fold is never loaded: ``_load_fold_ids`` raises if anything asks for ``"test"``.

Writes ``results/step07b/search_runs.csv`` (16 rows) and ``search_summary.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from adni.preprocess import Preprocessor
from adni.train import TrainConfig, make_tensors, train_a0

ROOT = Path(__file__).resolve().parents[1]

LR_GRID = (3e-4, 1e-4)
WD_GRID = (0.05, 0.3)
DROPOUT_GRID = ((0.2, 0.1), (0.4, 0.2))   # (encoder, transformer/attn)
FOLDS = (0, 1)
SEED = 0
MAX_EPOCHS = 60
PATIENCE = 10


class _NoTestIds(dict):
    """Dict wrapper that refuses any access to the test fold (validation-only pre-registration)."""

    def __getitem__(self, key):
        if key == "test":
            raise AssertionError("the test fold must not be accessed during the validation-only search")
        return super().__getitem__(key)


def _load_fold_ids(splits: Path, seed: int, fold: int):
    blob = json.loads(Path(splits).read_text())
    node = _NoTestIds(blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"])
    return np.asarray(node["train"], dtype=np.int64), np.asarray(node["val"], dtype=np.int64)


def configurations() -> list[dict]:
    grid = []
    for i, (lr, wd, (enc_do, attn_do)) in enumerate(
            itertools.product(LR_GRID, WD_GRID, DROPOUT_GRID)):
        grid.append({"config_id": f"c{i}", "lr": lr, "weight_decay": wd,
                     "enc_dropout": enc_do, "attn_dropout": attn_do})
    return grid


def _select(rows: pd.DataFrame) -> dict:
    """Highest mean best-epoch val macro-F1; ties (<=0.005) -> larger wd, then lower lr, then smaller dropout."""
    per = rows.groupby("config_id", as_index=False)["val_macro_f1"].mean()
    best = float(per["val_macro_f1"].max())
    tied = set(per.loc[per["val_macro_f1"] >= best - 0.005, "config_id"])
    cand = rows.drop_duplicates("config_id").set_index("config_id").loc[sorted(tied)].reset_index()
    cand = cand.sort_values(
        ["weight_decay", "lr", "enc_dropout", "attn_dropout", "config_id"],
        ascending=[False, True, True, True, True])
    chosen = cand.iloc[0]["config_id"]
    return {"selected_config_id": chosen, "best_mean_val_macro_f1": best,
            "tied_within_0.005": sorted(tied),
            "mean_val_macro_f1_per_config": dict(zip(per["config_id"], per["val_macro_f1"]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, default=ROOT / "docs" / "episodes.csv")
    ap.add_argument("--blocks", type=Path, default=ROOT / "results/step01/stage1/blocks")
    ap.add_argument("--feature-spec", type=Path, default=ROOT / "results/step01/stage1/feature_spec.json")
    ap.add_argument("--splits", type=Path, default=ROOT / "results/step03/splits.json")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/model.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results/step07b")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    base = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")

    rows: list[dict] = []
    for cfg in configurations():
        for fold in FOLDS:
            tr, va = _load_fold_ids(a.splits, SEED, fold)
            pp = Preprocessor.from_files(a.episodes, a.blocks, a.feature_spec)
            pp.fit(tr)
            train_t = make_tensors(pp, tr, ep)
            val_t = make_tensors(pp, va, ep)
            tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                               batch_size=base["batch_size"], max_epochs=MAX_EPOCHS,
                               patience=PATIENCE, seed=SEED, device=a.device,
                               deterministic=base["deterministic"])
            kw = {k: base[k] for k in ("d_model", "enc_hidden", "n_heads", "n_layers",
                                       "ff_dim", "n_classes", "modality_dropout")}
            kw["enc_dropout"] = cfg["enc_dropout"]
            kw["attn_dropout"] = cfg["attn_dropout"]
            _, res, _ = train_a0(pp.kept_counts(), train_t, val_t, tcfg, kw)
            be = res["best"]["epoch"]
            hist = res["history"]
            row = {**cfg, "fold": fold, "best_epoch": int(be),
                   "val_macro_f1": float(hist[be]["val_macro_f1"]),
                   "val_macro_auroc": float(hist[be]["val_macro_auroc"]),
                   "val_loss_at_best": float(hist[be]["val_loss"]),
                   "epochs_run": int(res["epochs_run"])}
            rows.append(row)
            print(f"[{cfg['config_id']} fold {fold}] best_epoch={be} "
                  f"val_f1={row['val_macro_f1']:.4f} val_auroc={row['val_macro_auroc']:.4f} "
                  f"epochs_run={row['epochs_run']}", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(a.out / "search_runs.csv", index=False)

    sel = _select(runs)
    sel["grid"] = {"lr": list(LR_GRID), "weight_decay": list(WD_GRID),
                   "dropout": [list(d) for d in DROPOUT_GRID], "folds": list(FOLDS),
                   "seed": SEED, "max_epochs": MAX_EPOCHS, "patience": PATIENCE}
    sel["configurations"] = configurations()
    (a.out / "search_summary.json").write_text(json.dumps(sel, indent=2))

    print("\n=== mean best-epoch val macro-F1 per config ===")
    for k, v in sorted(sel["mean_val_macro_f1_per_config"].items()):
        print(f"  {k}: {v:.4f}")
    print(f"\nselected: {sel['selected_config_id']}  (tied {sel['tied_within_0.005']})")
    print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
