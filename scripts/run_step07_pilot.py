#!/usr/bin/env python3
"""Step 7 pilot - train A0 on S1 seed 0 fold 0 only and report curves + test metrics.

Trains the masked-attention fusion model (``src/adni/models.py``) with the A0 loss (plain
cross-entropy) on the frozen S1 seed-0 fold-0 split, saving:

  * ``results/step07/curves_s0f0.csv``  - per-epoch train loss, val loss, val macro-F1, val macro-AUROC
  * ``results/step07/curves_s0f0.png``  - the same as a plot
  * ``results/step07/pilot_s0f0.json``  - config, param count, epochs, best epoch, timing, GPU memory
  * ``results/step07/comparison_s0f0.md`` - fold-0 test metrics next to B1, B1-blind and B2_mri

No tuning is done against the test metrics; the test numbers are reported once.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adni.baselines import training_prior_table  # noqa: E402
from adni.metrics import (  # noqa: E402
    CLASSES,
    natural_metrics,
    pn_balanced_accuracy,
    pn_weights,
)
from adni.preprocess import Preprocessor  # noqa: E402
from adni.train import TrainConfig, Trainer, make_tensors, train_a0  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step07"
COMPARISON_ARMS = {
    "B1": PROJECT_ROOT / "results" / "step06b" / "oof_b1_i.parquet",
    "B1-blind": PROJECT_ROOT / "results" / "step06c" / "oof_b1_blind.parquet",
    "B2_mri": PROJECT_ROOT / "results" / "step06" / "oof_b2_mri.parquet",
}


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    arr = lambda xs: np.asarray(xs, dtype=np.int64)  # noqa: E731
    return arr(node["train"]), arr(node["val"]), arr(node["test"])


def _metrics_block(ep: pd.DataFrame, tr: np.ndarray, y: np.ndarray, proba: np.ndarray,
                   ids: np.ndarray) -> dict:
    prior = training_prior_table(ep, tr)
    pp = ep["pattern"].loc[list(ids)].to_numpy()
    m = natural_metrics(y, proba)
    wu = pn_weights(y, pp, prior, rho="uniform")
    pc = np.bincount(ep["y"].loc[list(tr)].to_numpy(), minlength=3).astype(float)
    pc = pc / pc.sum()
    wp = pn_weights(y, pp, prior, rho=pc)
    return {
        "n": int(len(ids)),
        "macro_auroc": float(m["macro_auroc"]),
        "macro_f1": float(m["macro_f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "f1_per_class": {CLASSES[c]: float(v) for c, v in m["f1_per_class"].items()},
        "pn_bacc_uniform": float(pn_balanced_accuracy(y, proba.argmax(1), wu)),
        "pn_bacc_prior": float(pn_balanced_accuracy(y, proba.argmax(1), wp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 7 pilot (A0, S1 seed 0 fold 0).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    tr, va, te = _fold_ids(blob, a.seed, a.fold)

    pp = Preprocessor.from_files(a.episodes, a.blocks, a.feature_spec)
    pp.fit(tr)
    feat_dims = pp.kept_counts()
    print(f"kept feature counts: {feat_dims}  (encoder input widths 2x)", flush=True)

    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_t = make_tensors(pp, te, ep)

    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                       max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=a.seed,
                       device=a.device, deterministic=cfg["deterministic"])
    model_kwargs = {k: cfg[k] for k in ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers",
                                        "ff_dim", "attn_dropout", "n_classes", "modality_dropout")}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, result, seconds = train_a0(feat_dims, train_t, val_t, tcfg, model_kwargs)
    hist = result["history"]
    peak_gpu = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0

    curves = pd.DataFrame(hist)
    curves.to_csv(a.out / "curves_s0f0.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(curves["epoch"], curves["train_loss"], label="train loss")
    ax[0].plot(curves["epoch"], curves["val_loss"], label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(); ax[0].set_title("loss")
    ax[1].plot(curves["epoch"], curves["val_macro_f1"], label="val macro-F1", color="tab:green")
    ax[1].plot(curves["epoch"], curves["val_macro_auroc"], label="val macro-AUROC", color="tab:orange")
    ax[1].axvline(result["best"]["epoch"], ls="--", color="grey", label="best epoch")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("validation metrics")
    fig.tight_layout()
    fig.savefig(a.out / "curves_s0f0.png", dpi=120)

    # ------------------------------------------------------------------ test metrics
    trainer = Trainer(model, tcfg)
    test_proba = trainer.predict_proba(test_t.to(tcfg.device))
    y_te = test_t.y.cpu().numpy()
    te_ids = np.unique(te)
    a0_all = _metrics_block(ep, tr, y_te, test_proba, te_ids)

    n_mod = ep["n_mod"].loc[te_ids].to_numpy()
    by_mod = {}
    for nm in (1, 2, 3, 4):
        sel = n_mod == nm
        if sel.sum():
            b = _metrics_block(ep, tr, y_te[sel], test_proba[sel], te_ids[sel])
            by_mod[f"n_mod={nm}"] = b

    # ------------------------------------------------------------------ comparison table
    comp_rows = [{"arm": "A0", **a0_all}]
    for arm, path in COMPARISON_ARMS.items():
        df = pd.read_parquet(path)
        df = df[(df["seed"] == a.seed) & (df["fold"] == a.fold)]
        ids = np.intersect1d(df["episode_id"].to_numpy(), te_ids)
        df = df.set_index("episode_id").loc[ids]
        proba = df[["p_CN", "p_MCI", "p_Dementia"]].to_numpy()
        yv = ep["y"].loc[ids].to_numpy()
        block = _metrics_block(ep, tr, yv, proba, ids)
        # also report A0 restricted to this arm's episodes for a like-for-like comparison
        a0_sel = test_proba[np.isin(te_ids, ids)]
        a0_block = _metrics_block(ep, tr, ep["y"].loc[ids].to_numpy(), a0_sel, ids)
        comp_rows.append({"arm": arm, **block, "a0_same_eps_macro_auroc": a0_block["macro_auroc"],
                          "a0_same_eps_pn_bacc": a0_block["pn_bacc_uniform"]})

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(a.out / "comparison_s0f0.csv", index=False)

    md = ["# Step 7 pilot - fold-0 test metrics (S1 seed 0 fold 0)", "",
          "| arm | n_ep | macro-AUROC | macro-F1 | BAcc | PN-BAcc (uniform) | PN-BAcc (P(c)) |",
          "|-----|------|-------------|----------|------|-------------------|----------------|"]
    for r in comp_rows:
        md.append(f"| {r['arm']} | {r['n']} | {r['macro_auroc']:.4f} | {r['macro_f1']:.4f} | "
                  f"{r['balanced_accuracy']:.4f} | {r['pn_bacc_uniform']:.4f} | {r['pn_bacc_prior']:.4f} |")
    md += ["", "A0 by number of observed modalities (natural metrics):", "",
           "| n_mod | n_ep | macro-AUROC | macro-F1 | PN-BAcc (uniform) |",
           "|-------|------|-------------|----------|-------------------|"]
    for k, b in by_mod.items():
        md.append(f"| {k.split('=')[1]} | {b['n']} | {b['macro_auroc']:.4f} | {b['macro_f1']:.4f} | "
                  f"{b['pn_bacc_uniform']:.4f} |")
    (a.out / "comparison_s0f0.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ summary + red flags
    best_ep = result["best"]["epoch"]
    val_best = hist[best_ep]
    sec_per_epoch = seconds / max(1, result["epochs_run"])
    red = {
        "test_macro_auroc_below_0.70": a0_all["macro_auroc"] < 0.70,
        "val_test_gap_above_0.05": abs(val_best["val_macro_auroc"] - a0_all["macro_auroc"]) > 0.05,
        "best_epoch_below_5": best_ep < 5,
        "loss_flat_from_epoch_0": len(hist) > 1 and abs(hist[-1]["train_loss"] - hist[0]["train_loss"]) < 1e-3,
    }
    summary = {
        "seed": a.seed, "fold": a.fold, "device": a.device,
        "feature_dims": feat_dims,
        "encoder_input_widths": {k: 2 * v for k, v in feat_dims.items()},
        "n_parameters": int(result["n_parameters"]),
        "epochs_run": int(result["epochs_run"]),
        "best_epoch": int(best_ep),
        "best_val_macro_f1": float(val_best["val_macro_f1"]),
        "val_macro_auroc_at_best": float(val_best["val_macro_auroc"]),
        "seconds_total": float(seconds),
        "seconds_per_epoch": float(sec_per_epoch),
        "peak_gpu_mem_mb": float(peak_gpu),
        "test": a0_all,
        "test_by_modality_count": by_mod,
        "red_flags": red,
    }
    (a.out / "pilot_s0f0.json").write_text(json.dumps(summary, indent=2))

    print("\n=== A0 pilot (S1 seed 0 fold 0) ===")
    print(f"params {result['n_parameters']:,}  epochs {result['epochs_run']}  best epoch {best_ep}  "
          f"sec/epoch {sec_per_epoch:.2f}  peak GPU {peak_gpu:.0f} MB")
    print(f"test macro-AUROC {a0_all['macro_auroc']:.4f}  macro-F1 {a0_all['macro_f1']:.4f}  "
          f"PN-BAcc(uniform) {a0_all['pn_bacc_uniform']:.4f}")
    print("red flags:", red)
    print(f"[wrote outputs under {a.out}]")


if __name__ == "__main__":
    main()
