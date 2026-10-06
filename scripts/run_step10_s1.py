#!/usr/bin/env python3
"""Step 10 (1-2) - S1 retrain A0/A1/A4 (+ controls C1, C-shuf), saving best-state weights.

One resumable grid: 5 run labels x S1 seeds [0,1,2] x folds [0..4] = 75 runs.

  A0     plain cross-entropy (retrained so its test-fold weights exist for the intervention).
  A1     class-only logit adjustment (retrained for the intervention).
  A4     joint logit adjustment (retrained for the intervention).
  C1     PLACEBO: A4 whose pi(c|m~) is estimated from labels PERMUTED WITHIN ORIGPROT of the
         TRAINING split; the training loss still uses the TRUE labels.
  Cshuf  CONTROL: A3 with the per-episode weights randomly permuted within each batch.

Frozen inputs and the two checkpoint rules are exactly those of Step 9 (results/step09/REPORT.md):
PRIMARY = best validation macro-F1 (drives early stopping); SECONDARY = best validation PN-BAcc
(rho uniform, fold TRAIN-split pi_hat, never stops training). Every run saves BOTH best states and
the test-fold OOF predictions for BOTH rules, so the intervention and the analysis can reuse them.

Per run it writes (under results/step10/s1/):
  runs/<run>_s<seed>_f<fold>.parquet     OOF rows (both checkpoint rules)
  curves/<run>_s<seed>_f<fold>.csv       per-epoch validation curves
  weights/<run>_s<seed>_f<fold>_<rule>.pt  best-state weights for both rules (gitignored)
and appends one JSON line per finished run to run_log.jsonl, then concatenates to oof_all.parquet.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from adni.baselines import training_prior_table
from adni.diagnostics import permute_labels_within
from adni.losses import inference_scores, priors_from_counts
from adni.metrics import CLASSES, pn_weights
from adni.preprocess import Preprocessor
from adni.priors import crosstab
from adni.train import TrainConfig, make_tensors, train_arm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step10" / "s1"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
OPS_A4 = (("uniform", "uniform"), ("P(c)", "class_prior"))   # A1/A4/C1
OPS_PLAIN = (("argmax", "uniform"),)                         # A0/Cshuf
# run label -> (loss arm actually trained, kind). kinds: plain | perm (C1) | shuf (C-shuf).
RUNS = {
    "A0": ("A0", "plain"),
    "A1": ("A1", "plain"),
    "A4": ("A4", "plain"),
    "C1": ("A4", "perm"),
    "Cshuf": ("A3", "shuf"),
}
RUN_LABELS = tuple(RUNS)
OPS = {"A0": OPS_PLAIN, "A1": OPS_A4, "A4": OPS_A4, "C1": OPS_A4, "Cshuf": OPS_PLAIN}
OOF_COLUMNS = ["episode_id", "seed", "fold", "arm", "ckpt_rule", "operating_point",
               "p_CN", "p_MCI", "p_Dementia", "logits", "best_epoch", "epochs_run"]
CURVE_COLUMNS = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_auroc", "val_pn_bacc"]


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _paths(out: Path, run: str, seed: int, fold: int):
    tag = f"{run}_s{seed}_f{fold}"
    return (out / "runs" / f"{tag}.parquet", out / "curves" / f"{tag}.csv",
            out / "weights" / f"{tag}_primary.pt", out / "weights" / f"{tag}_secondary.pt")


def _already_done(paths) -> bool:
    for p in paths:
        if not (p.exists() and p.stat().st_size):
            return False
    try:
        pd.read_parquet(paths[0])
        pd.read_csv(paths[1])
        torch.load(paths[2], map_location="cpu", weights_only=True)
        return True
    except Exception:
        return False


def _val_pn_weights(ep: pd.DataFrame, train_ids: np.ndarray, val_ids: np.ndarray) -> np.ndarray:
    """Fixed validation PN weights (rho uniform) from the fold's TRAIN-split P(c|m)."""
    prior = training_prior_table(ep, train_ids)
    ids = np.unique(val_ids)
    y = ep["y"].loc[ids].to_numpy()
    pat = ep["pattern"].loc[ids].to_numpy()
    return pn_weights(y, pat, prior, rho="uniform")


def _make_priors(ep, tr, kind: str, seed: int, fold: int, p_drop: float):
    """Loss priors; for the placebo C1 the labels are permuted within ORIGPROT (train split only)."""
    sub = ep.loc[tr]
    y = sub["y"].to_numpy()
    if kind == "perm":
        y = permute_labels_within(y, sub["ORIGPROT"].to_numpy(), seed=100_000 * int(seed) + int(fold))
    diag = pd.Series([CLASSES[int(c)] for c in y], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    return priors_from_counts(n_cm, pats, p_drop=p_drop)


def _test_logits(model, data, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.feats, data.mask, apply_modality_dropout=False)
    return logits


def _run_one(run, seed, fold, ep, blob, pp, cfg, device, out) -> dict:
    loss_arm, kind = RUNS[run]
    tr, va, te = _fold_ids(blob, seed, fold)
    pp.fit(tr)
    feat_dims = pp.kept_counts()
    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_t = make_tensors(pp, te, ep)
    test_dev = test_t.to(device)

    p_drop = float(cfg["modality_dropout"])
    priors = _make_priors(ep, tr, kind, seed, fold, p_drop)
    w_val = _val_pn_weights(ep, tr, va)
    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                       max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=seed,
                       device=device, deterministic=cfg["deterministic"])
    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}

    model, result, seconds = train_arm(loss_arm, feat_dims, train_t, val_t, tcfg, model_kwargs,
                                       priors, w_val, shuffle_a3=(kind == "shuf"))

    ids = np.unique(te)
    ops = OPS[run]
    rows: list[dict] = []
    parquet, curve, w_pri, w_sec = _paths(out, run, seed, fold)
    for rule, state_path in (("primary", w_pri), ("secondary", w_sec)):
        if rule == "primary":
            state, best_epoch = result["best_state"], int(result["best"]["epoch"])
        else:
            state, best_epoch = result["best_pn_bacc_state"], int(result["best_pn_bacc"]["epoch"])
        torch.save(state, state_path)
        model.load_state_dict(state)
        logits = _test_logits(model, test_dev, device)
        logits_np = logits.cpu().numpy()
        for op_label, rho in ops:
            probs = torch.softmax(inference_scores(logits, priors, rho), dim=1).cpu().numpy()
            for i, eid in enumerate(ids):
                rows.append({
                    "episode_id": int(eid), "seed": int(seed), "fold": int(fold), "arm": run,
                    "ckpt_rule": rule, "operating_point": op_label,
                    "p_CN": float(probs[i, 0]), "p_MCI": float(probs[i, 1]), "p_Dementia": float(probs[i, 2]),
                    "logits": json.dumps([float(x) for x in logits_np[i]]),
                    "best_epoch": best_epoch, "epochs_run": int(result["epochs_run"]),
                })

    hist = pd.DataFrame(result["history"])
    for c in CURVE_COLUMNS:
        if c not in hist.columns:
            hist[c] = np.nan
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(parquet, index=False)
    hist[CURVE_COLUMNS].to_csv(curve, index=False)

    nan_loss = bool(not np.isfinite(hist["train_loss"]).all() or not np.isfinite(hist["val_loss"]).all())
    return {
        "arm": run, "loss_arm": loss_arm, "kind": kind, "seed": int(seed), "fold": int(fold),
        "status": "ok", "epochs_run": int(result["epochs_run"]),
        "best_epoch_primary": int(result["best"]["epoch"]),
        "best_val_macro_f1": float(result["best"]["macro_f1"]),
        "best_epoch_secondary": int(result["best_pn_bacc"]["epoch"]),
        "best_val_pn_bacc": float(result["best_pn_bacc"]["pn_bacc"]),
        "n_parameters": int(result["n_parameters"]), "seconds": float(seconds), "nan_loss": nan_loss,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10 S1 retrain + controls (A0/A1/A4/C1/Cshuf).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=list(RUN_LABELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--combine-only", action="store_true")
    a = ap.parse_args()
    for sub in ("runs", "curves", "weights"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())

    def combine() -> None:
        parts = []
        for run in RUN_LABELS:
            for seed in a.seeds:
                for fold in a.folds:
                    p, _, _, _ = _paths(a.out, run, seed, fold)
                    if p.exists():
                        parts.append(pd.read_parquet(p))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(a.out / "oof_all.parquet", index=False)
            print(f"[combined] oof_all.parquet rows={len(df)} runs={len(parts)}", flush=True)

    if a.combine_only:
        combine()
        return

    pp = Preprocessor(ep, {k: pd.read_csv(a.blocks / f"{k}.csv").set_index("episode_id")
                           for k in ("mri", "fdg", "amy", "tau")},
                      json.loads(Path(a.feature_spec).read_text()))
    log_path = a.out / "run_log.jsonl"
    total = len(a.arms) * len(a.seeds) * len(a.folds)
    done = skipped = failed = 0
    for run in a.arms:
        for seed in a.seeds:
            for fold in a.folds:
                paths = _paths(a.out, run, seed, fold)
                if not a.force and _already_done(paths):
                    skipped += 1
                    print(f"[skip] {run} s{seed} f{fold} (already done)", flush=True)
                    continue
                try:
                    rec = _run_one(run, seed, fold, ep, blob, pp, cfg, a.device, a.out)
                    done += 1
                except Exception as exc:
                    rec = {"arm": run, "seed": int(seed), "fold": int(fold), "status": "error",
                           "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    failed += 1
                    print(f"[ERROR] {run} s{seed} f{fold}: {exc}", flush=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if rec["status"] == "ok":
                    print(f"[ok {done + skipped}/{total}] {run} s{seed} f{fold} ep={rec['epochs_run']:2d} "
                          f"best={rec['best_epoch_primary']:2d}/{rec['best_epoch_secondary']:2d} "
                          f"valF1={rec['best_val_macro_f1']:.4f} valPN={rec['best_val_pn_bacc']:.4f} "
                          f"{rec['seconds']:.1f}s nan={rec['nan_loss']}", flush=True)

    combine()
    print(f"[finished] ran={done} skipped={skipped} failed={failed} (log: {log_path})", flush=True)


if __name__ == "__main__":
    main()
