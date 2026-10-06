#!/usr/bin/env python3
"""Step 9 - main experiment: arms A0-A4 x S1 seeds [0, 1, 2] x folds [0..4] = 75 runs.

Pre-declared protocol (frozen before any grid result was seen; see results/step09/REPORT.md):

  * PRIMARY checkpoint rule: best validation macro-F1 (as in Steps 7-8). Early stopping and
    patience (15) use this rule only.
  * SECONDARY checkpoint rule: best validation PN balanced accuracy (rho uniform, weights from the
    fold's TRAIN-split pi_hat, exactly as ``metrics.pn_weights``). It is tracked on every epoch the
    run actually trains and never affects when training stops. Both best states are kept and the
    test fold is evaluated with BOTH rules.
  * Validation macro-F1 uses the argmax of the RAW logits ``s_c`` for every arm A0-A4. A uniform rho
    contributes ``log(1/3)`` to every class, so ``argmax(s + log rho_uniform) == argmax(s)``; the
    A1/A4 training-time adjustment (``log P(c)`` / ``tau * log pi``) does NOT enter the score.
  * Test operating points: A1/A4 -> ``uniform`` and ``P(c)``; A0/A2/A3 -> ``argmax``. The raw
    logits ``s`` are stored once per row.

Frozen inputs: ``configs/model.yaml`` (Step 7b), arms A0-A4 (Step 8), p_drop 0.25, alpha 0.5,
tau 1.0, clip [0.2, 5.0]. No per-arm tuning.

Each run writes, on completion:
  * ``results/step09/runs/<arm>_s<seed>_f<fold>.parquet``  OOF rows (both checkpoint rules)
  * ``results/step09/curves/<arm>_s<seed>_f<fold>.csv``    per-epoch validation curves
  * ``results/step09/run_log.jsonl``                      one JSON line per finished run

Restarting the script skips any run whose parquet AND curve already exist and parse (``--force``
re-runs everything). At the end every per-run parquet is concatenated to ``oof_all.parquet``.
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
from adni.losses import ARMS, inference_scores, priors_from_counts
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
DEFAULT_OUT = PROJECT_ROOT / "results" / "step09"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
# operating-point label -> inference rho argument (see `adni.losses.inference_scores`)
OPS_SECONDARY_ARMS = (("uniform", "uniform"), ("P(c)", "class_prior"))  # A1/A4
OPS_PLAIN = (("argmax", "uniform"),)                                    # A0/A2/A3
OOF_COLUMNS = ["episode_id", "seed", "fold", "arm", "ckpt_rule", "operating_point",
               "p_CN", "p_MCI", "p_Dementia", "logits", "best_epoch", "epochs_run"]
CURVE_COLUMNS = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_auroc", "val_pn_bacc"]


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _run_paths(out: Path, arm: str, seed: int, fold: int) -> tuple[Path, Path]:
    tag = f"{arm}_s{seed}_f{fold}"
    return out / "runs" / f"{tag}.parquet", out / "curves" / f"{tag}.csv"


def _already_done(parquet: Path, curve: Path) -> bool:
    if not (parquet.exists() and curve.exists() and parquet.stat().st_size and curve.stat().st_size):
        return False
    try:
        pd.read_parquet(parquet)
        pd.read_csv(curve)
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


def _test_logits(model, data, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.feats, data.mask, apply_modality_dropout=False)
    return logits


def _run_one(arm, seed, fold, ep, blob, pp, cfg, device, out) -> dict:
    tr, va, te = _fold_ids(blob, seed, fold)
    pp.fit(tr)
    feat_dims = pp.kept_counts()

    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_t = make_tensors(pp, te, ep)
    test_dev = test_t.to(device)

    p_drop = float(cfg["modality_dropout"])
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    priors = priors_from_counts(n_cm, pats, p_drop=p_drop)

    w_val = _val_pn_weights(ep, tr, va)
    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                       max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=seed,
                       device=device, deterministic=cfg["deterministic"])
    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}

    model, result, seconds = train_arm(arm, feat_dims, train_t, val_t, tcfg, model_kwargs, priors, w_val)

    ids = np.unique(te)
    ops = OPS_SECONDARY_ARMS if arm in ("A1", "A4") else OPS_PLAIN
    rows: list[dict] = []
    for rule in CKPT_RULES:
        if rule == "primary":
            state, best_epoch = result["best_state"], int(result["best"]["epoch"])
        else:
            state, best_epoch = result["best_pn_bacc_state"], int(result["best_pn_bacc"]["epoch"])
        model.load_state_dict(state)
        logits = _test_logits(model, test_dev, device)
        logits_np = logits.cpu().numpy()
        for op_label, rho in ops:
            probs = torch.softmax(inference_scores(logits, priors, rho), dim=1).cpu().numpy()
            for i, eid in enumerate(ids):
                rows.append({
                    "episode_id": int(eid), "seed": int(seed), "fold": int(fold), "arm": arm,
                    "ckpt_rule": rule, "operating_point": op_label,
                    "p_CN": float(probs[i, 0]), "p_MCI": float(probs[i, 1]), "p_Dementia": float(probs[i, 2]),
                    "logits": json.dumps([float(x) for x in logits_np[i]]),
                    "best_epoch": best_epoch, "epochs_run": int(result["epochs_run"]),
                })

    hist = pd.DataFrame(result["history"])
    for c in CURVE_COLUMNS:
        if c not in hist.columns:
            hist[c] = np.nan
    parquet, curve = _run_paths(out, arm, seed, fold)
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(parquet, index=False)
    hist[CURVE_COLUMNS].to_csv(curve, index=False)

    nan_loss = bool(not np.isfinite(hist["train_loss"]).all() or not np.isfinite(hist["val_loss"]).all())
    return {
        "arm": arm, "seed": int(seed), "fold": int(fold), "status": "ok",
        "epochs_run": int(result["epochs_run"]),
        "best_epoch_primary": int(result["best"]["epoch"]),
        "best_val_macro_f1": float(result["best"]["macro_f1"]),
        "best_epoch_secondary": int(result["best_pn_bacc"]["epoch"]),
        "best_val_pn_bacc": float(result["best_pn_bacc"]["pn_bacc"]),
        "n_parameters": int(result["n_parameters"]), "seconds": float(seconds),
        "nan_loss": nan_loss,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 9 main experiment (A0-A4 x seeds x folds).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--combine-only", action="store_true", help="only (re)build oof_all.parquet")
    a = ap.parse_args()
    (a.out / "runs").mkdir(parents=True, exist_ok=True)
    (a.out / "curves").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())

    def combine() -> None:
        parts = []
        for arm in ARMS:
            for seed in a.seeds:
                for fold in a.folds:
                    p, _ = _run_paths(a.out, arm, seed, fold)
                    if p.exists():
                        parts.append(pd.read_parquet(p))
        if parts:
            all_df = pd.concat(parts, ignore_index=True)
            all_df.to_parquet(a.out / "oof_all.parquet", index=False)
            print(f"[combined] oof_all.parquet rows={len(all_df)} runs={len(parts)}", flush=True)

    if a.combine_only:
        combine()
        return

    pp = Preprocessor(ep, {k: pd.read_csv(a.blocks / f"{k}.csv").set_index("episode_id")
                           for k in ("mri", "fdg", "amy", "tau")},
                      json.loads(Path(a.feature_spec).read_text()))

    log_path = a.out / "run_log.jsonl"
    total = len(a.arms) * len(a.seeds) * len(a.folds)
    done = skipped = failed = 0
    for arm in a.arms:
        for seed in a.seeds:
            for fold in a.folds:
                parquet, curve = _run_paths(a.out, arm, seed, fold)
                if not a.force and _already_done(parquet, curve):
                    skipped += 1
                    print(f"[skip] {arm} s{seed} f{fold} (already done)", flush=True)
                    continue
                try:
                    rec = _run_one(arm, seed, fold, ep, blob, pp, cfg, a.device, a.out)
                    done += 1
                except Exception as exc:  # keep the grid going; log the failure
                    rec = {"arm": arm, "seed": int(seed), "fold": int(fold), "status": "error",
                           "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    failed += 1
                    print(f"[ERROR] {arm} s{seed} f{fold}: {exc}", flush=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                if rec["status"] == "ok":
                    print(f"[ok {done + skipped}/{total}] {arm} s{seed} f{fold} "
                          f"ep={rec['epochs_run']:2d} best={rec['best_epoch_primary']:2d}/"
                          f"{rec['best_epoch_secondary']:2d} "
                          f"valF1={rec['best_val_macro_f1']:.4f} valPN={rec['best_val_pn_bacc']:.4f} "
                          f"{rec['seconds']:.1f}s nan={rec['nan_loss']}", flush=True)

    combine()
    print(f"[finished] ran={done} skipped={skipped} failed={failed} (log: {log_path})", flush=True)


if __name__ == "__main__":
    main()
