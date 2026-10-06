#!/usr/bin/env python3
"""Step 11a (robustness only) - S1 grid for the R2 recipe and the recipe-1 sensitivities.

Frozen design: every run reuses the Step 9 protocol exactly (uniform protocol of Step 6b, both
checkpoint rules). Nothing here changes the primary conclusions; the settings are pre-registered in
``configs/step11a_robustness.yaml`` and were frozen before any run was seen.

Run-sets on the S1 grid (seeds [0,1,2] x folds [0..4]):

  R2_A0, R2_A1, R2_A4     the second training recipe: lr 3e-5, weight decay 0.3, batch 128,
                          encoder dropout 0.4, transformer dropout 0.2, max 150 epochs,
                          patience 25 (recipe 1 keeps all other values).
  A4_tau0p5 / A4_tau1p5   A4 at recipe 1 with tau in {0.5, 1.5}.
  A3_clip0p5_2 / _0p1_10  A3 at recipe 1 with the training weight clip [0.5, 2] / [0.1, 10].
  A0_pdrop0 / A0_pdrop0p5 A0 at recipe 1 with modality dropout p_drop = 0 / 0.5.
  A4_pdrop0 / A4_pdrop0p5 A4 at recipe 1 with p_drop = 0 / 0.5 (the prior is built with the
                          MATCHING kernel: ``priors_from_counts(..., p_drop=p_drop)``).

Recipe 1 itself is the Step 9 ``results/step09/oof_all.parquet`` (A0, A1, A2, A3, A4); it is NOT
retrained here. Sensitivity settings are compared against their recipe-1 counterpart in the
analysis.

Checkpoint protocol (identical to Step 9; see results/step09/REPORT.md):

  * PRIMARY   = best validation macro-F1 (drives early stopping / patience). Validation macro-F1
                uses the argmax of the RAW logits for every arm (a uniform rho is constant).
  * SECONDARY = best validation PN balanced accuracy (rho uniform, fold TRAIN-split pi_hat). It is
                tracked every epoch but never stops training. Both best states are kept and the
                test fold is evaluated with BOTH rules.

Seeds are fixed before the model is built (``train_arm`` calls ``set_seed`` before constructing the
``FusionModel``), so every (run-set, seed, fold) is deterministic.

Per run it writes under ``results/step11a/``::

  runs/<runset>_s<seed>_f<fold>.parquet   OOF rows (both checkpoint rules, all operating points)
  curves/<runset>_s<seed>_f<fold>.csv     per-epoch validation curves

Restarting skips any run whose parquet AND curve already exist and parse (``--force`` re-runs
everything). ``--shard K --nshards N`` runs a deterministic round-robin subset in parallel; a final
``--combine-only`` pass concatenates every per-run parquet into ``oof_all.parquet`` and merges the
shard logs into ``run_log.jsonl``.
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
DEFAULT_ROBUSTNESS = PROJECT_ROOT / "configs" / "step11a_robustness.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step11a"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
R2_ARMS = ("A0", "A1", "A4")
OPS_SECONDARY_ARMS = (("uniform", "uniform"), ("P(c)", "class_prior"))   # A1/A4
OPS_PLAIN = (("argmax", "uniform"),)                                     # A0/A3
OPS_BY_ARM = {"A0": OPS_PLAIN, "A1": OPS_SECONDARY_ARMS, "A3": OPS_PLAIN, "A4": OPS_SECONDARY_ARMS}
PRIMARY_OP_BY_ARM = {"A0": "argmax", "A1": "uniform", "A3": "argmax", "A4": "uniform"}
OOF_COLUMNS = ["episode_id", "seed", "fold", "runset", "arm", "ckpt_rule", "operating_point",
               "p_CN", "p_MCI", "p_Dementia", "logits", "best_epoch", "epochs_run"]
CURVE_COLUMNS = ["epoch", "train_loss", "val_loss", "val_macro_f1", "val_macro_auroc", "val_pn_bacc"]


def _mk_runset(arm: str, cfg: dict, tau: float, clip, label: str) -> dict:
    return {"label": label, "arm": arm, "cfg": dict(cfg), "tau": float(tau), "clip": clip,
            "primary_op": PRIMARY_OP_BY_ARM[arm], "ops": OPS_BY_ARM[arm]}


def build_runsets(base_cfg: dict, spec: dict) -> dict[str, dict]:
    """Resolve the pre-registered grid into ``{run-set: {arm, cfg, tau, clip, ops, ...}}``.

    Pure (no I/O), so the override plumbing is unit-testable. ``base_cfg`` is the frozen recipe-1
    ``configs/model.yaml`` dict; ``spec`` is ``configs/step11a_robustness.yaml``.
    """
    runsets: dict[str, dict] = {}
    r2 = spec.get("recipe_2", {}) or {}
    for arm in R2_ARMS:
        label = f"R2_{arm}"
        runsets[label] = _mk_runset(arm, {**base_cfg, **r2}, tau=1.0, clip=None, label=label)
    for item in spec.get("sensitivity", []) or []:
        name = str(item["name"])
        arm = str(item["arm"])
        if name in runsets:
            raise ValueError(f"duplicate run-set name {name!r} in the robustness grid")
        over = {k: v for k, v in item.items() if k not in ("name", "arm")}
        tau = float(over.pop("tau", 1.0))
        wmin = over.pop("wmin", None)
        wmax = over.pop("wmax", None)
        if (wmin is None) != (wmax is None):
            raise ValueError(f"{name}: give both wmin and wmax, or neither")
        clip = None if wmin is None else (float(wmin), float(wmax))
        runsets[name] = _mk_runset(arm, {**base_cfg, **over}, tau=tau, clip=clip, label=name)
    return runsets


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _paths(out: Path, runset: str, seed: int, fold: int) -> tuple[Path, Path]:
    tag = f"{runset}_s{seed}_f{fold}"
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
    """Fixed validation PN weights (rho uniform) from the fold's TRAIN-split P(c|m) (Step 9 rule)."""
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


def _run_one(runset: str, rs: dict, seed: int, fold: int, ep, blob, pp, device, out) -> dict:
    arm = rs["arm"]
    cfg = rs["cfg"]
    tr, va, te = _fold_ids(blob, seed, fold)
    pp.fit(tr)
    feat_dims = pp.kept_counts()
    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_t = make_tensors(pp, te, ep)
    test_dev = test_t.to(device)

    # The prior uses the MATCHING mask-process kernel: the same p_drop as the model's dropout.
    p_drop = float(cfg["modality_dropout"])
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    priors = priors_from_counts(n_cm, pats, p_drop=p_drop)
    w_val = _val_pn_weights(ep, tr, va)

    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                       max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=seed,
                       device=device, deterministic=cfg["deterministic"],
                       tau=rs["tau"], clip=rs["clip"])
    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}

    model, result, seconds = train_arm(arm, feat_dims, train_t, val_t, tcfg, model_kwargs,
                                       priors, w_val)

    ids = np.unique(te)
    rows: list[dict] = []
    for rule in CKPT_RULES:
        if rule == "primary":
            state, best_epoch = result["best_state"], int(result["best"]["epoch"])
        else:
            state, best_epoch = result["best_pn_bacc_state"], int(result["best_pn_bacc"]["epoch"])
        model.load_state_dict(state)
        logits = _test_logits(model, test_dev, device)
        logits_np = logits.cpu().numpy()
        for op_label, rho in rs["ops"]:
            probs = torch.softmax(inference_scores(logits, priors, rho), dim=1).cpu().numpy()
            for i, eid in enumerate(ids):
                rows.append({
                    "episode_id": int(eid), "seed": int(seed), "fold": int(fold),
                    "runset": runset, "arm": arm, "ckpt_rule": rule, "operating_point": op_label,
                    "p_CN": float(probs[i, 0]), "p_MCI": float(probs[i, 1]),
                    "p_Dementia": float(probs[i, 2]),
                    "logits": json.dumps([float(x) for x in logits_np[i]]),
                    "best_epoch": best_epoch, "epochs_run": int(result["epochs_run"]),
                })

    hist = pd.DataFrame(result["history"])
    for c in CURVE_COLUMNS:
        if c not in hist.columns:
            hist[c] = np.nan
    parquet, curve = _paths(out, runset, seed, fold)
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(parquet, index=False)
    hist[CURVE_COLUMNS].to_csv(curve, index=False)

    nan_loss = bool(not np.isfinite(hist["train_loss"]).all() or not np.isfinite(hist["val_loss"]).all())
    return {
        "runset": runset, "arm": arm, "tau": rs["tau"], "clip": rs["clip"],
        "modality_dropout": p_drop, "lr": cfg["lr"], "batch_size": cfg["batch_size"],
        "seed": int(seed), "fold": int(fold), "status": "ok",
        "epochs_run": int(result["epochs_run"]),
        "best_epoch_primary": int(result["best"]["epoch"]),
        "best_val_macro_f1": float(result["best"]["macro_f1"]),
        "best_epoch_secondary": int(result["best_pn_bacc"]["epoch"]),
        "best_val_pn_bacc": float(result["best_pn_bacc"]["pn_bacc"]),
        "n_parameters": int(result["n_parameters"]), "seconds": float(seconds), "nan_loss": nan_loss,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 11a robustness grid (R2 recipe + recipe-1 sensitivities).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--robustness", type=Path, default=DEFAULT_ROBUSTNESS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runsets", nargs="+", default=None, help="subset of run-set names")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-run even if outputs exist")
    ap.add_argument("--combine-only", action="store_true", help="only (re)build oof_all.parquet")
    ap.add_argument("--no-combine", action="store_true",
                    help="skip the end-of-run combine (use with --nshards to avoid concurrent writes)")
    a = ap.parse_args()
    (a.out / "runs").mkdir(parents=True, exist_ok=True)
    (a.out / "curves").mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(a.config).read_text())
    spec = yaml.safe_load(Path(a.robustness).read_text())
    runsets = build_runsets(base_cfg, spec)
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())

    def combine() -> None:
        parts = []
        for rs in runsets:
            for seed in a.seeds:
                for fold in a.folds:
                    p, _ = _paths(a.out, rs, seed, fold)
                    if p.exists():
                        parts.append(pd.read_parquet(p))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(a.out / "oof_all.parquet", index=False)
            print(f"[combined] oof_all.parquet rows={len(df)} "
                  f"runs={df.groupby(['runset', 'seed', 'fold']).ngroups}", flush=True)
        merged: dict = {}
        for p in sorted(a.out.glob("run_log*.jsonl")):
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    merged[(rec.get("runset"), rec.get("seed"), rec.get("fold"))] = rec
        if merged:
            with (a.out / "run_log.jsonl").open("w", encoding="utf-8") as fh:
                for k in sorted(merged, key=lambda x: (str(x[0]), x[1] or -1, x[2] or -1)):
                    fh.write(json.dumps(merged[k]) + "\n")

    if a.combine_only:
        combine()
        return

    pp = Preprocessor(ep, {k: pd.read_csv(a.blocks / f"{k}.csv").set_index("episode_id")
                           for k in ("mri", "fdg", "amy", "tau")},
                      json.loads(Path(a.feature_spec).read_text()))
    log_path = a.out / (f"run_log_shard{a.shard}.jsonl" if a.nshards > 1 else "run_log.jsonl")

    selected = [rs for rs in runsets if (not a.runsets or rs in set(a.runsets))]
    unknown = set(a.runsets or []) - set(runsets)
    if unknown:
        raise SystemExit(f"unknown run-set(s): {sorted(unknown)}; have {sorted(runsets)}")
    jobs = [(rs, seed, fold) for rs in selected for seed in a.seeds for fold in a.folds]
    if a.nshards > 1:
        jobs = jobs[a.shard::a.nshards]
    total = len(jobs)
    done = skipped = failed = 0
    for runset, seed, fold in jobs:
        parquet, curve = _paths(a.out, runset, seed, fold)
        if not a.force and _already_done(parquet, curve):
            skipped += 1
            print(f"[skip] {runset} s{seed} f{fold} (already done)", flush=True)
            continue
        try:
            rec = _run_one(runset, runsets[runset], seed, fold, ep, blob, pp, a.device, a.out)
            done += 1
        except Exception as exc:
            rec = {"runset": runset, "seed": int(seed), "fold": int(fold), "status": "error",
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            failed += 1
            print(f"[ERROR] {runset} s{seed} f{fold}: {exc}", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["status"] == "ok":
            print(f"[ok {done + skipped}/{total}] {runset} s{seed} f{fold} ep={rec['epochs_run']:3d} "
                  f"best={rec['best_epoch_primary']:3d}/{rec['best_epoch_secondary']:3d} "
                  f"valF1={rec['best_val_macro_f1']:.4f} valPN={rec['best_val_pn_bacc']:.4f} "
                  f"{rec['seconds']:.1f}s nan={rec['nan_loss']}", flush=True)

    if not a.no_combine:
        combine()
    print(f"[finished shard {a.shard}/{a.nshards}] ran={done} skipped={skipped} failed={failed} "
          f"(log: {log_path})", flush=True)


if __name__ == "__main__":
    main()
