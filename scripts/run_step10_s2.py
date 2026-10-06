#!/usr/bin/env python3
"""Step 10 (3) - S2 enrolment-cohort shift.

Train on the subjects first enrolled in ADNI1/ADNI-GO/ADNI2 (S2 ``train`` side, with the frozen
S2 ``val`` holdout) and test on the ADNI3/ADNI4-enrolled subjects (S2 ``test``). Three model seeds
per arm (the S2 split is a single fixed split, so a seed only varies the initialisation).

  * neural A0-A4: same encoders/fusion/optimiser/early stopping as S1; priors, val PN weights and
    test PN weights all come from the S2 TRAIN split. Both Step 9 checkpoint rules are saved.
  * B1: LightGBM on the raw concatenated blocks + mask bits, early-stopping on the S2 val split.

Writes results/step10/s2/{runs/*.parquet, run_log.jsonl, oof_all.parquet}.
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

from adni.baselines import build_raw_matrix, fit_predict_b1, training_prior_table
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
DEFAULT_OUT = PROJECT_ROOT / "results" / "step10" / "s2"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
OPS_A4 = (("uniform", "uniform"), ("P(c)", "class_prior"))
OPS_PLAIN = (("argmax", "uniform"),)
OPS = {"A0": OPS_PLAIN, "A1": OPS_A4, "A2": OPS_PLAIN, "A3": OPS_PLAIN, "A4": OPS_A4}
OOF_COLUMNS = ["episode_id", "seed", "fold", "arm", "ckpt_rule", "operating_point",
               "p_CN", "p_MCI", "p_Dementia", "logits", "best_epoch", "epochs_run"]


def _s2_ids(blob):
    node = blob["s2"]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _run_path(out: Path, arm: str, seed: int) -> Path:
    return out / "runs" / f"{arm}_s{seed}.parquet"


def _test_logits(model, data, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.feats, data.mask, apply_modality_dropout=False)
    return logits


def _run_neural(arm, seed, ep, tr, va, te, pp, cfg, device, out) -> dict:
    feat_dims = pp.kept_counts()
    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_t = make_tensors(pp, te, ep)
    test_dev = test_t.to(device)

    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    priors = priors_from_counts(n_cm, pats, p_drop=float(cfg["modality_dropout"]))

    prior = training_prior_table(ep, tr)
    vids = np.unique(va)
    w_val = pn_weights(ep["y"].loc[vids].to_numpy(), ep["pattern"].loc[vids].to_numpy(), prior, "uniform")

    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                       max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=seed,
                       device=device, deterministic=cfg["deterministic"])
    model, result, seconds = train_arm(arm, feat_dims, train_t, val_t, tcfg,
                                       {k: cfg[k] for k in MODEL_KEYS}, priors, w_val)
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
        for op_label, rho in OPS[arm]:
            probs = torch.softmax(inference_scores(logits, priors, rho), dim=1).cpu().numpy()
            for i, eid in enumerate(ids):
                rows.append({"episode_id": int(eid), "seed": int(seed), "fold": 0, "arm": arm,
                             "ckpt_rule": rule, "operating_point": op_label,
                             "p_CN": float(probs[i, 0]), "p_MCI": float(probs[i, 1]),
                             "p_Dementia": float(probs[i, 2]),
                             "logits": json.dumps([float(x) for x in logits_np[i]]),
                             "best_epoch": best_epoch, "epochs_run": int(result["epochs_run"])})
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(_run_path(out, arm, seed), index=False)
    return {"arm": arm, "seed": int(seed), "fold": 0, "status": "ok",
            "epochs_run": int(result["epochs_run"]), "best_epoch_primary": int(result["best"]["epoch"]),
            "best_val_macro_f1": float(result["best"]["macro_f1"]),
            "best_epoch_secondary": int(result["best_pn_bacc"]["epoch"]),
            "best_val_pn_bacc": float(result["best_pn_bacc"]["pn_bacc"]),
            "n_parameters": int(result["n_parameters"]), "seconds": float(seconds), "nan_loss": False}


def _run_b1(seed, X, ep, tr, va, te, out) -> dict:
    proba, booster = fit_predict_b1(X, ep, tr, va, te, seed=seed)
    ids = np.unique(te)
    rows = []
    for i, eid in enumerate(ids):
        rows.append({"episode_id": int(eid), "seed": int(seed), "fold": 0, "arm": "B1",
                     "ckpt_rule": "single", "operating_point": "proba",
                     "p_CN": float(proba[i, 0]), "p_MCI": float(proba[i, 1]), "p_Dementia": float(proba[i, 2]),
                     "logits": json.dumps([float(np.log(max(x, 1e-12))) for x in proba[i]]),
                     "best_epoch": int(booster.best_iteration or 0), "epochs_run": int(booster.best_iteration or 0)})
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(_run_path(out, "B1", seed), index=False)
    return {"arm": "B1", "seed": int(seed), "fold": 0, "status": "ok", "epochs_run": 0,
            "best_epoch_primary": -1, "best_val_macro_f1": float("nan"), "best_epoch_secondary": -1,
            "best_val_pn_bacc": float("nan"), "n_parameters": 0,
            "seconds": 0.0, "nan_loss": False, "best_iteration": int(booster.best_iteration or 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10 S2 protocol-shift runner (A0-A4 + B1).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=list(ARMS) + ["B1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (a.out / "runs").mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    tr, va, te = _s2_ids(blob)
    print(f"[S2] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    spec = json.loads(Path(a.feature_spec).read_text())
    blocks = {k: pd.read_csv(a.blocks / f"{k}.csv").set_index("episode_id")
              for k in ("mri", "fdg", "amy", "tau")}
    pp = Preprocessor(ep, blocks, spec)
    pp.fit(tr)
    X = build_raw_matrix(ep, blocks, spec)

    log_path = a.out / "run_log.jsonl"
    done = skipped = failed = 0
    for arm in a.arms:
        for seed in a.seeds:
            p = _run_path(a.out, arm, seed)
            if not a.force and p.exists() and p.stat().st_size:
                skipped += 1
                print(f"[skip] {arm} s{seed} (already done)", flush=True)
                continue
            try:
                if arm == "B1":
                    rec = _run_b1(seed, X, ep, tr, va, te, a.out)
                else:
                    rec = _run_neural(arm, seed, ep, tr, va, te, pp, cfg, a.device, a.out)
                done += 1
            except Exception as exc:
                rec = {"arm": arm, "seed": int(seed), "fold": 0, "status": "error",
                       "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                failed += 1
                print(f"[ERROR] {arm} s{seed}: {exc}", flush=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if rec["status"] == "ok":
                print(f"[ok {done + skipped}] {arm} s{seed} ep={rec['epochs_run']:2d} "
                      f"best={rec['best_epoch_primary']:2d} valF1={rec['best_val_macro_f1']:.4f} "
                      f"{rec['seconds']:.1f}s", flush=True)

    parts = [pd.read_parquet(p) for arm in a.arms for seed in a.seeds
             for p in [_run_path(a.out, arm, seed)] if p.exists()]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(a.out / "oof_all.parquet", index=False)
        print(f"[combined] oof_all.parquet rows={len(df)} runs={len(parts)}", flush=True)
    print(f"[finished] ran={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
