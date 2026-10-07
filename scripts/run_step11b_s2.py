#!/usr/bin/env python3
"""Step 11b (A) - confirmatory S2 check for A4 with tau = 0.5.

Held-out check of the tau = 0.5 setting that was selected from the S1 Step 11a results: train on
the S2 train side (ADNI1/GO/2) with the frozen S2 validation holdout, test on the ADNI3/4-enrolled
subjects. Three model seeds (the split is fixed, so a seed varies the initialisation only).

New run-sets (pre-registered in configs/step11b.yaml -> confirmatory.new_runsets):

  S2R1_A4_tau0p5   recipe 1, A4, tau 0.5
  S2R2_A0 / S2R2_A1 / S2R2_A4      the R2 recipe (Step 11a recipe_2) at tau 1
  S2R2_A4_tau0p5   the R2 recipe, A4, tau 0.5

Comparators reused WITHOUT retraining (results/step10/s2): S2 A0, A1 and A4 (tau 1, recipe 1).
The pre-declared reading: tau 0.5 is confirmed iff, for A4(tau 0.5) - A0,
  (i)  the PN-BAcc 95% CI lower bound is > 0 AND
  (ii) the macro-F1 95% CI lower bound is > -0.02.
Both checkpoint rules are evaluated; the rule is applied exactly as pre-declared and tau is not
changed afterwards.

Writes results/step11b/s2/{runs/<runset>_s<seed>.parquet, run_log.jsonl, oof_all.parquet}.
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
DEFAULT_SPEC11B = PROJECT_ROOT / "configs" / "step11b.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step11b" / "s2"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
OPS_PLAIN = (("argmax", "uniform"),)
OPS_SECONDARY_ARMS = (("uniform", "uniform"), ("P(c)", "class_prior"))
OPS_BY_ARM = {"A0": OPS_PLAIN, "A1": OPS_SECONDARY_ARMS, "A4": OPS_SECONDARY_ARMS}
PRIMARY_OP_BY_ARM = {"A0": "argmax", "A1": "uniform", "A4": "uniform"}
OOF_COLUMNS = ["episode_id", "seed", "fold", "runset", "arm", "ckpt_rule", "operating_point",
               "p_CN", "p_MCI", "p_Dementia", "logits", "best_epoch", "epochs_run"]


def build_runsets(base_cfg: dict, spec: dict) -> dict[str, dict]:
    """Resolve the five confirmatory run-sets from the pre-registered yaml (pure, testable)."""
    conf = spec["confirmatory"]
    r2 = spec["exploratory"]["r2"]
    out: dict[str, dict] = {}
    for item in conf["new_runsets"]:
        name = str(item["name"])
        if name in out:
            raise ValueError(f"duplicate confirmatory run-set {name!r}")
        over = dict(r2) if item["recipe"] == "R2" else {}
        cfg = {**base_cfg, **over}
        out[name] = {"arm": str(item["arm"]), "tau": float(item["tau"]), "cfg": cfg,
                     "primary_op": PRIMARY_OP_BY_ARM[str(item["arm"])],
                     "ops": OPS_BY_ARM[str(item["arm"])]}
    return out


def _s2_ids(blob):
    node = blob["s2"]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _run_path(out: Path, runset: str, seed: int) -> Path:
    return out / "runs" / f"{runset}_s{seed}.parquet"


def _test_logits(model, data, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.feats, data.mask, apply_modality_dropout=False)
    return logits


def _run_one(runset: str, rs: dict, seed: int, ep, tr, va, te, pp, device, out) -> dict:
    arm = rs["arm"]
    cfg = rs["cfg"]
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
    prior = training_prior_table(ep, tr)
    vids = np.unique(va)
    w_val = pn_weights(ep["y"].loc[vids].to_numpy(), ep["pattern"].loc[vids].to_numpy(),
                       prior, rho="uniform")

    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                       batch_size=cfg["batch_size"], max_epochs=cfg["max_epochs"],
                       patience=cfg["patience"], seed=seed, device=device,
                       deterministic=cfg["deterministic"], tau=rs["tau"])
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
        for op_label, rho in rs["ops"]:
            probs = torch.softmax(inference_scores(logits, priors, rho), dim=1).cpu().numpy()
            for i, eid in enumerate(ids):
                rows.append({"episode_id": int(eid), "seed": int(seed), "fold": 0,
                             "runset": runset, "arm": arm, "ckpt_rule": rule,
                             "operating_point": op_label,
                             "p_CN": float(probs[i, 0]), "p_MCI": float(probs[i, 1]),
                             "p_Dementia": float(probs[i, 2]),
                             "logits": json.dumps([float(x) for x in logits_np[i]]),
                             "best_epoch": best_epoch, "epochs_run": int(result["epochs_run"])})
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(_run_path(out, runset, seed), index=False)
    return {"runset": runset, "arm": arm, "tau": rs["tau"], "seed": int(seed), "fold": 0,
            "status": "ok", "epochs_run": int(result["epochs_run"]),
            "best_epoch_primary": int(result["best"]["epoch"]),
            "best_val_macro_f1": float(result["best"]["macro_f1"]),
            "best_epoch_secondary": int(result["best_pn_bacc"]["epoch"]),
            "best_val_pn_bacc": float(result["best_pn_bacc"]["pn_bacc"]),
            "n_parameters": int(result["n_parameters"]), "seconds": float(seconds),
            "nan_loss": False}


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 11b confirmatory S2 runner (A4 tau 0.5).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--step11b-spec", type=Path, default=DEFAULT_SPEC11B)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runsets", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--combine-only", action="store_true")
    a = ap.parse_args()
    (a.out / "runs").mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(a.config).read_text())
    spec = yaml.safe_load(Path(a.step11b_spec).read_text())
    runsets = build_runsets(base_cfg, spec)
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    tr, va, te = _s2_ids(blob)
    print(f"[S2] train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    def combine() -> None:
        parts = []
        for rs in runsets:
            for seed in a.seeds:
                p = _run_path(a.out, rs, seed)
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
    selected = [rs for rs in runsets if (not a.runsets or rs in set(a.runsets))]
    unknown = set(a.runsets or []) - set(runsets)
    if unknown:
        raise SystemExit(f"unknown run-set(s): {sorted(unknown)}; have {sorted(runsets)}")
    jobs = [(rs, s) for rs in selected for s in a.seeds]
    done = skipped = failed = 0
    for runset, seed in jobs:
        p = _run_path(a.out, runset, seed)
        if not a.force and p.exists() and p.stat().st_size:
            skipped += 1
            print(f"[skip] {runset} s{seed} (already done)", flush=True)
            continue
        try:
            pp.fit(tr)
            rec = _run_one(runset, runsets[runset], seed, ep, tr, va, te, pp, a.device, a.out)
            done += 1
        except Exception as exc:
            rec = {"runset": runset, "seed": int(seed), "fold": 0, "status": "error",
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            failed += 1
            print(f"[ERROR] {runset} s{seed}: {exc}", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["status"] == "ok":
            print(f"[ok {done + skipped}/{len(jobs)}] {runset} s{seed} ep={rec['epochs_run']:3d} "
                  f"best={rec['best_epoch_primary']:3d}/{rec['best_epoch_secondary']:3d} "
                  f"valF1={rec['best_val_macro_f1']:.4f} {rec['seconds']:.1f}s", flush=True)
    combine()
    print(f"[finished] ran={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
