#!/usr/bin/env python3
"""Step 11b (B) - exploratory arms A5-A8 under R2 on the S1 grid (EXPLORATORY).

Pre-registered in ``configs/step11b.yaml`` (exploratory). Everything follows the Step 9/11a
protocol exactly (R2 recipe, both checkpoint rules, Step 6b uniform aggregation, seeds fixed
before the model is built); only the arm differs. Comparator = A4(R2, tau 1) from Step 11a
(results/step11a, run-set R2_A4) - REUSED, not retrained.

Per (arm, seed, fold) run it writes under ``results/step11b/s1/``:

  runs/<arm>_s<seed>_f<fold>.parquet     OOF rows (both checkpoint rules, all operating points)
  curves/<arm>_s<seed>_f<fold>.csv       per-epoch loss components + validation curves
  swap/<arm>_s<seed>_f<fold>.parquet     embedding-swap test rows (rho uniform operating point)
  intervention/<arm>_s<seed>_f<fold>.csv Step 10 mask intervention (remove vs content swap)

Restarting skips a run whose outputs exist and parse (--force re-runs). --shard K --nshards N
runs a deterministic round-robin subset; --combine-only rebuilds oof_all.parquet and the
diagnostic aggregates without training.
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
from adni.diagnostics import choose_donors, remove_modality, swap_modality, tv_change
from adni.exploratory import (
    EXPLORATORY_ARMS,
    ExploratoryConfig,
    embedding_swap_probs,
    train_exploratory_arm,
)
from adni.losses import inference_scores, priors_from_counts
from adni.metrics import CLASSES, pn_weights
from adni.models import MODALITIES
from adni.preprocess import Preprocessor
from adni.priors import crosstab
from adni.train import TrainConfig, make_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_SPEC11B = PROJECT_ROOT / "configs" / "step11b.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step11b" / "s1"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
PRIMARY_OP = "uniform"          # the A4 primary operating point (rho uniform) for every arm here
OOF_COLUMNS = ["episode_id", "seed", "fold", "runset", "arm", "ckpt_rule", "operating_point",
               "p_CN", "p_MCI", "p_Dementia", "logits", "best_epoch", "epochs_run"]
CURVE_COLUMNS = ["epoch", "train_loss", "train_ce", "train_consistency", "train_prototype",
                 "val_loss", "val_macro_f1", "val_macro_auroc", "val_pn_bacc"]
SWAP_COLUMNS = ["episode_id", "seed", "fold", "arm", "ckpt_rule", "pattern_self",
                "pattern_donor", "p_CN", "p_MCI", "p_Dementia",
                "p_CN_swap", "p_MCI_swap", "p_Dementia_swap", "tv"]


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _paths(out: Path, arm: str, seed: int, fold: int):
    tag = f"{arm}_s{seed}_f{fold}"
    return {
        "oof": out / "runs" / f"{tag}.parquet",
        "curve": out / "curves" / f"{tag}.csv",
        "swap": out / "swap" / f"{tag}.parquet",
        "interv": out / "intervention" / f"{tag}.csv",
    }


def _already_done(paths: dict) -> bool:
    needed = (paths["oof"], paths["curve"], paths["swap"], paths["interv"])
    if not all(p.exists() and p.stat().st_size for p in needed):
        return False
    try:
        pd.read_parquet(paths["oof"])
        pd.read_parquet(paths["swap"])
        pd.read_csv(paths["curve"])
        pd.read_csv(paths["interv"])
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


@torch.no_grad()
def _probs(model, feats, mask) -> np.ndarray:
    logits, _, _, _ = model(feats, mask, apply_modality_dropout=False)
    return torch.softmax(logits, dim=1).cpu().numpy()


@torch.no_grad()
def _raw_logits(model, feats, mask) -> torch.Tensor:
    logits, _, _, _ = model(feats, mask, apply_modality_dropout=False)
    return logits


def _run_intervention(model, feats, mask, seed: int, fold: int, device: str) -> pd.DataFrame:
    """Step 10 mask-intervention diagnostic (remove vs content swap) at the rho-uniform op point."""
    n_mod = mask.sum(dim=1)
    records = []
    for ki, k in enumerate(MODALITIES):
        sel = (mask[:, ki].bool() & (n_mod >= 2)).cpu().numpy()
        rows_k = np.flatnonzero(sel)
        if rows_k.size == 0:
            continue
        pool_k = np.flatnonzero(mask[:, ki].cpu().numpy())
        rows_t = torch.as_tensor(rows_k, device=device)
        nf, nm = remove_modality(feats, mask, k, rows_t)
        p_rem = _probs(model, nf, nm)
        donors = choose_donors(rows_k, pool_k, seed=seed * 1_000_000 + fold * 100 + ki)
        donor_t = torch.as_tensor(donors, device=device)
        sf, sm = swap_modality(feats, mask, k, rows_t, donor_t)
        p_swp = _probs(model, sf, sm)
        p_orig_full = _probs(model, feats, mask)
        records.append({
            "modality": k, "n_episodes": int(rows_k.size),
            "mean_tv_remove": float(tv_change(p_orig_full[rows_k], p_rem[rows_k]).mean()),
            "mean_tv_swap": float(tv_change(p_orig_full[rows_k], p_swp[rows_k]).mean()),
        })
    return pd.DataFrame(records)


def _run_one(arm: str, seed: int, fold: int, ep, blob, pp, device, out, tcfg_kwargs: dict,
             model_kwargs: dict) -> dict:
    tr, va, te = _fold_ids(blob, seed, fold)
    pp.fit(tr)
    feat_dims = pp.kept_counts()
    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_t = make_tensors(pp, te, ep)
    test_dev = test_t.to(device)

    p_drop = float(model_kwargs["modality_dropout"])
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    priors = priors_from_counts(n_cm, pats, p_drop=p_drop)
    w_val = _val_pn_weights(ep, tr, va)

    tcfg = TrainConfig(arm=arm, seed=seed, device=device, **tcfg_kwargs)
    model, result, seconds = train_exploratory_arm(arm, feat_dims, train_t, val_t, tcfg,
                                                   model_kwargs, priors, w_val)
    ids = np.unique(te)
    rows: list[dict] = []
    for rule in CKPT_RULES:
        if rule == "primary":
            state, best_epoch = result["best_state"], int(result["best"]["epoch"])
        else:
            state, best_epoch = result["best_pn_bacc_state"], int(result["best_pn_bacc"]["epoch"])
        model.load_state_dict(state)
        logits = _raw_logits(model, test_dev.feats, test_dev.mask)
        logits_np = logits.cpu().numpy()
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        for i, eid in enumerate(ids):
            rows.append({"episode_id": int(eid), "seed": int(seed), "fold": int(fold),
                         "runset": arm, "arm": arm, "ckpt_rule": rule,
                         "operating_point": PRIMARY_OP,
                         "p_CN": float(probs[i, 0]), "p_MCI": float(probs[i, 1]),
                         "p_Dementia": float(probs[i, 2]),
                         "logits": json.dumps([float(x) for x in logits_np[i]]),
                         "best_epoch": best_epoch, "epochs_run": int(result["epochs_run"])})

    # ---- embedding-swap diagnostic (both checkpoint rules; rho-uniform operating point) ----
    donor_seed = seed * 10_000 + fold
    rng = np.random.default_rng(donor_seed)
    # a random OTHER episode as donor: shift by 1 + r (r in [0, n-1)) so donor != self always
    r = rng.integers(0, max(len(ids) - 1, 1), size=len(ids))
    donors = (np.arange(len(ids)) + 1 + r) % len(ids)
    swap_rows: list[dict] = []
    for rule in CKPT_RULES:
        state = result["best_state"] if rule == "primary" else result["best_pn_bacc_state"]
        model.load_state_dict(state)
        p_o, p_s, pat_o, pat_d = embedding_swap_probs(model, test_dev, donors, device)
        tv = 0.5 * np.abs(p_s - p_o).sum(axis=1)
        for i, eid in enumerate(ids):
            swap_rows.append({"episode_id": int(eid), "seed": int(seed), "fold": int(fold),
                              "arm": arm, "ckpt_rule": rule, "pattern_self": pat_o[i],
                              "pattern_donor": pat_d[i],
                              "p_CN": float(p_o[i, 0]), "p_MCI": float(p_o[i, 1]),
                              "p_Dementia": float(p_o[i, 2]),
                              "p_CN_swap": float(p_s[i, 0]), "p_MCI_swap": float(p_s[i, 1]),
                              "p_Dementia_swap": float(p_s[i, 2]), "tv": float(tv[i])})

    # ---- Step 10 mask-intervention diagnostic (primary-rule state) ----
    model.load_state_dict(result["best_state"])
    interv = _run_intervention(model, test_dev.feats, test_dev.mask, seed, fold, device)
    interv.insert(0, "arm", arm)
    interv.insert(1, "ckpt_rule", "primary")
    interv.insert(2, "seed", seed)
    interv.insert(3, "fold", fold)

    hist = pd.DataFrame(result["history"])
    for c in CURVE_COLUMNS:
        if c not in hist.columns:
            hist[c] = np.nan
    paths = _paths(out, arm, seed, fold)
    pd.DataFrame(rows, columns=OOF_COLUMNS).to_parquet(paths["oof"], index=False)
    hist[CURVE_COLUMNS].to_csv(paths["curve"], index=False)
    pd.DataFrame(swap_rows, columns=SWAP_COLUMNS).to_parquet(paths["swap"], index=False)
    interv.to_csv(paths["interv"], index=False)

    nan_loss = bool(not np.isfinite(hist["train_loss"]).all()
                    or not np.isfinite(hist["val_loss"]).all())
    return {"runset": arm, "arm": arm, "seed": int(seed), "fold": int(fold), "status": "ok",
            "epochs_run": int(result["epochs_run"]),
            "best_epoch_primary": int(result["best"]["epoch"]),
            "best_val_macro_f1": float(result["best"]["macro_f1"]),
            "best_epoch_secondary": int(result["best_pn_bacc"]["epoch"]),
            "best_val_pn_bacc": float(result["best_pn_bacc"]["pn_bacc"]),
            "n_parameters": int(result["n_parameters"]), "seconds": float(seconds),
            "nan_loss": nan_loss}

def main() -> None:
    ap = argparse.ArgumentParser(description="Step 11b exploratory S1 runner (A5-A8 under R2).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--step11b-spec", type=Path, default=DEFAULT_SPEC11B)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=list(EXPLORATORY_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--combine-only", action="store_true")
    ap.add_argument("--no-combine", action="store_true",
                    help="skip the end-of-run combine (use with --nshards)")
    a = ap.parse_args()
    for sub in ("runs", "curves", "swap", "intervention"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(a.config).read_text())
    spec = yaml.safe_load(Path(a.step11b_spec).read_text())
    exp = spec["exploratory"]
    r2 = exp["r2"]
    cfg = {**base_cfg, **r2}
    unknown = set(a.arms) - set(EXPLORATORY_ARMS)
    if unknown:
        raise SystemExit(f"unknown arm(s): {sorted(unknown)}; have {list(EXPLORATORY_ARMS)}")

    def combine() -> None:
        parts = []
        for arm in EXPLORATORY_ARMS:
            for seed in a.seeds:
                for fold in a.folds:
                    p = _paths(a.out, arm, seed, fold)["oof"]
                    if p.exists():
                        parts.append(pd.read_parquet(p))
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(a.out / "oof_all.parquet", index=False)
            print(f"[combined] oof_all.parquet rows={len(df)} "
                  f"runs={df.groupby(['runset', 'seed', 'fold']).ngroups}", flush=True)
        # aggregate swap + intervention diagnostics across runs
        swaps = [p for p in sorted((a.out / "swap").glob("*.parquet"))]
        if swaps:
            pd.concat([pd.read_parquet(p) for p in swaps], ignore_index=True).to_parquet(
                a.out / "swap_all.parquet", index=False)
        inter = [p for p in sorted((a.out / "intervention").glob("*.csv"))]
        if inter:
            pd.concat([pd.read_csv(p) for p in inter], ignore_index=True).to_csv(
                a.out / "intervention_raw.csv", index=False)

    if a.combine_only:
        combine()
        return

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    pp = Preprocessor(ep, {k: pd.read_csv(a.blocks / f"{k}.csv").set_index("episode_id")
                           for k in ("mri", "fdg", "amy", "tau")},
                      json.loads(Path(a.feature_spec).read_text()))
    tcfg_kwargs = {k: cfg[k] for k in ("lr", "weight_decay", "batch_size", "max_epochs",
                                       "patience", "deterministic")}
    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}
    xcfg = ExploratoryConfig.for_arm  # per-arm presets resolve inside train_exploratory_arm
    del xcfg

    log_path = a.out / (f"run_log_shard{a.shard}.jsonl" if a.nshards > 1 else "run_log.jsonl")
    jobs = [(arm, seed, fold) for arm in a.arms for seed in a.seeds for fold in a.folds]
    if a.nshards > 1:
        jobs = jobs[a.shard::a.nshards]
    total = len(jobs)
    done = skipped = failed = 0
    for arm, seed, fold in jobs:
        paths = _paths(a.out, arm, seed, fold)
        if not a.force and _already_done(paths):
            skipped += 1
            print(f"[skip] {arm} s{seed} f{fold} (already done)", flush=True)
            continue
        try:
            rec = _run_one(arm, seed, fold, ep, blob, pp, a.device, a.out, tcfg_kwargs,
                           model_kwargs)
            done += 1
        except Exception as exc:
            rec = {"runset": arm, "arm": arm, "seed": int(seed), "fold": int(fold),
                   "status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
            failed += 1
            print(f"[ERROR] {arm} s{seed} f{fold}: {exc}", flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if rec["status"] == "ok":
            print(f"[ok {done + skipped}/{total}] {arm} s{seed} f{fold} "
                  f"ep={rec['epochs_run']:3d} best={rec['best_epoch_primary']:3d}/"
                  f"{rec['best_epoch_secondary']:3d} valF1={rec['best_val_macro_f1']:.4f} "
                  f"valPN={rec['best_val_pn_bacc']:.4f} {rec['seconds']:.1f}s "
                  f"nan={rec['nan_loss']}", flush=True)

    if not a.no_combine:
        combine()
    print(f"[finished shard {a.shard}/{a.nshards}] ran={done} skipped={skipped} failed={failed} "
          f"(log: {log_path})", flush=True)


if __name__ == "__main__":
    main()
