#!/usr/bin/env python3
"""Step 8 pilot - train the loss arms A0-A4 on S1 seed 0 fold 0 only.

Every arm shares the frozen Step 7b config (``configs/model.yaml``) and the same trainer; only the
loss/prior changes. Priors are fitted on the TRAIN split only (Dirichlet alpha 0.5) and always use
the effective mask after modality dropout (p_drop = config modality_dropout).

Outputs under ``results/step08/``:
  * ``comparison.csv/.md``    - natural metrics, PN-BAcc (rho uniform and rho = P(c)), per-class F1,
                               best epoch / epochs run, for A0-A4 (A1/A4 at both operating points)
                               next to B1 and B1-blind on the same test episodes;
  * ``per_pattern.csv/.md``   - per-pattern metrics for patterns with >= 30 test episodes;
  * ``by_modality.csv/.md``   - metrics by number of observed modalities;
  * ``mask_sensitivity.csv``  - mean |change in predicted class probability| when one observed
                               modality is removed at test time, per arm;
  * ``curves_<arm>.csv``      - per-epoch training curves for each arm;
  * ``pilot_s0f0.json``       - full summary incl. red flags.

No tuning is done against test metrics; the test fold is read once per arm for reporting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from adni.baselines import training_prior_table  # noqa: E402
from adni.losses import ARMS, priors_from_counts, inference_scores  # noqa: E402
from adni.metrics import (  # noqa: E402
    CLASSES,
    natural_metrics,
    pn_balanced_accuracy,
    pn_weights,
)
from adni.models import MODALITIES  # noqa: E402
from adni.preprocess import Preprocessor  # noqa: E402
from adni.priors import crosstab  # noqa: E402
from adni.train import TrainConfig, make_tensors, train_arm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step08"
COMPARISON_ARMS = {
    "B1": PROJECT_ROOT / "results" / "step06b" / "oof_b1_i.parquet",
    "B1-blind": PROJECT_ROOT / "results" / "step06c" / "oof_b1_blind.parquet",
}
SECONDARY_ARMS = ("A1", "A4")   # arms reported at both rho=uniform and rho=P(c)
MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def _metrics_block(ep: pd.DataFrame, train_ids: np.ndarray, y: np.ndarray,
                   proba: np.ndarray, ids: np.ndarray) -> dict:
    prior = training_prior_table(ep, train_ids)
    pat = ep["pattern"].loc[list(ids)].to_numpy()
    m = natural_metrics(y, proba)
    wu = pn_weights(y, pat, prior, rho="uniform")
    pc = np.bincount(ep["y"].loc[list(train_ids)].to_numpy(), minlength=3).astype(float)
    pc = pc / pc.sum()
    wp = pn_weights(y, pat, prior, rho=pc)
    return {
        "n": int(len(ids)),
        "macro_auroc": float(m["macro_auroc"]),
        "macro_f1": float(m["macro_f1"]),
        "balanced_accuracy": float(m["balanced_accuracy"]),
        "pn_bacc_uniform": float(pn_balanced_accuracy(y, proba.argmax(1), wu)),
        "pn_bacc_prior": float(pn_balanced_accuracy(y, proba.argmax(1), wp)),
        "f1_CN": float(m["f1_per_class"][0]),
        "f1_MCI": float(m["f1_per_class"][1]),
        "f1_Dementia": float(m["f1_per_class"][2]),
    }


def _arm_proba(logits: torch.Tensor, priors, rho: str) -> np.ndarray:
    """Softmax probabilities at the requested operating point (argmax = the reported prediction)."""
    with torch.no_grad():
        return torch.softmax(inference_scores(logits, priors, rho), dim=1).cpu().numpy()


def _test_logits(model, data, device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(data.feats, data.mask, apply_modality_dropout=False)
    return logits


def _mask_sensitivity(model, data, device) -> float:
    """Mean |change in predicted class probability| over (episode, removed observed modality, class).

    Only episodes with >= 2 observed modalities and only currently-observed modalities are used.
    Probabilities are the arm's primary (rho uniform) softmax.
    """
    model.eval()
    mask = data.mask
    n_mod = mask.sum(dim=1)
    with torch.no_grad():
        full = torch.softmax(model(data.feats, mask, apply_modality_dropout=False)[0], dim=1)
    total, count = 0.0, 0
    for j in range(len(MODALITIES)):
        sel = mask[:, j] & (n_mod >= 2)
        if not bool(sel.any()):
            continue
        m2 = mask.clone()
        m2[sel, j] = False
        with torch.no_grad():
            p2 = torch.softmax(model(data.feats, m2, apply_modality_dropout=False)[0], dim=1)
        d = (full[sel] - p2[sel]).abs()
        total += float(d.sum())
        count += int(d.numel())
    return total / count if count else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 8 pilot (arms A0-A4, S1 seed 0 fold 0).")
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
    print(f"kept feature counts: {feat_dims}", flush=True)

    train_t = make_tensors(pp, tr, ep)
    val_t = make_tensors(pp, va, ep)
    test_dev = make_tensors(pp, te, ep).to(a.device)

    p_drop = float(cfg["modality_dropout"])
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats = crosstab(diag, sub["pattern"])
    priors = priors_from_counts(n_cm, pats, p_drop=p_drop)
    print(f"priors: {len(priors.submasks)} sub-masks, p_drop={p_drop}", flush=True)

    tcfg = TrainConfig(lr=cfg["lr"], weight_decay=cfg["weight_decay"], batch_size=cfg["batch_size"],
                       max_epochs=cfg["max_epochs"], patience=cfg["patience"], seed=a.seed,
                       device=a.device, deterministic=cfg["deterministic"])
    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}

    y_te = test_dev.y.cpu().numpy()
    te_ids = np.unique(te)
    n_mod_te = ep["n_mod"].loc[te_ids].to_numpy()
    pat_te = ep["pattern"].loc[te_ids].to_numpy()

    arm_rows: list[dict] = []
    per_pattern_rows: list[dict] = []
    by_mod_rows: list[dict] = []
    mask_rows: list[dict] = []
    summary_arms: dict[str, dict] = {}

    for arm in ARMS:
        model, result, seconds = train_arm(arm, feat_dims, train_t, val_t, tcfg, model_kwargs, priors)
        logits = _test_logits(model, test_dev, a.device)
        pd.DataFrame(result["history"]).to_csv(a.out / f"curves_{arm}.csv", index=False)

        best_ep = int(result["best"]["epoch"])
        val_best = result["history"][best_ep]
        proba = _arm_proba(logits, priors, "uniform")
        block = _metrics_block(ep, tr, y_te, proba, te_ids)
        common = {"best_epoch": best_ep, "epochs_run": int(result["epochs_run"]),
                  "n_parameters": int(result["n_parameters"]), "seconds": float(seconds)}
        arm_rows.append({"arm": arm, "operating_point": "rho=uniform (primary)", **block, **common})
        block2 = None
        if arm in SECONDARY_ARMS:
            proba2 = _arm_proba(logits, priors, "class_prior")
            block2 = _metrics_block(ep, tr, y_te, proba2, te_ids)
            arm_rows.append({"arm": arm, "operating_point": "rho=P(c) (secondary)", **block2, **common})

        for p in sorted(set(pat_te.tolist())):
            sel = pat_te == p
            if int(sel.sum()) < 30:
                continue
            per_pattern_rows.append({"arm": arm, "pattern": p,
                                     **_metrics_block(ep, tr, y_te[sel], proba[sel], te_ids[sel])})
        for nm in (1, 2, 3, 4):
            sel = n_mod_te == nm
            if not bool(sel.any()):
                continue
            by_mod_rows.append({"arm": arm, "n_mod": nm,
                                **_metrics_block(ep, tr, y_te[sel], proba[sel], te_ids[sel])})

        sens = _mask_sensitivity(model, test_dev, a.device)
        mask_rows.append({"arm": arm, "mean_abs_delta_prob": float(sens)})

        nan_loss = not all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"])
                           for h in result["history"])
        red = {
            "best_epoch_below_5": best_ep < 5,
            "val_test_gap_above_0.05": abs(float(val_best["val_macro_auroc"]) - block["macro_auroc"]) > 0.05,
            "nan_loss": bool(nan_loss),
        }
        summary_arms[arm] = {
            "arm": arm, **common,
            "val_macro_f1_at_best": float(val_best["val_macro_f1"]),
            "val_macro_auroc_at_best": float(val_best["val_macro_auroc"]),
            "test_primary": block,
            "test_secondary": block2,
            "mask_sensitivity": float(sens),
            "red_flags": red,
        }
        print(f"[{arm}] epochs {result['epochs_run']:2d} best {best_ep:2d} "
              f"test macro-AUROC {block['macro_auroc']:.4f} macro-F1 {block['macro_f1']:.4f} "
              f"PN-BAcc(u) {block['pn_bacc_uniform']:.4f} mask-sens {sens:.4f} red {red}",
              flush=True)

    for name, path in COMPARISON_ARMS.items():
        df = pd.read_parquet(path)
        df = df[(df["seed"] == a.seed) & (df["fold"] == a.fold)]
        ids = np.intersect1d(df["episode_id"].to_numpy(), te_ids)
        df = df.set_index("episode_id").loc[ids]
        proba = df[["p_CN", "p_MCI", "p_Dementia"]].to_numpy()
        block = _metrics_block(ep, tr, ep["y"].loc[ids].to_numpy(), proba, ids)
        arm_rows.append({"arm": name, "operating_point": "-", **block,
                         "best_epoch": None, "epochs_run": None, "n_parameters": None, "seconds": None})
        print(f"[{name}] n {block['n']} macro-AUROC {block['macro_auroc']:.4f} "
              f"macro-F1 {block['macro_f1']:.4f} PN-BAcc(u) {block['pn_bacc_uniform']:.4f}", flush=True)

    # ------------------------------------------------------------------ tables
    comp = pd.DataFrame(arm_rows)
    comp.to_csv(a.out / "comparison.csv", index=False)
    md = ["# Step 8 pilot - loss arms A0-A4 on S1 seed 0 fold 0", "",
          "Frozen config from Step 7b; priors fitted on the TRAIN split only (alpha 0.5, "
          f"p_drop={p_drop}). B1 and B1-blind use the same test episodes.", "",
          "| arm | operating point | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) "
          "| F1 CN | F1 MCI | F1 Dem | best epoch | epochs |",
          "|-----|-----------------|---|-------------|----------|------|------------|---------------"
          "|-------|--------|--------|------------|--------|"]
    for r in arm_rows:
        be = "-" if r["best_epoch"] is None else str(int(r["best_epoch"]))
        er = "-" if r["epochs_run"] is None else str(int(r["epochs_run"]))
        md.append(f"| {r['arm']} | {r['operating_point']} | {r['n']} | {r['macro_auroc']:.4f} | "
                  f"{r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['pn_bacc_uniform']:.4f} | "
                  f"{r['pn_bacc_prior']:.4f} | {r['f1_CN']:.4f} | {r['f1_MCI']:.4f} | "
                  f"{r['f1_Dementia']:.4f} | {be} | {er} |")
    (a.out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    per = pd.DataFrame(per_pattern_rows)
    per.to_csv(a.out / "per_pattern.csv", index=False)
    pmd = ["# Step 8 pilot - per-pattern test metrics (patterns with >= 30 test episodes)", "",
           "| arm | pattern | n | macro-AUROC | macro-F1 | PN-BAcc(u) |",
           "|-----|---------|---|-------------|----------|------------|"]
    for r in per_pattern_rows:
        pmd.append(f"| {r['arm']} | {r['pattern']} | {r['n']} | {r['macro_auroc']:.4f} | "
                   f"{r['macro_f1']:.4f} | {r['pn_bacc_uniform']:.4f} |")
    (a.out / "per_pattern.md").write_text("\n".join(pmd) + "\n", encoding="utf-8")

    bymod = pd.DataFrame(by_mod_rows)
    bymod.to_csv(a.out / "by_modality.csv", index=False)
    bmd = ["# Step 8 pilot - metrics by number of observed modalities", "",
           "| arm | n_mod | n | macro-AUROC | macro-F1 | PN-BAcc(u) |",
           "|-----|-------|---|-------------|----------|------------|"]
    for r in by_mod_rows:
        bmd.append(f"| {r['arm']} | {r['n_mod']} | {r['n']} | {r['macro_auroc']:.4f} | "
                   f"{r['macro_f1']:.4f} | {r['pn_bacc_uniform']:.4f} |")
    (a.out / "by_modality.md").write_text("\n".join(bmd) + "\n", encoding="utf-8")

    mask = pd.DataFrame(mask_rows)
    mask.to_csv(a.out / "mask_sensitivity.csv", index=False)
    smd = ["# Step 8 pilot - mask sensitivity", "",
           "Mean |change in predicted class probability| when one observed modality is removed at "
           "test time (episodes with >= 2 observed modalities; primary rho=uniform probabilities).", "",
           "| arm | mean abs delta prob |", "|-----|---------------------|"]
    for r in mask_rows:
        smd.append(f"| {r['arm']} | {r['mean_abs_delta_prob']:.4f} |")
    (a.out / "mask_sensitivity.md").write_text("\n".join(smd) + "\n", encoding="utf-8")

    summary = {
        "seed": a.seed, "fold": a.fold, "device": a.device,
        "config": str(a.config), "frozen_lr": cfg["lr"], "frozen_weight_decay": cfg["weight_decay"],
        "p_drop": p_drop, "n_submasks": len(priors.submasks), "feature_dims": feat_dims,
        "arms": summary_arms,
        "baselines": {r["arm"]: {k: r[k] for k in ("n", "macro_auroc", "macro_f1", "balanced_accuracy",
                                                   "pn_bacc_uniform", "pn_bacc_prior", "f1_CN",
                                                   "f1_MCI", "f1_Dementia")}
                      for r in arm_rows if r["arm"] in COMPARISON_ARMS},
    }
    (a.out / "pilot_s0f0.json").write_text(json.dumps(summary, indent=2))
    print(f"[wrote outputs under {a.out}]")


if __name__ == "__main__":
    main()
