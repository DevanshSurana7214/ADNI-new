#!/usr/bin/env python3
"""Step 10 (1) - counterfactual mask intervention on the S1 test folds (PIPELINE_v2 section 8.4).

Uses the best-state weights saved by ``run_step10_s1.py`` for arms A0, A1 and A4 (both checkpoint
rules). For every test episode with >= 2 observed modalities and every observed modality k:

  (a) REMOVE k: set mask bit k to 0 and re-predict (feature values are ignored once the mask is 0);
  (b) CONTENT SWAP k: replace k's values with those of a randomly chosen OTHER test episode that has
      k present, keeping the mask;
and record the total-variation (TV) change in the predicted probabilities versus the unmodified
prediction. A model that reacts to the mask more than to the content is using the shortcut:
report the ratio (a) / (b).

All predictions use the rho-uniform (pattern-neutral) operating point = softmax of the raw logits,
identical for every arm. Aggregated per (arm, checkpoint rule, modality) and pooled per (arm, rule).

Writes results/step10/diagnostics/intervention_{primary,secondary}.{csv,md} and intervention.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from adni.diagnostics import choose_donors, remove_modality, swap_modality, tv_change
from adni.models import MODALITIES, FusionModel, ModelConfig
from adni.preprocess import Preprocessor
from adni.train import make_tensors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_S1 = PROJECT_ROOT / "results" / "step10" / "s1"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step10" / "diagnostics"

MODEL_KEYS = ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers", "ff_dim",
              "attn_dropout", "n_classes", "modality_dropout")
CKPT_RULES = ("primary", "secondary")
ARMS = ("A0", "A1", "A4")


def _fold_ids(blob, seed, fold):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


@torch.no_grad()
def _probs(model, feats, mask, device) -> np.ndarray:
    logits, _, _ = model(feats, mask, apply_modality_dropout=False)
    return torch.softmax(logits, dim=1).cpu().numpy()


@torch.no_grad()
def _raw_logits(model, feats, mask, device) -> torch.Tensor:
    logits, _, _ = model(feats, mask, apply_modality_dropout=False)
    return logits


def _aggregate(records: pd.DataFrame) -> pd.DataFrame:
    """Mean TV and mean predicted class shares per (arm, rule, modality), plus pooled rows."""
    rows = []
    for (arm, rule, mod), g in records.groupby(["arm", "ckpt_rule", "modality"]):
        n = int(g["n_episodes"].sum())
        w = g["n_episodes"].to_numpy(dtype=float)
        mean_rm = float(np.average(g["mean_tv_remove"], weights=w))
        mean_sw = float(np.average(g["mean_tv_swap"], weights=w))
        rows.append({
            "arm": arm, "ckpt_rule": rule, "modality": mod, "n_episodes": n,
            "mean_tv_remove": mean_rm, "mean_tv_swap": mean_sw,
            "ratio_remove_over_swap": mean_rm / mean_sw if mean_sw > 0 else float("nan"),
            "meanp_CN_orig": float(np.average(g["meanp_CN_orig"], weights=w)),
            "meanp_MCI_orig": float(np.average(g["meanp_MCI_orig"], weights=w)),
            "meanp_Dem_orig": float(np.average(g["meanp_Dem_orig"], weights=w)),
            "meanp_CN_remove": float(np.average(g["meanp_CN_remove"], weights=w)),
            "meanp_MCI_remove": float(np.average(g["meanp_MCI_remove"], weights=w)),
            "meanp_Dem_remove": float(np.average(g["meanp_Dem_remove"], weights=w)),
            "meanp_CN_swap": float(np.average(g["meanp_CN_swap"], weights=w)),
            "meanp_MCI_swap": float(np.average(g["meanp_MCI_swap"], weights=w)),
            "meanp_Dem_swap": float(np.average(g["meanp_Dem_swap"], weights=w)),
        })
    # pooled across modalities (episode-weighted)
    for (arm, rule), g in records.groupby(["arm", "ckpt_rule"]):
        w = g["n_episodes"].to_numpy(dtype=float)
        mean_rm = float(np.average(g["mean_tv_remove"], weights=w))
        mean_sw = float(np.average(g["mean_tv_swap"], weights=w))
        rows.append({
            "arm": arm, "ckpt_rule": rule, "modality": "ALL", "n_episodes": int(w.sum()),
            "mean_tv_remove": mean_rm, "mean_tv_swap": mean_sw,
            "ratio_remove_over_swap": mean_rm / mean_sw if mean_sw > 0 else float("nan"),
            "meanp_CN_orig": float(np.average(g["meanp_CN_orig"], weights=w)),
            "meanp_MCI_orig": float(np.average(g["meanp_MCI_orig"], weights=w)),
            "meanp_Dem_orig": float(np.average(g["meanp_Dem_orig"], weights=w)),
            "meanp_CN_remove": float(np.average(g["meanp_CN_remove"], weights=w)),
            "meanp_MCI_remove": float(np.average(g["meanp_MCI_remove"], weights=w)),
            "meanp_Dem_remove": float(np.average(g["meanp_Dem_remove"], weights=w)),
            "meanp_CN_swap": float(np.average(g["meanp_CN_swap"], weights=w)),
            "meanp_MCI_swap": float(np.average(g["meanp_MCI_swap"], weights=w)),
            "meanp_Dem_swap": float(np.average(g["meanp_Dem_swap"], weights=w)),
        })
    return pd.DataFrame(rows)


def _write_md(df: pd.DataFrame, out: Path, rule: str) -> None:
    L = [f"# Step 10 - mask intervention ({rule} checkpoint rule)", "",
         "Test episodes with >= 2 observed modalities. (a) REMOVE an observed modality (mask -> 0); "
         "(b) CONTENT SWAP its values with a random other test episode that has it (mask kept). "
         "Predictions use the rho-uniform operating point (softmax of the raw logits). mean TV over "
         "episodes; ratio = (a)/(b) > 1 means the model reacts to the mask beyond the lost content.",
         "",
         "| arm | ckpt | modality | n | mean TV remove (a) | mean TV swap (b) | ratio (a)/(b) | "
         "predicted shares orig (CN/MCI/Dem) | remove | swap |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["arm", "ckpt_rule", "modality"]).iterrows():
        shares = lambda a, b, c: f"{a:.3f}/{b:.3f}/{c:.3f}"
        L.append(
            f"| {r['arm']} | {r['ckpt_rule']} | {r['modality']} | {int(r['n_episodes'])} | "
            f"{r['mean_tv_remove']:.4f} | {r['mean_tv_swap']:.4f} | {r['ratio_remove_over_swap']:.3f} | "
            f"{shares(r['meanp_CN_orig'], r['meanp_MCI_orig'], r['meanp_Dem_orig'])} | "
            f"{shares(r['meanp_CN_remove'], r['meanp_MCI_remove'], r['meanp_Dem_remove'])} | "
            f"{shares(r['meanp_CN_swap'], r['meanp_MCI_swap'], r['meanp_Dem_swap'])} |")
    (out / f"intervention_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10 mask-intervention diagnostic.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--s1", type=Path, default=DEFAULT_S1)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    pp = Preprocessor(ep, {k: pd.read_csv(a.blocks / f"{k}.csv").set_index("episode_id")
                           for k in MODALITIES}, json.loads(Path(a.feature_spec).read_text()))
    model_kwargs = {k: cfg[k] for k in MODEL_KEYS}

    records = []
    for arm in a.arms:
        for seed in a.seeds:
            for fold in a.folds:
                tr, va, te = _fold_ids(blob, seed, fold)
                pp.fit(tr)
                feat_dims = pp.kept_counts()
                test = make_tensors(pp, te, ep).to(a.device)
                feats, mask = test.feats, test.mask
                n_mod = mask.sum(dim=1)
                for rule in CKPT_RULES:
                    wpath = a.s1 / "weights" / f"{arm}_s{seed}_f{fold}_{rule}.pt"
                    if not wpath.exists():
                        print(f"[missing] {wpath}", flush=True)
                        continue
                    model = FusionModel(ModelConfig(feat_dims=feat_dims, **model_kwargs)).to(a.device)
                    model.load_state_dict(torch.load(wpath, map_location=a.device, weights_only=True))
                    model.eval()
                    p_orig = _probs(model, feats, mask, a.device)
                    for ki, k in enumerate(MODALITIES):
                        sel = (mask[:, ki].bool() & (n_mod >= 2)).cpu().numpy()
                        rows_k = np.flatnonzero(sel)
                        if rows_k.size == 0:
                            continue
                        pool_k = np.flatnonzero(mask[:, ki].cpu().numpy())
                        rows_t = torch.as_tensor(rows_k, device=a.device)

                        nf, nm = remove_modality(feats, mask, k, rows_t)
                        p_rem = _probs(model, nf, nm, a.device)

                        donors = choose_donors(rows_k, pool_k, seed=seed * 1_000_000 + fold * 100 + ki)
                        donor_t = torch.as_tensor(donors, device=a.device)
                        sf, sm = swap_modality(feats, mask, k, rows_t, donor_t)
                        p_swp = _probs(model, sf, sm, a.device)

                        tv_rm = tv_change(p_orig[rows_k], p_rem[rows_k])
                        tv_sw = tv_change(p_orig[rows_k], p_swp[rows_k])
                        mp_o = p_orig[rows_k].mean(axis=0)
                        mp_r = p_rem[rows_k].mean(axis=0)
                        mp_s = p_swp[rows_k].mean(axis=0)
                        records.append({
                            "arm": arm, "ckpt_rule": rule, "modality": k, "n_episodes": int(rows_k.size),
                            "mean_tv_remove": float(tv_rm.mean()), "mean_tv_swap": float(tv_sw.mean()),
                            "meanp_CN_orig": float(mp_o[0]), "meanp_MCI_orig": float(mp_o[1]),
                            "meanp_Dem_orig": float(mp_o[2]),
                            "meanp_CN_remove": float(mp_r[0]), "meanp_MCI_remove": float(mp_r[1]),
                            "meanp_Dem_remove": float(mp_r[2]),
                            "meanp_CN_swap": float(mp_s[0]), "meanp_MCI_swap": float(mp_s[1]),
                            "meanp_Dem_swap": float(mp_s[2]),
                        })
                print(f"[done] {arm} s{seed} f{fold}", flush=True)

    recs = pd.DataFrame(records)
    recs.to_csv(a.out / "intervention_raw.csv", index=False)
    agg = _aggregate(recs)
    for rule in CKPT_RULES:
        sub = agg[agg["ckpt_rule"] == rule]
        sub.to_csv(a.out / f"intervention_{rule}.csv", index=False)
        _write_md(sub, a.out, rule)
    summary = {
        "operating_point": "rho uniform (softmax of raw logits); identical for every arm",
        "modalities": list(MODALITIES),
        "rows": json.loads(agg.to_json(orient="records")),
    }
    (a.out / "intervention.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[wrote intervention tables under {a.out}]", flush=True)


if __name__ == "__main__":
    main()
