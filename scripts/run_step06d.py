#!/usr/bin/env python3
"""Step 6d: Run B1-imp+mask arm and compute decomposition analysis."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from adni.baselines import (
    MODALITIES,
    build_raw_matrix,
    fit_predict_b1,
    load_blocks,
    load_feature_spec,
    training_prior_table,
)
from adni.metrics import (
    paired_cluster_bootstrap,
    pn_balanced_accuracy,
    pn_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
CONFIG = PROJECT_ROOT / "configs" / "b1.yaml"
STEP6B = PROJECT_ROOT / "results" / "step06b"
OUT = PROJECT_ROOT / "results" / "step06d"

PROBA_COLS = ["p_CN", "p_MCI", "p_Dementia"]
MASK_COLS = [f"mask_{k}" for k in MODALITIES]

spec = importlib.util.spec_from_file_location(
    "step06b_baselines", PROJECT_ROOT / "scripts" / "step06b_baselines.py"
)
s6b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6b)


def _fold_ids(blob, seed, fold):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return (np.asarray(node["train"], dtype=np.int64),
            np.asarray(node["val"], dtype=np.int64),
            np.asarray(node["test"], dtype=np.int64))


def _impute_blind_with_mask(X_raw, present, tr, va, te, feature_spec, rng):
    """B1-imp+mask: same imputation as B1-blind but KEEP mask bits."""
    X = X_raw.copy()
    all_ids = np.concatenate([tr, va, te])
    for k in MODALITIES:
        cols = [f"{k}_{c}" for c in feature_spec[k]]
        donors = np.array([i for i in tr.tolist() if i in present[k]], dtype=np.int64)
        if len(donors) == 0:
            raise RuntimeError(f"no present training donors for block {k}")
        absent = [i for i in all_ids.tolist() if i not in present[k]]
        if not absent:
            continue
        pick = rng.choice(donors, size=len(absent), replace=True)
        X.loc[absent, cols] = X_raw.loc[pick, cols].to_numpy()
    return X


def _weights_for(ep, tr, te):
    prior = training_prior_table(ep, tr)
    ytr = ep["y"].loc[list(tr)].to_numpy()
    pc = np.bincount(ytr, minlength=3).astype(float)
    pc = pc / pc.sum()
    yy = ep["y"].loc[list(te)].to_numpy()
    pp = ep["pattern"].loc[list(te)].to_numpy()
    return pd.DataFrame({
        "episode_id": np.asarray(te),
        "w_uniform": pn_weights(yy, pp, prior, rho="uniform"),
        "w_prior": pn_weights(yy, pp, prior, rho=pc),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    
    cfg = yaml.safe_load(CONFIG.read_text())
    num_rounds = int(cfg.pop("num_rounds", 1000))
    early_stopping = int(cfg.pop("early_stopping", 50))
    
    import pandas as pd
    ep = pd.read_csv(EPISODES, dtype={"pattern": str}).set_index("episode_id")
    feature_spec = load_feature_spec(SPEC)
    blocks = load_blocks(BLOCKS)
    blob = json.loads(SPLITS.read_text())
    
    X_raw = build_raw_matrix(ep, blocks, feature_spec)
    present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}
    
    # Load existing OOF predictions
    oof_b1 = pd.read_parquet(STEP6B / "oof_b1_i.parquet")
    oof_b1_blind = pd.read_parquet(OUT.parent / "step06c" / "oof_b1_blind.parquet")
    
    # Fit B1-imp+mask
    rows = []
    wmap = {}
    
    for seed in [0, 1, 2]:
        for fold in [0, 1, 2, 3, 4]:
            tr, va, te = _fold_ids(blob, seed, fold)
            wmap[(seed, fold)] = _weights_for(ep, tr, te)
            
            t0 = time.time()
            rng = np.random.default_rng(1000 * (seed + 1) + fold)
            Xb = _impute_blind_with_mask(X_raw, present, tr, va, te, feature_spec, rng)
            p, _ = fit_predict_b1(Xb, ep, tr, va, te, seed=seed, params_override=cfg,
                                  num_rounds=num_rounds, early_stopping=early_stopping)
            
            for i, eid in enumerate(te):
                rows.append({
                    "episode_id": int(eid), "fold": fold, "seed": seed,
                    **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}
                })
        print(f"[seed {seed}] b1_imp_mask done", flush=True)
    
    oof_imp_mask = pd.DataFrame(rows)
    oof_imp_mask.to_parquet(OUT / "oof_b1_imp_mask.parquet", index=False)
    
    def with_weights(df):
        parts = []
        for (seed, fold), w in wmap.items():
            sub = df[(df["seed"] == seed) & (df["fold"] == fold)]
            if len(sub):
                parts.append(sub.merge(w, on="episode_id", how="left"))
        out = pd.concat(parts, ignore_index=True)
        assert out["w_uniform"].notna().all()
        return out
    
    frames = {
        "b1_i": with_weights(oof_b1),
        "b1_blind": with_weights(oof_b1_blind),
        "b1_imp_mask": with_weights(oof_imp_mask),
    }
    
    # Uniform CI
    uniform = {}
    for arm in ["b1_i", "b1_imp_mask", "b1_blind"]:
        tab, _ = s6b._uniform_table(frames[arm], ep, 2000, boot_seed=0)
        uniform[arm] = tab
    
    # Paired diffs
    def _paired(arm_a, arm_b):
        ga = s6b._uniform_geps(frames[arm_a], ep)
        gb = s6b._uniform_geps(frames[arm_b], ep)
        assert (ga.index == gb.index).all()
        y = ep["y"].loc[ga.index].to_numpy()
        groups = ep["RID"].loc[ga.index].to_numpy()
        pa = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_u"].to_numpy()])
        pb = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_u"].to_numpy()])
        mf = s6b.metric_family()
        fam = {"macro_auroc": mf["macro_auroc"], "pn_bacc_uniform": mf["pn_bacc_uniform"]}
        out = {}
        for k, fn in fam.items():
            ci = paired_cluster_bootstrap(lambda yy, pp, _f=fn: _f(yy, pp), y, pa, pb, groups,
                                          n=2000, seed=0)
            out[k] = {"diff": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high}
        return out
    
    diffs = {
        "noise_cost": _paired("b1_i", "b1_imp_mask"),
        "mask_info": _paired("b1_imp_mask", "b1_blind"),
        "total": _paired("b1_i", "b1_blind"),
    }
    
    # By-modality
    def _by_modality(arm):
        g = s6b._uniform_geps(frames[arm], ep)
        idx = g.index
        n_mod = ep["n_mod"].loc[idx].to_numpy()
        y = ep["y"].loc[idx].to_numpy()
        proba = g[PROBA_COLS].to_numpy()
        w = g["w_u"].to_numpy()
        out = {}
        for nm in (1, 2, 3, 4):
            sel = n_mod == nm
            if sel.sum() == 0:
                continue
            from adni.metrics import natural_metrics
            m = natural_metrics(y[sel], proba[sel])
            out[f"n_mod={nm}"] = {"n": int(sel.sum()),
                                  "macro_auroc": float(m["macro_auroc"]),
                                  "pn_bacc_uniform": float(pn_balanced_accuracy(
                                      y[sel], proba[sel].argmax(axis=1), w[sel]))}
        return out
    
    bymod = {arm: _by_modality(arm) for arm in ["b1_i", "b1_imp_mask", "b1_blind"]}
    
    # Write markdown
    md = ["# Step 6d - B1 variants uniform CI (seed-averaged, 2000 resamples)", "",
          "| arm | macro-AUROC | PN-BAcc (uniform) |",
          "|-----|-------------|------------------|"]
    for arm in ["b1_i", "b1_imp_mask", "b1_blind"]:
        t = uniform[arm]
        md.append(f"| {arm} | {t['macro_auroc']['point']:.4f} [{t['macro_auroc']['ci_low']:.4f}, {t['macro_auroc']['ci_high']:.4f}] | "
                  f"{t['pn_bacc_uniform']['point']:.4f} [{t['pn_bacc_uniform']['ci_low']:.4f}, {t['pn_bacc_uniform']['ci_high']:.4f}] |")
    md.append("")
    md.append("## Decomposition")
    md.append("")
    md.append("| Component | macro-AUROC diff |")
    md.append("|-----------|-----------------|")
    for name, key in [("noise cost (B1 - B1-imp+mask)", "noise_cost"),
                       ("mask information (B1-imp+mask - B1-blind)", "mask_info"),
                       ("total (B1 - B1-blind)", "total")]:
        d = diffs[key]["macro_auroc"]
        md.append(f"| {name} | {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] |")
    md.append("")
    md.append("## By number of observed modalities")
    md.append("")
    md.append("| n_mod | n_ep | B1 | B1-imp+mask | B1-blind |")
    md.append("|-------|------|-----|-------------|----------|")
    for nm in (1, 2, 3, 4):
        k = f"n_mod={nm}"
        if k not in bymod["b1_i"]:
            continue
        x, y, z = bymod["b1_i"][k], bymod["b1_imp_mask"][k], bymod["b1_blind"][k]
        md.append(f"| {nm} | {x['n']} | {x['macro_auroc']:.4f} | {y['macro_auroc']:.4f} | {z['macro_auroc']:.4f} |")
    
    (OUT / "b1_decomposition.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    
    # Save JSON
    with open(OUT / "b1_decomposition.json", "w") as f:
        json.dump({"uniform": uniform, "diffs": diffs, "by_modality": bymod}, f, indent=2)
    
    print("\n=== Results ===")
    for arm in ["b1_i", "b1_imp_mask", "b1_blind"]:
        t = uniform[arm]
        print(f"{arm:12s} AUROC {t['macro_auroc']['point']:.4f} [{t['macro_auroc']['ci_low']:.4f}, {t['macro_auroc']['ci_high']:.4f}]")
    d = diffs["noise_cost"]["macro_auroc"]
    print(f"noise cost: {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    d = diffs["mask_info"]["macro_auroc"]
    print(f"mask info: {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"[wrote {OUT}]")


if __name__ == "__main__":
    main()
