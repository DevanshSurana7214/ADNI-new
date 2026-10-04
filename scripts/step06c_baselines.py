#!/usr/bin/env python3
"""Step 6c - B0 encoding reconciliation, B1-blind modality imputation, uniform CI protocol.

This step addresses four items requested for Step 6c, and writes every output under
``results/step06c/`` (Step 6 / Step 6b result files are only READ):

1. Regenerated Step 3 fold tables (separate script ``step03_report_tables.py``).
2. B0 encoding. The primary pattern-only baseline is a multinomial logistic regression on the
   **15-column one-hot pattern** (``b0_oh``); the four-bit availability version is kept as
   ``b0_lin``. To test whether the one-hot advantage is representational capacity (a saturated
   15-cell model can express interactions between availability bits) rather than L2 shrinkage, we
   also fit the four-bit model **plus all six pairwise bit interactions** (``b0_lin_inter``).
   If the gap were only shrinkage, ``b0_lin_inter`` would still fall short of ``b0_oh``.
3. B1-blind. For every absent block, its feature columns are filled with the values of a randomly
   drawn PRESENT TRAINING row of that block (drawn per episode, donor pool = the fold's TRAINING
   rows only) and the four mask bits are removed. Everything else is identical to B1 (same
   config/folds/seeds). The difference B1 - B1-blind is the benefit of knowing which scans exist
   (which includes the availability shortcut).
4. A record of the fixed field-strength decision (kept from Step 6b) lives in the REPORT.

All arms are pooled with the **uniform CI protocol**: seed-average the predicted probabilities per
episode, pool the out-of-fold predictions, and run ONE subject-cluster bootstrap (2,000 resamples).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from adni.baselines import (
    MODALITIES,
    build_raw_matrix,
    fit_predict_b1,
    load_blocks,
    load_feature_spec,
    pattern_onehot,
    training_prior_table,
)
from adni.metrics import (
    CLASSES,
    natural_metrics,
    paired_cluster_bootstrap,
    pn_balanced_accuracy,
    pn_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "b1.yaml"
DEFAULT_STEP6 = PROJECT_ROOT / "results" / "step06"
DEFAULT_STEP6B = PROJECT_ROOT / "results" / "step06b"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step06c"

PROBA_COLS = ["p_CN", "p_MCI", "p_Dementia"]
MASK_COLS = [f"mask_{k}" for k in MODALITIES]
_REUSED = {
    "b0_lin": (DEFAULT_STEP6, "oof_b0.parquet"),
    "b2_mri": (DEFAULT_STEP6, "oof_b2_mri.parquet"),
    "b2_fdg": (DEFAULT_STEP6, "oof_b2_fdg.parquet"),
    "b2_amy": (DEFAULT_STEP6, "oof_b2_amy.parquet"),
    "b2_tau": (DEFAULT_STEP6, "oof_b2_tau.parquet"),
    "b1_i": (DEFAULT_STEP6B, "oof_b1_i.parquet"),
    "b1_ii": (DEFAULT_STEP6B, "oof_b1_ii.parquet"),
    "b1_iii": (DEFAULT_STEP6B, "oof_b1_iii.parquet"),
    "b0b": (DEFAULT_STEP6B, "oof_b0b.parquet"),
}
NEW_ARMS = ["b0_oh", "b0_lin_inter", "b1_blind"]


def _load_s6b():
    spec = importlib.util.spec_from_file_location(
        "step06b_baselines", PROJECT_ROOT / "scripts" / "step06b_baselines.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    arr = lambda xs: np.asarray(xs, dtype=np.int64)  # noqa: E731
    return arr(node["train"]), arr(node["val"]), arr(node["test"])


def _weights_for(ep: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> pd.DataFrame:
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


def _impute_blind(
    X_raw: pd.DataFrame,
    present: dict[str, set],
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    feature_spec: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """B1-blind matrix: absent blocks replaced by a random PRESENT TRAINING row of that block.

    Donor pool for block ``k`` = the fold's TRAINING episode_ids where ``k`` is present. Every
    episode (train/val/test) that lacks ``k`` gets that block's columns replaced. Mask bits are
    dropped. Returns a matrix indexed like ``X_raw``.
    """
    X = X_raw[[c for c in X_raw.columns if c not in MASK_COLS]].copy()
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 6c: B0 encoding, B1-blind, uniform CI.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s6b = _load_s6b()

    cfg = yaml.safe_load(Path(a.config).read_text())
    num_rounds = int(cfg.pop("num_rounds", 1000))
    early_stopping = int(cfg.pop("early_stopping", 50))

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(a.feature_spec)
    blocks = load_blocks(a.blocks)
    blob = json.loads(Path(a.splits).read_text())

    X_raw = build_raw_matrix(ep, blocks, spec)
    bits = pattern_onehot(ep).astype(float)
    X_inter = bits.copy()
    combos = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for i, j in combos:
        X_inter[f"inter_{MODALITIES[i]}_{MODALITIES[j]}"] = (bits.iloc[:, i] * bits.iloc[:, j]).to_numpy()
    X_oh = pd.get_dummies(ep["pattern"].astype(str)).astype(float)
    present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}

    rows: dict[str, list[dict]] = defaultdict(list)
    wmap: dict[tuple[int, int], pd.DataFrame] = {}
    timing: dict[str, float] = defaultdict(float)

    for seed in a.seeds:
        for fold in a.folds:
            tr, va, te = _fold_ids(blob, seed, fold)
            wmap[(seed, fold)] = _weights_for(ep, tr, te)
            ytr = ep["y"].loc[list(tr)].to_numpy()
            yte = ep["y"].loc[list(te)].to_numpy()

            for arm, X, kwargs in (
                ("b0_oh", X_oh, {}),
                ("b0_lin_inter", X_inter, {}),
            ):
                clf = LogisticRegression(max_iter=5000, C=1.0, random_state=seed, **kwargs)
                clf.fit(X.loc[tr], ytr)
                p = clf.predict_proba(X.loc[te])
                for i, eid in enumerate(te):
                    rows[arm].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                      **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})

            t0 = time.time()
            rng = np.random.default_rng(1000 * (seed + 1) + fold)
            Xb = _impute_blind(X_raw, present, tr, va, te, spec, rng)
            p, _ = fit_predict_b1(Xb, ep, tr, va, te, seed=seed, params_override=cfg,
                                  num_rounds=num_rounds, early_stopping=early_stopping)
            timing["b1_blind"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b1_blind"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                         **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})
        print(f"[seed {seed}] b0_oh / b0_lin_inter / b1_blind done", flush=True)

    def with_weights(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for (seed, fold), w in wmap.items():
            sub = df[(df["seed"] == seed) & (df["fold"] == fold)]
            if len(sub):
                parts.append(sub.merge(w, on="episode_id", how="left"))
        out = pd.concat(parts, ignore_index=True)
        assert out["w_uniform"].notna().all()
        return out

    frames: dict[str, pd.DataFrame] = {}
    for arm in NEW_ARMS:
        df = pd.DataFrame(rows[arm])
        df.to_parquet(a.out / f"oof_{arm}.parquet", index=False)
        frames[arm] = with_weights(df)
    for arm, (root, fname) in _REUSED.items():
        frames[arm] = with_weights(pd.read_parquet(root / fname))

    # ------------------------------------------------------------------ uniform CI protocol
    arms_order = ["b0_lin", "b0_lin_inter", "b0_oh", "b1_i", "b1_ii", "b1_iii", "b1_blind",
                  "b2_mri", "b2_fdg", "b2_amy", "b2_tau", "b0b"]
    uniform: dict[str, dict] = {}
    for arm in arms_order:
        tab, _ = s6b._uniform_table(frames[arm], ep, a.bootstrap, boot_seed=0)
        uniform[arm] = tab
    s6b._write_uniform_csv(uniform, a.out / "uniform_ci_table.csv")
    (a.out / "uniform_ci_table.md").write_text(
        s6b._uniform_md(uniform).replace("Step 6b", "Step 6c"), encoding="utf-8")

    # ------------------------------------------------------------------ B0 encoding table
    enc_rows = []
    for arm in ("b0_lin", "b0_lin_inter", "b0_oh"):
        t = uniform[arm]["macro_auroc"]
        enc_rows.append({"arm": arm, "n_features": int(bits.shape[1] if arm == "b0_lin"
                                                        else X_inter.shape[1] if arm == "b0_lin_inter"
                                                        else X_oh.shape[1]),
                         "macro_auroc": t["point"], "ci_low": t["ci_low"], "ci_high": t["ci_high"]})
    pd.DataFrame(enc_rows).to_csv(a.out / "b0_encoding.csv", index=False)
    enc_md = ["# Step 6c - B0 encoding reconciliation", "",
              "| B0 encoding | features | macro-AUROC (uniform CI) |",
              "|-------------|----------|---------------------------|"]
    for r in enc_rows:
        enc_md.append(f"| {r['arm']} | {r['n_features']} | {r['macro_auroc']:.4f} "
                      f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}] |")
    recovered = abs(enc_rows[1]["macro_auroc"] - enc_rows[2]["macro_auroc"])
    enc_md += ["",
               f"Gap one-hot minus four-bit = {enc_rows[2]['macro_auroc'] - enc_rows[0]['macro_auroc']:+.4f}.",
               f"Gap one-hot minus (four-bit + pairwise interactions) = {recovered:+.4f}.",
               "",
               "If the four-bit-plus-interactions model recovers the one-hot macro-AUROC, the encoding "
               "gap is representational capacity (the 15-cell model can express the interaction "
               "structure between availability bits), not L2 shrinkage on the four-bit model."]
    (a.out / "b0_encoding.md").write_text("\n".join(enc_md) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ paired diffs
    def _paired(arm_a: str, arm_b: str) -> dict:
        ga = s6b._uniform_geps(frames[arm_a], ep)
        gb = s6b._uniform_geps(frames[arm_b], ep)
        assert (ga.index == gb.index).all()
        y = ep["y"].loc[ga.index].to_numpy()
        groups = ep["RID"].loc[ga.index].to_numpy()
        pa = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_u"].to_numpy()])
        pb = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_u"].to_numpy()])
        mf = s6b.metric_family()
        fam = {"macro_auroc": mf["macro_auroc"], "macro_f1": mf["macro_f1"],
               "balanced_accuracy": mf["balanced_accuracy"], "pn_bacc_uniform": mf["pn_bacc_uniform"]}
        out = {}
        for k, fn in fam.items():
            ci = paired_cluster_bootstrap(lambda yy, pp, _f=fn: _f(yy, pp), y, pa, pb, groups,
                                          n=a.bootstrap, seed=0)
            out[k] = {"diff": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high, "n_valid": ci.n_valid}
        return out

    diffs = {
        "b0_oh_minus_b0_lin": _paired("b0_oh", "b0_lin"),
        "b0_lin_inter_minus_b0_lin": _paired("b0_lin_inter", "b0_lin"),
        "b0_oh_minus_b0_lin_inter": _paired("b0_oh", "b0_lin_inter"),
        "b1_i_minus_b1_blind": _paired("b1_i", "b1_blind"),
    }
    (a.out / "step06c_paired_diffs.json").write_text(json.dumps(diffs, indent=2))

    # ------------------------------------------------------------------ by-modality-count
    def _by_modality(arm: str) -> dict:
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
            m = natural_metrics(y[sel], proba[sel])
            out[f"n_mod={nm}"] = {"n": int(sel.sum()),
                                  "macro_auroc": float(m["macro_auroc"]),
                                  "macro_f1": float(m["macro_f1"]),
                                  "pn_bacc_uniform": float(pn_balanced_accuracy(
                                      y[sel], proba[sel].argmax(axis=1), w[sel]))}
        return out

    bymod = {"b1_i": _by_modality("b1_i"), "b1_blind": _by_modality("b1_blind")}
    (a.out / "by_modality_count.json").write_text(json.dumps(bymod, indent=2))
    bm = ["# Step 6c - B1 vs B1-blind by number of observed modalities (seed-averaged)", "",
          "| n_mod | n_ep | B1 macro-AUROC | B1-blind macro-AUROC | B1 PN-BAcc | B1-blind PN-BAcc |",
          "|-------|------|----------------|----------------------|-----------|------------------|"]
    for nm in (1, 2, 3, 4):
        k = f"n_mod={nm}"
        if k not in bymod["b1_i"]:
            continue
        x, z = bymod["b1_i"][k], bymod["b1_blind"][k]
        bm.append(f"| {nm} | {x['n']} | {x['macro_auroc']:.4f} | {z['macro_auroc']:.4f} | "
                  f"{x['pn_bacc_uniform']:.4f} | {z['pn_bacc_uniform']:.4f} |")
    (a.out / "by_modality_count.md").write_text("\n".join(bm) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ summary + console
    summary = {
        "n_bootstrap": a.bootstrap, "seeds": a.seeds, "folds": a.folds,
        "timing_seconds": {k: round(v, 2) for k, v in timing.items()},
        "uniform_ci_table": uniform,
        "b0_encoding": enc_rows,
        "paired_diffs": diffs,
        "by_modality_count": bymod,
    }
    (a.out / "summary_step06c.json").write_text(json.dumps(summary, indent=2))

    print("\n=== B0 encoding (uniform CI) ===")
    for r in enc_rows:
        print(f"  {r['arm']:14s} nfeat={r['n_features']:2d}  macro-AUROC {r['macro_auroc']:.4f} "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print("\n=== B1 vs B1-blind (uniform CI, seed-averaged) ===")
    for arm in ("b1_i", "b1_blind"):
        t = uniform[arm]
        print(f"  {arm:9s} macro-AUROC {t['macro_auroc']['point']:.4f} "
              f"[{t['macro_auroc']['ci_low']:.4f}, {t['macro_auroc']['ci_high']:.4f}]  "
              f"PN-BAcc {t['pn_bacc_uniform']['point']:.4f} "
              f"[{t['pn_bacc_uniform']['ci_low']:.4f}, {t['pn_bacc_uniform']['ci_high']:.4f}]")
    d = diffs["b1_i_minus_b1_blind"]["macro_auroc"]
    print(f"  B1 - B1-blind macro-AUROC {d['diff']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
    print(f"[wrote outputs under {a.out}]")


if __name__ == "__main__":
    main()
