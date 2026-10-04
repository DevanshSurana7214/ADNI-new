#!/usr/bin/env python3
"""Step 6b (B, C, E) — field-strength diagnostic, B1 variants, and the uniform CI protocol.

Runs, on the frozen S1 splits and the same configs/b1.yaml as Step 6:

* B0b — multinomial logistic regression on **only** ``mri_field_strength`` for episodes with MRI
  present (same folds/metrics as Step 6), plus the crosstab mri_field_strength x COLPROT x class.
* B1 variants — (i) as Step 6, (ii) without the four mask bits, (iii) without mri_field_strength.
* Paired subject-cluster-bootstrap differences (i)-(ii) and (i)-(iii).
* B1(i) importance: total-gain share of the mask bits and the rank of mri_field_strength.
* The uniform CI protocol for every arm (B0, B1 i/ii/iii, B2 mri/fdg/amy/tau, B0b): average the
  predicted probabilities over the 3 seeds per episode, pool the out-of-fold predictions, and run ONE
  subject-cluster bootstrap (2,000 resamples). The per-seed-then-average table is kept as the
  supplementary stability table.

Everything is written under results/step06b/. Step 6 result files are only read (B0 and the B2 arms
are reused from results/step06/oof_*.parquet, which are deterministic under the same config/folds).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from adni.baselines import (
    MODALITIES,
    build_raw_matrix,
    fit_predict_b1,
    load_blocks,
    load_feature_spec,
    training_prior_table,
)
from adni.metrics import (
    CLASSES,
    cluster_bootstrap_multi,
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
DEFAULT_OUT = PROJECT_ROOT / "results" / "step06b"
DEFAULT_STEP6_SUMMARY = PROJECT_ROOT / "results" / "step06" / "summary.json"

PROBA_COLS = ["p_CN", "p_MCI", "p_Dementia"]
MASK_COLS = [f"mask_{k}" for k in MODALITIES]
FIELD = "mri_mri_field_strength"
REUSED_FROM_STEP6 = {"b0": "oof_b0.parquet", **{f"b2_{k}": f"oof_b2_{k}.parquet" for k in MODALITIES}}
NEW_ARMS = ["b1_i", "b1_ii", "b1_iii", "b0b"]
ARMS = ["b0", "b1_i", "b1_ii", "b1_iii", "b2_mri", "b2_fdg", "b2_amy", "b2_tau", "b0b"]


# --------------------------------------------------------------------------- lean metrics
def _lean_macro_auroc(y: np.ndarray, proba: np.ndarray) -> float:
    out: list[float] = []
    for c in range(proba.shape[1]):
        pos = y == c
        n_pos, n_neg = int(pos.sum()), len(y) - int(pos.sum())
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(proba[:, c])
        out.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(out)) if out else float("nan")


def _lean_macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    return float(f1_score(y, proba.argmax(axis=1), average="macro", labels=[0, 1, 2], zero_division=0))


def _lean_bacc(y: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1)
    rec = [float((pred[y == c] == c).mean()) for c in range(3) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def metric_family() -> dict:
    """Metrics for ``cluster_bootstrap_multi``; PN metrics read weights from columns 3 / 4 of the
    augmented matrix ``[p_CN, p_MCI, p_Dementia, w_uniform, w_prior]``."""
    def aug(fn):
        return lambda y, a: fn(y, a[:, :3])

    def pn(col: int):
        return lambda y, a: pn_balanced_accuracy(y, a[:, :3].argmax(axis=1), a[:, col])

    return {
        "macro_auroc": aug(_lean_macro_auroc),
        "macro_f1": aug(_lean_macro_f1),
        "balanced_accuracy": aug(_lean_bacc),
        "pn_bacc_uniform": pn(3),
        "pn_bacc_prior": pn(4),
    }


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]
    arr = lambda xs: np.asarray(xs, dtype=np.int64)  # noqa: E731
    return arr(node["episode_ids"]["train"]), arr(node["episode_ids"]["val"]), arr(node["episode_ids"]["test"])


def _load_config(path: Path):
    cfg = yaml.safe_load(Path(path).read_text())
    return cfg, int(cfg.pop("num_rounds", 1000)), int(cfg.pop("early_stopping", 50))


# --------------------------------------------------------------------------- uniform protocol
def _uniform_table(arm_df: pd.DataFrame, ep: pd.DataFrame, n_boot: int, boot_seed: int) -> dict:
    """Seed-average probabilities and weights per episode, then ONE cluster bootstrap."""
    g = (arm_df.groupby("episode_id", sort=True)
         .agg(p_CN=("p_CN", "mean"), p_MCI=("p_MCI", "mean"), p_Dementia=("p_Dementia", "mean"),
              w_u=("w_uniform", "mean"), w_p=("w_prior", "mean"), n=("seed", "nunique")))
    ids = g.index.to_numpy()
    y = ep["y"].loc[ids].to_numpy()
    groups = ep["RID"].loc[ids].to_numpy()
    proba = g[["p_CN", "p_MCI", "p_Dementia"]].to_numpy()
    aug = np.column_stack([proba, g["w_u"].to_numpy(), g["w_p"].to_numpy()])
    m = natural_metrics(y, proba)
    ci = cluster_bootstrap_multi(metric_family(), y, aug, groups, n=n_boot, seed=boot_seed)
    out = {
        "n_episodes": int(len(ids)),
        "min_seed_coverage": int(g["n"].min()),
        "macro_auroc": {"point": float(m["macro_auroc"]), "ci_low": ci["macro_auroc"].ci_low,
                        "ci_high": ci["macro_auroc"].ci_high},
        "macro_f1": {"point": float(m["macro_f1"]), "ci_low": ci["macro_f1"].ci_low,
                     "ci_high": ci["macro_f1"].ci_high},
        "balanced_accuracy": {"point": float(m["balanced_accuracy"]),
                              "ci_low": ci["balanced_accuracy"].ci_low,
                              "ci_high": ci["balanced_accuracy"].ci_high},
        "f1_per_class": {CLASSES[c]: float(v) for c, v in m["f1_per_class"].items()},
        "pn_bacc_uniform": {"point": float(ci["pn_bacc_uniform"].point),
                            "ci_low": ci["pn_bacc_uniform"].ci_low,
                            "ci_high": ci["pn_bacc_uniform"].ci_high},
        "pn_bacc_prior": {"point": float(ci["pn_bacc_prior"].point),
                          "ci_low": ci["pn_bacc_prior"].ci_low,
                          "ci_high": ci["pn_bacc_prior"].ci_high},
    }
    return out, g.reset_index()


def _seed_metrics(arm_df: pd.DataFrame, ep: pd.DataFrame, n_boot: int) -> dict:
    """Per-seed natural + PN metrics (point + per-seed cluster-bootstrap CI), then mean over seeds."""
    acc: dict[str, dict[str, list]] = defaultdict(lambda: {"point": [], "lo": [], "hi": []})
    f1c = {c: [] for c in CLASSES}
    for seed in sorted(arm_df["seed"].unique()):
        d = arm_df[arm_df["seed"] == seed]
        ids = d["episode_id"].to_numpy()
        y = ep["y"].loc[ids].to_numpy()
        groups = ep["RID"].loc[ids].to_numpy()
        proba = d[PROBA_COLS].to_numpy()
        aug = np.column_stack([proba, d["w_uniform"].to_numpy(), d["w_prior"].to_numpy()])
        m = natural_metrics(y, proba)
        ci = cluster_bootstrap_multi(metric_family(), y, aug, groups, n=n_boot, seed=int(seed))
        for k in ("macro_auroc", "macro_f1", "balanced_accuracy", "pn_bacc_uniform", "pn_bacc_prior"):
            acc[k]["point"].append(float(m[k]) if k in m else float(ci[k].point))
            acc[k]["lo"].append(ci[k].ci_low)
            acc[k]["hi"].append(ci[k].ci_high)
        for c, v in m["f1_per_class"].items():
            f1c[CLASSES[c]].append(float(v))
    return {
        k: {"point": float(np.mean(v["point"])), "sd": float(np.std(v["point"])),
            "ci_low": float(np.nanmean(v["lo"])), "ci_high": float(np.nanmean(v["hi"]))}
        for k, v in acc.items()
    } | {"f1_per_class": {c: float(np.mean(v)) for c, v in f1c.items()}}


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 6b: B0b, B1 variants, uniform CI protocol.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--step6", type=Path, default=DEFAULT_STEP6)
    ap.add_argument("--step6-summary", type=Path, default=DEFAULT_STEP6_SUMMARY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cfg, num_rounds, early_stopping = _load_config(a.config)
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(a.feature_spec)
    blocks = load_blocks(a.blocks)
    blob = json.loads(Path(a.splits).read_text())

    X_all = build_raw_matrix(ep, blocks, spec)
    X_i = X_all
    X_ii = X_all[[c for c in X_all.columns if c not in MASK_COLS]]
    X_iii = X_all[[c for c in X_all.columns if c != FIELD]]
    field = blocks["mri"]["mri_field_strength"].reindex(ep.index).astype(float)
    mri_present = ep.index[ep["m_mri"].to_numpy() == 1].to_numpy()

    rows: dict[str, list[dict]] = defaultdict(list)
    timing: dict[str, float] = defaultdict(float)
    gain_tot: dict[str, float] = defaultdict(float)
    # per (seed, fold) -> weight table for ALL test episodes
    wmap: dict[tuple[int, int], pd.DataFrame] = {}

    for seed in a.seeds:
        for fold in a.folds:
            tr, va, te = _fold_ids(blob, seed, fold)
            # --- weights (same for every arm; depend only on class, pattern and the fold's train prior)
            prior = training_prior_table(ep, tr)
            ytr = ep["y"].loc[list(tr)].to_numpy()
            pc = np.bincount(ytr, minlength=3).astype(float)
            pc = pc / pc.sum()
            yy = ep["y"].loc[list(te)].to_numpy()
            pp = ep["pattern"].loc[list(te)].to_numpy()
            wu = pn_weights(yy, pp, prior, rho="uniform")
            wp = pn_weights(yy, pp, prior, rho=pc)
            wmap[(seed, fold)] = pd.DataFrame({
                "episode_id": np.asarray(te), "w_uniform": wu, "w_prior": wp})

            # --- B1 variants
            for arm, Xv in (("b1_i", X_i), ("b1_ii", X_ii), ("b1_iii", X_iii)):
                t0 = time.time()
                p, bst = fit_predict_b1(Xv, ep, tr, va, te, seed=seed, params_override=cfg,
                                        num_rounds=num_rounds, early_stopping=early_stopping)
                timing[arm] += time.time() - t0
                if arm == "b1_i":
                    for name, g in zip(bst.feature_name(), bst.feature_importance("gain")):
                        gain_tot[name] += float(g)
                for i, eid in enumerate(te):
                    rows[arm].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                      **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})

            # --- B0b: logistic regression on mri_field_strength only, MRI-present episodes
            tr_p = np.array([i for i in tr if i in set(mri_present.tolist())], dtype=np.int64)
            te_p = np.array([i for i in te if i in set(mri_present.tolist())], dtype=np.int64)
            Xf = field.loc[tr_p].to_numpy().reshape(-1, 1)
            clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
            clf.fit(Xf, ep["y"].loc[list(tr_p)].to_numpy())
            pf = clf.predict_proba(field.loc[te_p].to_numpy().reshape(-1, 1))
            for i, eid in enumerate(te_p):
                rows["b0b"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                    **{c: float(pf[i, j]) for j, c in enumerate(PROBA_COLS)}})

    # ------------------------------------------------------------------ assemble arm frames
    print(f"B1 trees per variant fit accumulated; timing so far: "
          f"{ {k: round(v,1) for k,v in timing.items()} }", flush=True)

    def with_weights(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for (seed, fold), w in wmap.items():
            sub = df[(df["seed"] == seed) & (df["fold"] == fold)]
            if len(sub):
                parts.append(sub.merge(w, on="episode_id", how="left"))
        out = pd.concat(parts, ignore_index=True)
        assert out["w_uniform"].notna().all()
        return out

    arm_frames: dict[str, pd.DataFrame] = {}
    for arm in NEW_ARMS:
        df = pd.DataFrame(rows[arm])
        df.to_parquet(a.out / f"oof_{arm}.parquet", index=False)
        arm_frames[arm] = with_weights(df)
    for arm, fname in REUSED_FROM_STEP6.items():
        arm_frames[arm] = with_weights(pd.read_parquet(a.step6 / fname))

    # b1_i must reproduce the Step 6 B1 OOF prediction for prediction (deterministic).
    ref = pd.read_parquet(a.step6 / "oof_b1.parquet")
    new = arm_frames["b1_i"]
    keys = ["seed", "fold", "episode_id"]
    merged = ref[keys + PROBA_COLS].merge(new[keys + PROBA_COLS], on=keys, suffixes=("_ref", "_new"))
    max_dev = float(np.abs(merged[[f"{c}_ref" for c in PROBA_COLS]].to_numpy()
                           - merged[[f"{c}_new" for c in PROBA_COLS]].to_numpy()).max())
    print(f"[check] b1_i refit vs Step 6 oof_b1: compared {len(merged)} rows, "
          f"max abs prob deviation = {max_dev:.3e}", flush=True)
    if len(new) == len(ref):
        assert max_dev < 1e-9, f"b1_i refit is not identical to Step 6 B1 (max dev {max_dev})"

    # ------------------------------------------------------------------ uniform protocol (E.11)
    uniform: dict[str, dict] = {}
    for arm in ARMS:
        tab, _ = _uniform_table(arm_frames[arm], ep, a.bootstrap, boot_seed=0)
        uniform[arm] = tab
    _write_uniform_csv(uniform, a.out / "uniform_ci_table.csv")
    (a.out / "uniform_ci_table.md").write_text(_uniform_md(uniform), encoding="utf-8")

    # ------------------------------------------------------------------ stability table (per-seed then average)
    stability: dict[str, dict] = {}
    for arm in ARMS:
        if arm == "b1_i":
            # identical to Step 6 B1 by the determinism check above; reuse its reported stability.
            stability[arm] = _stability_from_step6(a.step6_summary, "b1")
        elif arm in ("b0",) or arm.startswith("b2_"):
            stability[arm] = _stability_from_step6(a.step6_summary, arm)
        else:
            stability[arm] = _seed_metrics(arm_frames[arm], ep, a.bootstrap)
    _write_stability_csv(stability, a.out / "stability_table.csv")
    (a.out / "stability_table.md").write_text(_stability_md(stability), encoding="utf-8")

    # ------------------------------------------------------------------ paired bootstrap diffs (C.8)
    def _paired(arm_a: str, arm_b: str) -> dict:
        ga = _uniform_geps(arm_frames[arm_a], ep)
        gb = _uniform_geps(arm_frames[arm_b], ep)
        assert (ga.index == gb.index).all()
        y = ep["y"].loc[ga.index].to_numpy()
        groups = ep["RID"].loc[ga.index].to_numpy()
        pa = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_u"].to_numpy()])
        pb = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_u"].to_numpy()])
        mf = metric_family()
        fam = {"macro_auroc": mf["macro_auroc"], "macro_f1": mf["macro_f1"],
               "balanced_accuracy": mf["balanced_accuracy"], "pn_bacc_uniform": mf["pn_bacc_uniform"]}
        out = {}
        for k, fn in fam.items():
            ci = paired_cluster_bootstrap(lambda yy, pp, _f=fn: _f(yy, pp), y, pa, pb, groups,
                                          n=a.bootstrap, seed=0)
            out[k] = {"diff": ci.point, "ci_low": ci.ci_low, "ci_high": ci.ci_high, "n_valid": ci.n_valid}
        return out

    diffs = {"b1_i_minus_b1_ii": _paired("b1_i", "b1_ii"),
             "b1_i_minus_b1_iii": _paired("b1_i", "b1_iii")}
    (a.out / "b1_variant_diffs.json").write_text(json.dumps(diffs, indent=2))

    # ------------------------------------------------------------------ B1(i) importance (C.9)
    total_gain = float(sum(gain_tot.values()))
    mask_gain = float(sum(v for k, v in gain_tot.items() if k in MASK_COLS))
    imp = sorted(gain_tot.items(), key=lambda kv: kv[1], reverse=True)
    field_rank = next((i + 1 for i, (n, _) in enumerate(imp) if n == FIELD), None)
    imp_rows = [{"rank": i + 1, "feature": n, "gain": float(g), "gain_share": float(g) / total_gain}
                for i, (n, g) in enumerate(imp)]
    pd.DataFrame(imp_rows).to_csv(a.out / "b1_importance.csv", index=False)
    importance = {
        "total_gain": total_gain,
        "mask_gain": mask_gain,
        "mask_gain_share": mask_gain / total_gain,
        "mask_bits": {c: float(gain_tot.get(c, 0.0)) for c in MASK_COLS},
        "field_strength_rank": field_rank,
        "field_strength_gain": float(gain_tot.get(FIELD, 0.0)),
        "field_strength_gain_share": float(gain_tot.get(FIELD, 0.0)) / total_gain,
        "n_features": len(gain_tot),
        "top20": imp_rows[:20],
    }
    (a.out / "b1_importance.json").write_text(json.dumps(importance, indent=2))

    # ------------------------------------------------------------------ B0b crosstab (B.6)
    mri_present_ep = ep.loc[ep["m_mri"] == 1].copy()
    mri_present_ep["mri_field_strength"] = field.loc[mri_present_ep.index].astype(int)
    cross = (mri_present_ep.groupby(["mri_field_strength", "COLPROT", "DIAGNOSIS"])
             .size().rename("n").reset_index())
    # ensure every (fs, colprot, class) cell appears even if zero
    fs_vals = sorted(cross["mri_field_strength"].unique())
    cp_vals = sorted(ep["COLPROT"].unique())
    grid = pd.MultiIndex.from_product([fs_vals, cp_vals, CLASSES],
                                      names=["mri_field_strength", "COLPROT", "DIAGNOSIS"]).to_frame(index=False)
    cross = grid.merge(cross, how="left").fillna({"n": 0})
    cross["n"] = cross["n"].astype(int)
    cross.to_csv(a.out / "b0b_field_strength_colprot_class.csv", index=False)

    # ------------------------------------------------------------------ decision rule (B.7)
    b0b_lo = uniform["b0b"]["macro_auroc"]["ci_low"]
    exclude = bool(b0b_lo > 0.55)
    decision = {
        "rule": "if B0b macro-AUROC CI lower bound > 0.55 -> EXCLUDE mri_field_strength from the "
                "primary neural models (keep as a sensitivity arm); else keep it",
        "threshold": 0.55,
        "b0b_macro_auroc_point": uniform["b0b"]["macro_auroc"]["point"],
        "b0b_macro_auroc_ci_low": b0b_lo,
        "b0b_macro_auroc_ci_high": uniform["b0b"]["macro_auroc"]["ci_high"],
        "branch": "EXCLUDE from primary models (sensitivity arm only)" if exclude
                  else "KEEP mri_field_strength in the primary models",
        "exclude": exclude,
    }
    (a.out / "decision_rule.json").write_text(json.dumps(decision, indent=2))

    # ------------------------------------------------------------------ summary + console
    summary = {
        "config": cfg, "num_rounds": num_rounds, "early_stopping": early_stopping,
        "n_bootstrap": a.bootstrap, "seeds": a.seeds, "folds": a.folds,
        "timing_seconds": {k: round(v, 2) for k, v in timing.items()},
        "b1_i_refit_max_prob_deviation_vs_step6": max_dev,
        "uniform_ci_table": uniform,
        "stability_table": stability,
        "b1_variant_diffs": diffs,
        "b1_importance": importance,
        "decision_rule": decision,
    }
    (a.out / "summary_baselines.json").write_text(json.dumps(summary, indent=2))

    print("\n=== UNIFORM CI PROTOCOL (seed-averaged, one bootstrap) ===")
    print(f"{'arm':8} {'n_ep':>6} {'macroAUROC':>11} {'macroF1':>8} {'BAcc':>7} {'PN-BAcc':>8}")
    for arm in ARMS:
        t = uniform[arm]
        print(f"{arm:8} {t['n_episodes']:6d} {t['macro_auroc']['point']:11.4f} "
              f"{t['macro_f1']['point']:8.4f} {t['balanced_accuracy']['point']:7.4f} "
              f"{t['pn_bacc_uniform']['point']:8.4f}")
    print("\n=== B1 variant paired diffs (i minus x) ===")
    for k, v in diffs.items():
        print(f"  {k}: AUROC {v['macro_auroc']['diff']:+.4f} "
              f"[{v['macro_auroc']['ci_low']:+.4f},{v['macro_auroc']['ci_high']:+.4f}]  "
              f"PN-BAcc {v['pn_bacc_uniform']['diff']:+.4f} "
              f"[{v['pn_bacc_uniform']['ci_low']:+.4f},{v['pn_bacc_uniform']['ci_high']:+.4f}]")
    print(f"\nmask-bit gain share = {importance['mask_gain_share']:.4f}  "
          f"field-strength rank = {importance['field_strength_rank']} of {importance['n_features']}  "
          f"share = {importance['field_strength_gain_share']:.5f}")
    print(f"\nDECISION RULE: {decision['branch']}  (B0b macro-AUROC {decision['b0b_macro_auroc_point']:.4f}, "
          f"CI low {b0b_lo:.4f}, threshold 0.55)")
    print(f"[wrote outputs under {a.out}]")


# --------------------------------------------------------------------------- small helpers
def _uniform_geps(arm_df: pd.DataFrame, ep: pd.DataFrame) -> pd.DataFrame:
    g = (arm_df.groupby("episode_id", sort=True)
         .agg(p_CN=("p_CN", "mean"), p_MCI=("p_MCI", "mean"), p_Dementia=("p_Dementia", "mean"),
              w_u=("w_uniform", "mean"), w_p=("w_prior", "mean")))
    g = g.loc[g.index.intersection(ep.index)]
    return g.rename(columns={"p_CN": "p_CN", "p_MCI": "p_MCI", "p_Dementia": "p_Dementia"})


def _stability_from_step6(path: Path, arm: str) -> dict:
    s = json.loads(Path(path).read_text())["arms"][arm]
    out = {"macro_auroc": s["natural"]["macro_auroc"], "macro_f1": s["natural"]["macro_f1"],
           "balanced_accuracy": s["natural"]["balanced_accuracy"],
           "pn_bacc_uniform": s["pn_bacc_uniform"], "pn_bacc_prior": s["pn_bacc_prior"],
           "f1_per_class": s["f1_per_class"], "source": "step06 summary.json"}
    return out


def _write_uniform_csv(table: dict, path: Path) -> None:
    recs = []
    for arm, t in table.items():
        for k in ("macro_auroc", "macro_f1", "balanced_accuracy", "pn_bacc_uniform", "pn_bacc_prior"):
            recs.append({"arm": arm, "metric": k, **t[k]})
        for c, v in t["f1_per_class"].items():
            recs.append({"arm": arm, "metric": f"f1_{c}", "point": v,
                         "ci_low": float("nan"), "ci_high": float("nan")})
    pd.DataFrame(recs).to_csv(path, index=False)


def _uniform_md(table: dict) -> str:
    lines = ["# Step 6b — uniform CI protocol (seed-averaged predictions, one 2,000-resample "
             "subject-cluster bootstrap)", "",
             "| arm | n_ep | macro-AUROC | macro-F1 | balanced acc | PN-BAcc (uniform) | PN-BAcc (P(c)) |",
             "|-----|------|-------------|----------|--------------|-------------------|----------------|"]
    for arm, t in table.items():
        def f(k, which="point"):
            return t[k][which]
        lines.append(f"| {arm} | {t['n_episodes']} | {f('macro_auroc'):.4f} "
                     f"[{f('macro_auroc','ci_low'):.4f}, {f('macro_auroc','ci_high'):.4f}] | "
                     f"{f('macro_f1'):.4f} [{f('macro_f1','ci_low'):.4f}, {f('macro_f1','ci_high'):.4f}] | "
                     f"{f('balanced_accuracy'):.4f} [{f('balanced_accuracy','ci_low'):.4f}, "
                     f"{f('balanced_accuracy','ci_high'):.4f}] | "
                     f"{f('pn_bacc_uniform'):.4f} [{f('pn_bacc_uniform','ci_low'):.4f}, "
                     f"{f('pn_bacc_uniform','ci_high'):.4f}] | {f('pn_bacc_prior'):.4f} |")
    lines.append("")
    lines.append("Per-class F1 (seed-averaged, natural distribution):")
    lines.append("")
    lines.append("| arm | CN | MCI | Dementia |")
    lines.append("|-----|----|-----|----------|")
    for arm, t in table.items():
        lines.append(f"| {arm} | {t['f1_per_class']['CN']:.4f} | {t['f1_per_class']['MCI']:.4f} | "
                     f"{t['f1_per_class']['Dementia']:.4f} |")
    return "\n".join(lines) + "\n"


def _write_stability_csv(table: dict, path: Path) -> None:
    recs = []
    for arm, t in table.items():
        for k in ("macro_auroc", "macro_f1", "balanced_accuracy", "pn_bacc_uniform", "pn_bacc_prior"):
            row = {"arm": arm, "metric": k}
            row.update({kk: t[k].get(kk, float("nan")) for kk in ("point", "sd", "ci_low", "ci_high")})
            recs.append(row)
        for c, v in t["f1_per_class"].items():
            recs.append({"arm": arm, "metric": f"f1_{c}", "point": v, "sd": float("nan"),
                         "ci_low": float("nan"), "ci_high": float("nan")})
    pd.DataFrame(recs).to_csv(path, index=False)


def _stability_md(table: dict) -> str:
    lines = ["# Step 6b — supplementary stability table (per-seed metric, then average over seeds)", "",
             "| arm | macro-AUROC (mean ± sd) | macro-F1 (mean ± sd) | BAcc (mean ± sd) | "
             "PN-BAcc (mean ± sd) |", "|-----|------------------------|----------------------|"
             "------------------|-------------------|"]
    for arm, t in table.items():
        lines.append(f"| {arm} | {t['macro_auroc']['point']:.4f} ± {t['macro_auroc']['sd']:.4f} | "
                     f"{t['macro_f1']['point']:.4f} ± {t['macro_f1']['sd']:.4f} | "
                     f"{t['balanced_accuracy']['point']:.4f} ± {t['balanced_accuracy']['sd']:.4f} | "
                     f"{t['pn_bacc_uniform']['point']:.4f} ± {t['pn_bacc_uniform']['sd']:.4f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
