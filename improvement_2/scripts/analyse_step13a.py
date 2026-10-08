#!/usr/bin/env python3
"""Step 13a analysis - script-generated tables (EXPLORATORY; post-hoc joint correction on existing OOF).

Protocol (mirrors steps 6b/9/10/11a/11b): seed-averaged probabilities (mean over seeds 0/1/2)
per episode, pooled OOF on the TEST split, ONE subject-cluster bootstrap (n=2000, seed=0) per arm,
both checkpoint rules for the neural arms. No new training. tau, alpha and margins are frozen by
configs/step13.yaml and are NOT tuned here.

Outputs under results/improvement2/:
  runs/          corrected + uncorrected OOF for each model X and E1
  tables/        main table, paired-diff table, within-pattern table, H3(v3) verdict table
  diagnostics/   mask_intervention.md (if run) or a "not run" statement
  analysis.json
  REPORT.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from adni.baselines import CLASSES, training_prior_table
from adni.metrics import (
    _group_index,
    _resample_indices,
    cluster_bootstrap_multi,
    holm,
    natural_metrics,
    pn_balanced_accuracy,
    pn_weights,
)
from adni.priors import crosstab

from improve2.e1_build import build_e1_corrected, build_e1_uncorrected
from improve2.post_hoc_correction import apply_correction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_OUT = PROJECT_ROOT / "improvement_2" / "results" / "improvement2"
DEFAULT_CONFIG = PROJECT_ROOT / "improvement_2" / "configs" / "step13.yaml"

PROBA_COLS = ("p_CN", "p_MCI", "p_Dementia")
CKPT_RULES = ("primary", "secondary")
N_BOOT = 2000
BOOT_SEED = 0
NI_MARGIN = -0.02
WITHIN_PATTERN_NI = -0.01


def _load_cfg() -> dict:
    import yaml
    return yaml.safe_load(Path(DEFAULT_CONFIG).read_text())


def _fold_ids(blob: dict, seed: int, fold: int):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64)
                 for k in ("train", "val", "test"))


def _fold_ids_s2(blob: dict):
    node = blob["s2"]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64)
                 for k in ("train", "val", "test"))


def seed_average(frame: pd.DataFrame) -> pd.DataFrame:
    g = (frame.groupby(["seed", "fold", "episode_id"], sort=True)
         .agg(**{c: (c, "mean") for c in PROBA_COLS},
              pattern=("pattern", "first"),
              RID=("RID", "first"),
              y=("y", "first")))
    return g


def augment(g: pd.DataFrame):
    ids = g["episode_id"].to_numpy()
    y = g["y"].to_numpy()
    groups = g["RID"].to_numpy()
    proba = g[PROBA_COLS].to_numpy()
    return ids, y, groups, proba


def _lean_macro_auroc(y, proba):
    out = []
    for c in range(proba.shape[1]):
        pos = y == c
        n_pos = int(pos.sum()); n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata_safe(proba[:, c])
        out.append((ranks[pos].sum() - n_pos*(n_pos+1)/2)/(n_pos*n_neg))
    return float(np.mean(out)) if out else float("nan")


def rankdata_safe(a):
    import numpy as np
    from scipy.stats import rankdata
    return rankdata(a)


def _lean_macro_f1(y, proba):
    return float(f1_score(y, proba.argmax(1), average="macro", labels=[0,1,2], zero_division=0))


def _lean_bacc(y, proba):
    pred = proba.argmax(1)
    rec = [float((pred[y==c]==c).mean()) for c in range(3) if (y==c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def _lean_ece(y, proba, n_bins=15):
    conf = proba.max(1); pred = proba.argmax(1); correct = (pred==y).astype(float)
    edges = np.linspace(0, 1, n_bins+1)
    b = np.digitize(conf, edges[1:-1], right=True); n = len(y); ece = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any(): ece += (m.sum()/n)*abs(correct[m].mean()-conf[m].mean())
    return float(ece)


def _lean_brier(y, proba):
    onehot = np.zeros_like(proba); onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba-onehot)**2, axis=1)))


def metric_family():
    def aug(fn): return lambda y, a: fn(y, a[:, :3])
    return {
        "macro_auroc": aug(_lean_macro_auroc),
        "macro_f1": aug(_lean_macro_f1),
        "balanced_accuracy": aug(_lean_bacc),
        "pn_bacc_uniform": lambda y, a: pn_balanced_accuracy(y, a[:, :3].argmax(1), np.full(len(y), 1/3)),
        "ece": aug(_lean_ece),
        "brier": aug(_lean_brier),
    }


def paired_boot(fn, y, a, b, groups, n=N_BOOT, seed=BOOT_SEED):
    order, starts = _group_index(groups)
    rng = np.random.default_rng(seed)
    point = float(fn(y, a)) - float(fn(y, b))
    ds = []
    for _ in range(n):
        idx = _resample_indices(order, starts, rng)
        try: d = float(fn(y[idx], a[idx])) - float(fn(y[idx], b[idx]))
        except Exception: d = float("nan")
        if np.isfinite(d): ds.append(d)
    ds = np.asarray(ds)
    if ds.size == 0:
        return {"diff": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "p": float("nan"), "n_valid": 0}
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p_less = (1+int((ds<=0).sum()))/(len(ds)+1)
    p_greater = (1+int((ds>=0).sum()))/(len(ds)+1)
    p = min(1.0, 2.0*min(p_less, p_greater))
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi),
            "p": float(p), "n_valid": int(len(ds))}


def _fmt(v, nd=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _ci(d, nd=4):
    return f"{_fmt(d['point'], nd)} [{_fmt(d['ci_low'], nd)}, {_fmt(d['ci_high'], nd)}]"


def within_pattern_macro_auroc(g: pd.DataFrame, ep: pd.DataFrame,
                               full_counts: pd.DataFrame,
                               min_episodes: int = 100, min_per_class: int = 10):
    ids = g["episode_id"].to_numpy()
    y = g["y"].to_numpy()
    proba = g[PROBA_COLS].to_numpy()
    pat = g["pattern"].to_numpy()
    pats = sorted(ep["pattern"].unique().tolist())
    included, records = [], []
    for pat_str in pats:
        sel = pat == pat_str
        n = int(sel.sum())
        per_cls = (full_counts.loc[CLASSES, pat_str].to_numpy()
                   if pat_str in full_counts.columns else np.zeros(3))
        ok = n >= min_episodes and int((per_cls >= min_per_class).all())
        if not ok:
            continue
        included.append(pat_str)
        aucs = []
        for c in range(3):
            pos = y[sel] == c
            n_pos = int(pos.sum()); n_neg = int(sel.sum()) - n_pos
            if n_pos == 0 or n_neg == 0:
                continue
            ranks = rankdata_safe(proba[sel, c])
            aucs.append((ranks[pos].sum() - n_pos*(n_pos+1)/2)/(n_pos*n_neg))
        wm = float(np.mean(aucs)) if aucs else float("nan")
        records.append({"pattern": pat_str, "n": n, "macro_auroc": wm})
    w = np.array([r["n"] for r in records], dtype=float)
    if w.sum() <= 0:
        return {"overall": float("nan"), "included": included,
                "records": records, "n_included": len(records)}
    overall = float(np.average([r["macro_auroc"] for r in records], weights=w))
    return {"overall": overall, "included": included,
            "records": records, "n_included": len(records)}


def eval_frame(df_corr: pd.DataFrame, df_uncorr: pd.DataFrame,
               model_name: str, rule: str,
               ep: pd.DataFrame, full_counts: pd.DataFrame,
               min_episodes: int, min_per_class: int):
    g_corr = seed_average(df_corr) if "seed" in df_corr.columns else df_corr
    g_uncorr = seed_average(df_uncorr) if "seed" in df_uncorr.columns else df_uncorr
    ids_c, y_c, groups_c, proba_c = augment(g_corr)
    ids_u, y_u, groups_u, proba_u = augment(g_uncorr)
    assert np.array_equal(ids_c, ids_u), "corrected/uncorrected ids must match"
    groups = groups_c
    y = y_c
    metrics = natural_metrics(y, proba_c)
    aug = np.column_stack([proba_c, np.zeros((len(y), 2))])
    cis = cluster_bootstrap_multi(metric_family(), y, aug, groups,
                                  n=N_BOOT, seed=BOOT_SEED)
    pn_c = pn_balanced_accuracy(y, proba_c.argmax(1), np.full(len(y), 1/3))
    pn_u = pn_balanced_accuracy(y, proba_u.argmax(1), np.full(len(y), 1/3))
    wp = within_pattern_macro_auroc(g_corr, ep, full_counts,
                                    min_episodes, min_per_class)
    return {
        "model": model_name, "rule": rule,
        "natural": metrics,
        "pn_bacc_uniform_corr": pn_c,
        "pn_bacc_uniform_uncorr": pn_u,
        "within_pattern": wp,
        "proba_corr": proba_c,
        "proba_uncorr": proba_u,
        "y": y,
        "groups": groups,
        "ids": ids_c,
    }


def corrected_frame(df_avg: pd.DataFrame, pi_lookup_fn,
                    tau: float, rho: np.ndarray) -> pd.DataFrame:
    out = df_avg.copy()
    proba = out[PROBA_COLS].to_numpy()
    corrected = np.zeros_like(proba)
    for (seed, fold), grp in out.groupby(["seed", "fold"], sort=False):
        g = grp.copy()
        plu = pi_lookup_fn(seed, fold)
        pi_g = plu.loc[g["episode_id"].to_numpy()].to_numpy()
        factor = 1.0 / (pi_g ** tau)
        proba_g = g[PROBA_COLS].to_numpy()
        corrected_g = proba_g * rho[None, :] * factor[:, None]
        s = corrected_g.sum(axis=1)
        s = np.where(s <= 0, 1.0, s)
        corrected_g = corrected_g / s[:, None]
        corrected[grp.index.values, :] = corrected_g
    out[PROBA_COLS] = corrected
    return out


def _build_pi_lookup_fn_s1_natural(ep: pd.DataFrame, blob: dict,
                                   alpha: float, seed: int, fold: int):
    tr, _, _ = _fold_ids(blob, seed, fold)
    prior = training_prior_table(ep, tr, alpha=alpha)
    pats = [str(c) for c in prior.columns]
    lut = pd.DataFrame(prior.to_numpy(), index=list(prior.index), columns=pats)
    def fn(_, __):
        ids = (prior
               .index
               .to_series()
               .reindex(ep.index)
               .loc[ep.loc[slice(None), "pattern"].to_numpy().tolist() if False else list(ep.index)]
               .to_numpy() if False else None)
        return None
    # simpler: build one per-call lookup on the fly from the fold's test episode patterns
    def pi_lookup(_, __):
        tr_f, _, te_f = _fold_ids(blob, seed, fold)
        pat_te = ep["pattern"].loc[te_f].astype(str).to_numpy()
        col_idx = np.searchsorted(pats, pat_te)
        return pd.Series(lut.lookup[lut.index, col_idx], index=te_f)
    return pi_lookup


def _pi_lookup_s1_natural(prior, pats, pat_vec):
    col_idx = np.searchsorted([str(c) for c in prior.columns], pat_vec)
    lut = pd.DataFrame(prior.to_numpy(), index=list(prior.index), columns=[str(c) for c in prior.columns])
    vals = lut.lookup[lut.index, col_idx]
    return pd.Series(vals)


def _pi_lookup_neural_s1(ep, blob, seed, fold, p_drop, alpha):
    tr, _, te = _fold_ids(blob, seed, fold)
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats_nat = crosstab(diag, sub["pattern"])
    from improve2.prior_loading import priors_from_counts as pfc
    priors = pfc(n_cm, pats_nat, p_drop=p_drop, alpha=alpha)
    pats_pi = [str(c) for c in priors.pi.columns]
    lut = pd.DataFrame(priors.pi.to_numpy(dtype=float), index=list(priors.pi.index), columns=pats_pi)
    pat_te = ep["pattern"].loc[te].astype(str).to_numpy()
    col_idx = np.searchsorted(pats_pi, pat_te)
    return pd.Series(lut.lookup[lut.index, col_idx], index=te)


def _pi_lookup_s2_natural(ep, blob, alpha):
    tr, _, te = _fold_ids_s2(blob)
    prior = training_prior_table(ep, tr, alpha=alpha)
    pats = [str(c) for c in prior.columns]
    lut = pd.DataFrame(prior.to_numpy(), index=list(prior.index), columns=pats)
    pat_te = ep["pattern"].loc[te].astype(str).to_numpy()
    col_idx = np.searchsorted(pats, pat_te)
    return pd.Series(lut.lookup[lut.index, col_idx], index=te)


def _pi_lookup_s2_neural(ep, blob, p_drop, alpha):
    tr, _, te = _fold_ids_s2(blob)
    sub = ep.loc[tr]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    n_cm, pats_nat = crosstab(diag, sub["pattern"])
    from improve2.prior_loading import priors_from_counts as pfc
    priors = pfc(n_cm, pats_nat, p_drop=p_drop, alpha=alpha)
    pats_pi = [str(c) for c in priors.pi.columns]
    lut = pd.DataFrame(priors.pi.to_numpy(dtype=float), index=list(priors.pi.index), columns=pats_pi)
    pat_te = ep["pattern"].loc[te].astype(str).to_numpy()
    col_idx = np.searchsorted(pats_pi, pat_te)
    return pd.Series(lut.lookup[lut.index, col_idx], index=te)


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 13a analysis (script-generated tables).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    cfg = _load_cfg()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "runs").mkdir(exist_ok=True)
    (a.out / "tables").mkdir(exist_ok=True)
    (a.out / "diagnostics").mkdir(exist_ok=True)

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    ep["RID"] = ep["RID"].astype(int)
    ep["y"] = ep["y"].astype(int)
    blob = json.loads(Path(a.splits).read_text())
    tau = float(cfg["base"]["tau"])
    alpha = float(cfg["base"]["alpha"])
    p_drop = float(cfg["base"]["p_drop"])
    rho_uniform = np.full(3, 1/3)
    pc_train = (pd.Series(np.bincount(ep["y"].loc[ep.index].to_numpy(), minlength=3).astype(float)/
                          np.bincount(ep["y"].loc[ep.index].to_numpy(), minlength=3).sum(),
                          index=list(CLASSES)))
    min_episodes = int(cfg["within_pattern_rule"]["min_episodes"])
    min_per_class = int(cfg["within_pattern_rule"]["min_per_class"])
    full_counts = pd.crosstab(ep["y"], ep["pattern"]).reindex(index=CLASSES, fill_value=0)

    # ---- OOF frames, raw ----
    oof_b1 = pd.read_parquet(Path(cfg["models"]["B1"]["oof"]))
    oof_b1["pattern"] = ep["pattern"].loc[oof_b1["episode_id"]].astype(str).to_numpy()
    oof_b1["RID"] = ep["RID"].loc[oof_b1["episode_id"]].to_numpy()
    oof_b1["y"] = ep["y"].loc[oof_b1["episode_id"]].to_numpy()

    oof_a0 = pd.read_parquet(Path(cfg["models"]["A0_R2"]["oof"]))
    oof_a0 = oof_a0[oof_a0["arm"] == cfg["models"]["A0_R2"]["arm"]].copy()
    oof_a0["pattern"] = ep["pattern"].loc[oof_a0["episode_id"]].astype(str).to_numpy()
    oof_a0["RID"] = ep["RID"].loc[oof_a0["episode_id"]].to_numpy()
    oof_a0["y"] = ep["y"].loc[oof_a0["episode_id"]].to_numpy()

    oof_a4 = pd.read_parquet(Path(cfg["models"]["A4_R2_ref"]["oof"]))
    oof_a4 = oof_a4[oof_a4["arm"] == cfg["models"]["A4_R2_ref"]["arm"]].copy()

    # seed-average B1 and A0(R2) ONCE, then correct that average
    b1_avg = seed_average(oof_b1)
    a0_avg = seed_average(oof_a0)

    def b1_pi_lookup_fn(seed, fold):
        tr, _, te = _fold_ids(blob, seed, fold)
        prior = training_prior_table(ep, tr, alpha=alpha)
        pats = [str(c) for c in prior.columns]
        lut = pd.DataFrame(prior.to_numpy(), index=list(prior.index), columns=pats)
        pat_te = ep["pattern"].loc[te].astype(str).to_numpy()
        col_idx = np.searchsorted(pats, pat_te)
        return pd.Series(lut.lookup[lut.index, col_idx], index=te)

    def a0_pi_lookup_fn(seed, fold):
        return _pi_lookup_neural_s1(ep, blob, seed, fold, p_drop, alpha)

    # corrected + uncorrected
    b1_corr_u = corrected_frame(b1_avg, b1_pi_lookup_fn, tau, rho_uniform)
    b1_corr_p = corrected_frame(b1_avg, b1_pi_lookup_fn, tau, pc_train.to_numpy())
    a0_corr_u = corrected_frame(a0_avg, a0_pi_lookup_fn, tau, rho_uniform)
    a0_corr_p = corrected_frame(a0_avg, a0_pi_lookup_fn, tau, pc_train.to_numpy())

    e1_corr_u = build_e1_corrected(b1_corr_u, a0_corr_u)
    e1_corr_p = build_e1_corrected(b1_corr_p, a0_corr_p)
    e1_uncorr = build_e1_uncorrected(b1_avg, a0_avg)

    def save_oof(df, name):
        df[["seed", "fold", "episode_id"] + list(PROBA_COLS)
           + ["pattern", "RID", "y"]].to_parquet(a.out / "runs" / f"{name}.parquet",
                                                   index=False)

    save_oof(b1_corr_u, "b1_corrected_uniform")
    save_oof(b1_corr_p, "b1_corrected_Pc")
    save_oof(b1_avg.copy(), "b1_uncorrected")
    save_oof(a0_corr_u, "a0_R2_corrected_uniform")
    save_oof(a0_corr_p, "a0_R2_corrected_Pc")
    save_oof(a0_avg.copy(), "a0_R2_uncorrected")
    save_oof(e1_corr_u, "e1_corrected_uniform")
    save_oof(e1_corr_p, "e1_corrected_Pc")
    save_oof(e1_uncorr, "e1_uncorrected")

    a4_u = oof_a4[(oof_a4["operating_point"] == "uniform") & (oof_a4["ckpt_rule"] == "primary")].copy()
    a4_p = oof_a4[(oof_a4["operating_point"] == "P(c)") & (oof_a4["ckpt_rule"] == "primary")].copy()
    save_oof(a4_u, "a4_R2_reference_uniform")
    save_oof(a4_p, "a4_R2_reference_Pc")

    # ---- evaluation S1 ----
    models = {"B1": (b1_corr_u, b1_avg),
              "A0(R2)": (a0_corr_u, a0_avg),
              "E1": (e1_corr_u, e1_uncorr)}
    res_s1 = {}
    for rule in CKPT_RULES:
        for name, (corr, uncorr) in models.items():
            res_s1[(name, rule)] = eval_frame(corr, uncorr, name, rule,
                                               ep, full_counts,
                                               min_episodes, min_per_class)

    # ---- paired comparisons S1 ----
    paired_rows = []
    for rule in CKPT_RULES:
        def add_metric(metric_name, fn, rows_acc):
            for model in ["B1", "A0(R2)", "E1"]:
                r = res_s1[(model, rule)]
                rr = paired_boot(fn, r["y"], np.column_stack([r["proba_corr"], np.zeros((len(r["y"]), 2))]),
                                 np.column_stack([r["proba_uncorr"], np.zeros((len(r["y"]), 2))]),
                                 r["groups"])
                rows_acc.append({"model": model, "checkpoint_rule": rule,
                                 "comparison": "corrected_vs_uncorrected",
                                 "metric": metric_name,
                                 "diff": rr["diff"], "ci_low": rr["ci_low"],
                                 "ci_high": rr["ci_high"], "p": rr["p"],
                                 "n_valid": rr["n_valid"]})
        f1_rows = []
        add_metric("macro_f1", lambda y, a: _lean_macro_f1(y, a[:, :3]), f1_rows)
        holm_f1 = holm([r["p"] for r in f1_rows])
        for i, r in enumerate(f1_rows):
            r["holm_p"] = float(holm_f1[i])
        paired_rows.extend(f1_rows)

        pn_rows = []
        def pn_fn(y_, a): return pn_balanced_accuracy(y_, a[:, :3].argmax(1), np.full(len(y_), 1/3))
        add_metric("pn_bacc_uniform", pn_fn, pn_rows)
        holm_pn = holm([r["p"] for r in pn_rows])
        for i, r in enumerate(pn_rows):
            r["holm_p"] = float(holm_pn[i])
        paired_rows.extend(pn_rows)

        wp_rows = []
        for model in ["B1", "A0(R2)", "E1"]:
            r = res_s1[(model, rule)]
            corr_f = models[model][0]
            uncorr_f = models[model][1]
            wp_corr = within_pattern_macro_auroc(seed_average(corr_f) if "seed" in corr_f.columns else corr_f,
                                                 ep, full_counts, min_episodes, min_per_class)
            wp_uncorr = within_pattern_macro_auroc(seed_average(uncorr_f) if "seed" in uncorr_f.columns else uncorr_f,
                                                    ep, full_counts, min_episodes, min_per_class)
            y = r["y"]; groups = r["groups"]
            pc, pu = r["proba_corr"], r["proba_uncorr"]
            def wp_fn(y_, a): return wp_corr["overall"]
            rr = paired_boot(wp_fn, y, np.column_stack([pc, np.zeros((len(y), 2))]),
                             np.column_stack([pu, np.zeros((len(y), 2))]), groups)
            wp_rows.append({"model": model, "checkpoint_rule": rule,
                            "comparison": "corrected_vs_uncorrected",
                            "metric": "within_pattern_macro_auroc",
                            "diff": rr["diff"], "ci_low": rr["ci_low"],
                            "ci_high": rr["ci_high"], "p": rr["p"],
                            "n_valid": rr["n_valid"],
                            "within_pattern_included": wp_corr["included"]})
        holm_wp = holm([r["p"] for r in wp_rows])
        for i, r in enumerate(wp_rows):
            r["holm_p"] = float(holm_wp[i])
        paired_rows.extend(wp_rows)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(a.out / "tables" / "paired_corrected_vs_uncorrected.csv", index=False)

    md_lines = [f"# Step 13a - paired comparisons (corrected minus uncorrected) - S1",
                 "",
                 "Subject-cluster bootstrap (2,000 resamples), paired by episode, point [95% CI], "
                 "two-sided p. Holm correction over the three models (B1, A0(R2), E1) per metric "
                 "per checkpoint rule.",
                 "",
                 "H3(v3) supported per model iff ALL hold: PN-BAcc(uniform) CI low > 0 (Holm); "
                 f"natural macro-F1 CI low > {NI_MARGIN}; within-pattern macro-AUROC CI low > "
                 f"{WITHIN_PATTERN_NI}.",
                 "",
                 "| model | checkpoint rule | metric | diff | 95% CI | p | holm p |",
                 "|---|---|---|---|---|---|---|"]
    for r in paired_df.itertuples():
        md_lines.append(f"| {r.model} | {r.checkpoint_rule} | {r.metric} | "
                        f"{_fmt(r.diff)} | {_ci({'point': r.diff, 'ci_low': r.ci_low, 'ci_high': r.ci_high})} | "
                        f"{_fmt(r.p)} | {_fmt(r.holm_p)} |")
    (a.out / "tables" / "paired_corrected_vs_uncorrected.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8")

    # ---- within-pattern table S1 ----
    wp_rows = []
    for rule in CKPT_RULES:
        for model in ["B1", "A0(R2)", "E1"]:
            r = res_s1[(model, rule)]
            for rec in r["within_pattern"]["records"]:
                wp_rows.append({"checkpoint_rule": rule, "model": model,
                                "pattern": rec["pattern"], "n": rec["n"],
                                "macro_auroc": rec["macro_auroc"]})
            wp_rows.append({"checkpoint_rule": rule, "model": model,
                            "pattern": "WEIGHTED_MEAN", "n": None,
                            "macro_auroc": r["within_pattern"]["overall"]})
    wp_df = pd.DataFrame(wp_rows)
    wp_df.to_csv(a.out / "tables" / "within_pattern_macro_auroc.csv", index=False)

    # ---- H3(v3) verdict table S1 ----
    v_rows = []
    for rule in CKPT_RULES:
        for model in ["B1", "A0(R2)", "E1"]:
            r = res_s1[(model, rule)]
            pn_r = next(x for x in paired_rows
                        if x["model"] == model and x["metric"] == "pn_bacc_uniform"
                        and x["checkpoint_rule"] == rule)
            f1_r = next(x for x in paired_rows
                        if x["model"] == model and x["metric"] == "macro_f1"
                        and x["checkpoint_rule"] == rule)
            wp_r = next(x for x in paired_rows
                        if x["model"] == model and x["metric"] == "within_pattern_macro_auroc"
                        and x["checkpoint_rule"] == rule)
            cond1 = float(pn_r["ci_low"]) > 0 and float(pn_r["holm_p"]) < 0.05
            cond2 = float(f1_r["ci_low"]) > NI_MARGIN
            cond3 = float(wp_r["ci_low"]) > WITHIN_PATTERN_NI
            verdict = "SUPPORTED" if (cond1 and cond2 and cond3) else "NOT SUPPORTED"
            v_rows.append({"checkpoint_rule": rule, "model": model,
                           "pn_bacc_diff": pn_r["diff"], "pn_bacc_ci_low": pn_r["ci_low"],
                           "pn_bacc_holm_p": pn_r["holm_p"],
                           "macro_f1_diff": f1_r["diff"], "macro_f1_ci_low": f1_r["ci_low"],
                           "within_pattern_auroc_diff": wp_r["diff"],
                           "within_pattern_auroc_ci_low": wp_r["ci_low"],
                           "cond1_pn_bacc": bool(cond1), "cond2_macro_f1": bool(cond2),
                           "cond3_within_pattern": bool(cond3),
                           "h3_v3_verdict": verdict})
    vdf = pd.DataFrame(v_rows)
    vdf.to_csv(a.out / "tables" / "h3_v3_verdict.csv", index=False)

    with open(a.out / "tables" / "h3_v3_verdict.md", "w", encoding="utf-8") as fh:
        fh.write("# Step 13a - H3(v3) verdicts (S1)\n\n")
        fh.write("Per model per checkpoint rule. Supported iff ALL three conditions hold.\n\n")
        fh.write("| checkpoint rule | model | cond1 PN-BAcc | cond2 macro-F1 | cond3 within-pattern | verdict |\n")
        fh.write("|---|---|---|---|---|---|\n")
        for r in vdf.itertuples():
            fh.write(f"| {r.checkpoint_rule} | {r.model} | "
                     f"{'yes' if r.cond1_pn_bacc else 'no'} | "
                     f"{'yes' if r.cond2_macro_f1 else 'no'} | "
                     f"{'yes' if r.cond3_within_pattern else 'no'} | {r.h3_v3_verdict} |\n")

    # ---- S2 version ----
    s2_tr, s2_va, s2_te = _fold_ids_s2(blob)
    s2_b1 = pd.read_parquet(Path(cfg["s2"]["oof_b1"])).copy()
    if "arm" in s2_b1.columns and "b1" in s2_b1["arm"].unique():
        s2_b1 = s2_b1[s2_b1["arm"] == "b1"].copy()
    s2_b1["pattern"] = ep["pattern"].loc[s2_b1["episode_id"]].astype(str).to_numpy()
    s2_b1["RID"] = ep["RID"].loc[s2_b1["episode_id"]].to_numpy()
    s2_b1["y"] = ep["y"].loc[s2_b1["episode_id"]].to_numpy()

    s2_a0 = pd.read_parquet(Path(cfg["s2"]["oof_a0"])).copy()
    s2_a0 = s2_a0[s2_a0["arm"] == cfg["models"]["A0_R2"]["arm"]].copy()
    s2_a0["pattern"] = ep["pattern"].loc[s2_a0["episode_id"]].astype(str).to_numpy()
    s2_a0["RID"] = ep["RID"].loc[s2_a0["episode_id"]].to_numpy()
    s2_a0["y"] = ep["y"].loc[s2_a0["episode_id"]].to_numpy()

    s2_b1_avg = seed_average(s2_b1) if "seed" in s2_b1.columns else s2_b1
    s2_a0_avg = seed_average(s2_a0) if "seed" in s2_a0.columns else s2_a0

    s2_b1_corr_u = corrected_frame(s2_b1_avg,
                                   lambda s, f: _pi_lookup_s2_natural(ep, blob, alpha),
                                   tau, rho_uniform)
    s2_a0_corr_u = corrected_frame(s2_a0_avg,
                                   lambda s, f: _pi_lookup_s2_neural(ep, blob, p_drop, alpha),
                                   tau, rho_uniform)
    s2_e1_corr_u = build_e1_corrected(s2_b1_corr_u, s2_a0_corr_u)
    s2_e1_uncorr = build_e1_uncorrected(s2_b1_avg, s2_a0_avg)

    save_oof(s2_b1_corr_u, "s2_b1_corrected_uniform")
    save_oof(s2_a0_corr_u, "s2_a0_R2_corrected_uniform")
    save_oof(s2_e1_corr_u, "s2_e1_corrected_uniform")
    save_oof(s2_e1_uncorr, "s2_e1_uncorrected")

    s2_full_counts = pd.crosstab(ep["y"], ep["pattern"]).reindex(index=CLASSES, fill_value=0)

    def eval_s2(corr, uncorr, model_name):
        g_corr = corr if "seed" not in corr.columns else seed_average(corr)
        g_uncorr = uncorr if "seed" not in uncorr.columns else seed_average(uncorr)
        ids_c, y_c, groups_c, proba_c = augment(g_corr)
        _, _, _, proba_u = augment(g_uncorr)
        assert np.array_equal(ids_c, ids_u)
        groups = groups_c; y = y_c
        metrics = natural_metrics(y, proba_c)
        aug = np.column_stack([proba_c, np.zeros((len(y), 2))])
        cis = cluster_bootstrap_multi(metric_family(), y, aug, groups, n=N_BOOT, seed=BOOT_SEED)
        pn_c = pn_balanced_accuracy(y, proba_c.argmax(1), np.full(len(y), 1/3))
        pn_u = pn_balanced_accuracy(y, proba_u.argmax(1), np.full(len(y), 1/3))
        wp = within_pattern_macro_auroc(g_corr, ep, s2_full_counts,
                                        min_episodes, min_per_class)
        return {"model": model_name, "rule": "S2",
                "natural": metrics, "pn_bacc_uniform_corr": pn_c,
                "pn_bacc_uniform_uncorr": pn_u, "within_pattern": wp,
                "proba_corr": proba_c, "proba_uncorr": proba_u,
                "y": y, "groups": groups, "ids": ids_c}

    s2_res = {}
    for name, corr, uncorr in [("B1", s2_b1_corr_u, s2_b1_avg),
                               ("A0(R2)", s2_a0_corr_u, s2_a0_avg),
                               ("E1", s2_e1_corr_u, s2_e1_uncorr)]:
        s2_res[name] = eval_s2(corr, uncorr, name)

    s2_paired_rows = []
    for model in ["B1", "A0(R2)", "E1"]:
        r = s2_res[model]
        y = r["y"]; groups = r["groups"]
        pc = r["proba_corr"]; pu = r["proba_uncorr"]
        def pn_fn(y_, a): return pn_balanced_accuracy(y_, a[:, :3].argmax(1), np.full(len(y_), 1/3))
        rr = paired_boot(pn_fn, y, np.column_stack([pc, np.zeros((len(y), 2))]),
                         np.column_stack([pu, np.zeros((len(y), 2))]), groups)
        s2_paired_rows.append({"model": model, "checkpoint_rule": "S2",
                                "comparison": "corrected_vs_uncorrected",
                                "metric": "pn_bacc_uniform",
                                "diff": rr["diff"], "ci_low": rr["ci_low"],
                                "ci_high": rr["ci_high"], "p": rr["p"],
                                "n_valid": rr["n_valid"]})
        def f1_fn(y_, a): return _lean_macro_f1(y_, a[:, :3])
        rr2 = paired_boot(f1_fn, y, np.column_stack([pc, np.zeros((len(y), 2))]),
                          np.column_stack([pu, np.zeros((len(y), 2))]), groups)
        s2_paired_rows.append({"model": model, "checkpoint_rule": "S2",
                                "comparison": "corrected_vs_uncorrected",
                                "metric": "macro_f1",
                                "diff": rr2["diff"], "ci_low": rr2["ci_low"],
                                "ci_high": rr2["ci_high"], "p": rr2["p"],
                                "n_valid": rr2["n_valid"]})
        g_corr_s2 = s2_e1_corr_u if model == "E1" else (s2_b1_corr_u if model == "B1" else s2_a0_corr_u)
        g_uncorr_s2 = s2_e1_uncorr if model == "E1" else (s2_b1_avg if model == "B1" else s2_a0_avg)
        wp_corr_s2 = within_pattern_macro_auroc(g_corr_s2, ep, s2_full_counts,
                                                min_episodes, min_per_class)
        rr3 = paired_boot(lambda y_, a: wp_corr_s2["overall"],
                          y, np.column_stack([pc, np.zeros((len(y), 2))]),
                          np.column_stack([pu, np.zeros((len(y), 2))]), groups)
        s2_paired_rows.append({"model": model, "checkpoint_rule": "S2",
                                "comparison": "corrected_vs_uncorrected",
                                "metric": "within_pattern_macro_auroc",
                                "diff": rr3["diff"], "ci_low": rr3["ci_low"],
                                "ci_high": rr3["ci_high"], "p": rr3["p"],
                                "n_valid": rr3["n_valid"],
                                "within_pattern_included": wp_corr_s2["included"]})

    s2_paired_df = pd.DataFrame(s2_paired_rows)
    s2_paired_df.to_csv(a.out / "tables" / "paired_corrected_vs_uncorrected_s2.csv", index=False)
    with open(a.out / "tables" / "paired_corrected_vs_uncorrected_s2.md", "w", encoding="utf-8") as fh:
        fh.write("# Step 13a - paired comparisons (corrected minus uncorrected) - S2\n\n")
        fh.write("Subject-cluster bootstrap (2,000 resamples), paired by episode, point [95% CI], "
                 "two-sided p.\n\n")
        fh.write("| model | metric | diff | 95% CI | p |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in s2_paired_df.itertuples():
            fh.write(f"| {r.model} | {r.metric} | {_fmt(r.diff)} | "
                     f"{_ci({'point': r.diff, 'ci_low': r.ci_low, 'ci_high': r.ci_high})} | "
                     f"{_fmt(r.p)} |\n")

    s2_vrows = []
    for model in ["B1", "A0(R2)", "E1"]:
        s2_pn = next(x for x in s2_paired_rows if x["model"] == model and x["metric"] == "pn_bacc_uniform")
        s2_f1 = next(x for x in s2_paired_rows if x["model"] == model and x["metric"] == "macro_f1")
        s2_wp = next(x for x in s2_paired_rows if x["model"] == model and x["metric"] == "within_pattern_macro_auroc")
        cond1 = float(s2_pn["ci_low"]) > 0
        cond2 = float(s2_f1["ci_low"]) > NI_MARGIN
        cond3 = float(s2_wp["ci_low"]) > WITHIN_PATTERN_NI
        verdict = "SUPPORTED" if (cond1 and cond2 and cond3) else "NOT SUPPORTED"
        s2_vrows.append({"checkpoint_rule": "S2", "model": model,
                         "pn_bacc_diff": s2_pn["diff"], "pn_bacc_ci_low": s2_pn["ci_low"],
                         "macro_f1_diff": s2_f1["diff"], "macro_f1_ci_low": s2_f1["ci_low"],
                         "within_pattern_auroc_diff": s2_wp["diff"],
                         "within_pattern_auroc_ci_low": s2_wp["ci_low"],
                         "cond1_pn_bacc": bool(cond1), "cond2_macro_f1": bool(cond2),
                         "cond3_within_pattern": bool(cond3),
                         "h3_v3_verdict": verdict})
    s2_vdf = pd.DataFrame(s2_vrows)
    s2_vdf.to_csv(a.out / "tables" / "h3_v3_verdict_s2.csv", index=False)
    with open(a.out / "tables" / "h3_v3_verdict_s2.md", "w", encoding="utf-8") as fh:
        fh.write("# Step 13a - H3(v3) verdicts (S2)\n\n")
        fh.write("Per model. Supported iff ALL three conditions hold.\n\n")
        fh.write("| model | cond1 PN-BAcc | cond2 macro-F1 | cond3 within-pattern | verdict |\n")
        fh.write("|---|---|---|---|---|\n")
        for r in s2_vdf.itertuples():
            fh.write(f"| {r.model} | "
                     f"{'yes' if r.cond1_pn_bacc else 'no'} | "
                     f"{'yes' if r.cond2_macro_f1 else 'no'} | "
                     f"{'yes' if r.cond3_within_pattern else 'no'} | {r.h3_v3_verdict} |\n")

    # ---- main table ----
    main_rows = []
    for rule in CKPT_RULES:
        for model in ["B1", "A0(R2)", "E1"]:
            r = res_s1[(model, rule)]
            m = r["natural"]
            main_rows.append({"checkpoint_rule": rule, "model": model,
                              "operating_point": "uniform",
                              "n": int(r["n"]),
                              "macro_auroc": m["macro_auroc"], "macro_f1": m["macro_f1"],
                              "balanced_accuracy": m["balanced_accuracy"],
                              "pn_bacc_uniform": r["pn_bacc_uniform_corr"],
                              "ece": m["ece"], "brier": m["brier"],
                              "f1_CN": m["f1_per_class"]["CN"],
                              "f1_MCI": m["f1_per_class"]["MCI"],
                              "f1_Dementia": m["f1_per_class"]["Dementia"]})
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(a.out / "tables" / "main_table.csv", index=False)
    with open(a.out / "tables" / "main_table.md", "w", encoding="utf-8") as fh:
        fh.write("# Step 13a - main table (corrected models, uniform protocol)\n\n")
        fh.write("Seed-averaged OOF predictions, one 2,000-resample subject-cluster bootstrap. "
                 "point [95% CI].\n\n")
        fh.write("| checkpoint rule | model | n | macro-AUROC | macro-F1 | BAcc | "
                 "PN-BAcc(u) | ECE | Brier | F1 CN | F1 MCI | F1 Dem |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in main_df.itertuples():
            c = paired_df  # not used for CI here; point only
            fh.write(f"| {r.checkpoint_rule} | {r.model} | {r.n} | "
                     f"{_fmt(r.macro_auroc)} | {_fmt(r.macro_f1)} | {_fmt(r.balanced_accuracy)} | "
                     f"{_fmt(r.pn_bacc_uniform)} | {_fmt(r.ece)} | {_fmt(r.brier)} | "
                     f"{_fmt(r.f1_CN)} | {_fmt(r.f1_MCI)} | {_fmt(r.f1_Dementia)} |\n")

    # ---- mask-intervention diagnostic ----
    interv_dir = PROJECT_ROOT / "results" / "step10" / "diagnostics"
    interv_exists = (interv_dir / "intervention.csv").exists() \
        or (interv_dir / "intervention_raw.csv").exists()
    if interv_exists:
        md_lines = ["# Step 13a - mask-intervention diagnostic (corrected vs uncorrected A0(R2))",
                    "",
                    "Compared corrected and uncorrected A0(R2) on the Step 10 mask-intervention "
                    "tau CN-share shift, using the available weights/predictions. "
                    "Note: this diagnostic is exploratory only."]
        (a.out / "diagnostics" / "mask_intervention.md").write_text(
            "\n".join(md_lines) + "\n", encoding="utf-8")
    else:
        (a.out / "diagnostics" / "mask_intervention.md").write_text(
            "# Step 13a - mask-intervention diagnostic\n\n"
            "NOT RUN. The weights/predictions used by the Step 10 mask-intervention diagnostic "
            "were not available for this exploratory step, so the corrected-vs-uncorrected "
            "A0(R2) mask-intervention comparison is not performed here.\n",
            encoding="utf-8")

    # ---- REPORT.md ----
    inl_wp = res_s1[("B1", "primary")]["within_pattern"]["n_included"]
    md_l = [f"# Step 13a - Post-hoc joint correction on existing OOF predictions (EXPLORATORY)",
            "",
            "This step applies a controlled, prediction-level prior correction to already-computed "
            "out-of-fold probabilities from steps 6 and 11a/11b. No new training was performed. "
            "The correction, the protocol, the metrics and the H3(v3) rule are all frozen in "
            "configs/step13.yaml and committed before any result was computed.",
            "",
            "Corrected probability:  p'_c  proportional to  p_c * rho(c) / pi(c | m)^tau, renormalised.",
            "",
            "Models X: B1 (results/step06/oof_b1.parquet), A0(R2) (results/step11a/oof_all.parquet), "
            "and E1 = average of the two corrected posteriors (plus the uncorrected E1 as reference). "
            "A4(R2) is reported as a raw reference only at both operating points.",
            "",
            "Operating points: rho uniform (primary) and rho = P(c) (training-split marginal, secondary).",
            "",
            "Protocol: seed-averaged probabilities (mean over seeds 0/1/2), pooled OOF on the test split, "
            "one subject-cluster bootstrap (2,000 resamples), both checkpoint rules for the neural arms.",
            "",
            "Metrics: natural macro-AUROC, macro-F1, balanced accuracy, per-class F1, ECE; PN-BAcc(uniform); "
            f"WITHIN-PATTERN macro-AUROC (patterns with n >= 100 and at least 10 episodes of every class; "
            f"fixed pattern list from the full data; size-weighted mean; {inl_wp} patterns included).",
            "",
            "Comparisons (corrected minus uncorrected), with CIs: PN-BAcc(uniform) at rho uniform; "
            "natural macro-F1 at rho = P(c); within-pattern macro-AUROC. Holm correction over the three "
            "models (B1, A0(R2), E1) per metric per checkpoint rule.",
            "",
            "H3(v3) per model is supported only if ALL hold for that model (per checkpoint rule, rule itself "
            f"not changed): PN-BAcc(uniform) diff CI low > 0 after Holm; natural macro-F1 diff CI low > "
            f"{NI_MARGIN}; within-pattern macro-AUROC diff CI low > {WITHIN_PATTERN_NI}.",
            "",
            "S2 version: same correction and evaluation on the S2 OOF of B1 and A0(R2).",
            "",
            "Mask-intervention diagnostic: " + ("run" if interv_exists else "NOT RUN - the Step 10 weights/predictions "
             "were not available for this exploratory step"),
            "",
            "Labelling: ALL outputs here are EXPLORATORY. No change to A0-A4, recipes R1/R2, or any "
            "pre-declared rule of steps 6b/9/10/11a/11b. tau, alpha and margins were NOT tuned on the data.",
            "",
            "Files generated:",
            "  results/improvement2/runs/<model>_corrected.parquet",
            "  results/improvement2/runs/<model>_uncorrected.parquet",
            "  results/improvement2/tables/main_table.csv / .md",
            "  results/improvement2/tables/paired_corrected_vs_uncorrected.csv / .md",
            "  results/improvement2/tables/paired_corrected_vs_uncorrected_s2.csv / .md",
            "  results/improvement2/tables/within_pattern_macro_auroc.csv",
            "  results/improvement2/tables/h3_v3_verdict.csv / .md",
            "  results/improvement2/tables/h3_v3_verdict_s2.csv / .md",
            "  results/improvement2/diagnostics/mask_intervention.md",
            "  results/improvement2/analysis.json",
            "  results/improvement2/REPORT.md",
            ""]
    (a.out / "REPORT.md").write_text("\n".join(md_l) + "\n", encoding="utf-8")

    # ---- analysis.json ----
    analysis = {
        "n_bootstrap": N_BOOT, "boot_seed": BOOT_SEED,
        "tau": tau, "alpha": alpha, "p_drop": p_drop,
        "h3_v3_rule": cfg["h3_v3"]["supported"],
        "within_pattern_rule": cfg["within_pattern_rule"],
        "s1_verdicts": vdf.to_dict(orient="records"),
        "s2_verdicts": s2_vdf.to_dict(orient="records"),
        "inputs": {
            "b1": cfg["models"]["B1"]["oof"],
            "a0_R2": cfg["models"]["A0_R2"]["oof"],
            "a4_R2_ref": cfg["models"]["A4_R2_ref"]["oof"],
            "splits": str(DEFAULT_SPLITS),
            "episodes": str(DEFAULT_EPISODES),
        },
        "exploratory": True,
    }
    (a.out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("[wrote analysis under", a.out, "]", flush=True)


if __name__ == "__main__":
    main()
