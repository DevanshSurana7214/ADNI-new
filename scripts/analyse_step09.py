#!/usr/bin/env python3
"""Step 9 analysis - script-generated tables (PIPELINE_v2 section 8, uniform protocol of Step 6b).

Protocol (identical to Step 6b): average the probabilities of each episode over the 3 seeds, pool
the out-of-fold predictions, and run ONE subject-cluster bootstrap (2,000 resamples) per arm. This
is done separately for each checkpoint rule (PRIMARY = best validation macro-F1, SECONDARY = best
validation PN-BAcc). Every table in results/step09/REPORT.md is written here.

Outputs under results/step09/:
  main_table_{primary,secondary}.{csv,md}     arms A0-A4 (+/- both operating points) and B0/B1/
                                              B1-blind/B2_mri, natural + PN metrics with CIs
  paired_{primary,secondary}.{csv,md}         paired A4 - {A0,A1,A2,A3} bootstrap, Holm over the
                                              four PN-BAcc comparisons, macro-F1 non-inferiority
  breakdown_{primary,secondary}.{csv,md}      A0/A1/A4 by pattern, n_mod, cell-count tertile,
                                              COLPROT and scan-gap tertile
  stability_per_seed_{rule}.{csv,md}          per-seed metric mean +/- SD per arm
  stability_per_fold_{rule}.{csv,md}          per (arm, seed) x fold PN-BAcc and macro-F1
  best_epochs_{rule}.{csv,md}                 best-epoch distribution per arm (flag median < 5)
  runtime.{csv,md}                            per-arm runtime and anomalies
  analysis.json                               machine-readable summary incl. the verdicts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from adni.baselines import training_prior_table
from adni.metrics import (
    CLASSES,
    _group_index,
    _resample_indices,
    cluster_bootstrap_multi,
    holm,
    natural_metrics,
    pn_balanced_accuracy,
    pn_weights,
)
from adni.priors import crosstab

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step09"
PROBA_COLS = ["p_CN", "p_MCI", "p_Dementia"]
MODALITIES = ("mri", "fdg", "amy", "tau")
CKPT_RULES = ("primary", "secondary")
BASELINES = {
    "B0": PROJECT_ROOT / "results" / "step06" / "oof_b0.parquet",
    "B1": PROJECT_ROOT / "results" / "step06b" / "oof_b1_i.parquet",
    "B1-blind": PROJECT_ROOT / "results" / "step06c" / "oof_b1_blind.parquet",
    "B2_mri": PROJECT_ROOT / "results" / "step06" / "oof_b2_mri.parquet",
}
N_BOOT = 2000
BOOT_SEED = 0
NI_MARGIN = 0.02          # pre-specified non-inferiority margin for natural macro-F1
ALPHA = 0.05
# arm label -> (rows-key, operating point). A1/A4 use both operating points; others argmax.
ARM_OP_LABEL = {
    "A0": [("A0", "argmax")], "A2": [("A2", "argmax")], "A3": [("A3", "argmax")],
    "A1": [("A1 (uniform)", "uniform"), ("A1 (P(c))", "P(c)")],
    "A4": [("A4 (uniform)", "uniform"), ("A4 (P(c))", "P(c)")],
}
PRIMARY_OP = {"A0": "argmax", "A1": "uniform", "A2": "argmax", "A3": "argmax", "A4": "uniform"}


# --------------------------------------------------------------------------- lean metrics
def _lean_macro_auroc(y, proba):
    out = []
    for c in range(proba.shape[1]):
        pos = y == c
        n_pos = int(pos.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(proba[:, c])
        out.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(out)) if out else float("nan")


def _lean_macro_f1(y, proba):
    return float(f1_score(y, proba.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0))


def _lean_bacc(y, proba):
    pred = proba.argmax(1)
    rec = [float((pred[y == c] == c).mean()) for c in range(3) if (y == c).any()]
    return float(np.mean(rec)) if rec else float("nan")


def _lean_ece(y, proba, n_bins=15):
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.digitize(conf, edges[1:-1], right=True)
    n = len(y)
    ece = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _lean_brier(y, proba):
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def metric_family():
    """Metrics for cluster_bootstrap_multi on the augmented matrix [p3, w_uniform, w_prior]."""
    def aug(fn):
        return lambda y, a: fn(y, a[:, :3])

    return {
        "macro_auroc": aug(_lean_macro_auroc),
        "macro_f1": aug(_lean_macro_f1),
        "balanced_accuracy": aug(_lean_bacc),
        "pn_bacc_uniform": lambda y, a: pn_balanced_accuracy(y, a[:, :3].argmax(1), a[:, 3]),
        "pn_bacc_prior": lambda y, a: pn_balanced_accuracy(y, a[:, :3].argmax(1), a[:, 4]),
        "ece": aug(_lean_ece),
        "brier": aug(_lean_brier),
    }


def paired_boot(fn, y, aug_a, aug_b, groups, n=None, seed=BOOT_SEED):
    """Paired subject-cluster bootstrap for fn(a) - fn(b): point, 95% CI and a two-sided p-value."""
    n = N_BOOT if n is None else n
    order, starts = _group_index(groups)
    rng = np.random.default_rng(seed)
    point = float(fn(y, aug_a)) - float(fn(y, aug_b))
    ds = []
    for _ in range(n):
        idx = _resample_indices(order, starts, rng)
        try:
            d = float(fn(y[idx], aug_a[idx])) - float(fn(y[idx], aug_b[idx]))
        except Exception:
            d = float("nan")
        if np.isfinite(d):
            ds.append(d)
    ds = np.asarray(ds)
    if ds.size == 0:
        return {"diff": point, "ci_low": float("nan"), "ci_high": float("nan"), "p": float("nan"),
                "n_valid": 0}
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p_less = (1 + int((ds <= 0).sum())) / (len(ds) + 1)
    p_greater = (1 + int((ds >= 0).sum())) / (len(ds) + 1)
    p = min(1.0, 2.0 * min(p_less, p_greater))
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi), "p": float(p),
            "n_valid": int(len(ds))}


# --------------------------------------------------------------------------- data assembly
def _fold_ids(blob, seed, fold):
    node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
    return tuple(np.asarray(node[k], dtype=np.int64) for k in ("train", "val", "test"))


def build_weights(ep, blob, seeds=(0, 1, 2), folds=range(5)) -> pd.DataFrame:
    """PN weights for every (seed, fold, episode) from that fold's TRAIN-split P(c|m)."""
    recs = []
    for seed in seeds:
        for fold in folds:
            tr, _, te = _fold_ids(blob, seed, fold)
            prior = training_prior_table(ep, tr)
            yy = ep["y"].loc[list(te)].to_numpy()
            pp = ep["pattern"].loc[list(te)].to_numpy()
            wu = pn_weights(yy, pp, prior, rho="uniform")
            pc = np.bincount(ep["y"].loc[list(tr)].to_numpy(), minlength=3).astype(float)
            pc = pc / pc.sum()
            wp = pn_weights(yy, pp, prior, rho=pc)
            recs.append(pd.DataFrame({"seed": int(seed), "fold": int(fold),
                                      "episode_id": np.asarray(te, dtype=np.int64),
                                      "w_uniform": wu, "w_prior": wp}))
    return pd.concat(recs, ignore_index=True)


def seed_average(frame: pd.DataFrame) -> pd.DataFrame:
    """Average probabilities and weights over the 3 seeds per episode (Step 6b uniform protocol)."""
    g = (frame.groupby("episode_id", sort=True)
         .agg(p_CN=("p_CN", "mean"), p_MCI=("p_MCI", "mean"), p_Dementia=("p_Dementia", "mean"),
              w_uniform=("w_uniform", "mean"), w_prior=("w_prior", "mean"), n_seeds=("seed", "nunique")))
    return g


def augment(g: pd.DataFrame, ep: pd.DataFrame):
    ids = g.index.to_numpy()
    y = ep["y"].loc[ids].to_numpy()
    groups = ep["RID"].loc[ids].to_numpy()
    aug = np.column_stack([g[PROBA_COLS].to_numpy(), g["w_uniform"].to_numpy(), g["w_prior"].to_numpy()])
    return ids, y, groups, aug


def evaluate(g, ep, n=None, seed=BOOT_SEED):
    """Point + bootstrap CIs for the full metric family, plus per-class F1."""
    n = N_BOOT if n is None else n
    ids, y, groups, aug = augment(g, ep)
    m = natural_metrics(y, g[PROBA_COLS].to_numpy())
    ci = cluster_bootstrap_multi(metric_family(), y, aug, groups, n=n, seed=seed)
    out = {"n": int(len(ids)), "min_seed_coverage": int(g["n_seeds"].min())}
    for k in ("macro_auroc", "macro_f1", "balanced_accuracy", "pn_bacc_uniform", "pn_bacc_prior",
              "ece", "brier"):
        out[k] = {"point": float(m[k]) if k in m else float(ci[k].point),
                  "ci_low": float(ci[k].ci_low), "ci_high": float(ci[k].ci_high)}
    out["f1_per_class"] = {c: float(m["f1_per_class"][i]) for i, c in enumerate(CLASSES)}
    return out


# --------------------------------------------------------------------------- tables
def _fmt(v, nd=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _ci(d, nd=4):
    return f"{_fmt(d['point'], nd)} [{_fmt(d['ci_low'], nd)}, {_fmt(d['ci_high'], nd)}]"


def _all_row_keys():
    return (ARM_OP_LABEL["A0"] + ARM_OP_LABEL["A1"] + ARM_OP_LABEL["A2"] + ARM_OP_LABEL["A3"]
            + ARM_OP_LABEL["A4"] + [(n, "-") for n in BASELINES])


METRIC_KEYS = ("macro_auroc", "macro_f1", "balanced_accuracy", "pn_bacc_uniform",
               "pn_bacc_prior", "ece", "brier")


def main_table_rows(cis):
    rows = []
    for key, op in _all_row_keys():
        c = cis[key]
        rows.append({"arm": key, "operating_point": op,
                     **{k: c[k]["point"] for k in METRIC_KEYS},
                     "f1_CN": c["f1_per_class"]["CN"], "f1_MCI": c["f1_per_class"]["MCI"],
                     "f1_Dementia": c["f1_per_class"]["Dementia"], "n": c["n"]})
    return rows


def eval_cis(arm_frame_provider, ep, rule):
    cis = {}
    for key, op in _all_row_keys():
        cis[key] = evaluate(arm_frame_provider(key, op, rule), ep)
    return cis


def write_main(rows, cis, out, rule):
    df = pd.DataFrame(rows)
    df.to_csv(out / f"main_table_{rule}.csv", index=False)
    L = [f"# Step 9 - main table ({rule} checkpoint rule)", "",
         "Seed-averaged OOF predictions, one 2,000-resample subject-cluster bootstrap (Step 6b "
         "uniform protocol). point [95% CI].", "",
         "| arm | op | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | PN-BAcc(P(c)) | ECE | Brier | "
         "F1 CN | F1 MCI | F1 Dem |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = cis[r["arm"]]
        L.append(f"| {r['arm']} | {r['operating_point']} | {r['n']} | {_ci(c['macro_auroc'])} | "
                 f"{_ci(c['macro_f1'])} | {_ci(c['balanced_accuracy'])} | {_ci(c['pn_bacc_uniform'])} | "
                 f"{_ci(c['pn_bacc_prior'])} | {_ci(c['ece'])} | {_ci(c['brier'])} | "
                 f"{_fmt(r['f1_CN'])} | {_fmt(r['f1_MCI'])} | {_fmt(r['f1_Dementia'])} |")
    (out / f"main_table_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return df


def write_paired(rows, out, rule):
    df = pd.DataFrame(rows)
    df.to_csv(out / f"paired_{rule}.csv", index=False)
    L = [f"# Step 9 - primary paired comparisons ({rule} checkpoint rule)", "",
         "A4(primary/rho-uniform) minus each of A0 (argmax), A1 (rho-uniform), A2 (argmax), "
         "A3 (argmax); paired subject-cluster bootstrap (2,000), point [95% CI], two-sided p. "
         f"Holm correction over the four PN-BAcc(uniform) comparisons. Non-inferiority margin for "
         f"macro-F1: {NI_MARGIN}.", "",
         "| comparison | metric | diff | 95% CI | p | holm p |", "|---|---|---|---|---|---|"]
    for r in rows:
        hp = "-" if r.get("holm_p") is None or not np.isfinite(r.get("holm_p", float("nan"))) else f"{r['holm_p']:.4f}"
        L.append(f"| {r['comparison']} | {r['metric']} | {_fmt(r['diff'])} | "
                 f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | {_fmt(r['p'])} | {hp} |")
    (out / f"paired_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return df


def stratum_point(y, proba, w_u, w_p):
    m = natural_metrics(y, proba)
    pred = proba.argmax(1)
    return {"n": int(len(y)), "macro_auroc": float(m["macro_auroc"]), "macro_f1": float(m["macro_f1"]),
            "balanced_accuracy": float(m["balanced_accuracy"]),
            "pn_bacc_uniform": float(pn_balanced_accuracy(y, pred, w_u)),
            "pn_bacc_prior": float(pn_balanced_accuracy(y, pred, w_p)),
            "ece": float(m["ece"]), "brier": float(m["brier"])}


def breakdown_table(arm_frame_provider, ep, blob, rule, arm, op, strata_kind, n_mod, cell_count, max_gap):
    g = arm_frame_provider(arm, op, rule)
    ids, y, _, _ = augment(g, ep)
    proba = g[PROBA_COLS].to_numpy()
    w_u = g["w_uniform"].to_numpy()
    w_p = g["w_prior"].to_numpy()
    rows = []
    if strata_kind == "pattern":
        pat = ep["pattern"].loc[ids].to_numpy()
        vals, counts = np.unique(pat, return_counts=True)
        small = vals[counts < 30]
        for v, c in zip(vals, counts):
            if c >= 30:
                sel = pat == v
                rows.append({"stratum": v, **stratum_point(y[sel], proba[sel], w_u[sel], w_p[sel])})
        if small.size:
            sel = np.isin(pat, small)
            rows.append({"stratum": "other (<30)", **stratum_point(y[sel], proba[sel], w_u[sel], w_p[sel])})
    elif strata_kind == "n_mod":
        nm = n_mod.loc[ids].to_numpy()
        for v in sorted(np.unique(nm)):
            sel = nm == v
            rows.append({"stratum": f"n_mod={int(v)}", **stratum_point(y[sel], proba[sel], w_u[sel], w_p[sel])})
    elif strata_kind == "cell_count":
        cc = cell_count.loc[ids].to_numpy()
        q = np.quantile(cc, [1 / 3, 2 / 3])
        labels = np.where(cc <= q[0], "low", np.where(cc <= q[1], "mid", "high"))
        for lab in ("low", "mid", "high"):
            sel = labels == lab
            rows.append({"stratum": f"cell_count {lab} (<= {q[0]:.0f} / <= {q[1]:.0f})",
                         **stratum_point(y[sel], proba[sel], w_u[sel], w_p[sel])})
    elif strata_kind == "colprot":
        cp = ep["COLPROT"].loc[ids].to_numpy()
        for v, c in zip(*np.unique(cp, return_counts=True)):
            sel = cp == v
            rows.append({"stratum": str(v), **stratum_point(y[sel], proba[sel], w_u[sel], w_p[sel])})
    elif strata_kind == "gap":
        gp = max_gap.loc[ids].to_numpy()
        q = np.quantile(gp, [1 / 3, 2 / 3])
        labels = np.where(gp <= q[0], "low", np.where(gp <= q[1], "mid", "high"))
        for lab in ("low", "mid", "high"):
            sel = labels == lab
            rows.append({"stratum": f"max|gap| {lab} (<= {q[0]:.0f} / <= {q[1]:.0f} d)",
                         **stratum_point(y[sel], proba[sel], w_u[sel], w_p[sel])})
    return [{"arm": arm, "operating_point": op, "breakdown": strata_kind, **r} for r in rows]


# --------------------------------------------------------------------------- main
def main() -> None:
    global N_BOOT
    ap = argparse.ArgumentParser(description="Step 9 analysis (script-generated tables).")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--oof", type=Path, default=DEFAULT_OUT / "oof_all.parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=N_BOOT)
    a = ap.parse_args()
    N_BOOT = a.bootstrap
    a.out.mkdir(parents=True, exist_ok=True)

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    oof = pd.read_parquet(a.oof)
    wdf = build_weights(ep, blob)
    oof = oof.merge(wdf, on=["seed", "fold", "episode_id"], how="left")
    assert oof["w_uniform"].notna().all(), "every OOF row must have fold PN weights"
    print(f"[data] oof rows={len(oof)}, arms={sorted(oof['arm'].unique())}, "
          f"runs={oof.groupby(['arm','seed','fold']).ngroups}", flush=True)

    # episode-level covariates
    n_mod = ep["n_mod"].astype(int)
    gap_cols = [f"{k}_gap_days" for k in MODALITIES]
    gaps = ep[gap_cols].abs()
    mcols = [f"m_{k}" for k in MODALITIES]
    max_gap = gaps.where(ep[mcols].to_numpy() == 1).max(axis=1)
    # per (seed, fold) training cell count of the (true class, pattern) cell
    cc_recs = []
    for seed in (0, 1, 2):
        for fold in range(5):
            tr, _, te = _fold_ids(blob, seed, fold)
            sub = ep.loc[tr]
            diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
            n_cm, pats = crosstab(diag, sub["pattern"])
            lookup = pd.DataFrame(n_cm, index=CLASSES, columns=pats)
            for eid in te:
                y = CLASSES[int(ep["y"].loc[eid])]
                pat = str(ep["pattern"].loc[eid])
                cc_recs.append({"seed": int(seed), "fold": int(fold), "episode_id": int(eid),
                                "cell_count": float(lookup.loc[y, pat]) if pat in lookup.columns else 0.0})
    ccdf = pd.DataFrame(cc_recs)
    cc_mean = ccdf.groupby("episode_id")["cell_count"].mean()
    cell_count = cc_mean  # seed-averaged training cell count per episode

    # frames: (arm label / op / rule) -> seed-averaged
    base_frames: dict[str, pd.DataFrame] = {}
    for name, path in BASELINES.items():
        d = pd.read_parquet(path)
        if "arm" in d.columns:
            d = d.drop(columns=["arm"])
        d["arm"] = name
        base_frames[name] = seed_average(d.merge(wdf, on=["seed", "fold", "episode_id"], how="left"))

    def provider(key, op, rule):
        """Return the seed-averaged frame for a table row key and operating point."""
        if key in BASELINES:
            return base_frames[key]
        arm = key.split(" ")[0]
        d = oof[(oof["arm"] == arm) & (oof["ckpt_rule"] == rule) & (oof["operating_point"] == op)]
        assert len(d), f"no OOF rows for {arm} {rule} {op}"
        return seed_average(d)

    # main tables (per checkpoint rule)
    main_dfs = {}
    cis_by_rule = {}
    for rule in CKPT_RULES:
        cis = eval_cis(provider, ep, rule)
        cis_by_rule[rule] = cis
        rows = main_table_rows(cis)
        main_dfs[rule] = write_main(rows, cis, a.out, rule)
        print(f"[main] wrote main_table_{rule}.csv", flush=True)

    # paired comparisons (per checkpoint rule)
    paired_out = {}
    verdicts = {}
    for rule in CKPT_RULES:
        g_a4 = provider("A4 (uniform)", "uniform", rule)
        rows = []
        pn_ps = []
        for x, xop in (("A0", "argmax"), ("A1", "uniform"), ("A2", "argmax"), ("A3", "argmax")):
            gx = provider(x, xop, rule)
            common = g_a4.index.intersection(gx.index)
            ga = g_a4.loc[common]
            gb = gx.loc[common]
            y = ep["y"].loc[common].to_numpy()
            groups = ep["RID"].loc[common].to_numpy()
            aug_a = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_uniform"].to_numpy(), ga["w_prior"].to_numpy()])
            aug_b = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_uniform"].to_numpy(), gb["w_prior"].to_numpy()])
            mf = metric_family()
            for metric in ("pn_bacc_uniform", "macro_f1", "macro_auroc"):
                r = paired_boot(mf[metric], y, aug_a, aug_b, groups)
                row = {"comparison": f"A4 - {x}", "metric": metric, "diff": r["diff"],
                       "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"], "n_valid": r["n_valid"],
                       "holm_p": None}
                rows.append(row)
                if metric == "pn_bacc_uniform":
                    pn_ps.append(r["p"])
        holm_p = holm(pn_ps)
        for i, row in enumerate([r for r in rows if r["metric"] == "pn_bacc_uniform"]):
            row["holm_p"] = float(holm_p[i])
        df = write_paired(rows, a.out, rule)
        paired_out[rule] = df

        pn_rows = [r for r in rows if r["metric"] == "pn_bacc_uniform"]
        f1_rows = [r for r in rows if r["metric"] == "macro_f1"]
        wins = all(r["diff"] > 0 and np.isfinite(r["holm_p"]) and r["holm_p"] < ALPHA for r in pn_rows)
        ni_ok = all(r["ci_low"] > -NI_MARGIN for r in f1_rows)
        verdict = ("H3 supported" if (wins and ni_ok)
                   else "trade-off (A4 wins PN-BAcc but fails macro-F1 non-inferiority)" if wins
                   else "negative result")
        verdicts[rule] = {"wins_pn_bacc_holm": bool(wins), "macro_f1_non_inferior": bool(ni_ok),
                          "verdict": verdict,
                          "pn_bacc_diffs": {r["comparison"]: r["diff"] for r in pn_rows},
                          "pn_bacc_holm_p": {r["comparison"]: r["holm_p"] for r in pn_rows},
                          "macro_f1_ci_low": {r["comparison"]: r["ci_low"] for r in f1_rows}}
        print(f"[paired {rule}] {verdict}", flush=True)

    # breakdowns
    breakdown_frames = {}
    for rule in CKPT_RULES:
        rows = []
        for arm in ("A0", "A1", "A4"):
            op = PRIMARY_OP[arm]
            for kind in ("pattern", "n_mod", "cell_count", "colprot", "gap"):
                rows += breakdown_table(provider, ep, blob, rule, arm, op, kind, n_mod, cell_count, max_gap)
        bdf = pd.DataFrame(rows)
        bdf.to_csv(a.out / f"breakdown_{rule}.csv", index=False)
        L = [f"# Step 9 - breakdowns ({rule} checkpoint rule; A0/A1/A4 at their primary operating point)",
             "", "Point estimates on the seed-averaged OOF predictions.", "",
             "| arm | op | breakdown | stratum | n | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) | "
             "PN-BAcc(P(c)) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['arm']} | {r['operating_point']} | {r['breakdown']} | {r['stratum']} | "
                     f"{r['n']} | {_fmt(r['macro_auroc'])} | {_fmt(r['macro_f1'])} | "
                     f"{_fmt(r['balanced_accuracy'])} | {_fmt(r['pn_bacc_uniform'])} | "
                     f"{_fmt(r['pn_bacc_prior'])} |")
        (a.out / f"breakdown_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        breakdown_frames[rule] = bdf
        print(f"[breakdown] wrote breakdown_{rule}.csv", flush=True)

    # stability: per-seed mean +/- SD (point metrics, one seed at a time)
    stability = {}
    for rule in CKPT_RULES:
        arms = ["A0", "A1 (uniform)", "A1 (P(c))", "A2", "A3", "A4 (uniform)", "A4 (P(c))"]
        recs = []
        for key in arms:
            arm = key.split(" ")[0]
            op = PRIMARY_OP[arm]
            if "(" in key:
                op = "uniform" if "uniform" in key else "P(c)"
            d = oof[(oof["arm"] == arm) & (oof["ckpt_rule"] == rule) & (oof["operating_point"] == op)]
            vals = {"macro_auroc": [], "macro_f1": [], "pn_bacc_uniform": [], "balanced_accuracy": []}
            for seed in sorted(d["seed"].unique()):
                ds = d[d["seed"] == seed]
                ids = ds["episode_id"].to_numpy()
                y = ep["y"].loc[ids].to_numpy()
                proba = ds[PROBA_COLS].to_numpy()
                m = natural_metrics(y, proba)
                wu = ds["w_uniform"].to_numpy()
                vals["macro_auroc"].append(float(m["macro_auroc"]))
                vals["macro_f1"].append(float(m["macro_f1"]))
                vals["balanced_accuracy"].append(float(m["balanced_accuracy"]))
                vals["pn_bacc_uniform"].append(float(pn_balanced_accuracy(y, proba.argmax(1), wu)))
            row = {"rule": rule, "arm": key}
            for k, v in vals.items():
                row[f"{k}_mean"] = float(np.mean(v))
                row[f"{k}_sd"] = float(np.std(v, ddof=0))
            recs.append(row)
        sdf = pd.DataFrame(recs)
        sdf.to_csv(a.out / f"stability_per_seed_{rule}.csv", index=False)
        L = [f"# Step 9 - per-seed stability ({rule} checkpoint rule)", "",
             "Pooled OOF metrics per seed, then mean +/- SD over the 3 seeds.", "",
             "| arm | macro-AUROC | macro-F1 | BAcc | PN-BAcc(u) |", "|---|---|---|---|---|"]
        for r in recs:
            L.append(f"| {r['arm']} | {r['macro_auroc_mean']:.4f} +/- {r['macro_auroc_sd']:.4f} | "
                     f"{r['macro_f1_mean']:.4f} +/- {r['macro_f1_sd']:.4f} | "
                     f"{r['balanced_accuracy_mean']:.4f} +/- {r['balanced_accuracy_sd']:.4f} | "
                     f"{r['pn_bacc_uniform_mean']:.4f} +/- {r['pn_bacc_uniform_sd']:.4f} |")
        (a.out / f"stability_per_seed_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        stability[rule] = recs

    # stability: per-fold PN-BAcc and macro-F1 for every arm
    per_fold = {}
    for rule in CKPT_RULES:
        recs = []
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            op = PRIMARY_OP[arm]
            d = oof[(oof["arm"] == arm) & (oof["ckpt_rule"] == rule) & (oof["operating_point"] == op)]
            for (seed, fold), ds in d.groupby(["seed", "fold"]):
                ids = ds["episode_id"].to_numpy()
                y = ep["y"].loc[ids].to_numpy()
                proba = ds[PROBA_COLS].to_numpy()
                m = natural_metrics(y, proba)
                wu = ds["w_uniform"].to_numpy()
                recs.append({"rule": rule, "arm": arm, "seed": int(seed), "fold": int(fold),
                             "macro_f1": float(m["macro_f1"]),
                             "pn_bacc_uniform": float(pn_balanced_accuracy(y, proba.argmax(1), wu)),
                             "n": int(len(ids))})
        pdf = pd.DataFrame(recs)
        pdf.to_csv(a.out / f"stability_per_fold_{rule}.csv", index=False)
        for metric in ("pn_bacc_uniform", "macro_f1"):
            piv = pdf.pivot_table(index=["arm", "seed"], columns="fold", values=metric)
            L = [f"# Step 9 - per-fold {metric} ({rule} checkpoint rule)", "",
                 "| arm | seed | f0 | f1 | f2 | f3 | f4 |", "|---|---|---|---|---|---|---|"]
            for (arm, seed), r in piv.iterrows():
                L.append(f"| {arm} | {seed} | " + " | ".join(f"{r[f]:.4f}" for f in range(5)) + " |")
            (a.out / f"stability_per_fold_{rule}_{metric}.md").write_text("\n".join(L) + "\n",
                                                                          encoding="utf-8")
        per_fold[rule] = pdf

    # best-epoch distribution per arm
    best_epochs = {}
    for rule in CKPT_RULES:
        recs = []
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            op = PRIMARY_OP[arm]
            d = oof[(oof["arm"] == arm) & (oof["ckpt_rule"] == rule) & (oof["operating_point"] == op)]
            be = d.groupby(["seed", "fold"])["best_epoch"].first().to_numpy()
            recs.append({"rule": rule, "arm": arm, "n_runs": int(len(be)),
                         "median": float(np.median(be)), "mean": float(np.mean(be)),
                         "min": int(be.min()), "max": int(be.max()),
                         "n_below_5": int((be < 5).sum()),
                         "flag_median_below_5": bool(np.median(be) < 5)})
        bdf = pd.DataFrame(recs)
        bdf.to_csv(a.out / f"best_epochs_{rule}.csv", index=False)
        L = [f"# Step 9 - best-epoch distribution ({rule} checkpoint rule)", "",
             "| arm | runs | median | mean | min | max | #(<5) | flag (median < 5) |",
             "|---|---|---|---|---|---|---|---|"]
        for r in recs:
            L.append(f"| {r['arm']} | {r['n_runs']} | {r['median']:.1f} | {r['mean']:.2f} | {r['min']} | "
                     f"{r['max']} | {r['n_below_5']} | {'YES' if r['flag_median_below_5'] else 'no'} |")
        (a.out / f"best_epochs_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        best_epochs[rule] = recs

    # runtime and anomalies from the run log
    log_path = a.out / "run_log.jsonl"
    log = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                log.append(json.loads(line))
    # keep only the last record per (arm, seed, fold) if the log has duplicates
    latest = {}
    for rec in log:
        latest[(rec["arm"], rec["seed"], rec["fold"])] = rec
    log = list(latest.values())
    ok = [r for r in log if r.get("status") == "ok"]
    rt_rows = []
    for arm in ("A0", "A1", "A2", "A3", "A4"):
        sub = [r for r in ok if r["arm"] == arm]
        rt_rows.append({"arm": arm, "runs": len(sub), "seconds_total": float(sum(r["seconds"] for r in sub)),
                        "seconds_mean": float(np.mean([r["seconds"] for r in sub])) if sub else float("nan"),
                        "errors": int(sum(1 for r in log if r["arm"] == arm and r.get("status") == "error")),
                        "nan_loss_runs": int(sum(1 for r in sub if r.get("nan_loss")))})
    rt_rows.append({"arm": "TOTAL", "runs": len(ok), "seconds_total": float(sum(r["seconds"] for r in ok)),
                    "seconds_mean": float(np.mean([r["seconds"] for r in ok])) if ok else float("nan"),
                    "errors": int(sum(1 for r in log if r.get("status") == "error")),
                    "nan_loss_runs": int(sum(1 for r in ok if r.get("nan_loss")))})
    rtdf = pd.DataFrame(rt_rows)
    rtdf.to_csv(a.out / "runtime.csv", index=False)
    errs = [r for r in log if r.get("status") == "error"]
    L = ["# Step 9 - runtime and anomalies", "", "| arm | runs | total s | mean s | errors | NaN-loss runs |",
         "|---|---|---|---|---|---|"]
    for r in rt_rows:
        L.append(f"| {r['arm']} | {r['runs']} | {r['seconds_total']:.1f} | {r['seconds_mean']:.1f} | "
                 f"{r['errors']} | {r['nan_loss_runs']} |")
    L += ["", f"Errors: {len(errs)}"]
    if errs:
        for e in errs:
            L.append(f"* {e['arm']} s{e['seed']} f{e['fold']}: {e.get('error')}")
    (a.out / "runtime.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    analysis = {
        "n_bootstrap": N_BOOT, "boot_seed": BOOT_SEED, "ni_margin": NI_MARGIN,
        "uniform_protocol": "seed-averaged probabilities and weights per episode, pooled OOF, one "
                            "subject-cluster bootstrap per arm (Step 6b)",
        "verdicts": verdicts,
        "primary_checkpoint_rule": "primary",
        "best_epochs": {rule: best_epochs[rule] for rule in CKPT_RULES},
        "stability_per_seed": {rule: stability[rule] for rule in CKPT_RULES},
        "runtime": rt_rows,
    }
    (a.out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print("\n=== VERDICTS ===")
    for rule in CKPT_RULES:
        print(f"{rule}: {verdicts[rule]['verdict']}")
    print(f"[wrote analysis under {a.out}]")


if __name__ == "__main__":
    main()
