#!/usr/bin/env python3
"""Step 6 - reference baselines B0, B1, B2 on S1 (3 seeds x 5 folds).

Runs the non-neural baselines over the Step 3 splits using only the Step 1 block tables and the
Step 4 recipe (B1/B2 use the RAW features with NaNs left as NaN, plus the 4 mask bits for B1):

* B0 - multinomial logistic regression on the four availability bits;
* B1 - LightGBM on the concatenation of all four blocks + the 4 mask bits;
* B2 - LightGBM per single block, restricted to episodes where that modality is present.

Writes one out-of-fold prediction file per arm (``episode_id, fold, seed, arm, p_CN, p_MCI,
p_Dementia``) and ``summary.json`` with:

* natural metrics + per-class F1 (mean over seeds, subject-cluster bootstrap CI per seed then averaged);
* the same split by number of observed modalities and by pattern (n >= 30, others pooled);
* PN balanced accuracy (rho uniform = primary, rho = P(c) alongside) with bootstrap CIs;
* B1 top-15 gain importance and wall-clock timings.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.metrics import f1_score

from adni.baselines import (
    MODALITIES,
    build_raw_matrix,
    fit_predict_b0,
    fit_predict_b1,
    fit_predict_b2,
    load_blocks,
    load_feature_spec,
    training_prior_table,
)
from adni.metrics import (
    CLASSES,
    cluster_bootstrap_multi,
    natural_metrics,
    pn_balanced_accuracy,
    pn_weights,
    stratified_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_BLOCKS = PROJECT_ROOT / "results" / "step01" / "stage1" / "blocks"
DEFAULT_SPEC = PROJECT_ROOT / "results" / "step01" / "stage1" / "feature_spec.json"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "b1.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step06"
PROBA_COLS = ["p_CN", "p_MCI", "p_Dementia"]
ARMS = ["b0", "b1"] + [f"b2_{k}" for k in MODALITIES]


# --------------------------------------------------------------------------- helpers
def _load_config(path: Path) -> tuple[dict, int, int]:
    cfg = yaml.safe_load(Path(path).read_text())
    num_rounds = int(cfg.pop("num_rounds", 1000))
    early_stopping = int(cfg.pop("early_stopping", 50))
    return cfg, num_rounds, early_stopping


def _fold_ids(blob: dict, seed: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = blob["s1"][f"seed_{seed}"][str(fold)]
    to_arr = lambda xs: np.asarray(xs, dtype=np.int64)  # noqa: E731
    return (to_arr(node["episode_ids"]["train"]), to_arr(node["episode_ids"]["val"]),
            to_arr(node["episode_ids"]["test"]))


# --- lean metric functions for the bootstrap: identical definitions to metrics.natural_metrics, but
# only the required quantity is computed so a 2,000-resample bootstrap stays tractable.
def _lean_macro_auroc(y: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-vs-rest AUROC via the Mann-Whitney rank identity (equal to sklearn's trapezoidal
    AUROC, ties handled by average ranks)."""
    out: list[float] = []
    for c in range(proba.shape[1]):
        pos = y == c
        n_pos = int(pos.sum())
        n_neg = len(y) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        ranks = rankdata(proba[:, c])
        out.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(out)) if out else float("nan")


def _lean_macro_f1(y: np.ndarray, proba: np.ndarray) -> float:
    return float(f1_score(y, proba.argmax(axis=1), average="macro", labels=[0, 1, 2], zero_division=0))


def _lean_balanced_accuracy(y: np.ndarray, proba: np.ndarray) -> float:
    pred = proba.argmax(axis=1)
    recalls = [float((pred[y == c] == c).mean()) for c in range(3) if (y == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def bootstrap_metric_fns() -> dict:
    """Metric family for ``cluster_bootstrap_multi``. The PN metrics read their weights from columns
    3 / 4 of the augmented matrix ``[p_CN, p_MCI, p_Dementia, w_uniform, w_prior]``."""
    def aug(fn):
        return lambda y, a: fn(y, a[:, :3])

    def pn(col: int):
        return lambda y, a: pn_balanced_accuracy(y, a[:, :3].argmax(axis=1), a[:, col])

    return {
        "macro_auroc": aug(_lean_macro_auroc),
        "macro_f1": aug(_lean_macro_f1),
        "balanced_accuracy": aug(_lean_balanced_accuracy),
        "pn_bacc_uniform": pn(3),
        "pn_bacc_prior": pn(4),
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 6 - non-neural baselines B0/B1/B2 on S1.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--blocks", type=Path, default=DEFAULT_BLOCKS)
    ap.add_argument("--feature-spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--importance-top", type=int, default=15)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    # sklearn's f1 warns when a bootstrap resample contains a class that was never predicted;
    # that is expected here and the metric is still well defined (zero_division=0).
    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
    cfg, num_rounds, early_stopping = _load_config(a.config)

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    spec = load_feature_spec(a.feature_spec)
    blocks = load_blocks(a.blocks)
    X = build_raw_matrix(ep, blocks, spec)
    blob = json.loads(Path(a.splits).read_text())

    rows: dict[str, list[dict]] = defaultdict(list)
    wrows: list[dict] = []
    gain_tot: dict[str, float] = defaultdict(float)
    timing: dict[str, float] = defaultdict(float)
    n_iters: dict[str, list[int]] = defaultdict(list)

    for seed in a.seeds:
        for fold in a.folds:
            tr, va, te = _fold_ids(blob, seed, fold)

            # --- B0
            t0 = time.time()
            p0 = fit_predict_b0(ep, tr, te, seed=seed)
            timing["b0"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b0"].append({"episode_id": int(eid), "fold": fold, "seed": seed, "arm": "b0",
                                   **{c: float(p0[i, j]) for j, c in enumerate(PROBA_COLS)}})

            # --- B1
            t0 = time.time()
            p1, bst = fit_predict_b1(X, ep, tr, va, te, seed=seed, params_override=cfg,
                                     num_rounds=num_rounds, early_stopping=early_stopping)
            timing["b1"] += time.time() - t0
            n_iters["b1"].append(int(bst.num_trees()))
            for name, g in zip(bst.feature_name(), bst.feature_importance("gain")):
                gain_tot[name] += float(g)
            for i, eid in enumerate(te):
                rows["b1"].append({"episode_id": int(eid), "fold": fold, "seed": seed, "arm": "b1",
                                   **{c: float(p1[i, j]) for j, c in enumerate(PROBA_COLS)}})

            # --- B2 (one model per block, present episodes only)
            te_by_arm = {"b0": te, "b1": te}
            for k in MODALITIES:
                Xk = X[[c for c in X.columns if c.startswith(f"{k}_")]]
                present = ep.index[ep[f"m_{k}"].to_numpy() == 1].to_numpy()
                t0 = time.time()
                p2, te2, b2 = fit_predict_b2(Xk, ep, present, tr, va, te, seed=seed,
                                             params_override=cfg, num_rounds=num_rounds,
                                             early_stopping=early_stopping)
                arm = f"b2_{k}"
                timing[arm] += time.time() - t0
                n_iters[arm].append(int(b2.num_trees()))
                for i, eid in enumerate(te2):
                    rows[arm].append({"episode_id": int(eid), "fold": fold, "seed": seed, "arm": arm,
                                      **{c: float(p2[i, j]) for j, c in enumerate(PROBA_COLS)}})
                te_by_arm[arm] = te2

            # --- PN weights for every arm, from this fold's TRAIN-only P(c|m)
            prior = training_prior_table(ep, tr)
            ytr = ep["y"].loc[list(tr)].to_numpy()
            pc = np.bincount(ytr, minlength=3).astype(float)
            pc = pc / pc.sum()
            for arm, tids in te_by_arm.items():
                yy = ep["y"].loc[list(tids)].to_numpy()
                pp = ep["pattern"].loc[list(tids)].to_numpy()
                wu = pn_weights(yy, pp, prior, rho="uniform")
                wp = pn_weights(yy, pp, prior, rho=pc)
                for i, eid in enumerate(tids):
                    wrows.append({"episode_id": int(eid), "seed": seed, "arm": arm,
                                  "w_uniform": float(wu[i]), "w_prior": float(wp[i])})

    # --- write per-arm OOF predictions
    oof: dict[str, pd.DataFrame] = {}
    for arm in ARMS:
        df = pd.DataFrame(rows[arm]).sort_values(["seed", "fold", "episode_id"]).reset_index(drop=True)
        df.to_parquet(a.out / f"oof_{arm}.parquet", index=False)
        oof[arm] = df
    weights = pd.DataFrame(wrows)

    # --- metrics
    summary: dict = {"arms": {}, "config": cfg, "num_rounds": num_rounds,
                     "early_stopping": early_stopping, "n_bootstrap": a.bootstrap,
                     "seeds": list(a.seeds), "folds": list(a.folds)}

    for arm in ARMS:
        df = oof[arm]
        per_seed_rows: dict[int, pd.DataFrame] = {}
        for seed in a.seeds:
            per_seed_rows[seed] = df[df.seed == seed].reset_index(drop=True)

        # natural metrics: per-seed point + CI, then averaged over seeds
        nat: dict[str, dict[str, float]] = defaultdict(lambda: {"point": [], "ci_low": [], "ci_high": []})
        f1_seed: dict[str, list[float]] = defaultdict(list)
        by_nmod: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        by_pat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        pn_acc: dict[str, dict[str, list]] = {
            "uniform": {"point": [], "ci_low": [], "ci_high": []},
            "prior": {"point": [], "ci_low": [], "ci_high": []},
        }

        for seed in a.seeds:
            d = per_seed_rows[seed]
            ids = d.episode_id.to_numpy()
            proba = d[PROBA_COLS].to_numpy()
            y = ep["y"].loc[list(ids)].to_numpy()
            groups = ep["RID"].loc[list(ids)].to_numpy()
            m = natural_metrics(y, proba)
            w = weights[(weights.arm == arm) & (weights.seed == seed)]
            w = d[["episode_id"]].merge(w, on="episode_id", how="left")
            assert w.w_uniform.notna().all() and len(w) == len(d)
            aug = np.column_stack([proba, w.w_uniform.to_numpy(), w.w_prior.to_numpy()])
            ci = cluster_bootstrap_multi(bootstrap_metric_fns(), y, aug, groups,
                                         n=a.bootstrap, seed=seed)
            for kind in ("macro_auroc", "macro_f1", "balanced_accuracy"):
                nat[kind]["point"].append(float(m[kind]))
                nat[kind]["ci_low"].append(ci[kind].ci_low)
                nat[kind]["ci_high"].append(ci[kind].ci_high)
            for c, v in m["f1_per_class"].items():
                f1_seed[CLASSES[c]].append(float(v))
            for tag, key in (("uniform", "pn_bacc_uniform"), ("prior", "pn_bacc_prior")):
                pn_acc[tag]["point"].append(
                    pn_balanced_accuracy(y, proba.argmax(axis=1), aug[:, 3 if tag == "uniform" else 4]))
                pn_acc[tag]["ci_low"].append(ci[key].ci_low)
                pn_acc[tag]["ci_high"].append(ci[key].ci_high)

            # stratified: by number of observed modalities and by pattern (n >= 30)
            nmod = ep["n_mod"].loc[list(ids)].to_numpy()
            sr_n = stratified_report(y, proba, np.array([str(x) for x in nmod]), min_n=30)
            for k, v in sr_n.items():
                for kind in ("macro_auroc", "macro_f1", "balanced_accuracy"):
                    by_nmod[k][kind].append(float(v[kind]))
            pat = ep["pattern"].loc[list(ids)].to_numpy()
            sr_p = stratified_report(y, proba, pat, min_n=30)
            for k, v in sr_p.items():
                for kind in ("macro_auroc", "macro_f1", "balanced_accuracy"):
                    by_pat[k][kind].append(float(v[kind]))

        def _agg(dct):
            return {"point": float(np.mean(dct["point"])),
                    "sd": float(np.std(dct["point"])),
                    "ci_low": float(np.nanmean(dct["ci_low"])),
                    "ci_high": float(np.nanmean(dct["ci_high"]))}

        arm_out = {
            "n_episodes": int(len(df)),
            "natural": {k: _agg(v) for k, v in nat.items()},
            "f1_per_class": {k: float(np.mean(v)) for k, v in f1_seed.items()},
            "pn_bacc_uniform": _agg(pn_acc["uniform"]),
            "pn_bacc_prior": _agg(pn_acc["prior"]),
            "by_modality_count": {
                k: {kind: float(np.mean(vals)) for kind, vals in v.items()}
                for k, v in sorted(by_nmod.items())
            },
            "by_pattern": {
                k: {kind: float(np.mean(vals)) for kind, vals in v.items()}
                for k, v in sorted(by_pat.items())
            },
        }
        summary["arms"][arm] = arm_out

    imp = sorted(gain_tot.items(), key=lambda kv: kv[1], reverse=True)
    summary["importance_b1_top"] = [[name, float(g)] for name, g in imp[: a.importance_top]]
    summary["importance_b1_total_gain"] = float(sum(gain_tot.values()))
    summary["timing_seconds"] = {k: float(v) for k, v in sorted(timing.items())}
    summary["b1_num_trees"] = n_iters["b1"]

    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- tabular CSV exports
    def _write_nat_csv():
        recs = []
        for arm, v in summary["arms"].items():
            for kind, s in v["natural"].items():
                recs.append({"arm": arm, "metric": kind, **s})
            for cls, f1 in v["f1_per_class"].items():
                recs.append({"arm": arm, "metric": f"f1_{cls}", "point": f1,
                             "sd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")})
            recs.append({"arm": arm, "metric": "pn_bacc_uniform",
                         **{k: v for k, v in v["pn_bacc_uniform"].items()}})
            recs.append({"arm": arm, "metric": "pn_bacc_prior", **v["pn_bacc_prior"]})
        pd.DataFrame(recs).to_csv(a.out / "natural_metrics.csv", index=False)

    _write_nat_csv()

    # --- console report
    print(f"\nStep 6 baselines - {len(a.seeds)} seeds x {len(a.folds)} folds  "
          f"(bootstrap n={a.bootstrap})")
    print(f"{'arm':8} {'n_ep':>6} {'macroAUROC':>11} {'macroF1':>8} {'BAcc':>7} {'PN-BAcc':>8} "
          f"{'PN-P(c)':>8} {'sec':>7}")
    for arm in ARMS:
        v = summary["arms"][arm]
        print(f"{arm:8} {v['n_episodes']:6d} {v['natural']['macro_auroc']['point']:11.4f} "
              f"{v['natural']['macro_f1']['point']:8.4f} {v['natural']['balanced_accuracy']['point']:7.4f} "
              f"{v['pn_bacc_uniform']['point']:8.4f} {v['pn_bacc_prior']['point']:8.4f} "
              f"{summary['timing_seconds'].get(arm, float('nan')):7.1f}")
    print("\nB1 top gain importance:")
    for name, g in summary["importance_b1_top"]:
        print(f"  {name:40s} {g:12.1f}")
    print(f"\nB1 trees per fit: {summary['b1_num_trees']}")
    print(f"[wrote {a.out / 'summary.json'}]")


if __name__ == "__main__":
    main()
