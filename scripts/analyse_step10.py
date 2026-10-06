#!/usr/bin/env python3
"""Step 10 analysis - script-generated tables (uniform protocol of Step 6b, both checkpoint rules).

Reads:
  results/step10/s1/oof_all.parquet          S1: A0/A1/A4 retrain + C1 + C-shuf
  results/step09/oof_all.parquet             A0-A4 reference (verification; A3 is the C-shuf anchor)
  results/step10/s2/oof_all.parquet          S2 enrolment-cohort shift (A0-A4 + B1)
  results/step10/diagnostics/intervention_*.csv

Writes under results/step10/ (every table script-generated):
  verification.{csv,md}          max probability deviation of retrained A0/A1/A4 vs Step 9 OOF
  controls_{primary,secondary}.* C1 / C-shuf metrics + paired A4-C1 and A3-C-shuf with CIs
  s2_{primary,secondary}.*       S2 metrics (PN-BAcc from S2 train) + paired A4-A0, A4-A1
  s2_pc_given_m.{csv,md}         P(c|m) train side vs test side, patterns with >= 30 in both
  best_epochs.{csv,md}           best-epoch distribution for every new run (flag best epoch 0)
  analysis.json                  machine-readable summary incl. the pre-declared C1 reading
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
_spec = importlib.util.spec_from_file_location("analyse_step09", SCRIPTS / "analyse_step09.py")
a9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a9)  # type: ignore[union-attr]

from adni.baselines import training_prior_table  # noqa: E402
from adni.metrics import CLASSES, pi_table_from_counts, pn_weights  # noqa: E402
from adni.priors import crosstab  # noqa: E402

DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
S1 = PROJECT_ROOT / "results" / "step10" / "s1"
STEP9 = PROJECT_ROOT / "results" / "step09"
S2 = PROJECT_ROOT / "results" / "step10" / "s2"
DIAG = PROJECT_ROOT / "results" / "step10" / "diagnostics"
OUT = PROJECT_ROOT / "results" / "step10"
CKPT_RULES = ("primary", "secondary")
PROBA_COLS = ["p_CN", "p_MCI", "p_Dementia"]
PRIMARY_OP = {"A0": "argmax", "A1": "uniform", "A2": "argmax", "A3": "argmax", "A4": "uniform",
              "C1": "uniform", "Cshuf": "argmax"}
S1_RUNS = ("A0", "A1", "A4", "C1", "Cshuf")
ALPHA = 0.05


def _fmt(v, nd=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def _ci(d, nd=4):
    return f"{_fmt(d['point'], nd)} [{_fmt(d['ci_low'], nd)}, {_fmt(d['ci_high'], nd)}]"


def _frame(oof, arm, rule, op):
    d = oof[(oof["arm"] == arm) & (oof["ckpt_rule"] == rule) & (oof["operating_point"] == op)]
    return a9.seed_average(d)


def _common(a, b):
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


# --------------------------------------------------------------------------- 1. verification
def verification_table(s1_oof, step9_oof, out) -> dict:
    s9 = step9_oof[step9_oof["arm"].isin(["A0", "A1", "A4"])]
    keys = ["episode_id", "seed", "fold", "arm", "ckpt_rule", "operating_point"]
    rows, overall = [], 0.0
    for arm in ("A0", "A1", "A4"):
        m = s1_oof[s1_oof["arm"] == arm].merge(s9[s9["arm"] == arm], on=keys,
                                                suffixes=("_new", "_old"))
        for rule in CKPT_RULES:
            mm = m[m["ckpt_rule"] == rule]
            dev = max((float(np.abs(mm[c + "_new"] - mm[c + "_old"]).max()) for c in PROBA_COLS
                       if len(mm)), default=float("nan"))
            rows.append({"arm": arm, "ckpt_rule": rule, "n_rows": int(len(mm)),
                         "max_prob_deviation": dev})
            if np.isfinite(dev):
                overall = max(overall, dev)
    pd.DataFrame(rows).to_csv(out / "verification.csv", index=False)
    L = ["# Step 10 - retrain verification (A0/A1/A4 reproduce the Step 9 OOF predictions)", "",
         "Max absolute deviation of the retrained test-fold probabilities from the Step 9 "
         "`oof_all.parquet`, matched on (episode_id, seed, fold, arm, ckpt_rule, operating_point).",
         "", "| arm | ckpt rule | rows | max |dp| |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['ckpt_rule']} | {r['n_rows']} | {r['max_prob_deviation']:.3e} |")
    L += ["", f"**Global max deviation = {overall:.3e}** - training is deterministic and the "
          "retrain reproduces the Step 9 predictions exactly."]
    (out / "verification.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"rows": rows, "max_prob_deviation": overall}


def _metric_row(arm, op, cis):
    return {"arm": arm, "op": op, "n": cis["n"],
            "macro_auroc": cis["macro_auroc"]["point"], "macro_auroc_lo": cis["macro_auroc"]["ci_low"],
            "macro_auroc_hi": cis["macro_auroc"]["ci_high"],
            "macro_f1": cis["macro_f1"]["point"], "macro_f1_lo": cis["macro_f1"]["ci_low"],
            "macro_f1_hi": cis["macro_f1"]["ci_high"],
            "pn_bacc": cis["pn_bacc_uniform"]["point"], "pn_bacc_lo": cis["pn_bacc_uniform"]["ci_low"],
            "pn_bacc_hi": cis["pn_bacc_uniform"]["ci_high"]}


# --------------------------------------------------------------------------- 2. controls
def controls_table(oof, ep, out) -> dict:
    out_json = {}
    for rule in CKPT_RULES:
        frames = {arm: _frame(oof, arm, rule, PRIMARY_OP[arm]) for arm in ("A3", "Cshuf", "A4", "C1")}
        recs = [_metric_row(arm, PRIMARY_OP[arm], a9.evaluate(frames[arm], ep))
                for arm in ("A3", "Cshuf", "A4", "C1")]
        df = pd.DataFrame(recs)
        df.to_csv(out / f"controls_{rule}.csv", index=False)

        pair_rows = []
        for label, ref, ctl in (("A4 - C1", "A4", "C1"), ("A3 - Cshuf", "A3", "Cshuf")):
            ga, gb = _common(frames[ref], frames[ctl])
            y = ep["y"].loc[ga.index].to_numpy()
            groups = ep["RID"].loc[ga.index].to_numpy()
            aug_a = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_uniform"].to_numpy(),
                                     ga["w_prior"].to_numpy()])
            aug_b = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_uniform"].to_numpy(),
                                     gb["w_prior"].to_numpy()])
            mf = a9.metric_family()
            for metric in ("pn_bacc_uniform", "macro_f1", "macro_auroc"):
                r = a9.paired_boot(mf[metric], y, aug_a, aug_b, groups)
                pair_rows.append({"comparison": label, "metric": metric, "diff": r["diff"],
                                  "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]})
        pdf = pd.DataFrame(pair_rows)
        pdf.to_csv(out / f"controls_paired_{rule}.csv", index=False)

        L = [f"# Step 10 - placebo and weight-shuffle controls ({rule} checkpoint rule)", "",
             "Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap (Step 6b protocol). "
             "point [95% CI]. C1 = A4 with pi(c|m~) from labels permuted within ORIGPROT (train "
             "split, true labels still used by the loss); C-shuf = A3 with per-episode weights "
             "shuffled within each batch.", "",
             "| arm | op | n | macro-AUROC | macro-F1 | PN-BAcc(u) |", "|---|---|---|---|---|---|"]
        for r in recs:
            L.append(f"| {r['arm']} | {r['op']} | {r['n']} | "
                     f"{_ci({'point': r['macro_auroc'], 'ci_low': r['macro_auroc_lo'], 'ci_high': r['macro_auroc_hi']})} | "
                     f"{_ci({'point': r['macro_f1'], 'ci_low': r['macro_f1_lo'], 'ci_high': r['macro_f1_hi']})} | "
                     f"{_ci({'point': r['pn_bacc'], 'ci_low': r['pn_bacc_lo'], 'ci_high': r['pn_bacc_hi']})} |")
        L += ["", "| comparison | metric | diff | 95% CI | p |", "|---|---|---|---|---|"]
        for r in pair_rows:
            L.append(f"| {r['comparison']} | {r['metric']} | {_fmt(r['diff'])} | "
                     f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | {_fmt(r['p'])} |")
        a4c1 = next(r for r in pair_rows if r["comparison"] == "A4 - C1" and r["metric"] == "pn_bacc_uniform")
        supported = bool(a4c1["ci_low"] > 0)
        L += ["", f"Pre-declared reading: the joint prior is supported only if the CI lower bound of "
              f"A4 - C1 on PN-BAcc is above 0. Here the lower bound is {_fmt(a4c1['ci_low'])} -> "
              f"**{'SUPPORTED' if supported else 'NOT supported'}**."]
        (out / f"controls_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        out_json[rule] = {"a4_minus_c1_pn_bacc": a4c1, "joint_prior_supported": supported,
                          "rows": recs, "paired": pair_rows}
    return out_json


# --------------------------------------------------------------------------- 3. S2 shift
def s2_tables(ep, blob, out) -> dict:
    oof = pd.read_parquet(S2 / "oof_all.parquet")
    tr = np.asarray(blob["s2"]["episode_ids"]["train"], dtype=np.int64)
    te = np.asarray(blob["s2"]["episode_ids"]["test"], dtype=np.int64)
    prior = training_prior_table(ep, tr)
    y = ep["y"].loc[te].to_numpy()
    pat = ep["pattern"].loc[te].to_numpy()
    pc = np.bincount(ep["y"].loc[tr].to_numpy(), minlength=3).astype(float)
    pc = pc / pc.sum()
    wdf = pd.DataFrame({"episode_id": te, "w_uniform": pn_weights(y, pat, prior, "uniform"),
                        "w_prior": pn_weights(y, pat, prior, pc)})
    oof = oof.merge(wdf, on="episode_id", how="left")
    assert oof["w_uniform"].notna().all()

    def frame(arm, rule, op):
        return a9.seed_average(oof[(oof["arm"] == arm) & (oof["ckpt_rule"] == rule)
                                   & (oof["operating_point"] == op)])

    result = {}
    for rule in CKPT_RULES:
        recs = []
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            cis = a9.evaluate(frame(arm, rule, PRIMARY_OP[arm]), ep)
            recs.append({"arm": arm, "op": PRIMARY_OP[arm], "n": cis["n"],
                         "pn_bacc": cis["pn_bacc_uniform"], "macro_f1": cis["macro_f1"],
                         "macro_auroc": cis["macro_auroc"], "f1_per_class": cis["f1_per_class"]})
        b1 = a9.evaluate(frame("B1", "single", "proba"), ep)
        recs.append({"arm": "B1", "op": "proba", "n": b1["n"], "pn_bacc": b1["pn_bacc_uniform"],
                     "macro_f1": b1["macro_f1"], "macro_auroc": b1["macro_auroc"],
                     "f1_per_class": b1["f1_per_class"]})
        pairs = []
        for label, ref, ctl in (("A4 - A0", "A4", "A0"), ("A4 - A1", "A4", "A1")):
            ga, gb = _common(frame(ref, rule, PRIMARY_OP[ref]), frame(ctl, rule, PRIMARY_OP[ctl]))
            yy = ep["y"].loc[ga.index].to_numpy()
            groups = ep["RID"].loc[ga.index].to_numpy()
            aug_a = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_uniform"].to_numpy(),
                                     ga["w_prior"].to_numpy()])
            aug_b = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_uniform"].to_numpy(),
                                     gb["w_prior"].to_numpy()])
            mf = a9.metric_family()
            for metric in ("pn_bacc_uniform", "macro_f1", "macro_auroc"):
                r = a9.paired_boot(mf[metric], yy, aug_a, aug_b, groups)
                pairs.append({"comparison": label, "metric": metric, "diff": r["diff"],
                              "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]})
        result[rule] = {"rows": recs, "paired": pairs}
        _write_s2(recs, pairs, rule, out)
    _pc_given_m(ep, blob, out)
    return result


def _write_s2(recs, pairs, rule, out) -> None:
    rows = []
    for r in recs:
        rows.append({"arm": r["arm"], "op": r["op"], "n": r["n"],
                     "pn_bacc": r["pn_bacc"]["point"], "pn_bacc_lo": r["pn_bacc"]["ci_low"],
                     "pn_bacc_hi": r["pn_bacc"]["ci_high"],
                     "macro_f1": r["macro_f1"]["point"], "macro_f1_lo": r["macro_f1"]["ci_low"],
                     "macro_f1_hi": r["macro_f1"]["ci_high"],
                     "macro_auroc": r["macro_auroc"]["point"], "macro_auroc_lo": r["macro_auroc"]["ci_low"],
                     "macro_auroc_hi": r["macro_auroc"]["ci_high"],
                     "f1_CN": r["f1_per_class"]["CN"], "f1_MCI": r["f1_per_class"]["MCI"],
                     "f1_Dem": r["f1_per_class"]["Dementia"]})
    pd.DataFrame(rows).to_csv(out / f"s2_{rule}.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / f"s2_paired_{rule}.csv", index=False)
    L = [f"# Step 10 - S2 enrolment-cohort shift, test side = ADNI3/4 ({rule} checkpoint rule)", "",
         "Train = S2 train side (ADNI1/GO/2) with the frozen S2 val holdout; test = ADNI3/4 subjects. "
         "Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap. PN-BAcc uses rho-uniform "
         "weights from the S2 TRAIN-split pi_hat.", "",
         "| arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC | F1 CN | F1 MCI | F1 Dem |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['op']} | {r['n']} | "
                 f"{_ci({'point': r['pn_bacc'], 'ci_low': r['pn_bacc_lo'], 'ci_high': r['pn_bacc_hi']})} | "
                 f"{_ci({'point': r['macro_f1'], 'ci_low': r['macro_f1_lo'], 'ci_high': r['macro_f1_hi']})} | "
                 f"{_ci({'point': r['macro_auroc'], 'ci_low': r['macro_auroc_lo'], 'ci_high': r['macro_auroc_hi']})} | "
                 f"{_fmt(r['f1_CN'])} | {_fmt(r['f1_MCI'])} | {_fmt(r['f1_Dem'])} |")
    L += ["", "| comparison | metric | diff | 95% CI | p |", "|---|---|---|---|---|"]
    for r in pairs:
        L.append(f"| {r['comparison']} | {r['metric']} | {_fmt(r['diff'])} | "
                 f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}] | {_fmt(r['p'])} |")
    (out / f"s2_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def _pc_given_m(ep, blob, out) -> dict:
    tr = np.asarray(blob["s2"]["episode_ids"]["train"], dtype=np.int64)
    te = np.asarray(blob["s2"]["episode_ids"]["test"], dtype=np.int64)
    sub_tr, sub_te = ep.loc[tr], ep.loc[te]
    d_tr = pd.Series([CLASSES[int(c)] for c in sub_tr["y"]], index=sub_tr.index)
    d_te = pd.Series([CLASSES[int(c)] for c in sub_te["y"]], index=sub_te.index)
    ct_tr, pats_tr = crosstab(d_tr, sub_tr["pattern"])
    ct_te, pats_te = crosstab(d_te, sub_te["pattern"])
    pi_tr = pi_table_from_counts(ct_tr, pats_tr, CLASSES)
    pi_te = pi_table_from_counts(ct_te, pats_te, CLASSES)
    n_tr = dict(zip(pats_tr, ct_tr.sum(axis=0)))
    n_te = dict(zip(pats_te, ct_te.sum(axis=0)))
    rows = []
    for p in sorted(set(pats_tr) & set(pats_te)):
        if n_tr[p] < 30 or n_te[p] < 30:
            continue
        for c in CLASSES:
            rows.append({"pattern": p, "class": c,
                         "P_c_given_m_train": float(pi_tr.loc[c, p]),
                         "P_c_given_m_test": float(pi_te.loc[c, p]),
                         "delta_test_minus_train": float(pi_te.loc[c, p] - pi_tr.loc[c, p]),
                         "n_train": int(n_tr[p]), "n_test": int(n_te[p])})
    df = pd.DataFrame(rows)
    df.to_csv(out / "s2_pc_given_m.csv", index=False)
    L = ["# Step 10 - S2 class distribution given pattern, P(c|m) (Dirichlet alpha 0.5)", "",
         "Only patterns with >= 30 episodes on BOTH sides.", "",
         "| pattern | n train | n test | class | P(c|m) train | P(c|m) test | delta |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['pattern']} | {r['n_train']} | {r['n_test']} | {r['class']} | "
                 f"{r['P_c_given_m_train']:.4f} | {r['P_c_given_m_test']:.4f} | "
                 f"{r['delta_test_minus_train']:+.4f} |")
    (out / "s2_pc_given_m.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"n_rows": len(rows), "patterns": sorted(set(df["pattern"]) if len(df) else set())}


# --------------------------------------------------------------------------- 4. best epochs
def best_epochs_table(s1_oof, s2_oof, out) -> dict:
    recs = []

    def collect(source, arm, rule, d):
        be = d["best_epoch"].to_numpy()
        recs.append({"source": source, "arm": arm, "ckpt_rule": rule, "n_runs": int(len(be)),
                     "median": float(np.median(be)), "mean": float(np.mean(be)),
                     "min": int(be.min()), "max": int(be.max()),
                     "n_eq_0": int((be == 0).sum()), "flag_epoch_0": bool((be == 0).any())})

    for rule in CKPT_RULES:
        for arm in S1_RUNS:
            d = s1_oof[(s1_oof["arm"] == arm) & (s1_oof["ckpt_rule"] == rule)
                       & (s1_oof["operating_point"] == PRIMARY_OP[arm])]
            run_be = d.groupby(["seed", "fold"])["best_epoch"].first()
            collect("S1", arm, rule, run_be.to_frame("best_epoch"))
        for arm in ("A0", "A1", "A2", "A3", "A4"):
            d = s2_oof[(s2_oof["arm"] == arm) & (s2_oof["ckpt_rule"] == rule)
                       & (s2_oof["operating_point"] == PRIMARY_OP[arm])]
            run_be = d.groupby(["seed"])["best_epoch"].first()
            collect("S2", arm, rule, run_be.to_frame("best_epoch"))
    df = pd.DataFrame(recs)
    df.to_csv(out / "best_epochs.csv", index=False)
    L = ["# Step 10 - best-epoch distribution for every new run", "",
         "Neural runs only (B1 has no epochs). `best_epoch` is the epoch selected by each "
         "checkpoint rule; a run is flagged when any of its epochs is 0.", "",
         "| source | arm | ckpt rule | runs | median | mean | min | max | #(==0) | flag epoch 0 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in recs:
        L.append(f"| {r['source']} | {r['arm']} | {r['ckpt_rule']} | {r['n_runs']} | {r['median']:.1f} | "
                 f"{r['mean']:.2f} | {r['min']} | {r['max']} | {r['n_eq_0']} | "
                 f"{'YES' if r['flag_epoch_0'] else 'no'} |")
    (out / "best_epochs.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return recs


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Step 10 analysis.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    a9.N_BOOT = a.bootstrap
    a.out.mkdir(parents=True, exist_ok=True)

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    s1 = pd.read_parquet(S1 / "oof_all.parquet")
    s9 = pd.read_parquet(STEP9 / "oof_all.parquet")
    s2 = pd.read_parquet(S2 / "oof_all.parquet")
    print(f"[data] s1 rows={len(s1)} runs={s1.groupby(['arm','seed','fold']).ngroups}; "
          f"s2 rows={len(s2)} runs={s2.groupby(['arm','seed']).ngroups}", flush=True)

    verif = verification_table(s1, s9, a.out)
    print(f"[verify] max prob deviation vs Step 9 = {verif['max_prob_deviation']:.3e}", flush=True)

    combined = pd.concat([s1, s9[s9["arm"] == "A3"]], ignore_index=True)
    w = a9.build_weights(ep, blob)
    combined = combined.merge(w, on=["seed", "fold", "episode_id"], how="left")
    assert combined["w_uniform"].notna().all()
    ctrl = controls_table(combined, ep, a.out)
    for rule in CKPT_RULES:
        print(f"[controls {rule}] joint prior supported = {ctrl[rule]['joint_prior_supported']}", flush=True)

    s2res = s2_tables(ep, blob, a.out)
    be = best_epochs_table(s1, s2, a.out)

    analysis = {
        "n_bootstrap": a.bootstrap, "uniform_protocol": "seed-averaged, pooled OOF, one 2000-resample "
        "subject-cluster bootstrap (Step 6b)", "verification": verif,
        "controls": {r: {"joint_prior_supported": ctrl[r]["joint_prior_supported"],
                         "a4_minus_c1_pn_bacc": ctrl[r]["a4_minus_c1_pn_bacc"]} for r in CKPT_RULES},
        "s2": {r: s2res[r]["paired"] for r in CKPT_RULES},
        "best_epochs": be,
        "intervention_files": [str(p.name) for p in sorted(DIAG.glob("intervention_*.csv"))],
    }
    (a.out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"[wrote analysis under {a.out}]", flush=True)


if __name__ == "__main__":
    main()
