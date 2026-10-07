#!/usr/bin/env python3
"""Step 11b analysis (EXPLORATORY labels throughout section B).

Reads (all produced by the Step 11b runners; nothing is retrained here):

  results/step11b/s2/oof_all.parquet        confirmatory S2 runs (5 run-sets x 3 seeds)
  results/step10/s2/oof_all.parquet         S2 comparators A0/A1/A4 (Step 10, reused as-is)
  results/step11b/s1/oof_all.parquet        A5-A8 exploratory OOF (R2, S1 grid)
  results/step11a/oof_all.parquet           comparator R2_A4 (Step 11a, reused as-is)
  results/step11b/s1/swap_all.parquet       embedding-swap test rows
  results/step11b/s1/intervention_raw.csv   mask-intervention diagnostic rows

Writes under results/step11b/:

  confirmatory_{primary,secondary}.{csv,md}     S2 metrics + paired differences + verdict
  exploratory_{primary,secondary}.{csv,md}      A5-A8 + R2_A4 metrics with CIs
  exploratory_paired_{primary,secondary}.{csv,md}  paired vs A4(R2) + Holm over 4 PN-BAcc rows
  best_epochs.{csv,md}                          per arm x rule distribution + flags
  embedding_swap.{csv,md} + swap.json           swap TV by arm/rule (+ same-pattern share)
  intervention.{csv,md}                         remove-vs-swap ratio per arm/modality
  analysis.json                                 verdicts, counts, flags, EXPLORATORY notice
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyse_step09 as a9  # noqa: E402  (scripts/ on sys.path when run from scripts/)
import analyse_step10 as a10  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "step11b"
CKPT_RULES = a10.CKPT_RULES
PROBA_COLS = a10.PROBA_COLS
EXPLORATORY_ARMS = ("A5", "A6", "A7", "A8")
CONFIRMED = "confirmed"
NOT_CONFIRMED = "not confirmed"
EXPLORATORY_NOTICE = ("EXPLORATORY: no selection among A5-A8; these results must not be used to "
                      "change A0-A4, the recipes, or any pre-declared rule.")


def _load_oof(path: Path, require_cols=("episode_id", "seed", "fold", "runset", "arm",
                                        "ckpt_rule", "operating_point")) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in require_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}; rerun the Step 11b runners")
    return df


def _s2_weights(ep, blob):
    tr = np.asarray(blob["s2"]["episode_ids"]["train"], dtype=np.int64)
    te = np.asarray(blob["s2"]["episode_ids"]["test"], dtype=np.int64)
    prior = a10.training_prior_table(ep, tr)
    y = ep["y"].loc[te].to_numpy()
    pat = ep["pattern"].loc[te].to_numpy()
    pc = np.bincount(ep["y"].loc[tr].to_numpy(), minlength=3).astype(float)
    pc = pc / pc.sum()
    wdf = pd.DataFrame({"episode_id": te, "w_uniform": pn_weights_u(y, pat, prior),
                        "w_prior": pn_weights_p(y, pat, prior, pc)})
    return wdf


def pn_weights_u(y, pat, prior):
    return a10.pn_weights(y, pat, prior, "uniform")


def pn_weights_p(y, pat, prior, pc):
    return a10.pn_weights(y, pat, prior, pc)


def _frame_s2(oof, runset_arm: str, rule: str, op: str):
    """Seed-average an S2 frame for either a step11b run-set (by runset) or a Step 10 arm."""
    d = oof[oof["ckpt_rule"] == rule]
    d = d[d["operating_point"] == op]
    d = d[d["runset"] == runset_arm] if "runset" in d.columns else d[d["arm"] == runset_arm]
    return a9.seed_average(d)


def _eval(g, ep):
    return a9.evaluate(g, ep)


def _row(label, op, cis):
    return {"label": label, "op": op, "n": cis["n"],
            "pn_bacc": cis["pn_bacc_uniform"]["point"],
            "pn_bacc_lo": cis["pn_bacc_uniform"]["ci_low"],
            "pn_bacc_hi": cis["pn_bacc_uniform"]["ci_high"],
            "macro_f1": cis["macro_f1"]["point"], "macro_f1_lo": cis["macro_f1"]["ci_low"],
            "macro_f1_hi": cis["macro_f1"]["ci_high"],
            "macro_auroc": cis["macro_auroc"]["point"],
            "macro_auroc_lo": cis["macro_auroc"]["ci_low"],
            "macro_auroc_hi": cis["macro_auroc"]["ci_high"]}


def _paired(label: str, metric: str, ga, gb, ep) -> dict:
    yy = ep["y"].loc[ga.index].to_numpy()
    groups = ep["RID"].loc[ga.index].to_numpy()
    aug_a = np.column_stack([ga[PROBA_COLS].to_numpy(), ga["w_uniform"].to_numpy(),
                             ga["w_prior"].to_numpy()])
    aug_b = np.column_stack([gb[PROBA_COLS].to_numpy(), gb["w_uniform"].to_numpy(),
                             gb["w_prior"].to_numpy()])
    r = a9.paired_boot(a9.metric_family()[metric], yy, aug_a, aug_b, groups)
    return {"comparison": label, "metric": metric, "diff": r["diff"], "ci_low": r["ci_low"],
            "ci_high": r["ci_high"], "p": r["p"]}


def _write_tables(rows, pairs, rule, out, verdict: dict, title: str) -> None:
    pd.DataFrame(rows).to_csv(out / f"confirmatory_{rule}.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / f"confirmatory_paired_{rule}.csv", index=False)
    L = [f"# Step 11b (A) - confirmatory S2 check, {title} ({rule} checkpoint rule)", "",
         "Train = S2 train side (ADNI1/GO/2) with the frozen S2 val holdout; test = ADNI3/4 "
         "subjects. Seed-averaged OOF, one 2,000-resample subject-cluster bootstrap. PN-BAcc "
         "weights from the S2 TRAIN-split pi_hat (rho uniform). Step 10 comparators reused "
         "as-is; tau = 0.5 was chosen from S1 Step 11a results, so this is the held-out check.",
         ""]
    hdr = ("| run-set | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |")
    L += [hdr, "|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['label']} | {r['op']} | {r['n']} | {r['pn_bacc']:.4f} "
                 f"[{r['pn_bacc_lo']:.4f}, {r['pn_bacc_hi']:.4f}] | "
                 f"{r['macro_f1']:.4f} [{r['macro_f1_lo']:.4f}, {r['macro_f1_hi']:.4f}] | "
                 f"{r['macro_auroc']:.4f} [{r['macro_auroc_lo']:.4f}, {r['macro_auroc_hi']:.4f}] |")
    L += ["", "| comparison | metric | diff | 95% CI | p |", "|---|---|---|---|---|"]
    for p in pairs:
        L.append(f"| {p['comparison']} | {p['metric']} | {p['diff']:.4f} "
                 f"[{p['ci_low']:.4f}, {p['ci_high']:.4f}] | {p['p']:.4f} |")
    L += ["", f"**Pre-declared reading ({rule}):** PN-BAcc(A4 tau 0.5 - A0) CI low = "
              f"{verdict['pn_low']:.4f} (> 0: {verdict['pn_ok']}); macro-F1 CI low = "
              f"{verdict['f1_low']:.4f} (> -0.02: {verdict['f1_ok']}) -> **{verdict['verdict']}**.",
          ""]
    (out / f"confirmatory_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")

def confirmatory_section(ep, blob, out) -> dict:
    """Part A: S2 metrics for the five run-sets + Step 10 comparators, paired diffs, verdict."""
    new = _load_oof(OUT / "s2" / "oof_all.parquet")
    old = pd.read_parquet(PROJECT_ROOT / "results" / "step10" / "s2" / "oof_all.parquet")
    wdf = _s2_weights(ep, blob)
    new = new.merge(wdf, on="episode_id", how="left")
    old = old.merge(wdf, on="episode_id", how="left")
    assert new["w_uniform"].notna().all() and old["w_uniform"].notna().all()

    # labels: (source, key, op) -> display label
    spec = {("new", "S2R1_A4_tau0p5"): "A4(tau 0.5) [R1]",
            ("new", "S2R2_A0"): "A0 [R2]",
            ("new", "S2R2_A1"): "A1 [R2]",
            ("new", "S2R2_A4"): "A4(tau 1) [R2]",
            ("new", "S2R2_A4_tau0p5"): "A4(tau 0.5) [R2]",
            ("old", "A0"): "A0 [R1, Step 10]",
            ("old", "A1"): "A1 [R1, Step 10]",
            ("old", "A4"): "A4(tau 1) [R1, Step 10]"}
    OPS = {"A4(tau 0.5) [R1]": "uniform", "A4(tau 0.5) [R2]": "uniform", "A4(tau 1) [R2]": "uniform",
           "A4(tau 1) [R1, Step 10]": "uniform", "A1 [R2]": "uniform", "A1 [R1, Step 10]": "uniform",
           "A0 [R2]": "argmax", "A0 [R1, Step 10]": "argmax"}
    result = {}
    for rule in CKPT_RULES:
        cis = {}
        for (src, key), label in spec.items():
            oof = new if src == "new" else old
            g = _frame_s2(oof, key, rule, OPS[label])
            cis[label] = _eval(g, ep)
        rows = [_row(label, OPS[label], cis[label]) for label in spec.values()]
        pairs = []
        # R1: A4(tau 0.5) [R1] vs A0 [R1, Step 10]; A1 [R1, Step 10]; A4(tau 1) [R1, Step 10]
        r1_new = _frame_s2(new, "S2R1_A4_tau0p5", rule, "uniform")
        r2_new = _frame_s2(new, "S2R2_A4_tau0p5", rule, "uniform")
        combos = [
            ("A4(tau 0.5) - A0 [R1]", r1_new, _frame_s2(old, "A0", rule, "argmax")),
            ("A4(tau 0.5) - A1 [R1]", r1_new, _frame_s2(old, "A1", rule, "uniform")),
            ("A4(tau 0.5) - A4(tau 1) [R1]", r1_new, _frame_s2(old, "A4", rule, "uniform")),
            ("A4(tau 0.5) - A0 [R2]", r2_new, _frame_s2(new, "S2R2_A0", rule, "argmax")),
            ("A4(tau 0.5) - A1 [R2]", r2_new, _frame_s2(new, "S2R2_A1", rule, "uniform")),
            ("A4(tau 0.5) - A4(tau 1) [R2]", r2_new, _frame_s2(new, "S2R2_A4", rule, "uniform")),
        ]
        for label, ga, gb in combos:
            idx = ga.index.intersection(gb.index)
            for metric in ("pn_bacc_uniform", "macro_f1", "macro_auroc"):
                pairs.append(_paired(label, metric, ga.loc[idx], gb.loc[idx], ep))
        # the pre-declared rule, under each recipe
        def verdict_for(pn_pair, f1_pair):
            pn = next(p for p in pairs if p["comparison"] == pn_pair and p["metric"] == "pn_bacc_uniform")
            f1 = next(p for p in pairs if p["comparison"] == pn_pair and p["metric"] == "macro_f1")
            pn_ok = bool(pn["ci_low"] > 0.0)
            f1_ok = bool(f1["ci_low"] > -0.02)
            return {"comparison": pn_pair, "pn_low": pn["ci_low"], "pn_ok": pn_ok,
                    "f1_low": f1["ci_low"], "f1_ok": f1_ok,
                    "verdict": CONFIRMED if (pn_ok and f1_ok) else NOT_CONFIRMED}
        verdicts = {"R1": verdict_for("A4(tau 0.5) - A0 [R1]", "macro_f1"),
                    "R2": verdict_for("A4(tau 0.5) - A0 [R2]", "macro_f1")}
        _write_tables(rows, pairs, rule, out, verdicts["R1"],
                      "recipe 1 (tau 0.5 pre-declared rule applied under R1; R2 replication below)")
        # R2 verdict into the same md: append a short block
        v2 = verdicts["R2"]
        with (out / f"confirmatory_{rule}.md").open("a", encoding="utf-8") as fh:
            fh.write(f"**R2 replication ({rule}):** PN-BAcc(A4 tau 0.5 [R2] - A0 [R2]) CI low = "
                     f"{v2['pn_low']:.4f} (> 0: {v2['pn_ok']}); macro-F1 CI low = "
                     f"{v2['f1_low']:.4f} (> -0.02: {v2['f1_ok']}) -> **{v2['verdict']}**.\n")
        result[rule] = {"rows": rows, "paired": pairs, "verdicts": verdicts}
    return result


def exploratory_section(ep, blob, out) -> dict:
    """Part B: A5-A8 metrics, paired vs A4(R2), best epochs, swap + intervention diagnostics."""
    s11 = _load_oof(OUT / "s1" / "oof_all.parquet")
    s11a = _load_oof(PROJECT_ROOT / "results" / "step11a" / "oof_all.parquet",
                     require_cols=("episode_id", "seed", "fold", "runset", "arm", "ckpt_rule",
                                   "operating_point"))
    weights = {}
    for seed in (0, 1, 2):
        for fold in range(5):
            node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
            tr = np.asarray(node["train"], dtype=np.int64)
            te = np.asarray(node["test"], dtype=np.int64)
            prior = a10.training_prior_table(ep, tr)
            y = ep["y"].loc[te].to_numpy()
            pat = ep["pattern"].loc[te].to_numpy()
            weights[(seed, fold)] = pd.DataFrame({
                "episode_id": te, "w_uniform": pn_weights_u(y, pat, prior),
                "w_prior": pn_weights_p(y, pat, prior,
                                        np.bincount(ep["y"].loc[tr].to_numpy(), minlength=3)
                                        .astype(float) / len(tr))})
    def attach(oof):
        parts = []
        for (seed, fold), w in weights.items():
            d = oof[(oof["seed"] == seed) & (oof["fold"] == fold)]
            parts.append(d.merge(w, on="episode_id", how="left"))
        df = pd.concat(parts, ignore_index=True)
        assert df["w_uniform"].notna().all()
        return df

    s11 = attach(s11)
    s11a = attach(s11a[s11a["runset"] == "R2_A4"])
    result = {}
    for rule in CKPT_RULES:
        def frame(df, rs):
            d = df[(df["runset"] == rs) & (df["ckpt_rule"] == rule)
                   & (df["operating_point"] == "uniform")]
            return a9.seed_average(d)
        labels = [("R2_A4 (comparator)", frame(s11a, "R2_A4"))] + [
            (f"{arm} (EXPLORATORY)", frame(s11, arm)) for arm in EXPLORATORY_ARMS]
        rows, cis = [], {}
        for label, g in labels:
            cis[label] = _eval(g, ep)
            rows.append(_row(label, "uniform", cis[label]))
        comp = frame(s11a, "R2_A4")
        pairs = []
        for arm in EXPLORATORY_ARMS:
            ga = frame(s11, arm)
            idx = ga.index.intersection(comp.index)
            for metric in ("pn_bacc_uniform", "macro_f1", "macro_auroc"):
                pairs.append(_paired(f"{arm} - A4(R2)", metric, ga.loc[idx], comp.loc[idx], ep))
        pn = [p["p"] for p in pairs if p["metric"] == "pn_bacc_uniform"]
        holm_p = a9.holm(pn)
        for p, h in zip([p for p in pairs if p["metric"] == "pn_bacc_uniform"], holm_p):
            p["holm_p"] = h
        pd.DataFrame(rows).to_csv(out / f"exploratory_{rule}.csv", index=False)
        pd.DataFrame(pairs).to_csv(out / f"exploratory_paired_{rule}.csv", index=False)
        L = [f"# Step 11b (B) - exploratory arms under R2, S1 grid ({rule} checkpoint rule)", "",
             EXPLORATORY_NOTICE, "",
             "Seed-averaged OOF, pooled, one 2,000-resample subject-cluster bootstrap. A5-A8 are "
             "trained with the R2 recipe on the S1 grid; the comparator is A4(R2, tau 1) from "
             "Step 11a (reused, not retrained). Holm over the four PN-BAcc comparisons.", "",
             "| arm | op | n | PN-BAcc(u) | macro-F1 | macro-AUROC |", "|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['label']} | {r['op']} | {r['n']} | {r['pn_bacc']:.4f} "
                     f"[{r['pn_bacc_lo']:.4f}, {r['pn_bacc_hi']:.4f}] | {r['macro_f1']:.4f} "
                     f"[{r['macro_f1_lo']:.4f}, {r['macro_f1_hi']:.4f}] | {r['macro_auroc']:.4f} "
                     f"[{r['macro_auroc_lo']:.4f}, {r['macro_auroc_hi']:.4f}] |")
        L += ["", "| comparison | metric | diff | 95% CI | p | holm p |", "|---|---|---|---|---|---|"]
        for p in pairs:
            hp = p.get("holm_p")
            L.append(f"| {p['comparison']} | {p['metric']} | {p['diff']:.4f} "
                     f"[{p['ci_low']:.4f}, {p['ci_high']:.4f}] | {p['p']:.4f} | "
                     + ("-" if hp is None else f"{hp:.4f}") + " |")
        (out / f"exploratory_{rule}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        result[rule] = {"rows": rows, "paired": pairs}
    _best_epochs(out, s11, s11a)
    _swap_diagnostics(out)
    _intervention_tables(out)
    return result


def _best_epochs(out, s11, s11a) -> pd.DataFrame:
    rows = []
    for df, tag in ((s11a, "R2_A4 (Step 11a comparator)"), (s11, None)):
        arms = ["R2_A4"] if tag else EXPLORATORY_ARMS
        for arm in arms:
            for rule in CKPT_RULES:
                d = df[(df["runset"] == arm) & (df["ckpt_rule"] == rule)
                       & (df["operating_point"] == "uniform")]
                be = d.groupby(["seed", "fold"])["best_epoch"].first()
                rows.append({"arm": arm, "ckpt_rule": rule, "runs": int(be.size),
                             "median": float(be.median()), "mean": float(be.mean()),
                             "min": int(be.min()), "max": int(be.max()),
                             "n_below_5": int((be < 5).sum()),
                             "flag_median_below_5": bool(be.median() < 5),
                             "n_eq_0": int((be == 0).sum()), "flag_epoch_0": bool((be == 0).any())})
    df = pd.DataFrame(rows)
    df.to_csv(out / "best_epochs.csv", index=False)
    L = ["# Step 11b - best-epoch distributions (flag median < 5 and any epoch-0 selection)", "",
         EXPLORATORY_NOTICE, "",
         "| arm | rule | runs | median | mean | min | max | #(<5) | flag median<5 | #(==0) | flag epoch 0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        L.append(f"| {r['arm']} | {r['ckpt_rule']} | {r['runs']} | {r['median']:.1f} | "
                 f"{r['mean']:.2f} | {r['min']} | {r['max']} | {r['n_below_5']} | "
                 f"{'YES' if r['flag_median_below_5'] else 'no'} | {r['n_eq_0']} | "
                 f"{'YES' if r['flag_epoch_0'] else 'no'} |")
    (out / "best_epochs.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return df


def _swap_diagnostics(out) -> dict:
    df = pd.read_parquet(out / "s1" / "swap_all.parquet")
    rows = []
    for (arm, rule), g in df.groupby(["arm", "ckpt_rule"]):
        same = g["pattern_self"] == g["pattern_donor"]
        rows.append({"arm": arm, "ckpt_rule": rule, "n": int(len(g)),
                     "mean_tv": float(g["tv"].mean()), "median_tv": float(g["tv"].median()),
                     "p95_tv": float(g["tv"].quantile(0.95)),
                     "same_pattern_share": float(same.mean()),
                     "mean_tv_diff_pattern": float(g.loc[~same, "tv"].mean())
                     if (~same).any() else float("nan")})
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out / "embedding_swap.csv", index=False)
    L = ["# Step 11b - embedding-swap test for A5-A8 (EXPLORATORY)", "",
         EXPLORATORY_NOTICE, "",
         "At test time every episode's pattern embedding is replaced by that of a random OTHER test "
         "episode (class-agnostic). TV = total-variation change of the predicted probabilities at "
         "the rho-uniform operating point. A large TV on different-pattern swaps means the model "
         "relies on the availability pattern beyond the content.", "",
         "| arm | rule | n | mean TV | median TV | 95th pct TV | same-pattern share | mean TV (diff pattern) |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in out_df.iterrows():
        L.append(f"| {r['arm']} | {r['ckpt_rule']} | {r['n']} | {r['mean_tv']:.4f} | "
                 f"{r['median_tv']:.4f} | {r['p95_tv']:.4f} | {r['same_pattern_share']:.3f} | "
                 f"{r['mean_tv_diff_pattern']:.4f} |")
    (out / "embedding_swap.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return {"rows": json.loads(out_df.to_json(orient="records"))}


def _intervention_tables(out) -> pd.DataFrame:
    df = pd.read_csv(out / "s1" / "intervention_raw.csv")
    rows = []
    for (arm, mod), g in df.groupby(["arm", "modality"]):
        n = g["n_episodes"].sum()
        rm = float(np.average(g["mean_tv_remove"], weights=g["n_episodes"]))
        sw = float(np.average(g["mean_tv_swap"], weights=g["n_episodes"]))
        rows.append({"arm": arm, "modality": mod, "n_episodes": int(n), "mean_tv_remove": rm,
                     "mean_tv_swap": sw, "ratio_remove_over_swap": rm / sw if sw > 0 else np.nan})
    pooled = []
    for arm, g in df.groupby("arm"):
        n = g["n_episodes"].sum()
        rm = float(np.average(g["mean_tv_remove"], weights=g["n_episodes"]))
        sw = float(np.average(g["mean_tv_swap"], weights=g["n_episodes"]))
        pooled.append({"arm": arm, "modality": "ALL", "n_episodes": int(n), "mean_tv_remove": rm,
                       "mean_tv_swap": sw, "ratio_remove_over_swap": rm / sw if sw > 0 else np.nan})
    out_df = pd.DataFrame(rows + pooled)
    out_df.to_csv(out / "intervention.csv", index=False)
    L = ["# Step 11b - mask-intervention diagnostic for A5-A8 (EXPLORATORY, primary rule)", "",
         EXPLORATORY_NOTICE, "",
         "Step 10 protocol: (a) REMOVE an observed modality (mask -> 0); (b) CONTENT SWAP its values "
         "with a random other test episode that has it (mask kept). ratio = (a)/(b) > 1 means the "
         "model reacts to the mask beyond the lost content.", "",
         "| arm | modality | n | mean TV remove | mean TV swap | ratio (a)/(b) |",
         "|---|---|---|---|---|---|"]
    for _, r in out_df.iterrows():
        L.append(f"| {r['arm']} | {r['modality']} | {r['n_episodes']} | {r['mean_tv_remove']:.4f} | "
                 f"{r['mean_tv_swap']:.4f} | {r['ratio_remove_over_swap']:.3f} |")
    ref = _step10_intervention_reference()
    if ref:
        L += ["",
              "Reference rows (Step 10, recipe R1, same S1 grid and same rho-uniform protocol). "
              "Step 10 has no intervention for the R2 arms, so these are context only: they show the "
              "same sub-1 pooled ratio under the earlier recipe.", "",
              "| ref arm | ckpt | modality | n | mean TV remove | mean TV swap | ratio (a)/(b) |",
              "|---|---|---|---|---|---|---|"]
        for r in ref:
            L.append(f"| A4 [Step 10, R1] | {r['ckpt_rule']} | ALL | {r['n_episodes']} | "
                     f"{r['mean_tv_remove']:.4f} | {r['mean_tv_swap']:.4f} | {r['ratio']:.3f} |")
    (out / "intervention.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_df


def _step10_intervention_reference():
    """Pooled Step 10 A4 rows (context only; Step 10 used recipe R1)."""
    rows = []
    for rule in ("primary", "secondary"):
        p = PROJECT_ROOT / "results" / "step10" / "diagnostics" / f"intervention_{rule}.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        s = d[(d["arm"] == "A4") & (d["modality"] == "ALL")]
        for _, r in s.iterrows():
            rows.append({"ckpt_rule": rule, "n_episodes": int(r["n_episodes"]),
                         "mean_tv_remove": float(r["mean_tv_remove"]),
                         "mean_tv_swap": float(r["mean_tv_swap"]),
                         "ratio": float(r["ratio_remove_over_swap"])})
    return rows

def main() -> None:
    ap = argparse.ArgumentParser(description="Step 11b analysis (confirmatory + exploratory).")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--episodes", type=Path,
                    default=PROJECT_ROOT / "docs" / "episodes.csv")
    ap.add_argument("--splits", type=Path,
                    default=PROJECT_ROOT / "results" / "step03" / "splits.json")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a9.N_BOOT = a.bootstrap

    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())

    print("[confirmatory] S2 metrics + paired + verdicts", flush=True)
    conf = confirmatory_section(ep, blob, a.out)
    for rule in CKPT_RULES:
        v = conf[rule]["verdicts"]
        print(f"[confirmatory {rule}] R1: {v['R1']['verdict']} | R2: {v['R2']['verdict']}",
              flush=True)

    print("[exploratory] A5-A8 metrics + paired + Holm + diagnostics", flush=True)
    expl = exploratory_section(ep, blob, a.out)

    analysis = {
        "n_bootstrap": a.bootstrap,
        "uniform_protocol": "Step 6b: seed-averaged probabilities, pooled OOF, one "
                            "2,000-resample subject-cluster bootstrap, per checkpoint rule",
        "confirmatory": {
            rule: {"verdicts": conf[rule]["verdicts"],
                   "paired": conf[rule]["paired"]} for rule in CKPT_RULES},
        "exploratory_notice": EXPLORATORY_NOTICE,
        "exploratory": {rule: {"rows": expl[rule]["rows"], "paired": expl[rule]["paired"]}
                        for rule in CKPT_RULES},
        "files": ["confirmatory_{primary,secondary}.{csv,md}",
                  "confirmatory_paired_{primary,secondary}.csv",
                  "exploratory_{primary,secondary}.{csv,md}",
                  "exploratory_paired_{primary,secondary}.csv",
                  "best_epochs.{csv,md}", "embedding_swap.{csv,md}", "intervention.{csv,md}"],
    }
    (a.out / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"[wrote analysis under {a.out}]", flush=True)


if __name__ == "__main__":
    main()
