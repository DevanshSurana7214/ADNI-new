#!/usr/bin/env python3
"""
Stage 3 - Joint class x modality-pattern profiling (PIPELINE_v2 section 4).

Computes, from episodes.csv:  P(c), P(m), P(c,m) (Dirichlet-smoothed), PMI I(c,m),
independence weight w=exp(-I)=P(c)P(m)/P(c,m), bounded rarity sigmoid(-I), Cramer's V with a
subject-cluster bootstrap CI, mutual-information decomposition against the enrolment/protocol
stratum, a mask-only shortcut benchmark, and weight diagnostics (effective sample size).

In the real pipeline every table is fitted on TRAIN folds only; running it on all episodes here is
DESCRIPTIVE profiling of the cohort, not a modelling step.
"""
import argparse, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

CLASSES = ["CN", "MCI", "Dementia"]


def joint_tables(df, alpha=0.5, pats=None):
    pats = pats or sorted(df.pattern.unique())
    n = pd.crosstab(df.DIAGNOSIS, df.pattern).reindex(index=CLASSES, columns=pats, fill_value=0)
    P = (n + alpha) / (n.values.sum() + alpha * n.size)
    Pc, Pm = P.sum(1), P.sum(0)
    I = np.log(P / np.outer(Pc, Pm))
    return n, P, Pc, Pm, I


def cramers_v(tab):
    t = np.asarray(tab, float); t = t[:, t.sum(0) > 0]
    e = np.outer(t.sum(1), t.sum(0)) / t.sum()
    chi2 = ((t - e) ** 2 / e).sum()
    return np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1)))


def mi(a, b):
    t = pd.crosstab(a, b).values.astype(float); p = t / t.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def cond_mi(c, m, s):
    d = pd.DataFrame(dict(c=c.values, m=m.values, s=s.values))
    return sum(len(g) / len(d) * mi(g.c, g.m) for _, g in d.groupby("s"))


def perm_null(c, m, s, within=None, reps=200, seed=0, fn=None):
    rng = np.random.default_rng(seed); out = []
    for _ in range(reps):
        cc = c.copy()
        if within is None:
            cc = pd.Series(rng.permutation(c.values), index=c.index)
        else:
            cc = c.groupby(within).transform(lambda x: rng.permutation(x.values))
        out.append(fn(cc))
    return float(np.mean(out))


def shortcut_benchmark(df, feats, seed=0, folds=5):
    y = df.DIAGNOSIS.map({c: i for i, c in enumerate(CLASSES)}).values
    sk = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = {}
    for name, cols in feats.items():
        X = pd.get_dummies(df[cols].astype(str)).values.astype(float)
        P = np.zeros((len(df), 3))
        for tr, te in sk.split(X, y, df.RID.values):
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr]); P[te] = clf.predict_proba(X[te])
        res[name] = (roc_auc_score(y, P, multi_class="ovr", average="macro"),
                     balanced_accuracy_score(y, P.argmax(1)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="./stage1/episodes.csv")
    ap.add_argument("--out", default="./joint_profile.csv")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--wmin", type=float, default=0.2)
    ap.add_argument("--wmax", type=float, default=5.0)
    ap.add_argument("--metrics-json", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.episodes, dtype={"pattern": str})
    pats = sorted(df.pattern.unique())
    n, P, Pc, Pm, I = joint_tables(df, a.alpha, pats)

    print(f"N={len(df)} subjects={df.RID.nunique()} classes={df.DIAGNOSIS.value_counts().to_dict()}")
    print("P(c):", Pc.round(3).to_dict())
    print("patterns with n<10 in some class:", int((n < 10).values.sum()), "of", n.size, "cells;",
          "empty cells:", int((n == 0).values.sum()))

    # ---- association strength with a subject-cluster bootstrap -------------------------------------
    V = cramers_v(pd.crosstab(df.DIAGNOSIS, df.pattern))
    rng = np.random.default_rng(0)
    per = {r: pd.crosstab(d.DIAGNOSIS, d.pattern).reindex(index=CLASSES, columns=pats, fill_value=0).values
           for r, d in df.groupby("RID")}
    rids = np.array(list(per)); bs = []
    for _ in range(500):
        bs.append(cramers_v(sum(per[r] for r in rng.choice(rids, len(rids)))))
    print(f"Cramer's V(class,pattern)={V:.3f}  95% subject-bootstrap CI [{np.percentile(bs,2.5):.3f}, {np.percentile(bs,97.5):.3f}]")

    # ---- is the dependence just the protocol?  I(C;M), I(C;S), I(C;M|S) ----------------------------
    print("Mutual information in nats  (observed | permutation-null mean | excess):")
    for sname in ("ORIGPROT", "COLPROT"):
        S = df[sname]
        o1, o2, o3 = mi(df.DIAGNOSIS, df.pattern), mi(df.DIAGNOSIS, S), cond_mi(df.DIAGNOSIS, df.pattern, S)
        n3 = perm_null(df.DIAGNOSIS, df.pattern, S, S, 100, 2, lambda cc: cond_mi(cc, df.pattern, S))
        print(f"   S={sname}:  I(C;M)={o1:.4f}   I(C;S)={o2:.4f}   I(C;M|S)={o3:.4f} (null {n3:.4f}, excess {o3-n3:.4f})")
        print(f"      share of I(C;M) that survives conditioning on S: {(o3-n3)/o1:.0%}")

    # ---- mask-only shortcut benchmark (subject-grouped 5-fold CV) -----------------------------------
    df["ORIG"] = df.ORIGPROT; df["COL"] = df.COLPROT
    r = shortcut_benchmark(df, {"pattern only": ["pattern"], "protocol only (ORIGPROT)": ["ORIG"],
                                "protocol only (visit COLPROT)": ["COL"],
                                "pattern + ORIGPROT": ["pattern", "ORIG"], "pattern + COLPROT": ["pattern", "COL"]})
    print("Classifier that sees NO imaging values (macro-OvR AUROC, balanced acc; chance = 0.5, 0.333):")
    for k, (au, ba) in r.items(): print(f"   {k:32s} {au:.3f}  {ba:.3f}")

    # ---- joint rarity / independence weights -------------------------------------------------------
    w_cell = np.exp(-I)                                   # = P(c)P(m)/P(c,m)
    w = df.apply(lambda r_: w_cell.loc[r_.DIAGNOSIS, r_.pattern], axis=1).values
    wc = np.clip(w, a.wmin, a.wmax); wc = wc / wc.mean()
    ess = lambda x: (x.sum() ** 2) / (x ** 2).sum() / len(x)
    print(f"independence weight exp(-I): raw range [{w.min():.2f}, {w.max():.2f}] ESS/N={ess(w):.3f} | "
          f"clipped [{a.wmin},{a.wmax}] ESS/N={ess(wc):.3f}")
    # check that clipped weights remove most of the class-pattern association
    dfw = df.assign(w=wc)
    tw = dfw.pivot_table(index="DIAGNOSIS", columns="pattern", values="w", aggfunc="sum", fill_value=0).reindex(index=CLASSES, columns=pats, fill_value=0)
    print(f"Cramer's V after clipped reweighting = {cramers_v(tw):.3f}  (before {V:.3f})")
    sig = 1 / (1 + np.exp(I))                              # original bounded 'rarity' R=sigmoid(-I)
    cnt = n.values.ravel()
    from scipy.stats import spearmanr
    print(f"Spearman(cell count, sigmoid(-I))={spearmanr(cnt, sig.values.ravel())[0]:+.2f}   "
          f"Spearman(cell count, exp(-I))={spearmanr(cnt, w_cell.values.ravel())[0]:+.2f}")

    rows = []
    for c in CLASSES:
        for m in pats:
            rows.append(dict(DIAGNOSIS=c, pattern=m, n=int(n.loc[c, m]), P_cm=P.loc[c, m], P_c=Pc[c], P_m=Pm[m],
                             PMI=I.loc[c, m], R_sigmoid=sig.loc[c, m], w_indep=w_cell.loc[c, m],
                             P_c_given_m=P.loc[c, m] / Pm[m]))
    out = pd.DataFrame(rows); out.to_csv(a.out, index=False)
    if a.metrics_json:
        import json
        json.dump(dict(N=len(df), subjects=int(df.RID.nunique()), V=float(V), V_lo=float(np.percentile(bs, 2.5)), V_hi=float(np.percentile(bs, 97.5)),
                       shortcut={k: [float(x) for x in v] for k, v in r.items()}, ess_raw=float(ess(w)), ess_clip=float(ess(wc)),
                       w_min=float(w.min()), w_max=float(w.max()), V_after=float(cramers_v(tw)),
                       spearman_R=float(spearmanr(cnt, sig.values.ravel())[0]), n_small=int((n < 10).values.sum()),
                       n_empty=int((n == 0).values.sum()), Pc=Pc.to_dict()), open(a.metrics_json, "w"), indent=1)
    print(out.sort_values("PMI").head(6)[["DIAGNOSIS", "pattern", "n", "PMI", "w_indep"]].to_string(index=False))
    print(out.sort_values("PMI").tail(6)[["DIAGNOSIS", "pattern", "n", "PMI", "w_indep"]].to_string(index=False))


if __name__ == "__main__":
    main()
