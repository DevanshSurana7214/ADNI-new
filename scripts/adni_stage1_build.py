#!/usr/bin/env python3
"""
Stage 1 - Episode construction for the ADNI tabular multimodal pipeline (PIPELINE_v2).

Anchor      : one episode = one DXSUM diagnosis visit (label = CN / MCI / Dementia).
Modalities  : for each of MRI, FDG, amyloid, tau take the NEAREST scan within +/- WINDOW days
              of the diagnosis date. No scan in window => modality absent (mask bit 0).
Inclusion   : >= 1 imaging modality present (pattern 0000 is excluded: nothing to encode).
Outputs     : episodes.csv, feature_spec.json, and (with --write-blocks) blocks/<modality>.csv

No statistics are fitted here that depend on labels. Normalisation, feature filtering by
missingness and every profiling quantity are fitted on TRAIN folds only, in Stage 2 / 3.

Usage:
  python adni_stage1_build.py --uploads /mnt/user-data/uploads --out ./stage1 --window 180 --write-blocks
"""
import argparse, json, os, re
import numpy as np
import pandas as pd

FILES = dict(
    dx="DXSUM.csv",
    mri="UCSFFSX7_03Oct2026.csv",
    fdg="UCBERKELEYFDG_8mm_02_17_23_03Oct2026.csv",
    amy="UCBERKELEY_AMY_6MM_03Oct2026.csv",
    tau="UCBERKELEY_TAU_6MM_03Oct2026.csv",
)
CLASSES = ["CN", "MCI", "Dementia"]
MODS = ["mri", "fdg", "amy", "tau"]


def load(up, window, amy_tracers, tau_tracers):
    P = lambda k: os.path.join(up, FILES[k])

    dx = pd.read_csv(P("dx"), low_memory=False,
                     usecols=["RID", "ORIGPROT", "COLPROT", "VISCODE2", "EXAMDATE", "DIAGNOSIS", "update_stamp"])
    dx = dx.dropna(subset=["DIAGNOSIS", "EXAMDATE"])
    # drop (RID, date) groups whose duplicate rows disagree on the label
    nun = dx.groupby(["RID", "EXAMDATE"]).DIAGNOSIS.transform("nunique")
    dx = dx[nun == 1].sort_values("update_stamp").drop_duplicates(["RID", "EXAMDATE"], keep="last")
    dx["dx_date"] = pd.to_datetime(dx.EXAMDATE)
    dx = dx.sort_values("dx_date").reset_index(drop=True)
    dx["y"] = dx.DIAGNOSIS.map({c: i for i, c in enumerate(CLASSES)})

    mri = pd.read_csv(P("mri"), low_memory=False)
    mri = mri[mri.OVERALLQC.ne("Fail")].copy()
    mri["scan_date"] = pd.to_datetime(mri.EXAMDATE)
    # deterministic choice among several MRI rows on the same date: QC 'Pass' first, then lowest IMAGEUID
    mri["_qc_rank"] = mri.OVERALLQC.map({"Pass": 0}).fillna(1)
    mri = mri.sort_values(["RID", "scan_date", "_qc_rank", "IMAGEUID"]).drop_duplicates(["RID", "scan_date"])

    fdg = pd.read_csv(P("fdg"))
    fdg["scan_date"] = pd.to_datetime(fdg.EXAMDATE)

    amy = pd.read_csv(P("amy"), low_memory=False)
    amy = amy[amy.TRACER.isin(amy_tracers)].copy()
    amy["scan_date"] = pd.to_datetime(amy.SCANDATE)
    amy = amy.sort_values(["RID", "scan_date", "qc_flag"], ascending=[True, True, False]).drop_duplicates(["RID", "scan_date"])

    tau = pd.read_csv(P("tau"), low_memory=False)
    tau = tau[tau.TRACER.isin(tau_tracers)].copy()
    tau["scan_date"] = pd.to_datetime(tau.SCANDATE)
    tau = tau.sort_values(["RID", "scan_date", "qc_flag"], ascending=[True, True, False]).drop_duplicates(["RID", "scan_date"])

    return dx, dict(mri=mri, fdg=fdg, amy=amy, tau=tau)


def anchor(dx, scans, window):
    ep = dx[["RID", "ORIGPROT", "COLPROT", "VISCODE2", "dx_date", "DIAGNOSIS", "y"]].copy()
    tol = pd.Timedelta(days=window)
    for k in MODS:
        r = scans[k][["RID", "scan_date"]].dropna().drop_duplicates().sort_values("scan_date")
        r = r.rename(columns={"scan_date": f"{k}_date"}); r["_key"] = r[f"{k}_date"]
        ep = ep.sort_values("dx_date")
        ep = pd.merge_asof(ep, r, left_on="dx_date", right_on="_key", by="RID",
                           direction="nearest", tolerance=tol).drop(columns="_key")
        ep[f"m_{k}"] = ep[f"{k}_date"].notna().astype(int)
        ep[f"{k}_gap_days"] = (ep[f"{k}_date"] - ep.dx_date).dt.days
    ep["pattern"] = ep[[f"m_{k}" for k in MODS]].astype(str).agg("".join, axis=1)
    ep["n_mod"] = ep[[f"m_{k}" for k in MODS]].sum(axis=1)
    ep = ep[ep.n_mod >= 1].sort_values(["RID", "dx_date"]).reset_index(drop=True)
    ep.insert(0, "episode_id", np.arange(len(ep)))
    return ep


def blocks(ep, scans):
    out, spec = {}, {}
    # ---- MRI: FreeSurfer 7 cortical/subcortical volumes, surface area, thickness mean -------------
    m = scans["mri"]
    cols = [c for c in m.columns if re.match(r"^ST\d+(CV|SV|SA|TA)$", c) and c != "ST10CV"]  # ST10CV = ICV
    j = ep[["episode_id", "RID", "mri_date"]].dropna().merge(
        m, left_on=["RID", "mri_date"], right_on=["RID", "scan_date"], how="left")
    X = j[["episode_id"] + cols].copy()
    icv = j["ST10CV"].values
    for c in cols:                                   # volumes -> fraction of ICV
        if c.endswith(("CV", "SV")):
            X[c] = X[c].values / icv
    X = pd.concat([X, pd.Series(j.FIELD_STRENGTH.map({"1.5T": 0, "3T": 1}).values, name="mri_field_strength")], axis=1)
    out["mri"] = X; spec["mri"] = cols + ["mri_field_strength"]

    # ---- FDG: two summary ROIs only ---------------------------------------------------------------
    f = scans["fdg"]
    piv = f.pivot_table(index=["RID", "scan_date"], columns="ROINAME", values=["MEAN", "MAX", "STDEV"])
    piv.columns = [f"{roi}_{stat}" for stat, roi in piv.columns]
    piv = piv.reset_index()
    j = ep[["episode_id", "RID", "fdg_date"]].dropna().merge(
        piv, left_on=["RID", "fdg_date"], right_on=["RID", "scan_date"], how="left")
    fc = [c for c in piv.columns if c not in ("RID", "scan_date")]
    out["fdg"] = j[["episode_id"] + fc]; spec["fdg"] = fc

    # ---- Amyloid / tau PET: regional SUVRs; reference-region columns are constant 1.0 -> dropped ----
    for k, ref, extra in (("amy", "WHOLECEREBELLUM_SUVR", ["CENTILOIDS"]),
                          ("tau", "INFERIORCEREBELLUM_SUVR", [])):
        t = scans[k]
        pc = [c for c in t.columns if c.endswith("_SUVR") and c != ref] + extra
        j = ep[["episode_id", "RID", f"{k}_date"]].dropna().merge(
            t, left_on=["RID", f"{k}_date"], right_on=["RID", "scan_date"], how="left")
        out[k] = j[["episode_id"] + pc]; spec[k] = pc
    return out, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", default="/mnt/user-data/uploads")
    ap.add_argument("--out", default="./stage1")
    ap.add_argument("--window", type=int, default=180)
    ap.add_argument("--amy-tracers", nargs="+", default=["FBP"])   # florbetapir (AV45)
    ap.add_argument("--tau-tracers", nargs="+", default=["FTP"])   # flortaucipir (AV1451)
    ap.add_argument("--write-blocks", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dx, scans = load(a.uploads, a.window, a.amy_tracers, a.tau_tracers)
    ep = anchor(dx, scans, a.window)
    bl, spec = blocks(ep, scans)
    # Rule: a scan row with > 50 % of its features missing is treated as an ABSENT modality
    # (mask bit 0). Episodes left with no modality are dropped; episode_id values are kept so the
    # blocks stay aligned.
    n_before = len(ep)
    for k in MODS:
        feat = [c for c in bl[k].columns if c != "episode_id"]
        if k == "mri": feat = [c for c in feat if c != "mri_field_strength"]
        bad = bl[k].episode_id[bl[k][feat].notna().mean(axis=1) < 0.5]
        ep.loc[ep.episode_id.isin(bad), f"m_{k}"] = 0
        bl[k] = bl[k][~bl[k].episode_id.isin(bad)]
    ep["pattern"] = ep[[f"m_{k}" for k in MODS]].astype(str).agg("".join, axis=1)
    ep["n_mod"] = ep[[f"m_{k}" for k in MODS]].sum(axis=1)
    ep = ep[ep.n_mod >= 1].reset_index(drop=True)
    for k in MODS:
        bl[k] = bl[k][bl[k].episode_id.isin(ep.episode_id)]
    print(f"missing-feature rule: {n_before - len(ep)} episodes dropped (blocks with >50% missing features treated as absent)")
    ep.to_csv(os.path.join(a.out, "episodes.csv"), index=False)
    json.dump({k: v for k, v in spec.items()}, open(os.path.join(a.out, "feature_spec.json"), "w"), indent=1)

    print(f"episodes={len(ep)} subjects={ep.RID.nunique()} patterns={ep.pattern.nunique()} window=+/-{a.window}d")
    print(ep.DIAGNOSIS.value_counts().to_dict())
    for k in MODS:
        print(f"  {k}: n={int(ep[f'm_{k}'].sum())} features={len(spec[k])} "
              f"median|gap|={ep[f'{k}_gap_days'].abs().median():.0f}d")
    if a.write_blocks:
        os.makedirs(os.path.join(a.out, "blocks"), exist_ok=True)
        for k, X in bl.items():
            X.to_csv(os.path.join(a.out, "blocks", f"{k}.csv"), index=False)


if __name__ == "__main__":
    main()
