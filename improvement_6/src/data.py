"""Data layer for PIPELINE v6.

Loads the frozen v2 merged artefacts (read-only), runs the data audit
(section 2), builds the three cohorts (FULL / BASELINE / DIAL), and computes
the dial design distributions estimated once from the FULL cohort projected
onto BASE_MODS (section 7.2).

Nothing here writes to the frozen artefacts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

MODALITIES: tuple[str, ...] = ("mri", "fdg", "amy", "tau")
CLASSES: tuple[str, ...] = ("CN", "MCI", "Dementia")
# Pattern bit order used throughout the repo.
PATTERN_ORDER_BASE: tuple[str, ...] = tuple(
    f"{int(b_mri)}{int(b_fdg)}{int(b_amy)}"
    for b_mri in (0, 1)
    for b_fdg in (0, 1)
    for b_amy in (0, 1)
    if (b_mri or b_fdg or b_amy)
)
# 7 non-empty patterns over BASE_MODS.
PATTERNS_BASE: tuple[str, ...] = tuple(
    p for p in PATTERN_ORDER_BASE if p != "000"
)

J_LT_PATTERN = "pattern must equal concatenated m_mri m_fdg m_amy m_tau"
J_LT_NMOD = "n_mod must equal sum of mask bits"
J_LT_DXDATE_DUP = "no duplicate (RID, dx_date) rows allowed"
J_LT_EID_DUP = "no duplicate episode_id rows allowed"
J_LT_Y_DIAG = "y must match DIAGNOSIS"
J_LT_GAP = "|gap_days| must be <= 180 where a modality is present"
J_LT_MASK_FEATURES = "rows with m_<mod>=0 must have no block features"
J_LT_PRESENT_FEATURES = "rows with m_<mod>=1 must have block features"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

@dataclass
class Episodes:
    """The frozen merged episode table, with assertions from section 2.2-2.3."""

    df: pd.DataFrame  # indexed by episode_id
    blocks: dict[str, pd.DataFrame]
    spec: dict[str, list[str]]

    @property
    def n_episodes(self) -> int:
        return int(len(self.df))

    @property
    def n_subjects(self) -> int:
        return int(self.df["RID"].nunique())


def load_episodes(path: Path) -> pd.DataFrame:
    """Read episodes.csv with pattern as TEXT, parse dates, set index=episode_id."""
    df = pd.read_csv(path, dtype={"pattern": str})
    required_cols = [
        "episode_id", "RID", "ORIGPROT", "COLPROT", "VISCODE2", "dx_date",
        "DIAGNOSIS", "y", "pattern", "n_mod",
        "mri_date", "m_mri", "mri_gap_days",
        "fdg_date", "m_fdg", "fdg_gap_days",
        "amy_date", "m_amy", "amy_gap_days",
        "tau_date", "m_tau", "tau_gap_days",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"episodes table missing columns: {missing}")
    for c in ["dx_date", "mri_date", "fdg_date", "amy_date", "tau_date"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df = df.set_index("episode_id")
    if df.index.duplicated().any():
        raise ValueError(J_LT_EID_DUP)
    return df


def load_blocks(blocks_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for k in MODALITIES:
        df = pd.read_csv(blocks_dir / f"{k}.csv")
        if "episode_id" not in df.columns:
            raise KeyError(f"block {k}.csv missing episode_id column")
        df = df.set_index("episode_id")
        if df.index.duplicated().any():
            raise ValueError(f"block {k}.csv has duplicate episode_id rows")
        out[k] = df
    return out


def load_spec(spec_path: Path) -> dict[str, list[str]]:
    return json.loads(spec_path.read_text())


def load_merged(
    episodes_path: Path,
    blocks_dir: Path,
    spec_path: Path,
) -> Episodes:
    return Episodes(
        df=load_episodes(episodes_path),
        blocks=load_blocks(blocks_dir),
        spec=load_spec(spec_path),
    )


# ---------------------------------------------------------------------------
# assertions (section 2.2-2.3)
# ---------------------------------------------------------------------------

def audit_assertions(ep: Episodes) -> dict[str, Any]:
    """Run section 2.2-2.3 assertions; return a dict of results and raise on failure."""
    df = ep.df
    results: dict[str, Any] = {}

    results["n_rows"] = int(len(df))
    results["n_unique_episode_id"] = int(df.index.nunique())
    results["duplicate_episode_id_rows"] = int(df.index.duplicated().sum())
    dup_rid_date = df.reset_index().duplicated(subset=["RID", "dx_date"]).sum()
    results["duplicate_rid_dxdate_rows"] = int(dup_rid_date)
    results["pattern_matches_mask"] = int(
        (df["pattern"].astype(str)
         == (df["m_mri"].astype(str) + df["m_fdg"].astype(str)
             + df["m_amy"].astype(str) + df["m_tau"].astype(str))).sum()
    )
    results["n_mod_matches_bits"] = bool((df["n_mod"]
                                           == df[["m_mri", "m_fdg", "m_amy", "m_tau"]].sum(axis=1)).all())
    gap_cols = ["mri_gap_days", "fdg_gap_days", "amy_gap_days", "tau_gap_days"]
    gap_present = df[[f"m_{k}" for k in MODALITIES]].replace({0: np.nan})
    results["gap_within_180"] = bool(
        ((df[gap_cols].abs() <= 180) | df[gap_cols].isna()).all().all()
    )
    results["y_matches_diagnosis"] = bool(
        (df["DIAGNOSIS"] == df["y"].map({0: "CN", 1: "MCI", 2: "Dementia"})).all()
    )

    if results["duplicate_episode_id_rows"]:
        raise ValueError(J_LT_EID_DUP)
    if results["duplicate_rid_dxdate_rows"]:
        raise ValueError(J_LT_DXDATE_DUP)
    if results["pattern_matches_mask"] != len(df):
        raise ValueError(J_LT_PATTERN)
    if not results["n_mod_matches_bits"]:
        raise ValueError(J_LT_NMOD)
    if not results["gap_within_180"]:
        raise ValueError(J_LT_GAP)
    if not results["y_matches_diagnosis"]:
        raise ValueError(J_LT_Y_DIAG)

    # 2.3 block alignment.
    block_results: list[dict[str, Any]] = []
    for k in MODALITIES:
        df_blk = ep.blocks[k]
        present_ids = df.index[df[f"m_{k}"] == 1]
        absent_ids = df.index[df[f"m_{k}"] == 0]
        present_in_blk = df_blk.index.intersection(present_ids)
        absent_in_blk = df_blk.index.intersection(absent_ids)
        present_feature_cols = [c for c in df_blk.columns if c != "episode_id"]
        n_present_missing_any = int(
            df_blk.loc[present_in_blk, present_feature_cols].isna().any(axis=1).sum()
            if len(present_in_blk) and present_feature_cols
            else 0
        )
        br = {
            "modality": k,
            "block_rows": int(len(df_blk)),
            "present_ids_in_block": int(present_in_blk.nunique()),
            "absent_ids_in_block": int(absent_in_blk.nunique()),
            "present_feature_cols_null_any": n_present_missing_any,
        }
        block_results.append(br)
        # Formal section 2.3 assertions.
        if len(present_ids) and present_in_blk.nunique() != len(present_ids):
            raise ValueError(J_LT_PRESENT_FEATURES)
        if len(absent_ids) and absent_in_blk.nunique() != 0:
            raise ValueError(J_LT_MASK_FEATURES)
    results["blocks"] = block_results
    return results


# ---------------------------------------------------------------------------
# cohorts (section 2.4)
# ---------------------------------------------------------------------------

CohortName = str


def build_full(ep: Episodes) -> pd.DataFrame:
    """FULL = all episodes (a view, indexed like ep.df)."""
    return ep.df.copy()


def build_baseline(ep: Episodes) -> pd.DataFrame:
    """BASELINE: one episode per subject, earliest dx_date, prefer VISCODE2==bl."""
    g = ep.df.groupby("RID")
    order: list[int] = []
    for _rid, sub in g:
        sub = sub.copy()
        sub["_dx"] = pd.to_datetime(sub["dx_date"], errors="coerce")
        sub = sub.sort_values(["_dx", "VISCODE2"], na_position="last")
        prefer_bl = sub[sub["VISCODE2"] == "bl"]
        if len(prefer_bl):
            order.append(int(prefer_bl.index[0]))
        else:
            order.append(int(sub.index[0]))
    return ep.df.loc[order].copy()


def build_dial(ep: Episodes, dial_mods: Sequence[str] = ("mri", "fdg", "amy")) -> pd.DataFrame:
    """DIAL: one episode per subject (earliest dx), all BASE_MODS present."""
    mask = np.ones(len(ep.df), dtype=bool)
    for k in dial_mods:
        mask &= (ep.df[f"m_{k}"] == 1).to_numpy(dtype=bool)
    sub = ep.df.loc[mask]
    g = sub.groupby("RID")
    order: list[int] = []
    for _rid, grp in g:
        grp = grp.copy()
        grp["_dx"] = pd.to_datetime(grp["dx_date"], errors="coerce")
        grp = grp.sort_values("_dx")
        order.append(int(grp.index[0]))
    return ep.df.loc[order].copy()


def build_cohort(
    ep: Episodes,
    name: CohortName,
    dial_mods: Sequence[str] | None = None,
) -> pd.DataFrame:
    if name == "FULL":
        return build_full(ep)
    if name == "BASELINE":
        return build_baseline(ep)
    if name == "DIAL":
        return build_dial(ep, dial_mods=dial_mods or ("mri", "fdg", "amy"))
    raise ValueError(f"unknown cohort {name!r}")


def write_cohorts(
    ep: Episodes,
    out_dir: Path,
    dial_mods: Sequence[str] = ("mri", "fdg", "amy"),
) -> dict[CohortName, pd.DataFrame]:
    """Write each cohort as a list of episode_ids; return the DataFrames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cohorts: dict[CohortName, pd.DataFrame] = {}
    for name in ("FULL", "BASELINE", "DIAL"):
        c = build_cohort(ep, name, dial_mods=dial_mods)
        cohorts[name] = c
        (out_dir / f"{name}.txt").write_text(
            "\n".join(str(int(i)) for i in c.index.to_numpy()) + "\n"
        )
    return cohorts


def cohort_summary(ep: Episodes, cohorts: Mapping[CohortName, pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, c in cohorts.items():
        vc = pd.crosstab(c["y"], c["pattern"]).reindex(index=[0, 1, 2], fill_value=0)
        out[name] = {
            "n": int(len(c)),
            "n_subjects": int(c["RID"].nunique()),
            "class_counts": {CLASSES[i]: int((c["y"] == i).sum()) for i in range(3)},
            "pattern_counts": {p: int(vc[p].sum()) for p in vc.columns},
            "n_patterns": int(c["pattern"].nunique()),
        }
    return out


# ---------------------------------------------------------------------------
# joint imbalance tables (section 6.1)
# ---------------------------------------------------------------------------

def cramers_v_from_counts(conf: np.ndarray) -> float:
    conf = np.asarray(conf, dtype=float)
    chi2 = stats.chi2_contingency(conf, correction=False)[0]
    n = conf.sum()
    r, k = conf.shape
    return float(np.sqrt(chi2 / (n * min(r - 1, k - 1))))


def mi_class_pattern(ep_df: pd.DataFrame, alpha: float = 0.5) -> float:
    n_cm = pd.crosstab(ep_df["y"], ep_df["pattern"]).reindex(index=[0, 1, 2], fill_value=0)
    n = n_cm.values.astype(float)
    n += alpha
    P = n / n.sum()
    Pc = P.sum(axis=1, keepdims=True)
    Pm = P.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        el = np.where(P > 0, P * np.log(P / (Pc * Pm)), 0.0)
    return float(el.sum())


def modality_share_by_class(ep_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(CLASSES):
        sub = ep_df[ep_df["y"] == i]
        out[name] = {k: float(sub[f"m_{k}"].mean()) if len(sub) else float("nan")
                     for k in MODALITIES}
    out["all"] = {k: float(ep_df[f"m_{k}"].mean()) for k in MODALITIES}
    return out


def n_mod_by_class(ep_df: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, np.ndarray] = {}
    idx = list(range(1, len(MODALITIES) + 1))
    for i, name in enumerate(CLASSES):
        sub = ep_df[ep_df["y"] == i]
        out[name] = sub["n_mod"].value_counts().reindex(idx, fill_value=0).to_numpy(dtype=int)
    out["all"] = ep_df["n_mod"].value_counts().reindex(idx, fill_value=0).to_numpy(dtype=int)
    return pd.DataFrame(out, index=[f"n_mod={i}" for i in idx])


def class_pattern_table(ep_df: pd.DataFrame) -> pd.DataFrame:
    vc = pd.crosstab(ep_df["y"], ep_df["pattern"]).reindex(index=[0, 1, 2], fill_value=0)
    tab = pd.DataFrame(vc.values, index=list(CLASSES), columns=list(vc.columns))
    tab["Total"] = tab.sum(axis=1)
    tab.loc["Total"] = tab.sum(axis=0)
    return tab


def pattern_distribution(ep_df: pd.DataFrame) -> pd.DataFrame:
    vc = pd.crosstab(ep_df["y"], ep_df["pattern"]).reindex(index=[0, 1, 2], fill_value=0)
    total = vc.sum(axis=0).astype(int)
    rows: list[dict[str, Any]] = []
    for p in vc.columns:
        rows.append({
            "pattern": p,
            "n": int(total[p]),
            "share": float(total[p]) / len(ep_df) if len(ep_df) else 0.0,
            "n_mod": int(sum(int(b) for b in p)),
            **{CLASSES[i]: (float(vc.iloc[i][p]) / int(total[p]) if total[p] else 0.0)
               for i in range(3)},
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False, ignore_index=True)


def phase_tables(ep_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ph = ep_df["COLPROT"]
    avail = ep_df.groupby("COLPROT", observed=True)[[f"m_{k}" for k in MODALITIES]].mean()
    avail.columns = list(MODALITIES)
    cls = pd.crosstab(ep_df["COLPROT"], ep_df["y"], normalize="index")
    cls.columns = [CLASSES[int(c)] for c in cls.columns]
    per_phase: list[dict[str, Any]] = []
    for p in sorted(set(ep_df["COLPROT"])):
        sub = ep_df[ep_df["COLPROT"] == p]
        if len(sub) < 2:
            continue
        v = pd.crosstab(sub["y"], sub["pattern"]).reindex(index=[0, 1, 2], fill_value=0)
        per_phase.append({
            "phase": p,
            "n": int(len(sub)),
            "cramers_v": cramers_v_from_counts(v.values),
            "n_patterns": int(v.shape[1]),
        })
    return {
        "availability_by_phase": avail,
        "class_share_by_phase": cls,
        "per_phase_cramers": pd.DataFrame(per_phase),
    }


def year_table(ep_df: pd.DataFrame) -> pd.DataFrame:
    df = ep_df.copy()
    df["_year"] = pd.to_datetime(df["dx_date"], errors="coerce").dt.year
    g = df.groupby("_year", dropna=True)
    out = g.agg(
        n=("y", "size"),
        **{k: (f"m_{k}", "mean") for k in MODALITIES},
        share_dementia=("y", lambda s: float((s == 2).mean())),
    )
    out.index = out.index.astype(int)
    return out.reset_index().rename(columns={"_year": "year"})


def imbalance_tables(ep_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "class_counts": {CLASSES[i]: int((ep_df["y"] == i).sum()) for i in range(3)},
        "modality_share_by_class": modality_share_by_class(ep_df),
        "n_mod_by_class": n_mod_by_class(ep_df),
        "class_pattern_table": class_pattern_table(ep_df),
        "pattern_distribution": pattern_distribution(ep_df),
        "cramers_v": cramers_v_from_counts(
            pd.crosstab(ep_df["y"], ep_df["pattern"]).reindex(index=[0, 1, 2], fill_value=0).values
        ),
        "mutual_information_nats": mi_class_pattern(ep_df),
        "phase_tables": phase_tables(ep_df),
        "year_table": year_table(ep_df),
    }


# ---------------------------------------------------------------------------
# dial design distributions (section 7.2)
# ---------------------------------------------------------------------------

@dataclass
class DialDesign:
    """Estimated once from FULL projected onto BASE_MODS."""

    base_mods: tuple[str, ...]
    patterns: tuple[str, ...]
    P_marg: dict[str, float]          # P_marg(p) overall pattern distribution
    P_obs: dict[str, dict[str, float]]  # P_obs(p | c)
    classes: tuple[str, ...] = field(default_factory=lambda: CLASSES)

    def pattern_prob(self, p: str, c: int, lam: float) -> float:
        """P_lambda(p | c) = (1-lam)*P_marg(p) + lam*P_obs(p|c)."""
        pm = self.P_marg.get(p, 0.0)
        po = self.P_obs.get(CLASSES[c], {}).get(p, 0.0)
        return float((1.0 - lam) * pm + lam * po)


def estimate_dial_design(
    ep: Episodes,
    base_mods: Sequence[str] = ("mri", "fdg", "amy"),
    drop_zero_proj: bool = True,
) -> DialDesign:
    """Estimate P_marg(p) and P_obs(p|c) from FULL projected onto BASE_MODS.

    Projection: for each episode take the first len(base_mods) bits of pattern.
    Episodes whose projected pattern is 000 are dropped (no content over base mods).
    """
    bits = ep.df[[f"m_{k}" for k in base_mods]].to_numpy(dtype=np.uint8)
    proj = ["".join(str(int(b)) for b in row) for row in bits]
    proj = pd.Series(proj, index=ep.df.index)
    if drop_zero_proj:
        keep = proj != "000"
        proj = proj[keep]
        y = ep.df.loc[keep, "y"]
    else:
        y = ep.df["y"]

    patterns = tuple(p for p in PATTERN_ORDER_BASE if p != "000")
    vc = pd.crosstab(y, proj).reindex(index=[0, 1, 2], fill_value=0)
    P_obs: dict[str, dict[str, float]] = {}
    for i in range(3):
        row = vc.iloc[i].values.astype(float)
        s = row.sum()
        P_obs[CLASSES[i]] = {p: (float(row[p_idx]) / s if s else 0.0)
                             for p_idx, p in enumerate(patterns)}
    overall = vc.sum(axis=0).values.astype(float)
    P_marg = {p: float(overall[p_idx] / overall.sum()) for p_idx, p in enumerate(patterns)}
    return DialDesign(base_mods=tuple(base_mods), patterns=patterns, P_marg=P_marg, P_obs=P_obs)


def cramers_v_for_dial_design(dd: DialDesign, lam: float, n_per_class: int = 2000) -> float:
    """Simulate a train set of size n_per_class per class at dependence lambda, return Cramér's V.

    Used by CHECKPOINT 2 to validate the dial (section 7.3 validity check).
    """
    rng = np.random.default_rng(0)
    cells: list[tuple[int, str]] = []
    for c in range(3):
        ps = np.array([dd.pattern_prob(p, c, lam) for p in dd.patterns])
        ps = ps / ps.sum()
        drawn = rng.choice(dd.patterns, size=n_per_class, p=ps)
        for p in drawn:
            cells.append((c, p))
    conf = pd.crosstab(
        pd.Series([c for c, _ in cells], dtype=int),
        pd.Series([p for _, p in cells], dtype=str),
    )
    conf = conf.reindex(index=[0, 1, 2], columns=list(dd.patterns), fill_value=0)
    return cramers_v_from_counts(conf.values)


# ---------------------------------------------------------------------------
# train masking for the dial (section 7.3)
# ---------------------------------------------------------------------------

@dataclass
class MaskedEpisode:
    """An episode whose blocks have been masked to a target pattern over BASE_MODS."""

    episode_id: int
    rid: int
    y: int
    target_pattern: str        # over BASE_MODS, e.g. "101"
    base_mods: tuple[str, ...]


def mask_episode_to_pattern(ep_row: pd.Series, target_pattern: str,
                            base_mods: Sequence[str]) -> MaskedEpisode:
    """Return a MaskedEpisode. target_pattern length == len(base_mods)."""
    if len(target_pattern) != len(base_mods):
        raise ValueError("target_pattern length must equal len(base_mods)")
    return MaskedEpisode(
        episode_id=int(ep_row.name),
        rid=int(ep_row["RID"]),
        y=int(ep_row["y"]),
        target_pattern=target_pattern,
        base_mods=tuple(base_mods),
    )


def sample_pattern_for_class(c: int, lam: float, design: DialDesign, rng: np.random.Generator) -> str:
    ps = np.array([design.pattern_prob(p, c, lam) for p in design.patterns])
    ps = ps / ps.sum()
    return str(rng.choice(design.patterns, p=ps))
