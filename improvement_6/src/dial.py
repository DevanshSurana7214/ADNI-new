"""Imbalance dial (Part B) for PIPELINE v6 (section 7).

Design:
  - base cohort = DIAL (MRI+FDG+AMY all present), one episode per subject earliest.
  - patterns over BASE_MODS = [mri, fdg, amy]: 7 non-empty patterns.
  - P_marg(p), P_obs(p|c) estimated once from FULL projected onto BASE_MODS.
  - dependence dial lambda in {0, 0.25, 0.5, 0.75, 1.0}:
      P_lambda(p | c) = (1 - lambda) * P_marg(p) + lambda * P_obs(p | c)
  - imbalance dial keep_frac in {1.0, 0.5, 0.25}: keep that fraction of Dementia
    SUBJECTS in train+val only; test stays natural.

For each (lambda, keep_frac):
  - for each train/val episode with class c, draw a pattern p ~ P_lambda(p|c)
    and mask the blocks not in p (same encoding as section 3.3).
  - VALIDITY CHECK: for the train sets, Cramér's V(class, pattern) must increase
    monotonically with lambda (~0 at lambda=0, close to observed at lambda=1).

Test conditions for every trained model (same held-out test fold):
  - T_matched : test patterns drawn from the same P_lambda as training
  - T_indep   : test patterns drawn from lambda=0 (independent availability)
  - T_full    : all three modalities present

Pattern-response matrix (7.6): for models trained at lambda=0 and lambda=1
(keep_frac=1), evaluate every test episode under EVERY imposed pattern.

Joint-effect test (7.7): OLS on repeat-level results
  macro_cell_accuracy ~ lambda + skew + lambda:skew   (M1 only, each model)
where skew = 1 - keep_frac.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data import (DialDesign, MaskedEpisode, estimate_dial_design,
                 load_merged, sample_pattern_for_class)
from metrics import macro_cell_accuracy, overall_metrics, tmatch_minus_tindep
from models import LGBMConfig, MLPConfig, fit_lgbm, fit_mlp, predict_mlp
from methods import CellPrior, class_weights, joint_weights

MODALITIES: tuple[str, ...] = ("mri", "fdg", "amy", "tau")
CLASSES: tuple[str, ...] = ("CN", "MCI", "Dementia")


@dataclass
class DialSetting:
    lambda_: float
    keep_frac: float
    skew: float = field(init=False)

    def __post_init__(self) -> None:
        self.skew = 1.0 - self.keep_frac


@dataclass
class DialResult:
    setting: DialSetting
    model: str
    method: str
    repeat: int
    fold: int
    seed: int
    T_matched: dict[str, float]
    T_indep: dict[str, float]
    T_full: dict[str, float]
    tau: float | None = None


def mask_blocks_for_pattern(ep_row: pd.Series, target_pattern: str,
                            base_mods: Sequence[str], blocks: Mapping[str, pd.DataFrame],
                            spec: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Return per-modality feature array with blocks not in target_pattern set to NaN/zeros.

    This is a simplified helper: returns None for modalities not in target_pattern.
    """
    out: dict[str, np.ndarray] = {}
    present_in_pattern = {k: (target_pattern[i] == "1") for i, k in enumerate(base_mods)}
    for k in base_mods:
        if present_in_pattern[k]:
            if ep_row.name in blocks[k].index:
                row = blocks[k].loc[ep_row.name]
                feats = np.array([float(row[c]) if c in row.index and pd.notna(row[c]) else np.nan
                                  for c in spec.get(k, [])], dtype=float)
            else:
                feats = np.full(len(spec.get(k, [])), np.nan)
        else:
            feats = np.full(len(spec.get(k, [])), np.nan)
        out[k] = feats.astype(float)
    return out


def build_dial_train_episodes(
    ep_df: pd.DataFrame,
    design: DialDesign,
    lambda_: float,
    keep_frac: float,
    rng: np.random.Generator,
    base_mods: Sequence[str],
) -> list[MaskedEpisode]:
    """Build masked train episodes for one dial setting.

    - Keep all classes' subjects except Dementia: keep a fraction keep_frac of
      Dementia SUBJECTS in train+val; test is untouched (natural).
    - For each train/val episode with class c, draw pattern p ~ P_lambda(p|c).
    """
    # Dementia subject keep mask
    dementia_rids = set(int(r) for r in ep_df.index[ep_df["y"] == 2]["RID"])
    keep_dementia_rids = set(rng.choice(list(dementia_rids),
                                       size=max(1, int(round(len(dementia_rids) * keep_frac))),
                                       replace=False))
    out: list[MaskedEpisode] = []
    for eid, row in ep_df.iterrows():
        rid = int(row["RID"])
        y = int(row["y"])
        # Every subject appears in test naturally; we only mask train/val episodes.
        # Here we just produce masked episodes for all subjects (caller separates train/test).
        p = sample_pattern_for_class(y, lambda_, design, rng)
        out.append(MaskedEpisode(
            episode_id=int(eid),
            rid=rid,
            y=y,
            target_pattern=p,
            base_mods=tuple(base_mods),
        ))
    return out


# ---------------------------------------------------------------------------
# run one (lambda, keep_frac) x method x model x fold x repeat
# ---------------------------------------------------------------------------

def run_dial_setting(
    setting: DialSetting,
    model: str,
    method: str,
    design: DialDesign,
    base_mods: Sequence[str],
    train_ep_df: pd.DataFrame,
    val_ep_df: pd.DataFrame,
    test_ep_df: pd.DataFrame,
    train_rids: np.ndarray,
    val_rids: np.ndarray,
    test_rids: np.ndarray,
    spec: dict[str, list[str]],
    blocks: Mapping[str, pd.DataFrame],
    hide_mask: bool = False,
    tau_grid: Sequence[float] = (0, 0.25, 0.5, 1.0, 1.5),
    min_cell_n_val: int = 30,
    seed: int = 0,
    repeat: int = 0,
    fold: int = 0,
) -> DialResult:
    """Run one dial cell.

    Returns DialResult with T_matched / T_indep / T_full metrics.
    """
    rng = np.random.default_rng(seed + setting.lambda_ * 1000 + setting.keep_frac * 100 + repeat * 10 + fold)

    # Masked train episodes
    masked_train = build_dial_train_episodes(train_ep_df, design, setting.lambda_,
                                              setting.keep_frac, rng, base_mods)
    masked_val = build_dial_train_episodes(val_ep_df, design, setting.lambda_,
                                            1.0, rng, base_mods)  # val uses same lambda for T_matched

    # Build train/val/test feature matrices
    # For MLP/LGBM: concatenate features; mask absent modalities per target_pattern.
    def build_Xy(ep_df_iter, masked_list, is_train: bool):
        X_list: list[pd.DataFrame] = []
        y_list: list[int] = []
        pattern_list: list[str] = []
        for me in masked_list:
            row = ep_df_iter.loc[me.episode_id]
            y_list.append(me.y)
            pattern_list.append(me.target_pattern)
            feats = mask_blocks_for_pattern(row, me.target_pattern, base_mods, blocks, spec)
            # encode per modality: present -> feats; absent -> NaN
            # Here we use a simplified representation: for each modality in base_mods,
            # if present, take spec features; else NaN.
            parts: dict[str, Any] = {}
            for k in base_mods:
                f = feats.get(k)
                if f is None:
                    parts[f"{k}_missing"] = np.nan
                else:
                    for j, v in enumerate(f):
                        parts[f"{k}_{j}"] = v
            # mask bits
            for k in base_mods:
                parts[f"mask_{k}"] = 1 if me.target_pattern[base_mods.index(k)] == "1" else 0
            if not hide_mask:
                pass  # mask bits already added
            else:
                for k in base_mods:
                    parts.pop(f"mask_{k}", None)
            X_list.append(pd.DataFrame(parts, index=[me.episode_id]))
        X = pd.concat(X_list, ignore_index=True) if X_list else pd.DataFrame()
        y = pd.Series(y_list, index=X.index)
        pattern = pd.Series(pattern_list, index=X.index, dtype=str)
        return X, y, pattern

    train_X, train_y, train_pattern = build_Xy(train_ep_df, masked_train, True)
    val_X, val_y, val_pattern = build_Xy(val_ep_df, masked_val, False)
    # Test: use natural pattern for T_matched? No: T_matched uses lambda-same pattern,
    # T_indep uses lambda=0, T_full uses 111.
    # For test, we DON'T change the subject set; we re-mask the test episodes.
    def build_test_Xy(ep_df_iter, pattern_func):
        X_list: list[pd.DataFrame] = []
        y_list: list[int] = []
        pattern_list: list[str] = []
        for eid, row in ep_df_iter.iterrows():
            y_list.append(int(row["y"]))
            p = pattern_func(int(row["y"]))
            pattern_list.append(p)
            feats = mask_blocks_for_pattern(row, p, base_mods, blocks, spec)
            parts: dict[str, Any] = {}
            for k in base_mods:
                f = feats.get(k)
                if f is None:
                    parts[f"{k}_missing"] = np.nan
                else:
                    for j, v in enumerate(f):
                        parts[f"{k}_{j}"] = v
            for k in base_mods:
                parts[f"mask_{k}"] = 1 if p[base_mods.index(k)] == "1" else 0
            if hide_mask:
                for k in base_mods:
                    parts.pop(f"mask_{k}", None)
            X_list.append(pd.DataFrame(parts, index=[eid]))
        X = pd.concat(X_list, ignore_index=True)
        y = pd.Series(y_list, index=X.index)
        pattern = pd.Series(pattern_list, index=X.index, dtype=str)
        return X, y, pattern

    test_nat = test_ep_df.copy()
    # T_matched: sample from P_lambda
    p_match = lambda y: sample_pattern_for_class(y, setting.lambda_, design,
                                                 np.random.default_rng(seed + 9999))
    test_X_match, test_y_match, test_pat_match = build_test_Xy(test_nat, p_match)
    # T_indep: lambda=0
    p_indep = lambda y: sample_pattern_for_class(y, 0.0, design,
                                                  np.random.default_rng(seed + 8888))
    test_X_indep, test_y_indep, test_pat_indep = build_test_Xy(test_nat, p_indep)
    # T_full: 111
    p_full = lambda y: "111"
    test_X_full, test_y_full, test_pat_full = build_test_Xy(test_nat, p_full)

    # Choose method hooks
    class_w = class_weights(train_y.to_numpy(dtype=int)) if method == "M2" else None
    joint_w = joint_weights(train_y.to_numpy(dtype=int),
                            train_pattern.to_numpy(dtype=str)) if method == "M3" else None
    cell_prior = CellPrior.from_counts(train_y.to_numpy(dtype=int),
                                       train_pattern.to_numpy(dtype=str)) if method in ("M4", "M4c") else None
    tau = None
    use_class_only = method == "M4c"

    val_proba = None
    if model == "LGBM":
        # LightGBM doesn't use class/joint weights in this impl (apply later in metrics)
        # For M2 weight, we'd need sample_weight in lightgbm Dataset; skip for now.
        val_proba, test_proba_match, best_iter = fit_lgbm(train_X, train_y, val_X, val_y,
                                                          test_X_match, LGBMConfig(seed=seed))
        _, test_proba_indep, _ = fit_lgbm(train_X, train_y, val_X, val_y,
                                           test_X_indep, LGBMConfig(seed=seed))
        _, test_proba_full, _ = fit_lgbm(train_X, train_y, val_X, val_y,
                                          test_X_full, LGBMConfig(seed=seed))
    else:
        # MLP
        feat_dims = {k: len(spec.get(k, [])) for k in base_mods}
        cfg = MLPConfig(seed=seed)
        model_obj, info = fit_mlp(cfg, feat_dims,
                                  {k: train_X[k].to_numpy() for k in base_mods},
                                  train_X[[f"mask_{k}" for k in base_mods]].to_numpy().T
                                  if not hide_mask else np.zeros((len(train_X), 0)),
                                  train_y.to_numpy(dtype=int),
                                  {k: val_X[k].to_numpy() for k in base_mods},
                                  val_X[[f"mask_{k}" for k in base_mods]].to_numpy().T
                                  if not hide_mask else np.zeros((len(val_X), 0)),
                                  val_y.to_numpy(dtype=int),
                                  use_mask=not hide_mask,
                                  class_weights=class_w,
                                  joint_weights=joint_w)
        val_proba = predict_mlp(model_obj,
                                 {k: val_X[k].to_numpy() for k in base_mods},
                                 val_X[[f"mask_{k}" for k in base_mods]].to_numpy().T
                                 if not hide_mask else np.zeros((len(val_X), 0)),
                                 use_mask=not hide_mask)
        # choose tau on validation if M4/M4c
        if cell_prior is not None:
            meta_val = val_proba  # MLP outputs softmax; logits would be log(proba)
            # use log-prob as meta_logit
            meta_logit = np.log(np.clip(val_proba, 1e-12, 1.0))
            tau = choose_tau(meta_logit, val_y.to_numpy(dtype=int),
                             val_pattern.to_numpy(dtype=str), cell_prior,
                             tau_grid, min_cell_n=min_cell_n_val)
        else:
            meta_logit_val = None
        # T_matched
        probs_match = predict_mlp(model_obj,
                                  {k: test_X_match[k].to_numpy() for k in base_mods},
                                  test_X_match[[f"mask_{k}" for k in base_mods]].to_numpy().T
                                  if not hide_mask else np.zeros((len(test_X_match), 0)),
                                  use_mask=not hide_mask)
        test_proba_match = probs_match
        if cell_prior is not None and tau is not None:
            meta = np.log(np.clip(test_proba_match, 1e-12, 1.0))
            test_proba_match = np.exp(adjust_logits(meta, cell_prior, tau,
                                                    test_pat_match.to_numpy(dtype=str),
                                                    use_class_only=use_class_only))
        # T_indep
        probs_indep = predict_mlp(model_obj,
                                  {k: test_X_indep[k].to_numpy() for k in base_mods},
                                  test_X_indep[[f"mask_{k}" for k in base_mods]].to_numpy().T
                                  if not hide_mask else np.zeros((len(test_X_indep), 0)),
                                  use_mask=not hide_mask)
        test_proba_indep = probs_indep
        if cell_prior is not None and tau is not None:
            meta = np.log(np.clip(test_proba_indep, 1e-12, 1.0))
            test_proba_indep = np.exp(adjust_logits(meta, cell_prior, tau,
                                                     test_pat_indep.to_numpy(dtype=str),
                                                     use_class_only=use_class_only))
        # T_full
        probs_full = predict_mlp(model_obj,
                                 {k: test_X_full[k].to_numpy() for k in base_mods},
                                 test_X_full[[f"mask_{k}" for k in base_mods]].to_numpy().T
                                 if not hide_mask else np.zeros((len(test_X_full), 0)),
                                 use_mask=not hide_mask)
        test_proba_full = probs_full
        if cell_prior is not None and tau is not None:
            meta = np.log(np.clip(test_proba_full, 1e-12, 1.0))
            test_proba_full = np.exp(adjust_logits(meta, cell_prior, tau,
                                                    test_pat_full.to_numpy(dtype=str),
                                                    use_class_only=use_class_only))

    return DialResult(
        setting=setting,
        model=model,
        method=method,
        repeat=repeat,
        fold=fold,
        seed=seed,
        T_matched=overall_metrics(test_y_match.to_numpy(dtype=int),
                                  test_proba_match,
                                  test_pat_match.to_numpy(dtype=str)),
        T_indep=overall_metrics(test_y_indep.to_numpy(dtype=int),
                                test_proba_indep,
                                test_pat_indep.to_numpy(dtype=str)),
        T_full=overall_metrics(test_y_full.to_numpy(dtype=int),
                               test_proba_full,
                               test_pat_full.to_numpy(dtype=str)),
        tau=tau,
    )


def run_dial_grid(
    design: DialDesign,
    base_mods: Sequence[str],
    lam_grid: Sequence[float],
    keep_frac_grid: Sequence[float],
    train_ep_df: pd.DataFrame,
    val_ep_df: pd.DataFrame,
    test_ep_df: pd.DataFrame,
    train_rids: np.ndarray,
    val_rids: np.ndarray,
    test_rids: np.ndarray,
    spec: dict[str, list[str]],
    blocks: Mapping[str, pd.DataFrame],
    methods: Sequence[str],
    models: Sequence[str],
    hide_mask: bool = False,
    tau_grid: Sequence[float] = (0, 0.25, 0.5, 1.0, 1.5),
    min_cell_n_val: int = 30,
    seed: int = 0,
    repeat: int = 0,
    fold: int = 0,
) -> list[DialResult]:
    results: list[DialResult] = []
    for lam in lam_grid:
        for kf in keep_frac_grid:
            setting = DialSetting(lambda_=float(lam), keep_frac=float(kf))
            for model in models:
                model_methods = methods
                if model == "LGBM" and "M5" in model_methods:
                    model_methods = [m for m in model_methods if m != "M5"]
                for method in model_methods:
                    r = run_dial_setting(
                        setting, model, method, design, base_mods,
                        train_ep_df, val_ep_df, test_ep_df,
                        train_rids, val_rids, test_rids,
                        spec, blocks, hide_mask=hide_mask,
                        tau_grid=tau_grid, min_cell_n_val=min_cell_n_val,
                        seed=seed, repeat=repeat, fold=fold,
                    )
                    results.append(r)
    return results


def pattern_response_matrix(
    model: str,
    method: str,
    design: DialDesign,
    base_mods: Sequence[str],
    train_ep_df: pd.DataFrame,
    test_ep_df: pd.DataFrame,
    train_rids: np.ndarray,
    test_rids: np.ndarray,
    spec: dict[str, list[str]],
    blocks: Mapping[str, pd.DataFrame],
    lambda_train: float,
    keep_frac: float = 1.0,
    seed: int = 0,
    repeat: int = 0,
    fold: int = 0,
    hide_mask: bool = False,
) -> pd.DataFrame:
    """Evaluate every test episode under EVERY imposed pattern; return accuracy by (class, imposed pattern).

    Headline figure 3.
    """
    rng = np.random.default_rng(seed + lambda_train * 1000 + repeat)
    masked_train = build_dial_train_episodes(train_ep_df, design, lambda_train,
                                              keep_frac, rng, base_mods)
    train_X, train_y, train_pattern = _build_Xy(masked_train, train_ep_df, spec, blocks,
                                                 base_mods, hide_mask)
    # Fit model
    if model == "LGBM":
        _, test_proba, _ = fit_lgbm(train_X, train_y, train_X.iloc[:0], train_y.iloc[:0],
                                     _build_test_X(test_ep_df, base_mods, blocks, spec,
                                                   hide_mask, lambda y: None))
        # placeholder; real impl fits on train and applies to each pattern
        raise NotImplementedError("pattern-response matrix for LGBM not wired yet")
    # MLP
    feat_dims = {k: len(spec.get(k, [])) for k in base_mods}
    cfg = MLPConfig(seed=seed)
    model_obj, _ = fit_mlp(cfg, feat_dims,
                           {k: train_X[k].to_numpy() for k in base_mods},
                           train_X[[f"mask_{k}" for k in base_mods]].to_numpy().T
                           if not hide_mask else np.zeros((len(train_X), 0)),
                           train_y.to_numpy(dtype=int),
                           {k: train_X[k].to_numpy() for k in base_mods},
                           train_X[[f"mask_{k}" for k in base_mods]].to_numpy().T
                           if not hide_mask else np.zeros((len(train_X), 0)),
                           train_y.to_numpy(dtype=int),
                           use_mask=not hide_mask)
    patterns = list(design.patterns) + ["111"]
    rows: list[dict[str, Any]] = []
    for p in patterns:
        test_X, test_y, test_pat = _build_test_X(test_ep_df, base_mods, blocks, spec,
                                                   hide_mask, lambda y, _p=p: _p)
        probs = predict_mlp(model_obj,
                            {k: test_X[k].to_numpy() for k in base_mods},
                            test_X[[f"mask_{k}" for k in base_mods]].to_numpy().T
                            if not hide_mask else np.zeros((len(test_X), 0)),
                            use_mask=not hide_mask)
        pred = probs.argmax(axis=1)
        y = test_y.to_numpy(dtype=int)
        for c in range(3):
            m = y == c
            rows.append({"class": CLASSES[c], "imposed_pattern": p,
                         "n": int(m.sum()),
                         "accuracy": float((pred[m] == c).mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def _build_Xy(masked_list, ep_df, spec, blocks, base_mods, hide_mask):
    X_list: list[pd.DataFrame] = []
    y_list: list[int] = []
    for me in masked_list:
        row = ep_df.loc[me.episode_id]
        y_list.append(me.y)
        feats = mask_blocks_for_pattern(row, me.target_pattern, base_mods, blocks, spec)
        parts: dict[str, Any] = {}
        for k in base_mods:
            f = feats.get(k)
            if f is None:
                parts[f"{k}_missing"] = np.nan
            else:
                for j, v in enumerate(f):
                    parts[f"{k}_{j}"] = v
        for k in base_mods:
            parts[f"mask_{k}"] = 1 if me.target_pattern[base_mods.index(k)] == "1" else 0
        if hide_mask:
            for k in base_mods:
                parts.pop(f"mask_{k}", None)
        X_list.append(pd.DataFrame(parts, index=[me.episode_id]))
    X = pd.concat(X_list, ignore_index=True)
    y = pd.Series(y_list, index=X.index)
    return X, y


def _build_test_X(ep_df, base_mods, blocks, spec, hide_mask, pattern_func):
    X_list: list[pd.DataFrame] = []
    y_list: list[int] = []
    for eid, row in ep_df.iterrows():
        y_list.append(int(row["y"]))
        p = pattern_func(int(row["y"]))
        feats = mask_blocks_for_pattern(row, p, base_mods, blocks, spec)
        parts: dict[str, Any] = {}
        for k in base_mods:
            f = feats.get(k)
            if f is None:
                parts[f"{k}_missing"] = np.nan
            else:
                for j, v in enumerate(f):
                    parts[f"{k}_{j}"] = v
        for k in base_mods:
            parts[f"mask_{k}"] = 1 if p[base_mods.index(k)] == "1" else 0
        if hide_mask:
            for k in base_mods:
                parts.pop(f"mask_{k}", None)
        X_list.append(pd.DataFrame(parts, index=[eid]))
    X = pd.concat(X_list, ignore_index=True)
    y = pd.Series(y_list, index=X.index)
    return X, y
