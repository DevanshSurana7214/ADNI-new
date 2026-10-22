"""Preprocessing for PIPELINE v6 (section 3.2-3.3).

Per modality, fit on TRAIN rows that have the modality:
  - drop features with >50% missing on TRAIN modality rows
  - median impute
  - z-score (mean, std from TRAIN modality rows)
  - clip to [-5, 5]

Missing-modality encoding (section 3.3):
  MLP  : zeros for the whole block + explicit mask bit per modality; encoder output
         for an absent modality is multiplied by 0.
  LGBM : NaN for the whole block + mask bits as extra features.

`hide_mask` removes the explicit mask bits (used for the step-4B-style ablation
and for M0 which has no imaging features).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

MODALITIES: tuple[str, ...] = ("mri", "fdg", "amy", "tau")
CLASSES: tuple[str, ...] = ("CN", "MCI", "Dementia")


# ---------------------------------------------------------------------------
# per-modality scaler/imputer state
# ---------------------------------------------------------------------------

@dataclass
class ModalityPreproc:
    """Fitted per-modality preprocessing state.  All statistics from TRAIN modality rows only."""

    feature_names: tuple[str, ...]
    medians: np.ndarray           # shape (d_after_drop,)
    stds: np.ndarray             # shape (d_after_drop,); 1 where std is 0
    mask: np.ndarray             # bool, which features were kept (after drop>50% missing)
    all_feature_names: tuple[str, ...]  # original feature names from spec (for reference)


@dataclass
class Preprocessor:
    """Holds the unfitted spec + fitted modality states.  Fitted once per (modality, fold)."""

    spec: dict[str, list[str]]
    states: dict[str, ModalityPreproc] = field(default_factory=dict)

    def fit(self, blocks: Mapping[str, pd.DataFrame], train_episode_ids: np.ndarray,
            modality_present: pd.Series, drop_missing_frac: float = 0.5) -> "Preprocessor":
        """Fit per-modality state on TRAIN modality rows only."""
        for k in MODALITIES:
            self.states[k] = self._fit_modality(
                blocks[k], train_episode_ids, modality_present, drop_missing_frac
            )
        return self

    def _fit_modality(self, block: pd.DataFrame, train_episode_ids: np.ndarray,
                      modality_present: pd.Series, drop_missing_frac: float) -> ModalityPreproc:
        # rows that are in TRAIN and have the modality
        present_mask = modality_present.to_numpy(dtype=bool)
        ids_present = np.asarray([int(i) for i in train_episode_ids
                                 if i in block.index and present_mask[list(block.index).index(i)]
                                 ] if False else
                                 [int(i) for i in train_episode_ids if i in block.index
                                  and modality_present.get(i, False)])
        if not ids_present:
            # no training rows with this modality -> keep all features, medians=0, stds=1
            feat_names = tuple(self.spec.get(k, []))
            return ModalityPreproc(
                feature_names=feat_names,
                medians=np.zeros(len(feat_names)),
                stds=np.ones(len(feat_names)),
                mask=np.ones(len(feat_names), dtype=bool),
                all_feature_names=feat_names,
            )
        sub = block.loc[ids_present]
        feat_names = tuple(self.spec.get(k, []))
        # keep only features present in both spec and block
        spec_set = set(feat_names)
        cols = [c for c in sub.columns if c in spec_set]
        sub = sub[cols]
        arr = sub.to_numpy(dtype=float)
        n_missing = np.isnan(arr).sum(axis=0)
        n_rows = arr.shape[0]
        keep = (n_missing / n_rows) <= (1.0 - drop_missing_frac)
        kept_cols = [c for c, kk in zip(cols, keep) if kk]
        sub_kept = sub[kept_cols]
        arr_kept = sub_kept.to_numpy(dtype=float)
        medians = np.nanmedian(arr_kept, axis=0)
        stds = np.nanstd(arr_kept, axis=0, ddof=0)
        stds = np.where(stds > 0, stds, 1.0)
        idx = list(sub_kept.columns)
        return ModalityPreproc(
            feature_names=tuple(idx),
            medians=medians,
            stds=stds,
            mask=np.array([c in set(kept_cols) for c in cols], dtype=bool),
            all_feature_names=feat_names,
        )

    def transform(self, episode_ids: np.ndarray, blocks: Mapping[str, pd.DataFrame],
                  modality_present: pd.Series, hide_mask: bool = False) -> dict[str, Any]:
        """Return per-modality (X, mask) for the requested episode_ids.

        X[k] is [N, d_k] (z-scored, median-imputed, clipped).
        mask[k] is [N,] bool indicating whether the modality is present *in the data*.
        """
        ids = np.asarray([int(i) for i in episode_ids], dtype=np.int64)
        ids_set = set(int(i) for i in ids)
        present_in_ep = modality_present.to_numpy(dtype=bool)
        present_ids = np.asarray([int(i) for i in modality_present.index
                                 if present_in_ep[list(modality_present.index).index(i)]
                                 ] if False else
                                 [int(i) for i in modality_present.index
                                  if modality_present.get(i, False) and int(i) in ids_set])
        N = len(ids)
        out: dict[str, Any] = {}
        for k in MODALITIES:
            state = self.states[k]
            feat_names = list(state.feature_names)
            X = np.zeros((N, len(feat_names)), dtype=np.float32)
            mask_k = np.zeros(N, dtype=bool)
            if feat_names:
                col_idx = {c: j for j, c in enumerate(feat_names)}
                for ii, eid in enumerate(ids):
                    if eid in block.index and modality_present.get(eid, False):
                        row = block.loc[eid]
                        vals = np.full(len(feat_names), np.nan, dtype=float)
                        for c in feat_names:
                            if c in row.index:
                                v = row[c]
                                if pd.notna(v):
                                    vals[col_idx[c]] = float(v)
                        # median impute
                        vals = np.where(np.isnan(vals), state.medians, vals)
                        # z-score
                        vals = (vals - state.medians) / state.stds
                        # clip
                        vals = np.clip(vals, -5.0, 5.0)
                        X[ii] = vals.astype(np.float32)
                        mask_k[ii] = True
            out[k] = (X.astype(np.float32), mask_k)
        # build mask bits dataframe (per modality)
        masks_df = pd.DataFrame(
            {f"mask_{k}": out[k][1].astype(np.int8) for k in MODALITIES},
            index=ids,
        )
        out["masks"] = masks_df
        if hide_mask:
            out["masks"] = pd.DataFrame(index=ids)
        return out


def load_modality_present(ep_df: pd.DataFrame) -> pd.Series:
    """Series indexed by episode_id: True if at least one modality present (for block lookup)."""
    return ep_df[["m_mri", "m_fdg", "m_amym", "m_tau"]].any(axis=1) if False else \
        ep_df[[f"m_{k}" for k in MODALITIES]].any(axis=1)


def build_feature_spec_from_blocks(blocks: Mapping[str, pd.DataFrame],
                                   spec_path: Path | None = None) -> dict[str, list[str]]:
    """Load spec from file if given, else use all block columns except episode_id."""
    if spec_path is not None and spec_path.exists():
        return json.loads(spec_path.read_text())
    out: dict[str, list[str]] = {}
    for k, df in blocks.items():
        out[k] = [c for c in df.columns if c != "episode_id"]
    return out


def combine_features(preproc_out: dict[str, Any], hide_mask: bool = False) -> pd.DataFrame:
    """Combine per-modality X into one DataFrame indexed by episode_id.

    Columns are prefixed by modality. Mask bits are appended (unless hide_mask).
    """
    ids = [int(i) for i in preproc_out[MODALITIES[0]][0].__array_interface__["data"]] if False else \
        list(range(len(preproc_out[MODALITIES[0]][0])))
    # infer ids from masks index
    ids = list(map(int, preproc_out["masks"].index))
    parts: dict[str, np.ndarray] = {}
    for k in MODALITIES:
        X, _ = preproc_out[k]
        for j in range(X.shape[1]):
            parts[f"{k}_{j}"] = X[:, j]
    X = pd.DataFrame(parts, index=ids)
    if not hide_mask:
        for k in MODALITIES:
            X[f"mask_{k}"] = preproc_out["masks"][f"mask_{k}"].to_numpy(dtype=np.int8)
    return X
