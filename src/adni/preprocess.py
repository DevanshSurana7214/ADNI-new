"""Stage 2 preprocessing (PIPELINE_v2 §3.2) as a fit/transform object.

Everything is fitted on the TRAINING episodes only (Standing Rule 4 — no leakage):

* Features missing in more than ``max_missing`` of the training rows (where the modality is
  present) are dropped.
* The kept features are z-scored with the training mean and SD (NaN ignored); SD is floored at
  ``sd_floor``; z-scores are winsorised to ``[-clip, clip]``.
* Within-block missing values become 0 and a per-feature binary missing-indicator is concatenated
  (input width ``2·d_k``).
* Block-level absence is the mask bit ``m_k`` from ``episodes.csv``; absent rows are all-zero and
  flagged absent (``m_k = 0``).

Outputs per block: a float32 array ``[N, 2·d_k]`` and a ``uint8`` mask ``[N]``.

Usage
-----
>>> pp = Preprocessor.from_files(episodes_path, blocks_dir, feature_spec_path)
>>> pp.fit(train_episode_ids)                       # e.g. an S1 fold's training ids
>>> X, m = pp.transform(episode_ids)                # dict[block] -> (array, mask)
>>> joblib.dump(pp, "results/step04/artifacts/seed_0/fold_0.joblib")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

MODALITIES: Sequence[str] = ("mri", "fdg", "amy", "tau")

# Defaults per PIPELINE_v2 §3.2 (kept as constructor args so configs can override; no magic numbers).
DEFAULT_MAX_MISSING = 0.20   # drop features missing in > 20 % of training rows
DEFAULT_SD_FLOOR = 1e-6      # SD floor before z-scoring
DEFAULT_CLIP = 5.0           # winsorise z-scores to [-clip, clip]


def _read_episodes(path: Path) -> pd.DataFrame:
    # `pattern` is an all-digit string; without dtype=str pandas would parse it as an integer.
    return pd.read_csv(path, dtype={"pattern": str}).set_index("episode_id")


def _read_block(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("episode_id")


class Preprocessor:
    """Fit/transform Stage 2 preprocessing for the four modality blocks."""

    def __init__(
        self,
        episodes: pd.DataFrame,
        blocks: dict[str, pd.DataFrame],
        feature_spec: dict[str, list[str]],
        max_missing: float = DEFAULT_MAX_MISSING,
        sd_floor: float = DEFAULT_SD_FLOOR,
        clip: float = DEFAULT_CLIP,
    ) -> None:
        if tuple(feature_spec) != tuple(MODALITIES):
            raise ValueError(f"feature_spec must have keys {MODALITIES}, got {tuple(feature_spec)}")
        if tuple(blocks) != tuple(MODALITIES):
            raise ValueError(f"blocks must have keys {MODALITIES}, got {tuple(blocks)}")
        self.max_missing = float(max_missing)
        self.sd_floor = float(sd_floor)
        self.clip = float(clip)

        # Keep only the episode columns we need, aligned to the block tables.
        self._mask = episodes[[f"m_{k}" for k in MODALITIES]].astype(np.uint8)
        self._blocks: dict[str, pd.DataFrame] = {}
        for k in MODALITIES:
            blk = blocks[k]
            missing_cols = [c for c in feature_spec[k] if c not in blk.columns]
            if missing_cols:
                raise KeyError(f"block {k!r} is missing feature columns: {missing_cols[:5]}")
            # Restrict to the declared feature list (drops any extra columns like field-strength
            # bookkeeping) and to the episode index of episodes.csv.
            self._blocks[k] = blk.reindex(columns=feature_spec[k])

        self._fitted = False

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_files(
        cls,
        episodes_path: Path,
        blocks_dir: Path,
        feature_spec_path: Path,
        **kwargs: Any,
    ) -> "Preprocessor":
        ep = _read_episodes(episodes_path)
        blocks = {k: _read_block(blocks_dir / f"{k}.csv") for k in MODALITIES}
        spec = json.loads(Path(feature_spec_path).read_text())
        return cls(ep, blocks, spec, **kwargs)

    # ------------------------------------------------------------------ fitting
    def fit(self, train_episode_ids: Iterable[int]) -> "Preprocessor":
        """Fit feature filtering + z-score statistics on the TRAINING episodes only.

        For each block, only training episodes where the modality is *present* (``m_k == 1``)
        contribute: a feature is kept unless it is missing in more than ``max_missing`` of those
        rows, and the mean/SD are computed over the kept features ignoring NaN.
        """
        ids = self._resolve_ids(train_episode_ids)
        self.kept: dict[str, list[str]] = {}
        self.dropped: dict[str, list[str]] = {}
        self.mean_: dict[str, np.ndarray] = {}
        self.sd_: dict[str, np.ndarray] = {}

        for k in MODALITIES:
            m = self._mask.loc[ids, f"m_{k}"].to_numpy()
            present = ids[m == 1]
            feats = list(self._blocks[k].columns)
            block = self._blocks[k]
            # A present episode must have a row in the block table (Step 1 presence invariant).
            if not set(present).issubset(set(block.index)):
                missing = sorted(set(present) - set(block.index))[:5]
                raise KeyError(f"block {k!r} has no row for present episodes {missing}")

            vals = block.loc[present, feats].to_numpy(dtype=np.float64)
            nan_frac = np.isnan(vals).mean(axis=0) if len(vals) else np.ones(len(feats))
            keep_mask = nan_frac <= self.max_missing
            kept = [f for f, kp in zip(feats, keep_mask) if kp]
            dropped = [f for f, kp in zip(feats, keep_mask) if not kp]
            if not kept:
                raise ValueError(f"block {k!r}: the >{self.max_missing:.0%} missing filter dropped every feature")

            kept_vals = vals[:, keep_mask]
            with np.errstate(invalid="ignore"):
                mean = np.nanmean(kept_vals, axis=0)
                sd = np.nanstd(kept_vals, axis=0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > self.sd_floor), sd, self.sd_floor)

            self.kept[k] = kept
            self.dropped[k] = dropped
            self.mean_[k] = mean.astype(np.float64)
            self.sd_[k] = sd.astype(np.float64)

        self._fitted = True
        self.n_train_episodes_ = int(len(ids))
        return self

    # ------------------------------------------------------------------ transforming
    def transform(self, episode_ids: Iterable[int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Return ``{block: (X[N, 2·d_k] float32, mask[N] uint8)}``.

        Kept features are z-scored (NaN → 0) and clipped to ``[-clip, clip]``; the missing-indicator
        records which entries were NaN. Absent modalities produce all-zero rows with mask 0.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor.transform called before fit()")
        ids = self._resolve_ids(episode_ids)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for k in MODALITIES:
            d = len(self.kept[k])
            X = np.zeros((len(ids), 2 * d), dtype=np.float32)
            m = self._mask.loc[ids, f"m_{k}"].to_numpy(dtype=np.uint8)
            present = ids[m == 1]
            if len(present):
                raw = self._blocks[k].loc[present, self.kept[k]].to_numpy(dtype=np.float64)
                nan_mask = np.isnan(raw)
                z = (raw - self.mean_[k]) / self.sd_[k]
                z = np.nan_to_num(z, nan=0.0, posinf=self.clip, neginf=-self.clip)
                z = np.clip(z, -self.clip, self.clip)
                # Write present rows back in place, preserving the input id order.
                pos = np.searchsorted(ids, present)  # ids is sorted by _resolve_ids
                X[pos, :d] = z.astype(np.float32)
                X[pos, d:] = nan_mask.astype(np.float32)
            out[k] = (X, m)
        return out

    def fit_transform(self, train_episode_ids: Iterable[int]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self.fit(train_episode_ids)
        return self.transform(train_episode_ids)

    # ------------------------------------------------------------------ helpers
    def _resolve_ids(self, episode_ids: Iterable[int]) -> np.ndarray:
        """Validate ids, drop duplicates, and return a sorted int64 array (needed for searchsorted)."""
        arr = np.asarray(list(episode_ids))
        if arr.dtype == object:
            arr = arr.astype(np.int64)
        else:
            arr = arr.astype(np.int64)
        arr = np.unique(arr)
        unknown = np.setdiff1d(arr, self._mask.index.to_numpy(), assume_unique=False)
        if len(unknown):
            raise KeyError(f"unknown episode_ids: {unknown[:5].tolist()}")
        return arr

    # ------------------------------------------------------------------ reporting helpers
    def kept_counts(self) -> dict[str, int]:
        self._check_fitted()
        return {k: len(self.kept[k]) for k in MODALITIES}

    def raw_z(self, block: str, episode_ids: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return the UNCLIPPED z-scores and the present mask for *block* over *episode_ids*.

        NaN entries (missing values on present rows) are preserved. Used for reporting the
        fraction of z-scores that the ±clip winsorisation actually changes.
        """
        self._check_fitted()
        ids = self._resolve_ids(episode_ids)
        m = self._mask.loc[ids, f"m_{block}"].to_numpy(dtype=np.uint8)
        present = ids[m == 1]
        z = np.full((len(ids), len(self.kept[block])), np.nan, dtype=np.float64)
        if len(present):
            raw = self._blocks[block].loc[present, self.kept[block]].to_numpy(dtype=np.float64)
            pos = np.searchsorted(ids, present)
            z[pos] = (raw - self.mean_[block]) / self.sd_[block]
        return z, m

    def clip_fraction(self, block: str, episode_ids: Iterable[int]) -> tuple[int, int]:
        """(# raw z-scores with |z| > clip, # finite raw z-scores) over present rows."""
        z, _ = self.raw_z(block, episode_ids)
        finite = np.isfinite(z)
        n_finite = int(finite.sum())
        n_clipped = int((np.abs(z[finite]) > self.clip).sum())
        return n_clipped, n_finite

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")


def save_fitted(pp: Preprocessor, path: Path) -> None:
    """Save a fitted Preprocessor with joblib, creating parent directories as needed."""
    import joblib

    pp._check_fitted()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pp, path)
