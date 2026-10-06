"""Shortcut diagnostics and controls (PIPELINE_v2 section 8.4, section 9 C1).

Pure helpers used by the Step 10 runner scripts; kept separate from ``train.py`` so they can be
unit-tested without a GPU.

  * ``permute_labels_within`` - the placebo C1 prior: permute the TRAINING labels within each
    enrolment-protocol group, so the class-pattern association the joint prior relies on is
    destroyed while the per-protocol class marginal is kept.
  * ``tv_change`` - total-variation distance between two predicted-probability vectors.
  * ``choose_donors`` - pick, for each episode, a random OTHER episode from a donor pool
    (the content-swap control).
  * ``remove_modality`` / ``swap_modality`` - the two counterfactual interventions on a
    :class:`~adni.train.Tensors`-style feature dict + mask.

Nothing here trains or scores a model.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def permute_labels_within(labels: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    """Permute ``labels`` independently inside each distinct value of ``groups`` (deterministic).

    Returns a new array aligned to the input rows. The multiset of labels inside every group is
    preserved (so the per-group class marginal is unchanged), but the label assigned to any single
    episode is decoupled from that episode's row - including its availability pattern.
    """
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.shape[0] != groups.shape[0]:
        raise ValueError("labels and groups must have the same length")
    out = labels.copy()
    rng = np.random.default_rng(int(seed))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        out[idx] = labels[rng.permutation(idx)]
    return out


def tv_change(p_orig: np.ndarray, p_new: np.ndarray) -> np.ndarray:
    """Total-variation distance per row: ``0.5 * sum_c |p_new - p_orig|`` (0 = no change, 1 = max)."""
    p_orig = np.asarray(p_orig, dtype=float)
    p_new = np.asarray(p_new, dtype=float)
    if p_orig.shape != p_new.shape:
        raise ValueError(f"shape mismatch {p_orig.shape} vs {p_new.shape}")
    return 0.5 * np.abs(p_new - p_orig).sum(axis=-1)


def choose_donors(self_rows: np.ndarray, pool: np.ndarray, seed: int) -> np.ndarray:
    """For each row id in ``self_rows`` choose a random OTHER row id from ``pool``.

    ``self_rows`` must be a subset of ``pool``. The donor is ``!=`` the row itself whenever the pool
    has another element (a collision from the uniform draw is wrapped to the next pool element); if
    the pool contains only the row itself it is returned unchanged. Deterministic given ``seed``.
    """
    self_rows = np.asarray(self_rows)
    pool = np.asarray(pool)
    m = pool.shape[0]
    if m == 0:
        raise ValueError("donor pool is empty")
    rng = np.random.default_rng(int(seed))
    sel = rng.integers(0, m, size=self_rows.shape[0])
    donors = pool[sel]
    collide = donors == self_rows
    if bool(np.any(collide)) and m > 1:
        donors = np.where(collide, pool[(sel + 1) % m], donors)
    return donors


def remove_modality(
    feats: dict[str, torch.Tensor],
    mask: torch.Tensor,
    modality: str,
    rows: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Intervention (a): set the mask bit of *modality* to 0 for *rows*, keeping feature values.

    Feature values are left untouched (they are ignored once the mask bit is 0); a copy of the
    feature dict and mask is returned.
    """
    idx = _modality_index(modality)
    new_feats = {k: v.clone() for k, v in feats.items()}
    new_mask = mask.clone()
    new_mask[rows, idx] = False
    return new_feats, new_mask


def swap_modality(
    feats: dict[str, torch.Tensor],
    mask: torch.Tensor,
    modality: str,
    rows: torch.Tensor,
    donors: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Intervention (b): replace *modality*'s feature rows with donor rows, keeping the mask.

    ``feats[modality][rows] = feats[modality][donors]``; the mask (and every other modality) is
    unchanged, so the episode still reports the modality as observed but carries another subject's
    content.
    """
    new_feats = {k: v.clone() for k, v in feats.items()}
    new_mask = mask.clone()
    new_feats[modality][rows] = feats[modality][donors]
    return new_feats, new_mask


MODALITIES: Sequence[str] = ("mri", "fdg", "amy", "tau")


def _modality_index(modality: str) -> int:
    try:
        return list(MODALITIES).index(modality)
    except ValueError as exc:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown modality {modality!r}; expected one of {MODALITIES}") from exc
