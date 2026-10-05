"""Loss arms A0-A4 and the mask-process priors they use (PIPELINE_v2 section 6, Appendix B).

Every arm shares one model, trainer, optimiser and early-stopping rule; the arm is a config switch.
All priors are fitted on the TRAINING split only and always use the EFFECTIVE mask after modality
dropout. The exact enumeration over the (<= 15) sub-masks is in ``adni.priors.submask_kernel``.

  A0  cross-entropy.
  A1  logit-adjusted CE with log P(c) (class only).
  A2  CE reweighted by 1 / P(m_tilde), clipped to [0.2, 5], renormalised to mean 1 per batch.
  A3  CE reweighted by rho(c) / pi(c | m_tilde) (rho uniform), clipped [0.2, 5], mean-1 per batch.
  A4  logit-adjusted CE: softmax(s_c + tau * log pi(c | m_tilde)), tau = 1.

Inference: A0/A2/A3 use argmax s_c. A1 and A4 are reported at two operating points,
argmax(s_c + log rho(c)) with rho uniform (primary) and rho = P(c) (secondary). Under Appendix B2
the pattern adjustment is absorbed into s_c, so the pattern term does not appear at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .priors import (
    CLASSES,
    class_prior,
    pattern_marginal,
    pi_table,
    smoothed_joint,
)

ARMS: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")
WMIN, WMAX = 0.2, 5.0


def mask_to_pattern(mask) -> list[str]:
    """Convert a mask (torch tensor or array, [B, len(MODALITIES)]) to pattern strings.

    Bit order is the MODALITIES order, matching the ``pattern`` column of ``docs/episodes.csv``.
    """
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    if arr.ndim == 1:
        arr = arr[None, :]
    return ["".join("1" if b else "0" for b in row) for row in arr.astype(bool)]


@dataclass
class Priors:
    """Immutable prior tables for the loss arms, fitted on the TRAINING split only."""

    p_drop: float
    patterns: list[str]              # natural patterns (columns of P_cm)
    pi: pd.DataFrame                 # pi(c | m_tilde), index CLASSES, columns = sub-masks
    pattern_marginal: pd.Series      # P(m_tilde)
    class_prior: pd.Series           # P(c)
    log_pi: np.ndarray               # (n_sub, 3) log pi(c | sub)
    log_pc: np.ndarray               # (3,) log P(c)
    pc: np.ndarray                   # (3,) P(c)
    pm: np.ndarray                   # (n_sub,) P(sub)
    index: dict[str, int]            # sub-mask -> row of log_pi / pm

    @property
    def submasks(self) -> list[str]:
        return list(self.pi.columns)


def build_priors(P_cm, patterns, p_drop: float) -> Priors:
    """Build the loss-arm tables from a smoothed joint P(c,m) (training split) and ``p_drop``."""
    pats = [str(p) for p in patterns]
    pi = pi_table(P_cm, p_drop, pats)
    pm = pattern_marginal(P_cm, p_drop, pats)
    cp = class_prior(P_cm)
    subs = list(pi.columns)
    log_pi = np.log(pi.to_numpy(dtype=float).T)      # (n_sub, 3)
    log_pc = np.log(cp.to_numpy(dtype=float))
    return Priors(
        p_drop=float(p_drop), patterns=pats, pi=pi, pattern_marginal=pm, class_prior=cp,
        log_pi=log_pi, log_pc=log_pc, pc=cp.to_numpy(dtype=float),
        pm=pm.to_numpy(dtype=float), index={s: i for i, s in enumerate(subs)},
    )


def priors_from_counts(n_cm, patterns, p_drop: float, alpha: float = 0.5) -> Priors:
    """Build the tables from a class-by-pattern COUNT table (training split), smoothed with alpha."""
    P_cm, _, _, _ = smoothed_joint(np.asarray(n_cm, dtype=float), alpha=alpha)
    return build_priors(P_cm, patterns, p_drop)


def _sub_index(patterns, priors: Priors, device) -> torch.Tensor:
    return torch.as_tensor([priors.index[p] for p in patterns], dtype=torch.long, device=device)


def adjusted_logits(logits: torch.Tensor, patterns, priors: Priors, tau: float = 1.0) -> torch.Tensor:
    """A4 training logits ``s_c + tau * log pi(c | m_tilde)`` (tau = 0 reproduces the raw logits)."""
    if tau == 0.0:
        return logits
    idx = _sub_index(patterns, priors, logits.device)
    log_pi = torch.tensor(priors.log_pi, dtype=logits.dtype, device=logits.device)   # (S, 3)
    return logits + float(tau) * log_pi[idx]


def pattern_weights(patterns, priors: Priors, device) -> torch.Tensor:
    """A2 weights ``1 / P(m_tilde)``, clipped to [0.2, 5] and renormalised to mean 1 per batch."""
    idx = _sub_index(patterns, priors, device)
    pm = torch.tensor(priors.pm, dtype=torch.float32, device=device)
    w = torch.clamp(1.0 / pm[idx], WMIN, WMAX)
    return w / w.mean()


def joint_weights(patterns, y: torch.Tensor, priors: Priors, rho=None, device=None) -> torch.Tensor:
    """A3 weights ``rho(c) / pi(c | m_tilde)`` (rho uniform unless an explicit (3,) rho is passed),
    clipped to [0.2, 5] and renormalised to mean 1 per batch."""
    device = device if device is not None else y.device
    idx = _sub_index(patterns, priors, device)
    pi_t = torch.tensor(priors.pi.to_numpy(dtype=float), dtype=torch.float32, device=device)  # (3, S)
    pi_sel = pi_t[y, idx]                                  # pi(c_i | m_i)
    if rho is None:
        rho_y = torch.full_like(pi_sel, 1.0 / len(CLASSES))
    else:
        r = torch.as_tensor(rho, dtype=torch.float32, device=device)
        rho_y = r[y]
    w = torch.clamp(rho_y / pi_sel, WMIN, WMAX)
    return w / w.mean()


def arm_loss(
    arm: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    patterns,
    priors: Priors | None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Training loss for one batch under *arm*; *patterns* is the effective mask per row (strings)."""
    if arm == "A0":
        return F.cross_entropy(logits, y)
    if priors is None:
        raise ValueError(f"arm {arm} needs priors fitted on the training split")
    if arm == "A1":
        log_pc = torch.tensor(priors.log_pc, dtype=logits.dtype, device=logits.device)
        return F.cross_entropy(logits + log_pc[None, :], y)
    if arm == "A2":
        w = pattern_weights(patterns, priors, logits.device)
        return (w * F.cross_entropy(logits, y, reduction="none")).mean()
    if arm == "A3":
        w = joint_weights(patterns, y, priors, device=logits.device)
        return (w * F.cross_entropy(logits, y, reduction="none")).mean()
    if arm == "A4":
        return F.cross_entropy(adjusted_logits(logits, patterns, priors, tau), y)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def inference_scores(logits: torch.Tensor, priors: Priors | None, rho: str = "uniform") -> torch.Tensor:
    """Scores whose argmax defines the prediction at the requested operating point.

    ``rho="uniform"`` -> ``s_c`` (pattern-neutral / class-balanced, primary). ``rho="class_prior"``
    -> ``s_c + log P(c)`` (natural class prevalence, secondary). Used by every arm.
    """
    if rho in ("uniform", None):
        return logits
    if rho == "class_prior":
        if priors is None:
            raise ValueError("rho='class_prior' needs priors")
        log_pc = torch.tensor(priors.log_pc, dtype=logits.dtype, device=logits.device)
        return logits + log_pc[None, :]
    raise ValueError(f"unknown rho {rho!r}")


def infer_proba(logits: torch.Tensor, priors: Priors | None, rho: str = "uniform") -> torch.Tensor:
    """Softmax probabilities at the requested operating point (for mask-sensitivity diagnostics)."""
    return torch.softmax(inference_scores(logits, priors, rho), dim=1)
