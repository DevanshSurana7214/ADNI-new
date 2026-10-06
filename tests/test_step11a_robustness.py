"""Step 11a robustness tests: the clip override, the tau plumbing and the run-set resolution.

These exercise the new Step 11a code on tiny synthetic inputs (no cohort data, no GPU):

  * ``losses.joint_weights(..., clip=...)`` and ``arm_loss(..., clip=...)`` (A3 weight clip),
  * ``TrainConfig.tau`` / ``TrainConfig.clip`` threaded through ``train_arm`` into the batch loss,
  * ``run_step11a.build_runsets`` (the pre-registered recipe/sensitivity override plumbing) and the
    real ``configs/step11a_robustness.yaml``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from adni.losses import WMAX, WMIN, arm_loss, joint_weights, priors_from_counts
from adni.models import MODALITIES
from adni.train import Tensors, TrainConfig, train_arm

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_step11a", ROOT / "scripts" / "run_step11a.py")
run11 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run11)  # type: ignore[union-attr]

DIMS = {"mri": 3, "fdg": 2, "amy": 2, "tau": 2}
MODEL_KW = {"d_model": 16, "enc_hidden": 16, "ff_dim": 16, "n_heads": 4, "n_layers": 1,
            "enc_dropout": 0.0, "attn_dropout": 0.0, "modality_dropout": 0.25}
# an extreme class x pattern table so the raw A3 weights are far outside a tight clip
EXTREME_COUNTS = np.array([[1.0, 900.0], [900.0, 1.0], [1.0, 1.0]])
PATTERNS = ["1000", "1100"]


# --------------------------------------------------------------------------- clip on the A3 weights
def test_joint_weights_custom_clip_matches_manual_computation() -> None:
    priors = priors_from_counts(EXTREME_COUNTS, PATTERNS, p_drop=0.25)
    patterns = ["1100", "1000", "1100", "1000"]
    y = np.array([0, 0, 1, 1])
    got = joint_weights(patterns, torch.as_tensor(y), priors, clip=(0.5, 2.0)).numpy()

    pi = priors.pi.to_numpy()
    idx = [priors.index[p] for p in patterns]
    pi_sel = pi[y, idx]
    raw = np.clip((1.0 / 3.0) / pi_sel, 0.5, 2.0)
    expected = raw / raw.mean()
    assert np.allclose(got, expected, atol=1e-6)


def test_joint_weights_default_clip_equals_the_module_constants() -> None:
    priors = priors_from_counts(EXTREME_COUNTS, PATTERNS, p_drop=0.25)
    patterns = ["1100", "1000"] * 5
    y = torch.as_tensor([0, 1] * 5)
    default = joint_weights(patterns, y, priors, clip=None)
    explicit = joint_weights(patterns, y, priors, clip=(WMIN, WMAX))
    assert torch.allclose(default, explicit)


def test_tight_clip_reduces_the_weight_spread() -> None:
    priors = priors_from_counts(EXTREME_COUNTS, PATTERNS, p_drop=0.25)
    patterns = ["1100"] * 12
    y = torch.as_tensor([0, 1, 2] * 4)
    wide = joint_weights(patterns, y, priors, clip=(0.1, 10.0))
    tight = joint_weights(patterns, y, priors, clip=(0.5, 2.0))
    # both are mean-1; the tight one must have the smaller max/min ratio
    assert float(tight.max() / tight.min()) < float(wide.max() / wide.min())


def test_joint_weights_rejects_inverted_clip() -> None:
    priors = priors_from_counts(EXTREME_COUNTS, PATTERNS, p_drop=0.25)
    try:
        joint_weights(["1100"], torch.as_tensor([0]), priors, clip=(2.0, 0.5))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wmin > wmax")


def test_arm_loss_a3_clip_changes_the_batch_loss() -> None:
    priors = priors_from_counts(EXTREME_COUNTS, PATTERNS, p_drop=0.25)
    patterns = ["1100"] * 6
    logits = torch.randn(6, 3, generator=torch.Generator().manual_seed(0))
    y = torch.as_tensor([0, 1, 2, 0, 1, 2])
    wide = arm_loss("A3", logits, y, patterns, priors, 1.0, clip=(0.1, 10.0))
    tight = arm_loss("A3", logits, y, patterns, priors, 1.0, clip=(0.5, 2.0))
    assert torch.isfinite(wide) and torch.isfinite(tight)
    assert abs(float(wide) - float(tight)) > 1e-6


# --------------------------------------------------------------------------- plumbing through training
def _toy(n: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, len(MODALITIES)), dtype=bool)
    masks[:, 0] = True
    masks[:, 1] = rng.random(n) > 0.5
    mask = torch.as_tensor(masks)
    y = torch.as_tensor(rng.integers(0, 3, size=n))
    g = torch.Generator().manual_seed(seed)
    feats = {k: torch.randn(n, 2 * DIMS[k], generator=g) for k in MODALITIES}
    priors = priors_from_counts(EXTREME_COUNTS, PATTERNS, p_drop=0.25)
    ntr = int(n * 0.8)
    tr = Tensors({k: v[:ntr] for k, v in feats.items()}, mask[:ntr], y[:ntr])
    va = Tensors({k: v[ntr:] for k, v in feats.items()}, mask[ntr:], y[ntr:])
    return tr, va, priors


def _cfg(**kw):
    base = dict(lr=1e-3, weight_decay=0.0, batch_size=32, max_epochs=3, patience=10 ** 9,
                seed=0, device="cpu")
    base.update(kw)
    return TrainConfig(**base)


def test_train_arm_clip_is_threaded_and_changes_a3_training() -> None:
    tr, va, priors = _toy()
    w = np.ones(len(va))
    _, wide, _ = train_arm("A3", DIMS, tr, va, _cfg(clip=(0.1, 10.0)), MODEL_KW, priors, w)
    _, tight, _ = train_arm("A3", DIMS, tr, va, _cfg(clip=(0.5, 2.0)), MODEL_KW, priors, w)
    assert all(np.isfinite(h["train_loss"]) for h in wide["history"])
    assert all(np.isfinite(h["train_loss"]) for h in tight["history"])
    assert wide["history"] != tight["history"], "the clip override did not reach the A3 loss"


def test_train_arm_tau_zero_reproduces_a0() -> None:
    tr, va, priors = _toy()
    w = np.ones(len(va))
    _, a0, _ = train_arm("A0", DIMS, tr, va, _cfg(), MODEL_KW, None, w)
    _, a4, _ = train_arm("A4", DIMS, tr, va, _cfg(tau=0.0), MODEL_KW, priors, w)
    assert a0["history"] == a4["history"], "A4 with tau=0 must be identical to A0"


# --------------------------------------------------------------------------- run-set resolution
def _base_cfg() -> dict:
    return {"d_model": 128, "enc_hidden": 256, "enc_dropout": 0.2, "n_heads": 4, "n_layers": 2,
            "ff_dim": 256, "attn_dropout": 0.1, "n_classes": 3, "modality_dropout": 0.25,
            "lr": 1.0e-04, "weight_decay": 0.3, "batch_size": 256, "max_epochs": 100,
            "patience": 15, "seed": 0, "deterministic": True}


def test_build_runsets_applies_recipe_2_and_sensitivity_overrides() -> None:
    spec = {
        "recipe_2": {"lr": 3.0e-05, "weight_decay": 0.3, "batch_size": 128, "enc_dropout": 0.4,
                     "attn_dropout": 0.2, "max_epochs": 150, "patience": 25},
        "sensitivity": [
            {"name": "A4_tau0p5", "arm": "A4", "tau": 0.5},
            {"name": "A3_clip0p5_2", "arm": "A3", "wmin": 0.5, "wmax": 2.0},
            {"name": "A0_pdrop0", "arm": "A0", "modality_dropout": 0.0},
        ],
    }
    rs = run11.build_runsets(_base_cfg(), spec)
    assert set(rs) == {"R2_A0", "R2_A1", "R2_A4", "A4_tau0p5", "A3_clip0p5_2", "A0_pdrop0"}

    r2 = rs["R2_A4"]
    assert r2["arm"] == "A4" and r2["tau"] == 1.0 and r2["clip"] is None
    assert r2["cfg"]["lr"] == 3.0e-05 and r2["cfg"]["batch_size"] == 128
    assert r2["cfg"]["enc_dropout"] == 0.4 and r2["cfg"]["attn_dropout"] == 0.2
    assert r2["cfg"]["max_epochs"] == 150 and r2["cfg"]["patience"] == 25
    assert r2["cfg"]["weight_decay"] == 0.3 and r2["cfg"]["modality_dropout"] == 0.25

    assert rs["A4_tau0p5"]["tau"] == 0.5 and rs["A4_tau0p5"]["clip"] is None
    assert rs["A4_tau0p5"]["cfg"]["lr"] == 1.0e-04, "sensitivity must stay at recipe 1"
    assert rs["A3_clip0p5_2"]["clip"] == (0.5, 2.0) and rs["A3_clip0p5_2"]["primary_op"] == "argmax"
    assert rs["A0_pdrop0"]["cfg"]["modality_dropout"] == 0.0
    # every cfg keeps recipe-1 values it does not override
    for name, r in rs.items():
        assert r["cfg"]["d_model"] == 128 and r["cfg"]["n_layers"] == 2, name


def test_build_runsets_rejects_half_a_clip_and_duplicate_names() -> None:
    base = _base_cfg()
    for spec in ({"sensitivity": [{"name": "x", "arm": "A3", "wmin": 0.5}]},
                 {"sensitivity": [{"name": "x", "arm": "A3"}, {"name": "x", "arm": "A4"}]}):
        try:
            run11.build_runsets(base, spec)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_real_robustness_yaml_parses_to_the_pre_registered_grid() -> None:
    base = yaml.safe_load((ROOT / "configs" / "model.yaml").read_text())
    spec = yaml.safe_load((ROOT / "configs" / "step11a_robustness.yaml").read_text())
    rs = run11.build_runsets(base, spec)
    expected = {"R2_A0", "R2_A1", "R2_A4", "A4_tau0p5", "A4_tau1p5", "A3_clip0p5_2",
                "A3_clip0p1_10", "A0_pdrop0", "A0_pdrop0p5", "A4_pdrop0", "A4_pdrop0p5"}
    assert set(rs) == expected
    assert rs["R2_A0"]["cfg"]["lr"] == 3.0e-05
    assert rs["A3_clip0p1_10"]["clip"] == (0.1, 10.0)
    assert rs["A4_pdrop0p5"]["cfg"]["modality_dropout"] == 0.5
    assert rs["A4_pdrop0p5"]["arm"] == "A4"
    assert all(rs[n]["cfg"]["deterministic"] for n in rs)
