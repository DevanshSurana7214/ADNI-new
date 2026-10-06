"""Step 10 diagnostics/controls tests (PIPELINE_v2 section 8.4, section 9 C1/C-shuf).

These exercise the pure helpers added for Step 10 on tiny synthetic inputs; no cohort data and no
GPU are required:

  * ``permute_labels_within`` (placebo C1 prior),
  * ``tv_change`` / ``choose_donors`` / ``remove_modality`` / ``swap_modality`` (mask intervention),
  * ``shuffle_weights`` + ``arm_loss(..., shuffle_a3=True)`` and a through-the-entry-point C-shuf run.
"""
from __future__ import annotations

import numpy as np
import torch

from adni.diagnostics import (
    choose_donors,
    permute_labels_within,
    remove_modality,
    swap_modality,
    tv_change,
)
from adni.losses import arm_loss, priors_from_counts, shuffle_weights
from adni.models import MODALITIES
from adni.train import Tensors, TrainConfig, train_arm

DIMS = {"mri": 3, "fdg": 2, "amy": 2, "tau": 2}
PATTERNS = ["1111", "1100", "1000"]
MODEL_KW = {"d_model": 16, "enc_hidden": 16, "ff_dim": 16, "n_heads": 4, "n_layers": 1,
            "enc_dropout": 0.0, "attn_dropout": 0.0, "modality_dropout": 0.25}


# --------------------------------------------------------------------------- label permutation
def test_permute_labels_within_preserves_each_group_multiset() -> None:
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    groups = np.array(["ADNI1"] * 5 + ["ADNI2"] * 5)
    out = permute_labels_within(labels, groups, seed=0)
    for g in np.unique(groups):
        m = groups == g
        assert sorted(out[m].tolist()) == sorted(labels[m].tolist())
    assert sorted(out.tolist()) == sorted(labels.tolist())


def test_permute_labels_within_is_deterministic_and_actually_permutes() -> None:
    labels = np.arange(20) % 3
    groups = np.array(["A"] * 10 + ["B"] * 10)
    a = permute_labels_within(labels, groups, seed=7)
    b = permute_labels_within(labels, groups, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, labels), "permutation should move at least one label"


def test_permute_labels_within_rejects_length_mismatch() -> None:
    try:
        permute_labels_within(np.arange(3), np.array(["A"] * 2), seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


# --------------------------------------------------------------------------- total variation
def test_tv_change_bounds_and_symmetry() -> None:
    p = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
    assert np.allclose(tv_change(p, p), 0.0)
    q = np.array([[0.5, 0.25, 0.25], [0.0, 0.0, 1.0]])
    tv = tv_change(p, q)
    assert np.allclose(tv, tv_change(q, p))
    assert np.all(tv >= -1e-12) and np.all(tv <= 1.0 + 1e-12)
    # disjoint distributions -> TV 1
    assert abs(tv_change(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0] - 1.0) < 1e-12


# --------------------------------------------------------------------------- donor selection
def test_choose_donors_never_picks_self_and_is_deterministic() -> None:
    rows = np.array([10, 11, 12])
    pool = np.array([10, 11, 12, 13, 14])
    a = choose_donors(rows, pool, seed=3)
    b = choose_donors(rows, pool, seed=3)
    assert np.array_equal(a, b)
    assert all(d != r for d, r in zip(a, rows))
    assert set(a.tolist()).issubset(set(pool.tolist()))


def test_choose_donors_falls_back_to_self_when_pool_is_singleton() -> None:
    rows = np.array([5, 5])
    got = choose_donors(rows, np.array([5]), seed=0)
    assert np.array_equal(got, rows)


# --------------------------------------------------------------------------- tensor interventions
def _toy():
    n = 4
    feats = {k: torch.arange(n * 2 * DIMS[k], dtype=torch.float32).reshape(n, 2 * DIMS[k])
             for k in MODALITIES}
    mask = torch.ones(n, len(MODALITIES), dtype=torch.bool)
    return feats, mask


def test_remove_modality_clears_bit_keeps_feats_and_input() -> None:
    feats, mask = _toy()
    feats0 = {k: v.clone() for k, v in feats.items()}
    mask0 = mask.clone()
    rows = torch.tensor([0, 2])
    nf, nm = remove_modality(feats, mask, "fdg", rows)
    k = list(MODALITIES).index("fdg")
    assert not bool(nm[rows, k].any())
    assert bool(nm[[1, 3], k].all())
    for kk in MODALITIES:
        assert torch.equal(nf[kk], feats0[kk])
    # originals untouched
    assert torch.equal(mask, mask0)
    assert all(torch.equal(feats[kk], feats0[kk]) for kk in MODALITIES)


def test_swap_modality_copies_donor_rows_keeps_mask_and_input() -> None:
    feats, mask = _toy()
    feats0 = {k: v.clone() for k, v in feats.items()}
    mask0 = mask.clone()
    rows = torch.tensor([0, 1])
    donors = torch.tensor([3, 2])
    nf, nm = swap_modality(feats, mask, "amy", rows, donors)
    assert torch.equal(nm, mask0)
    assert torch.equal(nf["amy"][rows], feats0["amy"][donors])
    # other modalities untouched
    for kk in ("mri", "fdg", "tau"):
        assert torch.equal(nf[kk], feats0[kk])
    assert all(torch.equal(feats[kk], feats0[kk]) for kk in MODALITIES)


# --------------------------------------------------------------------------- C-shuf weights
def test_shuffle_weights_is_a_permutation_and_deterministic_with_generator() -> None:
    w = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    g = torch.Generator().manual_seed(0)
    a = shuffle_weights(w, generator=g)
    g2 = torch.Generator().manual_seed(0)
    b = shuffle_weights(w, generator=g2)
    assert torch.equal(a, b)
    assert sorted(a.tolist()) == sorted(w.tolist())
    assert a.shape == w.shape


def test_arm_loss_shuffle_a3_is_finite_and_can_differ() -> None:
    patterns = ["1100"] * 6
    logits = torch.randn(6, 3)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    counts = np.array([[4.0, 2.0], [2.0, 4.0], [1.0, 1.0]])
    priors = priors_from_counts(counts, ["1000", "1100"], p_drop=0.25)
    base = arm_loss("A3", logits, y, patterns, priors, 1.0, shuffle_a3=False)
    assert torch.isfinite(base)
    # across many shuffles the loss must move at least once (weights differ per row)
    varied = False
    for _ in range(10):
        s = arm_loss("A3", logits, y, patterns, priors, 1.0, shuffle_a3=True)
        assert torch.isfinite(s)
        if abs(float(s) - float(base)) > 1e-9:
            varied = True
    assert varied, "shuffling the A3 weights never changed the batch loss"


def test_cshuf_runs_through_train_arm_and_tracks_both_rules() -> None:
    rng = np.random.default_rng(0)
    n = 120
    # only mri (always) and fdg (random) can be observed, so every dropout sub-mask is one of
    # {1000, 1100, 0100} - exactly the sub-masks the two-pattern prior below covers.
    mask = torch.zeros(n, 4, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, 1] = torch.as_tensor(rng.random(n) > 0.5)
    y = torch.as_tensor(rng.choice(3, size=n))
    feats = {k: torch.randn(n, 2 * DIMS[k]) for k in MODALITIES}
    counts = np.stack([np.bincount(rng.choice(3, size=n), minlength=3) for _ in range(2)], axis=1).astype(float)
    priors = priors_from_counts(counts, ["1000", "1100"], p_drop=0.25)
    ntr = int(n * 0.8)
    tr = Tensors({k: v[:ntr] for k, v in feats.items()}, mask[:ntr], y[:ntr])
    va = Tensors({k: v[ntr:] for k, v in feats.items()}, mask[ntr:], y[ntr:])
    cfg = TrainConfig(lr=1e-3, weight_decay=0.0, batch_size=32, max_epochs=2, patience=10 ** 9,
                      seed=0, device="cpu")
    _, res, _ = train_arm("A3", DIMS, tr, va, cfg, MODEL_KW, priors, np.ones(len(va)),
                          shuffle_a3=True)
    assert res["epochs_run"] == 2
    assert all(np.isfinite(h["train_loss"]) and np.isfinite(h["val_loss"])
               for h in res["history"])
