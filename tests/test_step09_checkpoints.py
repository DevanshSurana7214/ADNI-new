"""Step 9 checkpoint-rule tests: both validation rules tracked in one run.

Pre-declared protocol being tested (results/step09/REPORT.md):

  * PRIMARY checkpoint rule = best validation macro-F1 (drives early stopping / patience).
  * SECONDARY checkpoint rule = best validation PN balanced accuracy (rho uniform, weights from the
    fold's TRAIN-split pi_hat). It is tracked every epoch but never stops training.
  * Validation macro-F1 uses the argmax of the RAW logits ``s_c`` for every arm A0-A4; a uniform rho
    adds a constant to all classes, so ``argmax(s + log rho_uniform) == argmax(s)``.

The tests use a tiny synthetic dataset; they do not touch the real cohort.
"""
from __future__ import annotations

import numpy as np
import torch

from adni.losses import ARMS, inference_scores, priors_from_counts
from adni.metrics import pn_balanced_accuracy
from adni.models import MODALITIES
from adni.train import Tensors, TrainConfig, train_arm

DIMS = {"mri": 3, "fdg": 2, "amy": 2, "tau": 2}
PATTERNS = ["1111", "1100", "1000"]
MODEL_KW = {"d_model": 16, "enc_hidden": 16, "ff_dim": 16, "n_heads": 4, "n_layers": 1,
            "enc_dropout": 0.0, "attn_dropout": 0.0, "modality_dropout": 0.25}


def _make(n_per: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    masks, ys, js = [], [], []
    for j, p in enumerate(PATTERNS):
        masks.extend([[int(b) for b in p]] * n_per)
        ys.append(rng.choice(3, size=n_per, p=[0.5, 0.3, 0.2]))
        js.append(np.full(n_per, j))
    mask = torch.as_tensor(np.asarray(masks, dtype=bool))
    y = torch.as_tensor(np.concatenate(ys))
    jidx = np.concatenate(js)
    n = len(y)
    g = torch.Generator().manual_seed(seed)
    feats = {k: torch.randn(n, 2 * DIMS[k], generator=g) for k in MODALITIES}
    counts = np.stack([np.bincount(y.numpy()[jidx == j], minlength=3)
                       for j in range(len(PATTERNS))], axis=1).astype(float)
    priors = priors_from_counts(counts, PATTERNS, p_drop=0.25)
    perm = torch.as_tensor(rng.permutation(n))
    feats = {k: v[perm] for k, v in feats.items()}
    return feats, mask[perm], y[perm], priors


def _split(feats, mask, y, frac=0.8):
    n = len(y)
    ntr = int(n * frac)
    tr = Tensors({k: v[:ntr] for k, v in feats.items()}, mask[:ntr], y[:ntr])
    va = Tensors({k: v[ntr:] for k, v in feats.items()}, mask[ntr:], y[ntr:])
    return tr, va


def _cfg(max_epochs=8, patience=10 ** 9):
    return TrainConfig(lr=1e-3, weight_decay=0.0, batch_size=64, max_epochs=max_epochs,
                       patience=patience, seed=0, device="cpu")


def test_both_rules_tracked_and_consistent() -> None:
    feats, mask, y, priors = _make()
    tr, va = _split(feats, mask, y)
    w = np.ones(len(va))          # uniform weights -> PN-BAcc == plain balanced accuracy
    _, res, _ = train_arm("A0", DIMS, tr, va, _cfg(), MODEL_KW, priors, w)

    hist = res["history"]
    assert all(np.isfinite(h["val_pn_bacc"]) for h in hist)
    # PRIMARY = best (history) macro-F1; SECONDARY = best (history) PN-BAcc.
    f1s = [h["val_macro_f1"] for h in hist]
    pns = [h["val_pn_bacc"] for h in hist]
    assert abs(res["best"]["macro_f1"] - max(f1s)) < 1e-12
    assert res["best"]["epoch"] == int(np.argmax(f1s))
    assert abs(res["best_pn_bacc"]["pn_bacc"] - max(pns)) < 1e-12
    assert res["best_pn_bacc"]["epoch"] == int(np.argmax(pns))
    # both states are kept and are valid state dicts
    assert set(res["best_state"]) and set(res["best_pn_bacc_state"])
    assert res["best_state"].keys() == res["best_pn_bacc_state"].keys()


def test_no_val_weights_gives_none_secondary() -> None:
    feats, mask, y, priors = _make()
    tr, va = _split(feats, mask, y)
    _, res, _ = train_arm("A0", DIMS, tr, va, _cfg(max_epochs=3), MODEL_KW, priors, None)
    # No val_weights -> PN-BAcc is never computed. The history must carry a NaN-free sentinel
    # (None), so that two identical histories compare equal (NaN != NaN).
    assert all(h["val_pn_bacc"] is None for h in res["history"])
    assert res["best_pn_bacc"]["epoch"] == -1


def test_primary_rule_drives_early_stopping() -> None:
    feats, mask, y, priors = _make()
    tr, va = _split(feats, mask, y)
    _, res, _ = train_arm("A0", DIMS, tr, va, _cfg(max_epochs=40, patience=3), MODEL_KW, priors,
                          np.ones(len(va)))
    assert res["epochs_run"] < 40, "patience-3 run must stop before max_epochs"


def test_uniform_operating_point_is_raw_logits_for_every_arm() -> None:
    feats, mask, y, priors = _make()
    tr, va = _split(feats, mask, y)
    for arm in ARMS:
        _, res, _ = train_arm(arm, DIMS, tr, va, _cfg(max_epochs=2), MODEL_KW, priors, np.ones(len(va)))
        assert res["history"][0]["val_pn_bacc"] >= 0.0
    # inference_scores(logits, priors, "uniform") is the identity; its argmax == argmax(raw logits)
    logits = torch.randn(7, 3)
    out = inference_scores(logits, priors, "uniform")
    assert torch.equal(out, logits)
    assert torch.equal(out.argmax(dim=1), logits.argmax(dim=1))


def test_pn_bacc_with_uniform_weights_equals_balanced_accuracy() -> None:
    rng = np.random.default_rng(0)
    y = rng.choice(3, size=500, p=[0.5, 0.3, 0.2])
    pred = rng.choice(3, size=500)
    from sklearn.metrics import balanced_accuracy_score

    assert abs(pn_balanced_accuracy(y, pred, np.ones(500)) - balanced_accuracy_score(y, pred)) < 1e-12


def test_two_state_tracking_is_deterministic() -> None:
    feats, mask, y, priors = _make()
    tr, va = _split(feats, mask, y)
    w = np.ones(len(va))
    runs = []
    for _ in range(2):
        _, res, _ = train_arm("A4", DIMS, tr, va, _cfg(), MODEL_KW, priors, w)
        runs.append(res)
    assert runs[0]["history"] == runs[1]["history"]
    assert runs[0]["best"] == runs[1]["best"]
    assert runs[0]["best_pn_bacc"] == runs[1]["best_pn_bacc"]
