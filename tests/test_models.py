"""Model + trainer tests (playbook Step 7).

Architecture tests operate on tiny feature dimensions and CPU only. They cover:
  1. absent-modality invariance (changing an absent modality's input leaves the logits bit-identical);
  2. present-modality sensitivity;
  3. all 15 patterns, batch size 1 and a mixed batch, without error;
  4. the model can overfit a fixed batch of 64 episodes (loss < 0.05 within 300 steps);
  5. determinism: two seeded runs give identical validation loss (CPU) or within 1e-5 (GPU);
  6. gradients reach every encoder when all modalities are present and are exactly zero for an
     entirely absent encoder;
  7. modality dropout never returns an empty mask and its empirical keep rate matches 1 - p;
  8. ``train_a0`` is reproducible *without* any external seeding (it seeds before building the
     model), which is how the pilot and every later arm call it.
"""
from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

from adni.models import MODALITIES, FusionModel, ModelConfig
from adni.train import Tensors, TrainConfig, Trainer, set_seed, train_a0

DIMS = {"mri": 3, "fdg": 2, "amy": 2, "tau": 2}


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(feat_dims=DIMS, **kw)


def _model(**kw) -> FusionModel:
    return FusionModel(_cfg(**kw))


def _feats(n: int, seed: int = 0, scale: float = 1.0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {k: torch.randn(n, 2 * DIMS[k], generator=g) * scale for k in MODALITIES}


def _mask(n: int, bits: tuple[int, int, int, int]) -> torch.Tensor:
    return torch.tensor([bits] * n, dtype=torch.bool)


# --------------------------------------------------------------------------- 1
def test_absent_modality_invariance():
    model = _model().eval()
    mask = _mask(5, (0, 1, 1, 1))          # mri absent
    f1 = _feats(5, seed=1)
    f2 = {k: v.clone() for k, v in f1.items()}
    f2["mri"] = torch.randn(5, 2 * DIMS["mri"])   # change the ABSENT modality only
    with torch.no_grad():
        out1, _, _ = model(f1, mask)
        out2, _, _ = model(f2, mask)
    assert torch.equal(out1, out2), "changing an absent modality changed the logits"


# --------------------------------------------------------------------------- 2
def test_present_modality_sensitivity():
    model = _model().eval()
    mask = _mask(5, (1, 1, 1, 1))
    f1 = _feats(5, seed=2)
    f2 = {k: v.clone() for k, v in f1.items()}
    # a constant offset is LayerNorm-invariant, so perturb with a fresh random draw
    f2["mri"] = torch.randn(5, 2 * DIMS["mri"], generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        out1, _, _ = model(f1, mask)
        out2, _, _ = model(f2, mask)
    assert not torch.allclose(out1, out2), "changing a present modality did not change logits"


# --------------------------------------------------------------------------- 3
def test_all_patterns_batch_size_one_and_mixed():
    model = _model().eval()
    patterns = [p for p in itertools.product((0, 1), repeat=4) if any(p)]
    assert len(patterns) == 15
    for p in patterns:
        feats = {k: v[:1] for k, v in _feats(1, seed=3).items()}
        with torch.no_grad():
            logits, h, eff = model(feats, _mask(1, p))
        assert logits.shape == (1, 3) and h.shape == (1, 128)
        assert torch.equal(eff, _mask(1, p))
    # mixed batch: one row per pattern
    n = len(patterns)
    g = torch.Generator().manual_seed(4)
    feats = {k: torch.randn(n, 2 * DIMS[k], generator=g) for k in MODALITIES}
    mask = torch.tensor(patterns, dtype=torch.bool)
    with torch.no_grad():
        logits, _, _ = model(feats, mask)
    assert logits.shape == (15, 3)


# --------------------------------------------------------------------------- 4
def test_overfit_fixed_batch():
    set_seed(0)
    model = _model()
    n = 64
    feats = _feats(n, seed=5)
    mask = _mask(n, (1, 1, 1, 1))
    y = torch.arange(n) % 3
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    loss_val = float("inf")
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        logits, _, _ = model(feats, mask)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        loss_val = float(loss.item())
        if loss_val < 0.05:
            break
    assert loss_val < 0.05, f"loss did not fall below 0.05 (got {loss_val:.4f})"


# --------------------------------------------------------------------------- 5
def test_determinism_val_loss_epoch3():
    n_tr, n_va = 128, 64
    tr = Tensors(_feats(n_tr, seed=6), _mask(n_tr, (1, 1, 1, 1)), torch.arange(n_tr) % 3)
    va = Tensors(_feats(n_va, seed=7), _mask(n_va, (1, 1, 1, 1)), torch.arange(n_va) % 3)
    hist = []
    for _ in range(2):
        set_seed(0)
        model = _model()
        cfg = TrainConfig(seed=0, max_epochs=3, batch_size=64, patience=10 ** 9, device="cpu")
        res = Trainer(model, cfg).fit(tr, va)
        hist.append(res["history"][2]["val_loss"])
    # the trainer is forced to CPU here, so the two runs must be bit-identical
    assert hist[0] == hist[1], (hist[0], hist[1])


# --------------------------------------------------------------------------- 6
def test_gradients_reach_present_encoders_absent_zero():
    model = _model()
    # all present -> every encoder gets a non-zero gradient
    feats = _feats(8, seed=8)
    mask = _mask(8, (1, 1, 1, 1))
    logits, _, _ = model(feats, mask)
    nn.functional.cross_entropy(logits, torch.arange(8) % 3).backward()
    for k in MODALITIES:
        grads = [p.grad for p in model.encoders[k].parameters()]
        assert all(g is not None for g in grads)
        assert max(float(g.abs().max()) for g in grads) > 0, f"encoder {k} got no gradient"

    # tau absent for every row -> its encoder gets exactly zero gradient
    model.zero_grad(set_to_none=True)
    mask2 = _mask(8, (1, 1, 1, 0))
    logits, _, _ = model(feats, mask2)
    nn.functional.cross_entropy(logits, torch.arange(8) % 3).backward()
    tau_grads = [p.grad for p in model.encoders["tau"].parameters()]
    assert all(g is None or float(g.abs().max()) == 0.0 for g in tau_grads), "absent encoder got gradient"


# --------------------------------------------------------------------------- 7
def test_modality_dropout_nonempty_and_rate():
    model = _model(modality_dropout=0.25)
    n = 30000
    mask = _mask(n, (1, 1, 1, 1))
    gen = torch.Generator().manual_seed(0)
    eff = model.modality_dropout(mask, generator=gen)
    assert torch.equal(eff, eff & mask), "dropout turned on an unobserved modality"
    assert bool(eff.any(dim=1).all()), "modality dropout produced an empty mask"
    rate = float(eff.float().mean())
    # conditioned on non-empty, q=0.75 gives E[keep] = 3 / (1 - 0.25**4) / 4 ~ 0.7529
    assert abs(rate - 0.75) < 0.02, f"empirical keep rate {rate:.4f} not within 0.02 of 0.75"


# --------------------------------------------------------------------------- 8
def test_train_a0_reproducible_without_external_seeding():
    """``train_a0`` must fix its own seed before building the model (no external set_seed)."""
    n_tr, n_va = 96, 48
    tr = Tensors(_feats(n_tr, seed=11), _mask(n_tr, (1, 1, 1, 1)), torch.arange(n_tr) % 3)
    va = Tensors(_feats(n_va, seed=12), _mask(n_va, (1, 1, 1, 1)), torch.arange(n_va) % 3)
    cfg = TrainConfig(seed=0, max_epochs=3, batch_size=32, patience=10 ** 9, device="cpu")
    runs = []
    for _ in range(2):
        model, res, _ = train_a0(DIMS, tr, va, cfg)
        runs.append((res["history"], {k: v.clone() for k, v in model.state_dict().items()}))
    assert runs[0][0] == runs[1][0], "train_a0 produced different training histories"
    for k, v in runs[0][1].items():
        assert torch.equal(v, runs[1][1][k]), f"train_a0 produced different weights for {k}"


# --------------------------------------------------------------------------- 9
def test_large_batch_inference_chunk_consistent():
    """A large single batch must not crash, and chunking must give the same predictions."""
    model = _model().eval()
    n, chunk = 8192, 1024
    feats = _feats(n, seed=13)
    mask = _mask(n, (1, 1, 1, 1))
    with torch.no_grad():
        full, _, _ = model(feats, mask)
        chunked = torch.cat([
            model({k: v[s:s + chunk] for k, v in feats.items()}, mask[s:s + chunk])[0]
            for s in range(0, n, chunk)
        ], dim=0)
    assert full.shape == (n, 3)
    assert torch.equal(full.argmax(1), chunked.argmax(1)), "chunking changed a prediction"
    assert torch.allclose(full, chunked, atol=1e-5), "chunked logits differ beyond float32 round-off"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
