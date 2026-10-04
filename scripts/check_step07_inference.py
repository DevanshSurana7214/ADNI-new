"""Step 7 close-out: large-batch inference check for the restored FusionModel.

Evaluates all episodes (13,240) with the masked-attention fusion model (a) in a single forward
call and (b) in chunks of 1024, and verifies the logits are identical. The restored model keeps a
token sequence of at most 1 + n_modalities (<= 5) per row, so the batch dimension is the only
thing that scales; this check confirms a 13,240-row batch neither segfaults nor changes the output.

Writes ``results/step07fix/inference_largebatch.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from adni.models import FusionModel, ModelConfig
from adni.preprocess import Preprocessor
from adni.train import make_tensors, set_seed

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, default=ROOT / "docs" / "episodes.csv")
    ap.add_argument("--blocks", type=Path, default=ROOT / "results/step01/stage1/blocks")
    ap.add_argument("--feature-spec", type=Path, default=ROOT / "results/step01/stage1/feature_spec.json")
    ap.add_argument("--splits", type=Path, default=ROOT / "results/step03/splits.json")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/model.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results/step07fix/inference_largebatch.json")
    ap.add_argument("--chunk", type=int, default=1024)
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    ep = pd.read_csv(a.episodes, dtype={"pattern": str}).set_index("episode_id")
    blob = json.loads(Path(a.splits).read_text())
    tr = blob["s1"]["seed_0"]["0"]["episode_ids"]["train"]

    pp = Preprocessor.from_files(a.episodes, a.blocks, a.feature_spec)
    pp.fit(tr)
    all_ids = ep.index.to_numpy()
    data = make_tensors(pp, all_ids, ep)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {k: cfg[k] for k in ("d_model", "enc_hidden", "enc_dropout", "n_heads", "n_layers",
                              "ff_dim", "attn_dropout", "n_classes", "modality_dropout")}
    set_seed(cfg["seed"], cfg["deterministic"])
    model = FusionModel(ModelConfig(feat_dims=pp.kept_counts(), **kw)).eval().to(device)
    data = data.to(device)
    n = len(data)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    one_call_ok = True
    one_call_err = None
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_all, _, _ = model(data.feats, data.mask, apply_modality_dropout=False)
        t_one = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        one_call_ok = False
        one_call_err = repr(exc)
        logits_all = None
        t_one = float("nan")

    chunk = a.chunk
    parts = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n, chunk):
            sl = slice(start, min(start + chunk, n))
            feats = {k: v[sl] for k, v in data.feats.items()}
            out, _, _ = model(feats, data.mask[sl], apply_modality_dropout=False)
            parts.append(out)
    logits_chunk = torch.cat(parts, dim=0) if parts else None
    t_chunk = time.perf_counter() - t0

    peak_gpu = (torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
    if one_call_ok:
        diff = (logits_all - logits_chunk).abs()
        max_abs_diff = float(diff.max().item())
        mean_abs_diff = float(diff.mean().item())
        logit_abs_max = float(logits_all.abs().max().item())
        bitwise_identical = bool(torch.equal(logits_all, logits_chunk))
        preds_identical = bool((logits_all.argmax(1) == logits_chunk.argmax(1)).all())
    else:
        max_abs_diff = mean_abs_diff = logit_abs_max = None
        bitwise_identical = preds_identical = False

    # NOTE: bitwise equality across different batch sizes is not attainable on GPU because the
    # reduction order of float32 matmuls depends on the batch shape. The decision-relevant check is
    # that the predicted classes agree exactly and the logits agree to float32 round-off.
    report = {
        "device": device,
        "n_episodes": int(n),
        "chunk_size": int(chunk),
        "n_chunks": int(-(-n // chunk)),
        "one_call_ok": one_call_ok,
        "one_call_error": one_call_err,
        "one_call_seconds": float(t_one),
        "chunked_seconds": float(t_chunk),
        "logits_shape": list(logits_chunk.shape) if logits_chunk is not None else None,
        "logits_bitwise_identical": bitwise_identical,
        "predictions_identical": preds_identical,
        "max_abs_logit_diff": max_abs_diff,
        "mean_abs_logit_diff": mean_abs_diff,
        "logit_abs_max": logit_abs_max,
        "peak_gpu_mem_mb": float(peak_gpu),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert one_call_ok, f"single-call inference failed: {one_call_err}"
    assert preds_identical, "chunked and single-call predictions differ"
    assert max_abs_diff < 1e-4, f"logits differ beyond float32 round-off (max abs diff {max_abs_diff})"


if __name__ == "__main__":
    main()
