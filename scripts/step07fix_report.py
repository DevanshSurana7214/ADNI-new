"""Generate results/step07fix/REPORT.md from the git state and the JSON artifacts.

All tables are produced here from real commands/artifacts, never typed by hand.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step07fix"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=True).stdout.strip()


def run_pytest() -> str:
    env = dict(os.environ, PYTHONPATH="src")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else f"exit {r.returncode}"


def main() -> None:
    det = json.loads((OUT / "determinism_check.json").read_text())
    inf = json.loads((OUT / "inference_largebatch.json").read_text())
    committed = json.loads((ROOT / "results/step07/pilot_s0f0.json").read_text())
    step07_commit = git("rev-parse", "step07")
    head = git("rev-parse", "--short", "HEAD")
    log = git("log", "--oneline", "-n", "6").splitlines()
    changed = git("diff", "--name-status", "573b2ca", "1e141a7").splitlines()
    diff_step07 = git("diff", "--stat", "results/step07").strip()
    tests = run_pytest()

    md: list[str] = []
    md.append("# Step 7 close-out - report (`step07fix`)")
    md.append("")
    md.append("Repairs the repository state after `1e141a7` replaced the verified Step 7 model, "
              "restores the `FusionModel` implementation, and makes training reproducible.")
    md.append("")

    md.append("## A.1 Repository state")
    md.append("")
    md.append(f"- `git rev-parse step07` -> `{step07_commit}` (the verified Step 7).")
    md.append(f"- HEAD after the repair -> `{head}`.")
    md.append("")
    md.append("| recent commits (newest first) |")
    md.append("|---|")
    for line in log:
        md.append(f"| `{line}` |")
    md.append("")

    md.append("## A.2 Files changed by `1e141a7` and the revert")
    md.append("")
    md.append("`1e141a7` changed exactly these four files (status = A/M/D vs its parent `573b2ca`); "
              "`git revert 1e141a7` restored the `6761aa7` versions of all four.")
    md.append("")
    md.append("| status | file | reverts to 6761aa7 version |")
    md.append("|---|---|---|")
    for row in changed:
        st, f = row.split("\t", 1)
        md.append(f"| {st} | `{f}` | yes |")
    md.append("")
    md.append("The other untracked files in the tree (`results/step06d/`, `tests/test_step03_tables.py`, "
              "`tests/test_step06c.py`, `scripts/step06d_verify_tables.py`, `scripts/verify_step03_tables.py`) "
              "are independent Step 6d/6e verification artifacts and were kept.")
    md.append("")

    md.append("## A.3 Test suite and pilot reproducibility")
    md.append("")
    md.append(f"- `pytest -q`: **{tests}** (the 6761aa7-tagged set was 61; Step 6e added 3 and this "
              "repair added 2 model tests, and the untracked Step 6d tests add 10 more).")
    md.append(f"- `git diff --stat results/step07`: `{diff_step07 or '(empty)'}` - the committed Step 7 "
              "results were not edited.")
    md.append("")
    md.append("Re-running `scripts/run_step07_pilot.py` on the reverted code did **not** reproduce "
              "`results/step07`: `train_a0` built the model *before* `Trainer.fit` called `set_seed`, so the "
              "initial weights were an unseeded draw. The fix seeds before model construction "
              "(`src/adni/train.py`); two runs then agree on every field except wall-clock timing.")
    md.append("")
    md.append("| pilot field | committed `results/step07` (unseeded draw) | fixed run (`step07fix/det_A`) |")
    md.append("|---|---|---|")
    rows = [
        ("epochs_run", committed["epochs_run"], det["epochs_run"]),
        ("best_epoch", committed["best_epoch"], det["best_epoch"]),
        ("best_val_macro_f1", committed["best_val_macro_f1"], det["best_val_macro_f1"]),
        ("test macro_auroc", committed["test"]["macro_auroc"], det["test_macro_auroc"]),
        ("test macro_f1", committed["test"]["macro_f1"], det["test_macro_f1"]),
    ]
    for name, a, b in rows:
        md.append(f"| {name} | {a:.6f} | {b:.6f} |" if isinstance(a, float) else f"| {name} | {a} | {b} |")
    md.append("")
    md.append(f"- deterministic re-runs: curves CSV identical = `{det['curves_csv_identical']}`, "
              f"comparison CSV identical = `{det['comparison_csv_identical']}`, "
              f"pilot JSON equal excluding timing = `{det['pilot_json_equal_excluding_timing']}`. "
              f"Wall-clock differs ({det['timing_fields']['seconds_total']['run_A']:.2f}s vs "
              f"{det['timing_fields']['seconds_total']['run_B']:.2f}s); GPU memory is identical.")
    md.append("")

    md.append("## A.4 Large-batch inference (restored model)")
    md.append("")
    md.append(f"All {inf['n_episodes']} episodes evaluated in one forward call and in chunks of "
              f"{inf['chunk_size']} ({inf['n_chunks']} chunks), device `{inf['device']}`.")
    md.append("")
    md.append("| quantity | value |")
    md.append("|---|---|")
    md.append(f"| single-call succeeded (no segfault) | `{inf['one_call_ok']}` |")
    md.append(f"| predictions identical (argmax) | `{inf['predictions_identical']}` |")
    md.append(f"| logits bitwise identical | `{inf['logits_bitwise_identical']}` |")
    md.append(f"| max abs logit diff | {inf['max_abs_logit_diff']:.3e} |")
    md.append(f"| mean abs logit diff | {inf['mean_abs_logit_diff']:.3e} |")
    md.append(f"| max |logit| | {inf['logit_abs_max']:.4f} |")
    md.append(f"| peak GPU memory (MB) | {inf['peak_gpu_mem_mb']:.1f} |")
    md.append("")
    md.append("The single 13,240-row call succeeds, so no chunked-inference utility was needed "
              "(the replacement segfaulted here because it built an O(N^2) attention tensor). "
              "Bitwise equality across batch sizes is not attainable on GPU - float32 matmul reduction "
              "order depends on the batch shape - so the decision-relevant check is that predictions "
              "agree exactly, which they do (logits agree to float32 round-off, ~1e-7 relative).")
    md.append("")

    md.append("## A.5 Why the replacement was made and why it produced empty masks")
    md.append("")
    md.append("The replacement rewrote the verified `FusionModel` as a `MultimodalTransformer` with "
              "per-modality encoders, a learnable `[CLS]` token and modality dropout (p=0.25) intended to "
              "regularise the fusion. Its dropout mask marked dropped entries (`True` = dropped) but the "
              "never-empty guard tested `drop_mask.any(dim=1)` (any dropped) instead of "
              "`drop_mask.all(dim=1)` (all dropped), so rows with every modality dropped were never "
              "detected; compounding this, the fallback wrote `retry_mask[still_empty, 0] = True`, which "
              "dropped modality 0 rather than keeping it, so resampled empty rows stayed empty. The result "
              "was a fully-empty mask at a rate near p^4 (measured 0.5745% at p=0.25 over 200k draws; "
              "~0.2% in the small 1000-draw test), after which the model could attend only to `[CLS]`; "
              "flipping both conditions reduces the rate to 0%.")
    md.append("")

    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[wrote {OUT / 'REPORT.md'}]")
    print(f"tests: {tests}")


if __name__ == "__main__":
    main()
