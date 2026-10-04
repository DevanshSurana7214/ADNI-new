"""Step 7 close-out: verify the fixed trainer makes the pilot reproducible.

Runs ``scripts/run_step07_pilot.py`` twice into ``results/step07fix/det_A`` and ``det_B`` with the
same fixed seed and compares the outputs. With ``set_seed`` applied before model construction the
curves and all metrics are identical; only wall-clock timing fields differ.

Writes ``results/step07fix/determinism_check.json``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step07fix"
TIMING_KEYS = {"seconds_total", "seconds_per_epoch", "peak_gpu_mem_mb"}


def _run(tag: str) -> Path:
    d = OUT / f"det_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH="src")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_step07_pilot.py"), "--out", str(d)],
                   cwd=str(ROOT), env=env, check=True)
    return d


def main() -> None:
    a, b = _run("A"), _run("B")
    curves_identical = pd.read_csv(a / "curves_s0f0.csv").equals(pd.read_csv(b / "curves_s0f0.csv"))
    comp_identical = (a / "comparison_s0f0.csv").read_bytes() == (b / "comparison_s0f0.csv").read_bytes()
    ja = json.loads((a / "pilot_s0f0.json").read_text())
    jb = json.loads((b / "pilot_s0f0.json").read_text())
    strip = lambda j: {k: v for k, v in j.items() if k not in TIMING_KEYS}  # noqa: E731
    report = {
        "seed": ja["seed"], "fold": ja["fold"],
        "curves_csv_identical": bool(curves_identical),
        "comparison_csv_identical": bool(comp_identical),
        "pilot_json_equal_excluding_timing": strip(ja) == strip(jb),
        "timing_fields": {k: {"run_A": ja[k], "run_B": jb[k]} for k in sorted(TIMING_KEYS)},
        "n_parameters": ja["n_parameters"],
        "epochs_run": ja["epochs_run"],
        "best_epoch": ja["best_epoch"],
        "best_val_macro_f1": ja["best_val_macro_f1"],
        "test_macro_auroc": ja["test"]["macro_auroc"],
        "test_macro_f1": ja["test"]["macro_f1"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "determinism_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    assert curves_identical and comp_identical and report["pilot_json_equal_excluding_timing"]


if __name__ == "__main__":
    main()
