"""Step 7b - re-run the A0 pilot (S1 seed 0 fold 0) with the frozen config and compare arms.

Runs ``scripts/run_step07_pilot.py`` with the frozen ``configs/model.yaml`` into
``results/step07b/pilot_frozen`` and builds the Step 7 comparison table (A0 old config, A0 new/frozen
config, B1, B1-blind, B2_mri). The pilot test fold is used here and nowhere else in Step 7b.

Writes ``results/step07b/comparison_frozen.csv`` and ``comparison_frozen.md``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/step07b"
OLD = ROOT / "results/step07fix/det_A/comparison_s0f0.csv"
COLS = ["arm", "n", "macro_auroc", "macro_f1", "balanced_accuracy", "pn_bacc_uniform", "pn_bacc_prior"]


def main() -> None:
    env = dict(os.environ, PYTHONPATH="src")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_step07_pilot.py"),
                    "--config", str(ROOT / "configs/model.yaml"), "--out", str(OUT / "pilot_frozen")],
                   cwd=str(ROOT), env=env, check=True)

    frozen = pd.read_csv(OUT / "pilot_frozen/comparison_s0f0.csv")
    old = pd.read_csv(OLD)

    a0_old = old[old["arm"] == "A0"].iloc[0]
    a0_new = frozen[frozen["arm"] == "A0"].iloc[0]
    arms = frozen[frozen["arm"] != "A0"]

    rows = [("A0 (old config, model_v0)", a0_old), ("A0 (frozen config)", a0_new)]
    rows += [(r["arm"], r) for _, r in arms.iterrows()]

    table = pd.DataFrame([{c: r[c] for c in COLS} | {"label": lbl} for lbl, r in rows])
    table.to_csv(OUT / "comparison_frozen.csv", index=False)

    md = ["# Step 7b - A0 pilot, S1 seed 0 fold 0 (frozen config)", "",
          "Test-fold metrics; the pilot test fold is used here and nowhere else in Step 7b.", "",
          "| arm | n_ep | macro-AUROC | macro-F1 | BAcc | PN-BAcc (uniform) | PN-BAcc (P(c)) |",
          "|---|---|---|---|---|---|---|"]
    for lbl, r in rows:
        md.append(f"| {lbl} | {int(r['n'])} | {r['macro_auroc']:.4f} | {r['macro_f1']:.4f} | "
                  f"{r['balanced_accuracy']:.4f} | {r['pn_bacc_uniform']:.4f} | {r['pn_bacc_prior']:.4f} |")
    md.append("")
    md.append("`A0 (old config)` is the deterministic rerun with `configs/model_v0.yaml` "
              "(lr 3e-4, wd 0.05, dropout 0.2/0.1); the committed `results/step07` was an unseeded draw "
              "of the same config. `B1`, `B1-blind` and `B2_mri` are the Step 6 reference arms on the "
              "same fold-0 test episodes.")
    (OUT / "comparison_frozen.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
