#!/usr/bin/env python3
"""Step 11b - append the decision row to Appendix E of both PIPELINE_v2.md copies. Idempotent."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PIPELINE_v2.md"
PARENT = ROOT.parent / "PIPELINE_v2.md"
ANALYSIS = ROOT / "results" / "step11b" / "analysis.json"


def _row() -> str:
    a = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    conf = a["confirmatory"]
    v1p = conf["primary"]["verdicts"]["R1"]["verdict"]
    v1s = conf["secondary"]["verdicts"]["R1"]["verdict"]
    v2p = conf["primary"]["verdicts"]["R2"]["verdict"]
    v2s = conf["secondary"]["verdicts"]["R2"]["verdict"]
    return (
        "| 11b | Confirmatory check and exploratory arms (frozen design, pre-registered in "
        "`configs/step11b.yaml` before any run, uniform Step 6b protocol, both checkpoint rules). "
        "(A) Held-out S2 confirmation of A4 with tau = 0.5 (selected from the S1 Step 11a "
        "results): new run-sets S2R1_A4_tau0p5 and S2R2_A0/A1/A4/A4_tau0p5 (3 seeds); the Step 10 "
        "S2 A0/A1/A4 are reused. Pre-declared reading: confirmed iff A4(tau 0.5) - A0 has a "
        "PN-BAcc CI low above 0 AND a macro-F1 CI low above -0.02. Verdicts - recipe 1 "
        f"(primary/secondary): {v1p} / {v1s}; R2 replication: {v2p} / {v2s}. tau is not changed "
        "afterwards. (B) EXPLORATORY arms A5-A8 under R2 on the S1 grid (3 seeds x 5 folds = 60 "
        "runs), comparator A4(R2, tau 1) reused from Step 11a: A5 FiLM pattern conditioning, A6 "
        "+ cross-view consistency (lambda 0.1, stop-gradient view 2), A7 + pattern-aware MoE (4 "
        "experts, router [h, e_m, mask], no label input), A8 + class-dependent prototypes "
        "mu_(c,m) = mu_c + B(h(mu_c) * e_m), prototype weight 0.1, with the exact loss stated in "
        "the report. Paired differences to A4(R2) on PN-BAcc/macro-F1/macro-AUROC with Holm over "
        "the four PN-BAcc comparisons; best-epoch distributions; embedding-swap test and the Step "
        "10 mask-intervention diagnostic for every arm. EVERY A5-A8 result is labelled "
        "EXPLORATORY; no selection among A5-A8; no use of these results to change A0-A4. "
        "Unit tests: A5-A8 absent-modality invariance; prototype class-dependent separation; MoE "
        "router uses no label. Window sensitivities and paper assets are NOT started. | "
        "`results/step11b/REPORT.md`, `scripts/run_step11b_s2.py`, `scripts/run_step11b_s1.py`, "
        "`scripts/analyse_step11b.py`, `configs/step11b.yaml`, "
        "`tests/test_step11b_exploratory.py` |"
    )


def main() -> None:
    row = _row()
    for path in (DOCS, PARENT):
        text = path.read_text(encoding="utf-8")
        if row in text:
            print(f"[skip] {path} already contains the 11b row")
            continue
        marker = "## Appendix E"
        pos = text.find(marker)
        if pos < 0:
            raise SystemExit(f"no '## Appendix E' section in {path}")
        lines = text.splitlines(keepends=True)
        insert_at = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("| 11a |"):
                insert_at = i + 1
                break
        lines.insert(insert_at, row + "\n")
        path.write_text("".join(lines), encoding="utf-8")
        print(f"[appended Appendix E row 11b to {path}]")


if __name__ == "__main__":
    main()
