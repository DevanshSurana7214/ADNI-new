"""Step 11a - append the decision row to Appendix E of both PIPELINE_v2.md copies. Idempotent.

The row text is filled from ``results/step11a/analysis.json`` so the verdicts in the log are never
hand-typed.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PIPELINE_v2.md"
PARENT = ROOT.parent / "PIPELINE_v2.md"
HEADER = "## Appendix E"
ANALYSIS = ROOT / "results" / "step11a" / "analysis.json"


def _row() -> str:
    a = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    c = a["conclusions"]
    n_runs = a.get("n_runs", 0)
    partial = {r: c[r]["r2_verdict_partial_comparators"] for r in c}
    full = {r: c[r]["r2_verdict_full_comparators"] for r in c}
    changed = any(c[r]["primary_conclusion_changes"] for r in c)
    sens = a["sensitivity_significant_settings"]
    n_sens = sum(len(v) for v in sens.values())
    return (
        "| 11a | Robustness only (frozen design, no tuning, uniform Step 6b protocol, both checkpoint "
        "rules). Second training recipe R2 = lr 3e-5, weight decay 0.3, batch 128, encoder dropout "
        "0.4, transformer dropout 0.2, max 150 epochs, patience 25, on arms A0/A1/A4; recipe 1 (the "
        "Step 9 config) is reused as-is for A0-A4 and for every sensitivity comparison. Sensitivities "
        "at recipe 1: A4 tau in {0.5, 1.5}; A3 training weight clip in {[0.5,2], [0.1,10]}; modality "
        f"dropout p_drop in {{0, 0.5}} for A0 and A4 with the prior rebuilt under the matching kernel. "
        f"{n_runs} runs (3 seeds x 5 folds). Report: PN-BAcc(u)/macro-F1/macro-AUROC with CIs per "
        "run-set, paired A4-A0 and A4-A1 under R2 with Holm over the two PN-BAcc comparisons and the "
        "H3 verdict under R2 per rule, best-epoch distributions flagged for median<5 and any epoch-0 "
        "selection, the paired (setting - recipe 1) difference for every sensitivity, and one "
        "PN-BAcc-vs-macro-F1 figure for every run-set. H3 under R2 (primary rule / secondary rule): "
        f"partial comparator set -> {partial.get('primary', '?')} / {partial.get('secondary', '?')}; "
        f"full four-comparator set (A2/A3 carried over from recipe 1) -> {full.get('primary', '?')} / "
        f"{full.get('secondary', '?')}. Primary conclusion changes = {changed}. Sensitivity settings "
        f"with a CI excluding 0: {n_sens} rows. A5-A9, C2 and the window sensitivities are not run. "
        "| `results/step11a/REPORT.md`, `scripts/run_step11a.py`, `scripts/analyse_step11a.py`, "
        "`configs/step11a_robustness.yaml`, `tests/test_step11a_robustness.py` |"
    )


def _add(section: str, row: str) -> str:
    section = section.rstrip("\n")
    if row not in section:
        section += "\n" + row
    return section + "\n"


def main() -> None:
    row = _row()
    for path in (DOCS, PARENT):
        text = path.read_text(encoding="utf-8")
        idx = text.find(HEADER)
        if idx == -1:
            raise AssertionError(f"{path} has no Appendix E")
        path.write_text(text[:idx] + _add(text[idx:], row), encoding="utf-8")
    print(f"[appended Appendix E row to {DOCS} and {PARENT}]")


if __name__ == "__main__":
    main()
