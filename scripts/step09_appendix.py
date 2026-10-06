"""Step 9 - append the decision row to Appendix E of both PIPELINE_v2.md copies. Idempotent."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PIPELINE_v2.md"
PARENT = ROOT.parent / "PIPELINE_v2.md"
HEADER = "## Appendix E"

ROW = (
    "| 9 | Main experiment: arms A0-A4 x S1 seeds [0,1,2] x folds [0..4] = 75 runs, frozen config "
    "(Step 7b), p_drop 0.25, alpha 0.5, tau 1, clip [0.2, 5]; one resumable runner. Two checkpoint "
    "rules were pre-declared and tracked in the SAME run: PRIMARY = best validation macro-F1 (drives "
    "early stopping/patience; validation macro-F1 is argmax of the RAW logits s for every arm A0-A4, "
    "since a uniform rho adds a constant), SECONDARY = best validation PN-BAcc (rho uniform, fold "
    "TRAIN-split pi_hat, never stops training). Both best states are kept and the test fold is "
    "evaluated with both. PIPELINE_v2 section 6 says early stopping on validation PN-BAcc and five "
    "seeds per fold; Step 9 follows the pre-declaration (macro-F1 primary + PN-BAcc secondary, three "
    "seeds). All tables are script-generated with the Step 6b uniform protocol (3-seed probability "
    "average, pooled OOF, one 2,000-resample subject-cluster bootstrap) separately per checkpoint "
    "rule. | `results/step09/REPORT.md`, `scripts/run_step09.py`, `scripts/analyse_step09.py`, "
    "`tests/test_step09_checkpoints.py` |"
)


def _add(section: str) -> str:
    section = section.rstrip("\n")
    if ROW not in section:
        section += "\n" + ROW
    return section + "\n"


def main() -> None:
    for path in (DOCS, PARENT):
        text = path.read_text(encoding="utf-8")
        idx = text.find(HEADER)
        if idx == -1:
            raise AssertionError(f"{path} has no Appendix E")
        path.write_text(text[:idx] + _add(text[idx:]), encoding="utf-8")
    print(f"[appended Appendix E row to {DOCS} and {PARENT}]")


if __name__ == "__main__":
    main()
