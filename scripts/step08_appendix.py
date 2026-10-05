"""Step 8 - append the decision row to Appendix E of both PIPELINE_v2.md copies. Idempotent."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PIPELINE_v2.md"
PARENT = ROOT.parent / "PIPELINE_v2.md"
HEADER = "## Appendix E"

ROW = (
    "| 8 | Mask-process prior and loss arms A0-A4 (`submask_kernel`, `pi_table`, `pattern_marginal`, "
    "`class_prior`; losses in `src/adni/losses.py`; one `train_arm` entry point). Priors fitted on the "
    "TRAIN split only and always on the effective mask after modality dropout. Unit test 4 was made "
    "statistically valid rather than loosened: it validates the kernel against `pi_table` from the "
    "UNSMOOTHED joint with 20,000,000 draws, because alpha=0.5 smoothing moves pi(c|m_tilde) by up to "
    "0.012 for the two rarest training patterns (0101, 0111) and 200,000 draws leave ~0.021 "
    "Monte-Carlo error on the rarest sub-masks. "
    "| `results/step08/REPORT.md`, `tests/test_priors_losses.py` |"
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
