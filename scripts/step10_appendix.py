"""Step 10 - append the decision row to Appendix E of both PIPELINE_v2.md copies. Idempotent."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PIPELINE_v2.md"
PARENT = ROOT.parent / "PIPELINE_v2.md"
HEADER = "## Appendix E"

ROW = (
    "| 10 | Diagnostics, controls and the shift split (frozen config, no tuning, uniform Step 6b "
    "protocol, both checkpoint rules). (1) A0/A1/A4 re-trained on S1 saving BOTH best states; the "
    "retrain reproduces the Step 9 OOF predictions with max probability deviation 0.0e+00. "
    "Counterfactual mask intervention on test episodes with >= 2 modalities: (a) remove one observed "
    "modality (mask -> 0) and (b) content-swap its values with a random other test episode that has "
    "it, at the rho-uniform operating point; report mean TV change and predicted class shares per "
    "modality x arm and the ratio (a)/(b). (2) Placebo C1 = A4 with pi(c|m~) from labels permuted "
    "within ORIGPROT on the TRAIN split only (loss still uses true labels); control C-shuf = A3 with "
    "the per-episode weights shuffled within each batch. Pre-declared reading: the joint prior is "
    "supported only if the CI lower bound of A4 - C1 on PN-BAcc is above 0. (3) S2 enrolment-cohort "
    "shift: train A0-A4 + B1 on the S2 train side (ADNI1/GO/2, frozen val holdout), test on ADNI3/4; "
    "PN-BAcc from the S2 train pi_hat, plus natural macro-F1/macro-AUROC/per-class F1, paired A4-A0 "
    "and A4-A1, and P(c|m) train vs test for patterns >= 30 both sides. (4) best-epoch distribution "
    "for every new run, flagging best epoch 0. A5-A9, C2 and the tau/clip/p_drop sensitivities are "
    "deferred to Step 11. | `results/step10/REPORT.md`, `scripts/run_step10_s1.py`, "
    "`scripts/run_step10_s2.py`, `scripts/run_step10_intervention.py`, `scripts/analyse_step10.py`, "
    "`src/adni/diagnostics.py`, `tests/test_step10_controls.py` |"
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
