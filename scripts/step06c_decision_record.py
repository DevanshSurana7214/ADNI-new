#!/usr/bin/env python3
"""Step 6c (item 4) - record the fixed field-strength decision, script-generated.

Reads the Step 6b diagnostic artifacts and writes ``results/step06c/decision_record.md``. Nothing is
recomputed: this only transcribes the frozen Step 6b numbers (decision rule, B1 gain importance, the
mri_field_strength x COLPROT x class crosstab) so the Step 6c deliverable folder records the decision
from a script rather than by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
S6B = PROJECT_ROOT / "results" / "step06b"
OUT = PROJECT_ROOT / "results" / "step06c" / "decision_record.md"


def main() -> None:
    decision = json.loads((S6B / "decision_rule.json").read_text())
    imp = json.loads((S6B / "b1_importance.json").read_text())
    cross = pd.read_csv(S6B / "b0b_field_strength_colprot_class.csv")

    lines = ["# Step 6c - record of the field-strength decision (script-generated)", "",
             "Source artifacts: ``results/step06b/decision_rule.json``, "
             "``results/step06b/b1_importance.json``, "
             "``results/step06b/b0b_field_strength_colprot_class.csv``.", "",
             "## Decision", "",
             f"- Rule: {decision['rule']}.",
             f"- B0b macro-AUROC = {decision['b0b_macro_auroc_point']:.4f} "
             f"[{decision['b0b_macro_auroc_ci_low']:.4f}, {decision['b0b_macro_auroc_ci_high']:.4f}] "
             f"(threshold {decision['threshold']}).",
             f"- Branch: **{decision['branch']}**.",
             "",
             "**Fixed decision:** ``mri_field_strength`` is KEPT in the primary neural models. A "
             "model trained without it is run only as a sensitivity arm; it is not the primary arm.",
             "",
             "## Supporting numbers (Step 6b)", "",
             f"- The four B1 mask bits together carry a gain share of "
             f"{imp['mask_gain_share']:.2e} (essentially zero).",
             f"- ``{imp['top20'][0]['feature']}`` is the top B1 feature by gain "
             f"(share {imp['top20'][0]['gain_share']:.4f}); ``mri_mri_field_strength`` ranks "
             f"{imp['field_strength_rank']} of {imp['n_features']} "
             f"(share {imp['field_strength_gain_share']:.5f}).",
             "",
             "### mri_field_strength x COLPROT x class (MRI-present episodes)", "",
             "| field_strength | COLPROT | class | n |", "|---|---|---|---|"]
    for _, r in cross.iterrows():
        lines.append(f"| {int(r['mri_field_strength'])} | {r['COLPROT']} | {r['DIAGNOSIS']} | "
                     f"{int(r['n'])} |")
    lines += ["",
              "field_strength = 0 occurs only in ADNI1 / ADNI-GO, so the binary flag is a near-perfect "
              "proxy for acquisition era/protocol; this is why Step 6b treats it as a shortcut-relevant "
              "feature and why removing it is a sensitivity analysis, not the primary specification."]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[wrote {OUT}]")


if __name__ == "__main__":
    main()
