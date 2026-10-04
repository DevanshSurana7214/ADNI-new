"""Step 7b - log the Step 7 repair and the frozen regularisation in Appendix E.

``docs/PIPELINE_v2.md`` in the repo is missing Appendix E, so this copies the authoritative log
from the parent ``PIPELINE_v2.md`` and appends the new rows to both copies. Idempotent.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "PIPELINE_v2.md"
PARENT = ROOT.parent / "PIPELINE_v2.md"
HEADER = "## Appendix E"


def _rows() -> list[str]:
    summary = json.loads((ROOT / "results/step07b/search_summary.json").read_text())
    sel = next(c for c in summary["configurations"]
               if c["config_id"] == summary["selected_config_id"])
    r7 = ("| 7 | Restored the verified Step 7 `FusionModel` after `1e141a7` and made training "
          "reproducible: `train_a0` now calls `set_seed` before constructing the model, so folds are "
          "deterministic, and a single 13,240-row forward pass succeeds with chunked inference matching. "
          "| `results/step07fix/REPORT.md` |")
    r7b = (f"| 7b | Neural-arm regularisation frozen by a pre-registered validation-only search over "
           f"lr {{3e-4,1e-4}} x weight_decay {{0.05,0.3}} x dropout {{(0.2,0.1),(0.4,0.2)}} (A0, S1 seed 0 "
           f"folds 0-1, no test fold loaded): lr={sel['lr']:.1e}, weight_decay={sel['weight_decay']}, "
           f"encoder dropout={sel['enc_dropout']}, transformer dropout={sel['attn_dropout']}. "
           f"Every arm A0-A9 reuses this configuration; no per-arm tuning. "
           f"| `results/step07b/search_runs.csv` (16 runs), `search_summary.json` |")
    return [r7, r7b]


def _add(section: str, rows: list[str]) -> str:
    section = section.rstrip("\n")
    for r in rows:
        if r not in section:
            section += "\n" + r
    return section + "\n"


def main() -> None:
    rows = _rows()
    parent = PARENT.read_text(encoding="utf-8")
    idx = parent.find(HEADER)
    assert idx != -1, "parent PIPELINE_v2.md has no Appendix E"
    appendix = _add(parent[idx:], rows)
    PARENT.write_text(parent[:idx] + appendix, encoding="utf-8")

    docs = DOCS.read_text(encoding="utf-8")
    if HEADER in docs:
        docs_new = _add(docs, rows)
    else:
        docs_new = docs.rstrip("\n") + "\n\n" + appendix
    DOCS.write_text(docs_new, encoding="utf-8")
    print(f"[appended Appendix E rows to {DOCS} and {PARENT}]")


if __name__ == "__main__":
    main()
