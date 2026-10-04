"""Step 7b - freeze the search-selected hyperparameters into configs/model.yaml.

Keeps the pre-search configuration as ``configs/model_v0.yaml`` and writes the selected
lr / weight_decay / encoder+transformer dropout into ``configs/model.yaml``. Everything else
(architecture, batch size, epoch cap, patience, modality dropout) is left untouched: the 60-epoch /
patience-10 cap was search-time only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary = json.loads((ROOT / "results/step07b/search_summary.json").read_text())
    sel = next(c for c in summary["configurations"] if c["config_id"] == summary["selected_config_id"])
    cfg = ROOT / "configs/model.yaml"
    v0 = ROOT / "configs/model_v0.yaml"
    text = cfg.read_text(encoding="utf-8")
    if not v0.exists():
        v0.write_text(text, encoding="utf-8")

    replacements = {
        "enc_dropout": f"{sel['enc_dropout']}",
        "attn_dropout": f"{sel['attn_dropout']}",
        "lr": f"{sel['lr']:.1e}",
        "weight_decay": f"{sel['weight_decay']}",
    }
    new = text
    for key, val in replacements.items():
        new, n = re.subn(rf"^{key}:.*$", f"{key}: {val}", new, count=1, flags=re.M)
        assert n == 1, f"key {key} not found exactly once"

    banner = (
        "# Hyperparameters frozen by the Step 7b validation-only search (results/step07b/search_summary.json).\n"
        "# Every arm A0-A9 trains with this configuration; do not tune per arm.\n"
        "# The pre-search configuration is preserved as configs/model_v0.yaml.\n"
    )
    anchor = "# Masked-attention fusion model A0 and its trainer"
    if banner not in new:
        new = new.replace(anchor, banner + anchor, 1)
    cfg.write_text(new, encoding="utf-8")

    print(f"selected {sel['config_id']}: lr={sel['lr']} wd={sel['weight_decay']} "
          f"enc_dropout={sel['enc_dropout']} attn_dropout={sel['attn_dropout']}")
    print(f"[wrote {cfg}] (old -> {v0})")


if __name__ == "__main__":
    main()
