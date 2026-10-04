"""Tests for the Step 7b validation-only regularisation search (pre-registration and selection)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("step07b_search", ROOT / "scripts/step07b_search.py")
search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search)


def test_grid_is_exactly_eight_configs():
    cfgs = search.configurations()
    assert len(cfgs) == 8
    assert {c["lr"] for c in cfgs} == {3e-4, 1e-4}
    assert {c["weight_decay"] for c in cfgs} == {0.05, 0.3}
    assert {(c["enc_dropout"], c["attn_dropout"]) for c in cfgs} == {(0.2, 0.1), (0.4, 0.2)}


def test_load_fold_ids_uses_only_train_and_val():
    tr, va = search._load_fold_ids(ROOT / "results/step03/splits.json", 0, 0)
    assert len(tr) > 0 and len(va) > 0
    node = search._NoTestIds({"train": [1], "val": [2], "test": [3]})
    assert node["train"] == [1] and node["val"] == [2]
    with pytest.raises(AssertionError):
        node["test"]


def test_selection_tie_break_prefers_weight_decay_then_lr_then_dropout():
    # all four are within 0.005 of the best (0.661); the tie-break must reach the dropout axis.
    rows = pd.DataFrame([
        {"config_id": "a", "lr": 1e-4, "weight_decay": 0.05, "enc_dropout": 0.2, "attn_dropout": 0.1, "val_macro_f1": 0.660},
        {"config_id": "b", "lr": 3e-4, "weight_decay": 0.3, "enc_dropout": 0.2, "attn_dropout": 0.1, "val_macro_f1": 0.661},
        {"config_id": "c", "lr": 1e-4, "weight_decay": 0.3, "enc_dropout": 0.4, "attn_dropout": 0.2, "val_macro_f1": 0.660},
        {"config_id": "d", "lr": 1e-4, "weight_decay": 0.3, "enc_dropout": 0.2, "attn_dropout": 0.1, "val_macro_f1": 0.659},
    ])
    sel = search._select(rows)
    assert sel["selected_config_id"] == "d"      # wd 0.3, lr 1e-4, smaller dropout
    assert set(sel["tied_within_0.005"]) == {"a", "b", "c", "d"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
