"""Path setup for the improvement_2 tests.

Keeps the change local to this folder: the repo-level pytest config is untouched, so
``pytest tests`` on the frozen sequence behaves exactly as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
IMPROVEMENT_ROOT = HERE.parents[1]          # improvement_2/
REPO_ROOT = IMPROVEMENT_ROOT.parent         # adni-shortcut/

for p in (REPO_ROOT / "src", IMPROVEMENT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
