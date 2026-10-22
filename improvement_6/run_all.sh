#!/usr/bin/env bash
# Reproduce everything in improvement_6 from the project venv.
# From adni-shortcut/:
#   PYTHONPATH="src;improvement_6/src" .venv/Scripts/python.exe improvement_6/run.py --part A
#   PYTHONPATH="src;improvement_6/src" .venv/Scripts/python.exe improvement_6/run.py --part B
#   PYTHONPATH="src;improvement_6/src" .venv/Scripts/python.exe improvement_6/run.py --part C
#   PYTHONPATH="src;improvement_6/src" .venv/Scripts/python.exe -m pytest improvement_6/tests -q
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="src;improvement_6/src" \
  .venv/Scripts/python.exe improvement_6/run.py "$@"
