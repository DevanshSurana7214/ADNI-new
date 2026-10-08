# Run metadata — improvement_2

## What was done

A single, controlled, exploratory experiment was injected and executed under `improvement_2/`.

Chunk A (indexed agent `a2t01`) implemented the post-hoc joint correction on existing out-of-fold
predictions, using the same uniform protocol, same metric family, same paired subject-cluster
bootstrap, same checkpoint-rule semantics and same S2 version used by steps 6b/9/10/11a/11b, but
applied to the frozen OOF predictions of B1, A0(R2) and A4(R2). Nothing in the frozen sequence was
re-run or re-tuned.

## What was added or changed

- `configs/step13.yaml` — pre-registration; written and committed before any result.
- `src/improve2/` — EXPLORATORY reusable library (correction, prior loading, E1 build).
- `code/` — run-specific helpers (EP/PD semantic split: correction + evaluation recipes).
- `tests/test_step13a.py` — tests for the new code.
- `scripts/analyse_step13a.py` — script that regenerates every table and the REPORT.md.
- `results/improvement2/` — outputs: runs/ (corrected + uncorrected E1 OOF), tables/, diagnostics/,
  `REPORT.md`, `analysis.json`.
- `AGENTS.md`, `pyproject.toml`, `.gitignore`, manipulation and prompt artifacts for this run,
  exactly as in `improvement_1/`.

## How to rerun

From the repo root with the project venv active:

```bash
cd improvement_2
PYTHONPATH=src:. .venv/../.venv/Scripts/python.exe -m pytest tests -q
PYTHONPATH=src:. .venv/../.venv/Scripts/python.exe scripts/analyse_step13a.py
```
