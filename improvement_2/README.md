# Improvement workstream 2 (EXPLORATORY)

Everything under this folder is exploratory and **outside the frozen step sequence**. It must be
labelled EXPLORATORY in any output, and it must not change A0-A4, recipes R1/R2, or any
pre-declared rule of steps 6b/9/10/11a/11b.

This workstream's goal (higher-level proposal, not yet a regulated step):
a smarter analytics/feedback loop that performs a controlled, post-hoc prior correction directly on
existing out-of-fold predictions — so the whole point is: *given what we already computed, can a
correction applied at the prediction level beat the raw model*, measured by the same family of
metrics and paired bootstrap used in the frozen steps.

The concrete experiment realised here is the injected task:
post-hoc joint correction on existing OOF predictions (tau=1, alpha=0.5), with B1 and the recipe-2
A0/A4 neural baselines, both checkpoint rules, S1 and S2, both operating points, within-pattern
macro-AUROC, controlled paired differences with Holm, and a small-number-of-statistics budget
(three comparisons per checkpoint rule). No per-model tuning of tau/alpha/margins.

## Layout (mirrors `improvement_1/`)

```
improvement_2/
  README.md
  AGENTS.md              run metadata (index, what changed)
  pyproject.toml         local test config (runs pytest from this folder only)
  .gitignore             excludes bulky / model-artifact outputs, plus __pycache__
  configs/step13.yaml    pre-registration (written and committed BEFORE any result)
  src/improve2/         reusable library code (EXPLORATORY)
  code/                 run-specific helpers (EP / PD split semantics)
  tests/test_step13a.py  tests for the new code (not just script output)
  scripts/analyse_step13a.py   script-generated tables + REPORT.md
  results/improvement2/       outputs (runs/, tables/, diagnostics/, REPORT.md, analysis.json)
```

## Run an analysis (local only)

From the repo root with the project venv:

```bash
cd improvement_2
PYTHONPATH=src:. .venv/../.venv/Scripts/python.exe -m pytest tests -q
PYTHONPATH=src:. .venv/../.venv/Scripts/python.exe scripts/analyse_step13a.py
```

The first command tests the new code. The second regenerates the tables/report.

## Status

Step 13a — post-hoc joint correction on existing OOF predictions. Pre-registration frozen in
`configs/step13.yaml` and committed before any result was computed.
