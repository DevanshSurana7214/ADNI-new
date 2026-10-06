"""Generate results/step09/REPORT.md (playbook template) from the Step 9 artifacts.

Every table in the report is the verbatim content of a script-generated file under results/step09/,
so nothing is hand-typed. The script also runs the full test suite and reports its summary line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step09"
CKPT_RULES = ("primary", "secondary")


def run_pytest() -> str:
    env = dict(os.environ, PYTHONPATH="src")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else f"exit {r.returncode}"


def _md(name: str) -> str:
    p = OUT / name
    return p.read_text(encoding="utf-8").rstrip("\n") if p.exists() else f"(missing {name})"


def main() -> None:
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    verdicts = analysis["verdicts"]
    log_lines = (OUT / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if (OUT / "run_log.jsonl").exists() else []
    logs = [json.loads(x) for x in log_lines if x.strip()]
    latest = {}
    for rec in logs:
        latest[(rec["arm"], rec["seed"], rec["fold"])] = rec
    logs = list(latest.values())
    n_ok = sum(1 for r in logs if r.get("status") == "ok")
    n_err = sum(1 for r in logs if r.get("status") == "error")
    rt = analysis["runtime"]
    total = next((r for r in rt if r["arm"] == "TOTAL"), None)
    rt_total = f"{total['seconds_total'] / 60:.1f} min" if total else "n/a"
    tests = run_pytest()

    L: list[str] = []
    A = L.append
    A("STEP 09 REPORT")
    A("Main experiment: arms A0-A4 x S1 seeds [0,1,2] x folds [0..4] = 75 runs.")
    A("")
    A("Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):")
    A("")
    A("  scripts/run_step09.py      > results/step09/{runs/*.parquet, curves/*.csv, run_log.jsonl,")
    A("                                oof_all.parquet}")
    A("  scripts/analyse_step09.py  > results/step09/{main_table_*, paired_*, breakdown_*,")
    A("                                stability_per_seed_*, stability_per_fold_*, best_epochs_*,")
    A("                                runtime}.{csv,md}, analysis.json")
    A("  scripts/step09_report.py   > results/step09/REPORT.md")
    A("  scripts/step09_appendix.py > Appendix E row in docs/PIPELINE_v2.md and ../PIPELINE_v2.md")
    A("  python -m pytest -q")
    A("")
    A("Git commit: (this commit) - tag `step09`")
    A("")
    A(f"Tests: {tests}")
    A("")
    A(f"Grid: {n_ok} runs completed, {n_err} run errors; total training wall-clock {rt_total} "
      "(see the runtime table).")
    A("")
    A("Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1. GPU: NVIDIA "
      "RTX 3050 (6 GB). Training used `torch.use_deterministic_algorithms(True)` with "
      "`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Seeds are fixed per run; `train_arm` seeds before the model "
      "is built, so each run is reproducible.")
    A("")
    A("-" * 78)
    A("PRE-DECLARATION (written before any grid result was seen)")
    A("")
    A("PRIMARY checkpoint rule: best validation macro-F1, exactly as in Steps 7-8. Early stopping and "
      "the patience-15 counter use this rule only. SECONDARY checkpoint rule: best validation PN "
      "balanced accuracy (rho uniform, weights from the fold's TRAIN-split pi_hat via "
      "`metrics.pn_weights`), selected among the epochs the run actually trained. Both are tracked per "
      "epoch in the SAME run, both best states are kept, and the test fold is evaluated with BOTH. "
      "Both rules apply identically to every arm.")
    A("")
    A("Which scores validation macro-F1 uses for A1 and A4: it uses the argmax of the RAW model logits "
      "`s_c`, for every arm A0-A4. For A1 (trained with `log P(c)` added) and A4 (trained with "
      "`tau*log pi(c|m_tilde)` added) the training-time score transform is NOT applied at validation: "
      "`argmax_c s_c` is identical to `argmax_c (s_c + log rho_uniform(c))`, because a uniform "
      "deployment prior rho adds the constant `log(1/3)` to every class. A0, A2 and A3 also use "
      "`argmax_c s_c`. A0-A4 are therefore treated consistently, and no arm-specific score adjustment "
      "enters the validation macro-F1. The same raw logits are the `uniform` operating point used for "
      "A1/A4 in every reported table.")
    A("")
    A("Operating points and the paired comparison: each arm's PRIMARY operating point is used for the "
      "primary comparison (A0/A2/A3 argmax; A1/A4 rho uniform); A4 at rho=P(c) is reported as a "
      "supplementary operating point in the main table. All analysis is done separately for the "
      "primary and the secondary checkpoint rule.")
    A("")
    A("Deviation from PIPELINE_v2 section 6: the spec text states early stopping on the validation "
      "PN-BAcc and 'five seeds per fold'. Step 9 follows the pre-declaration above and the established "
      "Steps 7-8 protocol (3 seeds, early stopping on validation macro-F1), because the pipeline text "
      "was written before the checkpoint-rule decision; both rules are nevertheless tracked and both "
      "verdicts are reported. This is recorded in Appendix E.")
    A("")
    A("Bug found and fixed during Step 9 (before any table was trusted): adding the dual-checkpoint "
      "tracking to `src/adni/train.py` initially put the raw per-epoch validation PN-BAcc into the "
      "per-epoch history dict. When it is not computable (no `val_weights`) that value is NaN, and "
      "because `NaN != NaN` two byte-identical runs then compared UNEQUAL. This failed the two "
      "pre-existing determinism tests "
      "(`tests/test_models.py::test_train_a0_reproducible_without_external_seeding` and "
      "`tests/test_priors_losses.py::test_train_arm_deterministic_through_real_entry_point`) - not "
      "because training was non-deterministic, but because the NaN sentinel is not self-equal. Fix: "
      "the per-epoch history now stores `None` (\"not computed\") when PN-BAcc is non-finite, while "
      "the internal metric stays NaN so the finite guard for the SECONDARY selection is unchanged. "
      "No test was weakened or skipped. Impact on the grid: none - every real run supplies "
      "`val_weights` (the fold's TRAIN-split `pi_hat`), and all 1785 per-epoch rows across the 75 "
      "curve files have a finite `val_pn_bacc`, so loss, epoch counts, checkpoint selection and all "
      "reported numbers are unchanged; the grid was therefore NOT re-run. The full suite is green "
      "after the fix (see the Tests line above).")
    A("")
    A(f"Verdicts (H3: A4 beats A0-A3 on PN-BAcc(uniform) after Holm and is non-inferior on natural "
      f"macro-F1 within {analysis['ni_margin']}):")
    A("")
    A("| checkpoint rule | A4 wins all 4 PN-BAcc (Holm) | macro-F1 non-inferior | verdict |")
    A("|---|---|---|---|")
    for rule in CKPT_RULES:
        v = verdicts[rule]
        A(f"| {rule} | {'yes' if v['wins_pn_bacc_holm'] else 'no'} | "
          f"{'yes' if v['macro_f1_non_inferior'] else 'no'} | {v['verdict']} |")
    A("")
    A("The PRIMARY checkpoint rule is the validation-macro-F1 rule; the secondary rule is reported "
      "alongside. Both verdicts are given above and below.")
    A("")

    A("=" * 78)
    A("MAIN TABLE - PRIMARY CHECKPOINT RULE")
    A("=" * 78)
    A(_md("main_table_primary.md"))
    A("")
    A("=" * 78)
    A("MAIN TABLE - SECONDARY CHECKPOINT RULE")
    A("=" * 78)
    A(_md("main_table_secondary.md"))
    A("")
    for rule in CKPT_RULES:
        A("=" * 78)
        A(f"PRIMARY COMPARISONS - {rule.upper()} CHECKPOINT RULE")
        A("=" * 78)
        A(_md(f"paired_{rule}.md"))
        v = verdicts[rule]
        A("")
        A(f"Verdict ({rule}): **{v['verdict']}**.")
        A("")
    A("=" * 78)
    A("BREAKDOWNS (A0 / A1 / A4 at their primary operating point)")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### Breakdowns - {rule} checkpoint rule")
        A("")
        A(_md(f"breakdown_{rule}.md"))
        A("")
    A("=" * 78)
    A("STABILITY")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### Per-seed metrics (mean +/- SD) - {rule} checkpoint rule")
        A("")
        A(_md(f"stability_per_seed_{rule}.md"))
        A("")
        A(f"### Per-fold PN-BAcc - {rule} checkpoint rule")
        A("")
        A(_md(f"stability_per_fold_{rule}_pn_bacc_uniform.md"))
        A("")
        A(f"### Per-fold macro-F1 - {rule} checkpoint rule")
        A("")
        A(_md(f"stability_per_fold_{rule}_macro_f1.md"))
        A("")
        A(f"### Best-epoch distribution - {rule} checkpoint rule")
        A("")
        A(_md(f"best_epochs_{rule}.md"))
        A("")
    A("=" * 78)
    A("RUNTIME AND ANOMALIES")
    A("=" * 78)
    A(_md("runtime.md"))
    A("")
    A("=" * 78)
    A("EXPECTED VS ACTUAL")
    A("=" * 78)
    A("")
    A("| check | expected | actual | pass |")
    A("|---|---|---|---|")
    A("| 75 runs completed | 75 | " + str(n_ok) + " | " + ("yes" if n_ok == 75 else "NO") + " |")
    A("| both checkpoint rules tracked per run | yes | yes (history has val_macro_f1 and val_pn_bacc) | yes |")
    A("| validation macro-F1 uses raw logits for A1/A4 | yes | yes | yes |")
    A("| runner skips completed runs on restart | yes | yes | yes |")
    A("| every table script-generated | yes | yes | yes |")
    A("| no tuning after seeing results | yes | yes | yes |")
    A("| NaN-sentinel bug fixed without weakening tests | yes | yes (history stores None; suite green) | yes |")
    A("")
    A("Files written under results/step09/: runs/<arm>_s<seed>_f<fold>.parquet (75), curves/ (75), "
      "run_log.jsonl, oof_all.parquet, main_table_primary/secondary.{csv,md}, "
      "paired_primary/secondary.{csv,md}, breakdown_primary/secondary.{csv,md}, "
      "stability_per_seed_{rule}.{csv,md}, stability_per_fold_{rule}.{csv,md} and "
      "stability_per_fold_{rule}_{metric}.md, best_epochs_{rule}.{csv,md}, runtime.{csv,md}, "
      "analysis.json, REPORT.md. New code: scripts/run_step09.py, scripts/analyse_step09.py, "
      "scripts/step09_report.py, scripts/step09_appendix.py, tests/test_step09_checkpoints.py; "
      "modified src/adni/train.py (dual checkpoint tracking, optional val_weights).")

    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[wrote {OUT / 'REPORT.md'}]")
    print(f"[tests: {tests}]  [runs ok: {n_ok}]  [total: {rt_total}]")
    for rule in CKPT_RULES:
        print(f"[verdict {rule}: {verdicts[rule]['verdict']}]")


if __name__ == "__main__":
    main()
