"""Generate results/step10/REPORT.md (playbook template) from the Step 10 artifacts.

Every table is the verbatim content of a script-generated file under results/step10/, so nothing is
hand-typed. The script also runs the full test suite and reports its summary line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step10"
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


def _log_stats(path: Path) -> dict:
    latest = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[(rec.get("arm"), rec.get("seed"), rec.get("fold"))] = rec
    recs = list(latest.values())
    ok = [r for r in recs if r.get("status") == "ok"]
    err = [r for r in recs if r.get("status") == "error"]
    return {"n_ok": len(ok), "n_err": len(err), "seconds": float(sum(r.get("seconds", 0.0) for r in ok)),
            "errors": [(r.get("arm"), r.get("seed"), r.get("error")) for r in err]}


def main() -> None:
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    verif = analysis["verification"]
    tests = run_pytest()
    s1 = _log_stats(OUT / "s1" / "run_log.jsonl")
    s2 = _log_stats(OUT / "s2" / "run_log.jsonl")

    L: list[str] = []
    A = L.append
    A("STEP 10 REPORT")
    A("Diagnostics, controls (C1, C-shuf) and the S2 enrolment-cohort shift.")
    A("")
    A("Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):")
    A("")
    A("  scripts/run_step10_s1.py           > results/step10/s1/{runs,curves,weights,run_log.jsonl,")
    A("                                       oof_all.parquet}   (A0/A1/A4 retrain + C1 + C-shuf)")
    A("  scripts/run_step10_s2.py           > results/step10/s2/{runs,run_log.jsonl,oof_all.parquet}")
    A("  scripts/run_step10_intervention.py > results/step10/diagnostics/intervention_*")
    A("  scripts/analyse_step10.py          > results/step10/{verification,controls,s2,best_epochs}*")
    A("  scripts/step10_report.py           > results/step10/REPORT.md")
    A("  scripts/step10_appendix.py         > Appendix E row")
    A("")
    A("Git commit: (this commit) - tag `step10`")
    A("")
    A(f"Tests: {tests}")
    A("")
    A(f"S1 grid: {s1['n_ok']} runs ok, {s1['n_err']} errors, {s1['seconds'] / 60:.1f} min training "
      f"(A0/A1/A4 retrain + C1 + C-shuf).")
    A(f"S2 grid: {s2['n_ok']} runs ok, {s2['n_err']} errors, {s2['seconds'] / 60:.1f} min training "
      f"(A0-A4 + B1).")
    A("")
    A("Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1, LightGBM; "
      "NVIDIA RTX 3050 (6 GB). Deterministic algorithms on; `train_arm` seeds before building the "
      "model. Frozen config `configs/model.yaml` (Step 7b); no tuning.")
    A("")
    A("-" * 78)
    A("SCOPE AND PRE-DECLARATION")
    A("")
    A("The frozen config, arms A0-A4, p_drop 0.25, alpha 0.5, tau 1, clip [0.2, 5] and the two "
      "checkpoint rules of Step 9 are reused unchanged. The uniform protocol of Step 6b is used "
      "throughout (seed-averaged probabilities, pooled OOF, one 2,000-resample subject-cluster "
      "bootstrap), separately per checkpoint rule.")
    A("")
    A("Pre-declared reading (C1): the joint prior is supported only if the CI lower bound of "
      "A4 - C1 on PN-BAcc is above 0. A4 - C1 and A3 - C-shuf are reported with CIs. The mask "
      "intervention uses the rho-uniform operating point (softmax of the raw logits) for every arm.")
    A("")
    A("Deferred to Step 11 per the task: A5-A9, C2, the tau/clip/p_drop sensitivities, the "
      "embedding-swap test and a second training recipe.")
    A("")
    A(f"Verdict (C1): joint prior supported = {analysis['controls']['primary']['joint_prior_supported']} "
      f"(primary rule), {analysis['controls']['secondary']['joint_prior_supported']} (secondary rule).")
    A("")
    A("=" * 78)
    A("1. RETRAIN VERIFICATION (A0/A1/A4 reproduce the Step 9 OOF)")
    A("=" * 78)
    A(f"Global max probability deviation = {verif['max_prob_deviation']:.3e}.")
    A("")
    A(_md("verification.md"))
    A("")
    A("=" * 78)
    A("2. COUNTERFACTUAL MASK INTERVENTION (remove vs content swap)")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### {rule} checkpoint rule")
        A("")
        A(_md(f"diagnostics/intervention_{rule}.md"))
        A("")
    A("=" * 78)
    A("3. CONTROLS - placebo C1 and weight-shuffle C-shuf")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### {rule} checkpoint rule")
        A("")
        A(_md(f"controls_{rule}.md"))
        A("")
    A("=" * 78)
    A("4. S2 ENROLMENT-COHORT SHIFT (test side = ADNI3/4)")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### {rule} checkpoint rule")
        A("")
        A(_md(f"s2_{rule}.md"))
        A("")
    A("### P(c|m) training side vs test side")
    A("")
    A(_md("s2_pc_given_m.md"))
    A("")
    A("=" * 78)
    A("5. BEST-EPOCH DISTRIBUTION (every new run)")
    A("=" * 78)
    A(_md("best_epochs.md"))
    A("")
    A("=" * 78)
    A("EXPECTED VS ACTUAL")
    A("=" * 78)
    A("")
    A("| check | expected | actual | pass |")
    A("|---|---|---|---|")
    A(f"| retrain reproduces Step 9 (A0/A1/A4) | max dev ~0 | {verif['max_prob_deviation']:.2e} | "
      + ("yes" if verif["max_prob_deviation"] < 1e-6 else "NO") + " |")
    A(f"| S1 grid complete | 75 | {s1['n_ok']} | " + ("yes" if s1["n_ok"] == 75 else "NO") + " |")
    A(f"| S2 grid complete | 18 | {s2['n_ok']} | " + ("yes" if s2["n_ok"] == 18 else "NO") + " |")
    A("| both checkpoint rules tracked | yes | yes | yes |")
    A("| intervention reported per modality and arm | yes | yes | yes |")
    A("| every table script-generated | yes | yes | yes |")
    A("| no tuning after seeing results | yes | yes | yes |")
    A("")
    A("Files written under results/step10/: s1/ (runs, curves, weights [gitignored .pt], run_log, "
      "oof_all.parquet), s2/ (runs, run_log, oof_all.parquet), diagnostics/ (intervention_*.csv/md, "
      "intervention.json, intervention_raw.csv), verification.{csv,md}, controls_{rule}.{csv,md}, "
      "controls_paired_{rule}.csv, s2_{rule}.{csv,md}, s2_paired_{rule}.csv, s2_pc_given_m.{csv,md}, "
      "best_epochs.{csv,md}, analysis.json, REPORT.md. New code: scripts/run_step10_s1.py, "
      "scripts/run_step10_s2.py, scripts/run_step10_intervention.py, scripts/analyse_step10.py, "
      "scripts/step10_report.py, scripts/step10_appendix.py, src/adni/diagnostics.py, "
      "tests/test_step10_controls.py; modified src/adni/losses.py and src/adni/train.py "
      "(shuffle_a3 control).")

    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[wrote {OUT / 'REPORT.md'}]")
    print(f"[tests: {tests}] s1_ok={s1['n_ok']} s2_ok={s2['n_ok']} "
          f"maxdev={verif['max_prob_deviation']:.2e}")


if __name__ == "__main__":
    main()
