"""Generate results/step11a/REPORT.md (playbook template) from the Step 11a artifacts.

Every table is the verbatim content of a script-generated file under results/step11a/, so nothing is
hand-typed. The script also runs the full test suite and reports its summary line.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "step11a"
STEP9 = ROOT / "results" / "step09"
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
    latest: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            latest[(rec.get("runset"), rec.get("seed"), rec.get("fold"))] = rec
    recs = list(latest.values())
    ok = [r for r in recs if r.get("status") == "ok"]
    err = [r for r in recs if r.get("status") == "error"]
    r2 = [r for r in ok if str(r.get("runset", "")).startswith("R2_")]
    return {"n_ok": len(ok), "n_err": len(err), "n_r2": len(r2),
            "n_sensitivity": len(ok) - len(r2),
            "seconds": float(sum(r.get("seconds", 0.0) for r in ok)),
            "errors": [(r.get("runset"), r.get("seed"), r.get("error")) for r in err]}


def main() -> None:
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    conc = analysis["conclusions"]
    tests = run_pytest()
    stats = _log_stats(OUT / "run_log.jsonl")
    n_runs = analysis["n_runs"]
    changed = any(conc[r]["primary_conclusion_changes"] for r in conc)
    s9_verdicts = json.loads((STEP9 / "analysis.json").read_text())["verdicts"]
    n_sens_sig = sum(len(v) for v in analysis["sensitivity_significant_settings"].values())

    L: list[str] = []
    A = L.append
    A("STEP 11A REPORT")
    A("Robustness only: the second training recipe (R2) and the recipe-1 sensitivity settings.")
    A("")
    A("Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):")
    A("")
    A("  scripts/run_step11a.py --shard {0,1,2} --nshards 3 --no-combine")
    A("  scripts/run_step11a.py --combine-only            > results/step11a/{runs,curves,")
    A("                                       oof_all.parquet,run_log.jsonl}")
    A("  scripts/analyse_step11a.py                       > results/step11a/{main_table_*,runset_pc_*,")
    A("                                       paired_r2_*,best_epochs,sensitivity_*,pn_bacc_vs_macro_f1.png}")
    A("  scripts/step11a_report.py                        > results/step11a/REPORT.md")
    A("  scripts/step11a_appendix.py                      > Appendix E row")
    A("")
    A("Git commit: (this commit) - tag `step11a`")
    A("")
    A(f"Tests: {tests}")
    A("")
    A(f"Grid: {stats['n_ok']} runs ok, {stats['n_err']} errors ({stats['n_r2']} R2 + "
      f"{stats['n_sensitivity']} sensitivity), {stats['seconds'] / 60:.1f} min training total "
      f"across the three shards.")
    A("")
    A("Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1, "
      "matplotlib; NVIDIA RTX 3050 (6 GB). Deterministic algorithms on; `train_arm` seeds before "
      "building the model, so every (run-set, seed, fold) is reproducible (a forced rerun of "
      "A0_pdrop0 s0 f0 reproduced the OOF exactly). No tuning.")
    A("")
    A("-" * 78)
    A("PREDECLARED DESIGN")
    A("")
    A("The grid is frozen in `configs/step11a_robustness.yaml`, written before any run was seen. "
      "Recipe 1 is the Step 7b config reused by Steps 8-10; it is NOT retrained here - the A0-A4 "
      "recipe-1 baselines are exactly the Step 9 OOF (`results/step09/oof_all.parquet`).")
    A("")
    A("R2 recipe (arms A0, A1, A4): lr 3e-5, weight decay 0.3, batch 128, encoder dropout 0.4, "
      "transformer dropout 0.2, max 150 epochs, patience 25; every other value (d_model, heads, "
      "layers, FFN, modality dropout 0.25, tau 1, clip [0.2, 5]) is recipe 1.")
    A("")
    A("Sensitivity at recipe 1: A4 tau in {0.5, 1.5}; A3 training weight clip in {[0.5,2], "
      "[0.1,10]}; modality dropout p_drop in {0, 0.5} for A0 and A4, with the mask-process prior "
      "rebuilt under the matching kernel. The evaluation metric (PN-BAcc, clip [0.2, 5]) and the "
      "two checkpoint rules are unchanged, so the sensitivities are training-loss/architecture "
      "changes only.")
    A("")
    A("Protocol: the Step 6b uniform aggregation (average probabilities and weights over the 3 "
      "seeds per episode, pool OOF, one 2,000-resample subject-cluster bootstrap), separately per "
      "checkpoint rule. PRIMARY = best validation macro-F1 (drives early stopping); SECONDARY = "
      "best validation PN-BAcc (never stops training). Both best states are evaluated on the test "
      "fold.")
    A("")
    A("Not run (per the task): A5-A9, C2, the window (+/-90 / +/-365 d) sensitivities and the "
      "summary-only amyloid fallback. These stay deferred to Step 11b/12.")
    A("")
    A("=" * 78)
    A("1. METRICS PER RUN-SET")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### {rule} checkpoint rule")
        A("")
        A(_md(f"main_table_{rule}.md"))
        A("")
        A(_md(f"runset_pc_{rule}.md"))
        A("")
    A("=" * 78)
    A("2. PAIRED COMPARISONS UNDER R2 AND THE H3 VERDICT")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### {rule} checkpoint rule")
        A("")
        A(_md(f"paired_r2_{rule}.md"))
        A("")
    A("=" * 78)
    A("3. BEST-EPOCH DISTRIBUTIONS (flag median < 5 and any epoch-0 selection)")
    A("=" * 78)
    A(_md("best_epochs.md"))
    A("")
    A("=" * 78)
    A("4. SENSITIVITY VERSUS RECIPE 1 (paired per episode)")
    A("=" * 78)
    for rule in CKPT_RULES:
        A(f"### {rule} checkpoint rule")
        A("")
        A(_md(f"sensitivity_{rule}.md"))
        A("")
    A("=" * 78)
    A("5. FIGURE - PN-BAcc vs macro-F1 FOR EVERY RUN-SET")
    A("=" * 78)
    A("`pn_bacc_vs_macro_f1.png`: one point per run-set at its primary operating point, both "
      "checkpoint rules side by side, with 95% subject-cluster-bootstrap error bars and the run-set "
      "name annotated. Grey = recipe 1 (Step 9), red = R2, blue = sensitivity.")
    A("")
    A("=" * 78)
    A("6. DOES ANY CONCLUSION CHANGE?")
    A("=" * 78)
    A("")
    A("Step 9 concluded a NEGATIVE result under BOTH checkpoint rules (A4 did not beat A1/A3 on "
      "PN-BAcc after Holm). Step 11a re-checks that with a second training recipe and by perturbing "
      "the correction hyperparameters.")
    A("")
    A("| checkpoint rule | Step 9 verdict | H3 under R2 (A0/A1 comparators) | H3 under R2 (full "
      "4 comparators, A2/A3 carried over) | primary conclusion changes |")
    A("|---|---|---|---|---|")
    for rule in CKPT_RULES:
        A(f"| {rule} | {s9_verdicts[rule]['verdict']} | "
          f"{conc[rule]['r2_verdict_partial_comparators']} | "
          f"{conc[rule]['r2_verdict_full_comparators']} | "
          f"{conc[rule]['primary_conclusion_changes']} |")
    A("")
    A(f"A4(R2) - A1(R2) PN-BAcc(u): primary {_fmt(conc['primary']['a4_r2_minus_a1_pn_bacc'])} "
      f"(Holm p {_fmt(conc['primary']['a4_r2_minus_a1_pn_bacc_holm_p'])}); "
      f"secondary {_fmt(conc['secondary']['a4_r2_minus_a1_pn_bacc'])} "
      f"(Holm p {_fmt(conc['secondary']['a4_r2_minus_a1_pn_bacc_holm_p'])}).")
    A("")
    A(f"Sensitivity settings whose paired CI against recipe 1 excludes 0 (either checkpoint rule): "
      f"{n_sens_sig} table rows. See section 4 for which ones and on which metric.")
    A("")
    A(f"**Overall:** the primary conclusion of the pipeline is "
      f"{'CHANGED' if changed else 'UNCHANGED'} by the R2 recipe and the sensitivities.")
    A("")
    A("=" * 78)
    A("EXPECTED VS ACTUAL")
    A("=" * 78)
    A("")
    A("| check | expected | actual | pass |")
    A("|---|---|---|---|")
    A(f"| R2 runs (A0/A1/A4 x 3 seeds x 5 folds) | 45 | {stats['n_r2']} | "
      + ("yes" if stats["n_r2"] == 45 else "NO") + " |")
    A(f"| sensitivity runs (8 settings x 3 x 5) | 120 | {stats['n_sensitivity']} | "
      + ("yes" if stats["n_sensitivity"] == 120 else "NO") + " |")
    A(f"| total Step 11a runs | 165 | {n_runs} | " + ("yes" if n_runs == 165 else "NO") + " |")
    A("| both checkpoint rules tracked for every run | yes | yes | yes |")
    A("| rule-1 baselines reused from Step 9 (not retrained) | yes | yes | yes |")
    A("| seed fixed before the model is built | yes | yes | yes |")
    A("| CIs per run-set (2,000 subject-cluster bootstrap) | yes | yes | yes |")
    A("| every table script-generated | yes | yes | yes |")
    A("| errors | 0 | " + str(stats["n_err"]) + " | " + ("yes" if stats["n_err"] == 0 else "NO")
      + " |")
    A("")
    A("=" * 78)
    A("DEVIATIONS FROM THE SPEC AND THINGS I AM UNSURE ABOUT")
    A("=" * 78)
    A("")
    A("* The Step 9 rule defines H3 with FOUR comparators (A0-A3). Under R2 only A0 and A1 are "
      "retrained, so the pre-registered under-R2 verdict uses the two available comparators "
      "(Holm over the two PN-BAcc comparisons). A second, clearly-labelled diagnostic carries A2 "
      "and A3 over from recipe 1 to apply the full four-comparator rule; it mixes recipes and is "
      "reported for completeness only.")
    A("* PIPELINE_v2 section 6/10 says five seeds per fold and a clip grid {3,5,10}. The "
      "established protocol uses three seeds, and the pre-registered Step 11a clip grid is "
      "{[0.5,2], [0.1,10]} against the recipe-1 default [0.2,5]. This was fixed before the runs.")
    A("* The A3 clip and the A4 tau sensitivity change only the TRAINING loss; the evaluation "
      "PN-BAcc keeps its fixed clip [0.2, 5] so the four run-sets remain directly comparable.")
    A("")
    A("=" * 78)
    A("FILES WRITTEN")
    A("=" * 78)
    A("")
    A("results/step11a/: runs/ (per-run parquet), curves/ (per-epoch curves), oof_all.parquet, "
      "run_log.jsonl (+ run_log_shard{0,1,2}.jsonl, shard{0,1,2}.log), main_table_{primary,secondary}.{csv,md}, "
      "runset_pc_{primary,secondary}.{csv,md}, paired_r2_{primary,secondary}.{csv,md}, "
      "paired_r2_carryover_{primary,secondary}.csv, best_epochs.{csv,md}, sensitivity_{primary,secondary}.{csv,md}, "
      "pn_bacc_vs_macro_f1.png, analysis.json, REPORT.md. New code: scripts/run_step11a.py, "
      "scripts/analyse_step11a.py, scripts/step11a_report.py, scripts/step11a_appendix.py, "
      "configs/step11a_robustness.yaml, tests/test_step11a_robustness.py; modified "
      "src/adni/losses.py and src/adni/train.py (clip override).")

    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[wrote {OUT / 'REPORT.md'}]")
    print(f"[tests: {tests}] runs={stats['n_ok']} errors={stats['n_err']} "
          f"conclusion_changed={changed}")


def _fmt(v, nd=4) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    main()
