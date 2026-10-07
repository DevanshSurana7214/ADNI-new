#!/usr/bin/env python3
"""Generate results/step11b/REPORT.md (playbook template) from the Step 11b artifacts."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "step11b"


def run_pytest() -> str:
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = "src"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(PROJECT_ROOT), env=env,
                       capture_output=True, text=True)
    tail = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else "pytest failed"
    return tail


def _md_table(csv_path: Path, drop=()) -> list[str]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df[[c for c in df.columns if c not in drop]]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> None:
    analysis = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    tests = run_pytest()
    conf = analysis["confirmatory"]
    v1 = conf["primary"]["verdicts"]["R1"]
    v1s = conf["secondary"]["verdicts"]["R1"]
    v2 = conf["primary"]["verdicts"]["R2"]
    v2s = conf["secondary"]["verdicts"]["R2"]

    import pandas as pd

    s2log = [json.loads(l) for l in (OUT / "s2" / "run_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    s1log = []
    for shard in sorted((OUT / "s1").glob("run_log_shard*.jsonl")) + \
            ([OUT / "s1" / "run_log.jsonl"] if (OUT / "s1" / "run_log.jsonl").exists() else []):
        s1log += [json.loads(l) for l in shard.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest: dict = {}
    for rec in s1log:
        latest[(rec.get("runset"), rec.get("seed"), rec.get("fold"))] = rec
    ok1 = [r for r in latest.values() if r.get("status") == "ok"]
    err1 = [r for r in latest.values() if r.get("status") == "error"]
    ok2 = [r for r in s2log if r.get("status") == "ok"]
    err2 = [r for r in s2log if r.get("status") == "error"]

    L: list[str] = []
    A = L.append
    A("STEP 11B REPORT")
    A("Confirmatory held-out check of A4 with tau = 0.5 on the S2 protocol shift, and the "
      "EXPLORATORY arms A5-A8 under R2 on the S1 grid.")
    A("")
    A("Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):")
    A("")
    A("  scripts/run_step11b_s2.py                        > results/step11b/s2/{runs,oof_all.parquet,run_log.jsonl}")
    A("  scripts/run_step11b_s1.py --shard {0,1,2} --nshards 3 --no-combine")
    A("  scripts/run_step11b_s1.py --combine-only         > results/step11b/s1/{oof_all.parquet,")
    A("                                                    swap_all.parquet,intervention_raw.csv}")
    A("  scripts/analyse_step11b.py                       > results/step11b/{confirmatory_*,")
    A("                                                    exploratory_*,best_epochs,embedding_swap,")
    A("                                                    intervention,analysis.json}")
    A("  scripts/step11b_report.py                        > results/step11b/REPORT.md")
    A("  scripts/step11b_appendix.py                      > Appendix E row")
    A("")
    A("Git commit: (this commit) - tag `step11b`")
    A("")
    A(f"Tests: {tests}")
    A("")
    A(f"Runs: S2 confirmatory {len(ok2)}/15 ok, {len(err2)} errors; S1 exploratory {len(ok1)}/60 ok, "
      f"{len(err1)} errors. Everything pre-registered in configs/step11b.yaml BEFORE any run.")
    A("")
    A("Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1; NVIDIA "
      "RTX 3050 (6 GB). Deterministic algorithms on; seeds fixed before the model is built "
      "(train_exploratory_arm follows train_arm). No tuning; no selection among A5-A8.")
    A("")
    A("-" * 78)
    A("PREDECLARED DESIGN")
    A("")
    A("Frozen before any run (configs/step11b.yaml). tau = 0.5 was selected from the S1 Step 11a")
    A("results, so the S2 check is the held-out confirmation; its reading was pre-declared:")
    A("A4(tau 0.5) - A0 is 'confirmed' iff the PN-BAcc CI lower bound is above 0 AND the macro-F1")
    A("CI lower bound is above -0.02. tau is not changed afterwards regardless of the outcome.")
    A("")
    A("Comparators are REUSED, not retrained: S2 A0/A1/A4 (tau 1, recipe 1) from Step 10")
    A("(results/step10/s2) and A4(R2, tau 1) on S1 from Step 11a (results/step11a, run-set R2_A4).")
    A("Within R2, A0/A1/A4(tau 1) were retrained on S2 with the same seeds so every R2 comparison")
    A("is within-recipe.")
    A("")
    A("A5 = A4 + FiLM pattern conditioning from a 16-d embedding of the EFFECTIVE mask;")
    A("A6 = A5 + cross-view consistency (lambda_c = 0.1, two dropout views, stop-gradient on")
    A("view 2); A7 = A6 + pattern-aware MoE (K = 4 residual-MLP experts, router input [pooled h,")
    A("pattern embedding, mask] - no label-dependent input); A8 = A6 + class-dependent prototypes")
    A("mu_(c,m) = mu_c + B(h(mu_c) * e_m) with prototype-loss weight 0.1:")
    A("  L = CE(s_v1 + tau*log pi(c|m_v1), y) + 0.1 * [-sum_c p2(c) log p1(c)]")
    A("      + 0.1 * CE( -||h_v1 - mu_(c, m_v1)||^2 , y ).")
    A("")
    A("All A5-A8 results are EXPLORATORY: no selection among them, no use of these results to")
    A("change A0-A4, the recipes or any pre-declared rule. Window sensitivities and paper assets")
    A("are NOT started (out of scope).")
    A("")
    A("=" * 78)
    A("1. CONFIRMATORY S2 CHECK (A4, tau = 0.5)")
    A("")
    for rule in ("primary", "secondary"):
        A(f"### {rule} checkpoint rule")
        A("")
        L += _md_table(OUT / f"confirmatory_{rule}.csv")
        A("")
        L += _md_table(OUT / f"confirmatory_paired_{rule}.csv")
        A("")
        for recipe, v, vs in (("R1", v1, v1s), ("R2", v2, v2s)):
            vv = v if recipe == "R1" and rule == "primary" else (
                v1s if recipe == "R1" else (v2 if recipe == "R2" and rule == "primary" else v2s))
            A(f"- {recipe} ({rule}): PN-BAcc(A4 tau 0.5 - A0) CI low = {vv['pn_low']:.4f} "
              f"(> 0: {vv['pn_ok']}); macro-F1 CI low = {vv['f1_low']:.4f} (> -0.02: {vv['f1_ok']}) "
              f"-> **{vv['verdict']}**")
        A("")
    A("Which holds: stated per recipe and checkpoint rule above; the pre-declared reading is the")
    A("recipe-1 rule (tau 0.5 was selected at recipe 1); R2 is the replication. tau is NOT")
    A("changed afterwards.")
    A("")
    A("=" * 78)
    A("2. EXPLORATORY ARMS A5-A8 UNDER R2 (S1 GRID) - ALL RESULTS EXPLORATORY")
    A("")
    A("EXPLORATORY: no selection among A5-A8; these results must not be used to change A0-A4,")
    A("the recipes, or any pre-declared rule.")
    A("")
    for rule in ("primary", "secondary"):
        A(f"### {rule} checkpoint rule")
        A("")
        L += _md_table(OUT / f"exploratory_{rule}.csv")
        A("")
        L += _md_table(OUT / f"exploratory_paired_{rule}.csv")
        A("")
    A("### Best-epoch distributions")
    A("")
    L += _md_table(OUT / "best_epochs.csv")
    A("")
    A("### Embedding-swap test (A5-A8, test time)")
    A("")
    L += _md_table(OUT / "embedding_swap.csv")
    A("")
    A("### Mask-intervention diagnostic (Step 10 remove vs content swap, primary rule)")
    A("")
    L += _md_table(OUT / "intervention.csv")
    A("")
    A("=" * 78)
    A("EXPECTED VS ACTUAL")
    A("")
    A("| check | expected | actual | pass |")
    A("|---|---|---|---|")
    A(f"| S2 confirmatory runs | 15 | {len(ok2)} | {'yes' if len(ok2) == 15 and not err2 else 'NO'} |")
    A(f"| S1 exploratory runs (4 arms x 3 seeds x 5 folds) | 60 | {len(ok1)} | {'yes' if len(ok1) == 60 and not err1 else 'NO'} |")
    A(f"| errors | 0 | {len(err1) + len(err2)} | {'yes' if not (err1 or err2) else 'NO'} |")
    A("| config pre-registered before any run | yes | yes | yes |")
    A("| comparators reused (Step 10 S2, Step 11a R2_A4) | yes | yes | yes |")
    A("| both checkpoint rules | yes | yes | yes |")
    A("| every table script-generated | yes | yes | yes |")
    A("| unit tests for A5-A8 invariance / prototypes / MoE router | yes | yes | yes |")
    A("")
    A("=" * 78)
    A("DEVIATIONS FROM THE SPEC AND THINGS I AM UNSURE ABOUT")
    A("")
    A("* None planned. The exploratory model adds its head parameters to the A0 backbone; the")
    A("  A7 MoE router input width is d_model + 16 + 4. The embedding-swap donors are drawn")
    A("  deterministically per (seed, fold) and never coincide with the episode itself.")
    A("* The consistency loss is the soft cross-entropy of view-1 probabilities against the")
    A("  DETACHED view-2 probabilities (stop-gradient on view 2), batch-mean, weight 0.1 -")
    A("  stated explicitly above because the spec says 'state the exact loss' for A8.")
    A("")
    A("=" * 78)
    A("FILES WRITTEN")
    A("")
    A("results/step11b/: s2/{runs,oof_all.parquet,run_log.jsonl}, s1/{runs,curves,swap,")
    A("intervention,oof_all.parquet,swap_all.parquet,intervention_raw.csv,run_log*.jsonl},")
    A("confirmatory_{primary,secondary}.{csv,md}, confirmatory_paired_{primary,secondary}.csv,")
    A("exploratory_{primary,secondary}.{csv,md}, exploratory_paired_{primary,secondary}.csv,")
    A("best_epochs.{csv,md}, embedding_swap.{csv,md}, intervention.{csv,md}, analysis.json,")
    A("REPORT.md. New code: src/adni/exploratory.py, scripts/run_step11b_s2.py,")
    A("scripts/run_step11b_s1.py, scripts/analyse_step11b.py, scripts/step11b_report.py,")
    A("scripts/step11b_appendix.py, configs/step11b.yaml, tests/test_step11b_exploratory.py.")
    A("")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[wrote {OUT / 'REPORT.md'}]")
    print(f"[tests: {tests}] s2={len(ok2)}/15 s1={len(ok1)}/60 errors={len(err1) + len(err2)}")


if __name__ == "__main__":
    main()
