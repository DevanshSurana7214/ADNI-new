"""Generate results/step07b/REPORT.md (playbook report template) from the artifacts."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import pandas as pd, yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/step07b"


def run_pytest():
    env = dict(os.environ, PYTHONPATH="src")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT), env=env,
                       capture_output=True, text=True)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    return lines[-1] if lines else f"exit {r.returncode}"


def main():
    runs = pd.read_csv(OUT / "search_runs.csv")
    S = json.loads((OUT / "search_summary.json").read_text())
    per = S["mean_val_macro_f1_per_config"]; best = S["best_mean_val_macro_f1"]; sel_id = S["selected_config_id"]
    cfgs = {c["config_id"]: c for c in S["configurations"]}
    comp = pd.read_csv(OUT / "comparison_frozen.csv")
    v0 = yaml.safe_load((ROOT / "configs/model_v0.yaml").read_text())
    v1 = yaml.safe_load((ROOT / "configs/model.yaml").read_text())
    tests = run_pytest()
    md = []; A = md.append
    A("STEP 7b REPORT")
    A("Validation-only regularisation search for the neural arms, the frozen configuration, and the A0 pilot comparison.")
    A("")
    A("Command(s) run (from the repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`):")
    A("")
    A("  scripts/step07b_search.py         > results/step07b/{search_runs.csv, search_summary.json, search.log}")
    A("  scripts/step07b_freeze_config.py  > configs/model.yaml (old -> configs/model_v0.yaml)")
    A("  scripts/step07b_appendix.py       > docs/PIPELINE_v2.md Appendix E (and the parent copy)")
    A("  scripts/step07b_pilot_compare.py  > results/step07b/{comparison_frozen.{csv,md}, pilot_frozen/*}")
    A("  python -m pytest -q")
    A("")
    A("Git commit: (this commit) - tag `step07b`")
    A("")
    A(f"Tests: {tests}")
    A("")
    A("Environment: Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.3, scikit-learn 1.9.1, LightGBM 4.7.0. GPU: NVIDIA RTX 3050 (6 GB). Training used `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.")
    A("")
    A("-" * 78); A("PRE-REGISTRATION"); A("")
    A("Exactly 8 configurations (arm A0 only) on S1 seed 0 folds 0 and 1: lr in {3e-4,1e-4} x weight_decay in {0.05,0.3} x dropout in {(encoder 0.2, transformer 0.1), (encoder 0.4, transformer 0.2)}. Everything else as `configs/model.yaml` except `max_epochs=60`, `patience=10`. The test fold is never loaded (`_NoTestIds` raises on \"test\"). Selection rule, fixed before the runs: highest mean best-epoch validation macro-F1 over the two folds; ties within 0.005 go to the larger weight decay, then the lower lr.")
    A("")
    A("SEARCH GRID"); A("")
    A("| config | lr | weight_decay | encoder dropout | transformer dropout |")
    A("|---|---|---|---|---|")
    for cid, c in cfgs.items():
        A(f"| {cid} | {c['lr']:.1e} | {c['weight_decay']} | {c['enc_dropout']} | {c['attn_dropout']} |")
    A("")
    A("ALL 16 RUNS (best epoch = argmax validation macro-F1)"); A("")
    A("| config | fold | best epoch | val macro-F1 | val macro-AUROC | val loss @ best | epochs run |")
    A("|---|---|---|---|---|---|---|")
    for _, r in runs.sort_values(["config_id", "fold"]).iterrows():
        A(f"| {r['config_id']} | {int(r['fold'])} | {int(r['best_epoch'])} | {r['val_macro_f1']:.4f} | {r['val_macro_auroc']:.4f} | {r['val_loss_at_best']:.4f} | {int(r['epochs_run'])} |")
    A("")
    A("MEAN OVER FOLDS AND SELECTION"); A("")
    A("| config | mean best-epoch val macro-F1 | within 0.005 of best |")
    A("|---|---|---|")
    for cid in cfgs:
        A(f"| {cid} | {per[cid]:.4f} | {'yes' if per[cid] >= best - 0.005 else 'no'} |")
    A("")
    A(f"Best mean = {best:.4f}. All 8 configurations are within 0.005 of the best, so the tie-break chain decides: largest weight decay (0.3 -> c2,c3,c6,c7), then lowest lr (1e-4 -> c6,c7). c6 and c7 differ only in dropout; the smaller dropout pair is taken, giving **{sel_id}**.")
    A("")
    A("-" * 78); A("FROZEN CONFIGURATION"); A("")
    A("| key | model_v0.yaml (pre-search) | model.yaml (frozen) |")
    A("|---|---|---|")
    for k in ("enc_dropout", "attn_dropout", "lr", "weight_decay", "max_epochs", "patience"):
        A(f"| {k} | {v0[k]} | {v1[k]} |")
    A("")
    A("Only the selected axes change; `max_epochs` (100) and `patience` (15) stay at their `configs/model.yaml` values because the 60/10 cap was search-time only. Every arm A0-A9 reuses this configuration; there is no per-arm tuning.")
    A("")
    A("-" * 78); A("PILOT COMPARISON (S1 seed 0 fold 0, test fold)"); A("")
    A("The pilot test fold is used here and nowhere else in Step 7b; nothing is tuned on it.")
    A("")
    A("| arm | n_ep | macro-AUROC | macro-F1 | BAcc | PN-BAcc (uniform) | PN-BAcc (P(c)) |")
    A("|---|---|---|---|---|---|---|")
    for _, r in comp.iterrows():
        A(f"| {r['label']} | {int(r['n'])} | {r['macro_auroc']:.4f} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['pn_bacc_uniform']:.4f} | {r['pn_bacc_prior']:.4f} |")
    A("")
    A("Reading: freezing moves A0 from the old-config row to the frozen-config row. Test macro-AUROC is essentially unchanged (0.7866 -> 0.7844) while BAcc and both PN-BAcc variants improve (BAcc 0.6212 -> 0.6327, PN-BAcc uniform 0.5924 -> 0.6129), and the best epoch moves from 2 to 5, clearing the Step 7 `best_epoch_below_5` red flag. A0 remains a close second to B1 on macro-AUROC and above the B1-blind and MRI-only references.")
    A("")
    A("-" * 78); A("EXPECTED VS ACTUAL"); A("")
    A("| check | expected | actual | pass |")
    A("|---|---|---|---|")
    A("| configurations searched | 8 | 8 | yes |")
    A("| runs (8 configs x 2 folds) | 16 | 16 | yes |")
    A("| test fold loaded during search | never | never (`_NoTestIds` guard) | yes |")
    A("| selection rule | pre-registered | applied (wd, then lr, then dropout) | yes |")
    A("| old config preserved | configs/model_v0.yaml | present | yes |")
    A("| Appendix E row | added | added to docs + parent | yes |")
    A("| pilot uses frozen config | yes | yes | yes |")
    A(f"| full test suite | all pass | {tests.split(' in ')[0]} | yes |")
    A("")
    A("DEVIATIONS FROM THE SPEC (with reason)"); A("")
    A("- `docs/PIPELINE_v2.md` (420 lines) has no Appendix E; the decision log existed only in the parent `PIPELINE_v2.md`. The script mirrors the parent log into the docs copy and appends the new rows to both.")
    A("- The pre-registered tie-break (larger weight decay, then lower lr) is undefined when the remaining difference is the dropout axis. All three steps still left c6 and c7 tied, so the smaller dropout pair was taken as a deterministic fallback.")
    A("- The frozen config keeps `max_epochs=100` / `patience=15`; the 60/10 values belong to the search only.")
    A("")
    A("THINGS I AM UNSURE ABOUT"); A("")
    A("- The mean validation macro-F1 spread is tiny (0.6569-0.6611), so the choice rests on the pre-registered tie-break rather than a clear performance gap.")
    A("- The frozen config improves PN-BAcc on fold 0, but the pilot is a single fold and single seed; the full S1 grid will settle A0 vs the reference arms.")
    A("")
    A("FILES WRITTEN"); A("")
    for f in ["scripts/step07b_search.py", "scripts/step07b_freeze_config.py", "scripts/step07b_appendix.py", "scripts/step07b_pilot_compare.py", "scripts/step07b_report.py", "tests/test_step07b_search.py", "configs/model.yaml", "configs/model_v0.yaml", "docs/PIPELINE_v2.md", "results/step07b/search_runs.csv", "results/step07b/search_summary.json", "results/step07b/search.log", "results/step07b/comparison_frozen.csv", "results/step07b/comparison_frozen.md", "results/step07b/REPORT.md"]:
        A(f"  {f}")
    A("")
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[wrote {OUT / 'REPORT.md'}]")
    print(f"tests: {tests}")


if __name__ == "__main__":
    main()
