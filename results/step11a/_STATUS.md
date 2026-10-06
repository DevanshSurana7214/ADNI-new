# Step 11a — STATUS (work stopped on request)

## Where it stopped
The S1 robustness grid was mid-run as **3 parallel background shards** and was **killed on request**.

- **140 / 165 runs done, 25 missing, 0 corrupt parquet files.**
- Missing runs (all in the last two run-sets):
  - `A0_pdrop0p5` (seed 2, fold 3) — 1 run
  - `A4_pdrop0` — 9 runs: (0,1) (0,3) (0,4) (1,1) (1,2) (1,4) (2,0) (2,2) (2,3)
  - `A4_pdrop0p5` — all 15 runs (seeds 0/1/2 × folds 0..4)
- Shard logs stopped at: shard0 46/55, shard1 44/55, shard2 50/55.
- No `python.exe` processes remain running.

## Already built and unit-tested (uncommitted)
- `configs/step11a_robustness.yaml` — pre-registered grid (R2 recipe + 8 sensitivities).
- `src/adni/losses.py` — added `clip` override to `joint_weights` / `arm_loss`.
- `src/adni/train.py` — added `TrainConfig.clip`, threaded into the batch loss.
- `scripts/run_step11a.py` — resumable S1 runner (R2 + sensitivities, both checkpoint rules,
  `--shard/--nshards/--no-combine/--combine-only/--force/--runsets`).
- `scripts/analyse_step11a.py` — tables, CIs, paired R2 + H3-under-R2 verdicts, best-epoch flags,
  sensitivity deltas, `pn_bacc_vs_macro_f1.png` (syntax-checked; not yet run on the data).
- `scripts/step11a_report.py`, `scripts/step11a_appendix.py` — report + Appendix E row.
- `tests/test_step11a_robustness.py` — 10 tests. Related suites: 36 passed.
- Determinism verified: forced rerun of `A0_pdrop0` s0 f0 reproduced the OOF exactly (max prob dev 0.0).
- Existing artifacts present: `results/step11a/runs/*.parquet`, `curves/*.csv`,
  `run_log_shard{0,1,2}.jsonl`, `shard{0,1,2}.log`.

## To resume (repo root, `PYTHONPATH=src ./.venv/Scripts/python.exe`)
1. Finish the grid (resumable — skips the 140 done):
   `scripts/run_step11a.py --no-combine`  (optionally re-shard with `--shard K --nshards N`)
2. Build the combined OOF + merged log: `scripts/run_step11a.py --combine-only`
3. `scripts/analyse_step11a.py`
4. `scripts/step11a_report.py`   (also runs pytest)
5. `scripts/step11a_appendix.py` (appends Appendix E row 11a to both `PIPELINE_v2.md`)
6. Run the full `pytest -q`, then commit `step11a: <title>` and tag `step11a`.

## Open questions to confirm
- H3 under R2 uses the two comparators actually retrained under R2 (A0, A1), Holm over 2; a clearly
  labelled carry-over diagnostic adds recipe-1 A2/A3 for the full 4-comparator rule.
- No A5–A9, no C2, no window (+/-90 / +/-365 d) sensitivities (out of scope for 11a).
