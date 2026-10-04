#!/usr/bin/env python3
"""Step 5 — reproduce and print the metrics synthetic checks with their ACTUAL values.

These are the same checks asserted in ``tests/test_metrics.py`` (playbook Step 5), but printed with
the concrete numbers so the report can quote them: the hand-computed 6-sample example (including the
probability matrix), the actual ECE of a perfectly calibrated synthetic sample, the random-AUROC
value, the cluster-bootstrap CI width against the analytic one, and the PN-weight balance.

Output is written to ``results/step05/synthetic_checks.txt`` (and echoed to stdout).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from adni.metrics import (
    CLASSES,
    cluster_bootstrap,
    cluster_bootstrap_multi,
    expected_calibration_error,
    holm,
    multiclass_brier,
    natural_metrics,
    pi_table_from_counts,
    pn_balanced_accuracy,
    pn_weights,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "step05" / "synthetic_checks.txt"

lines: list[str] = []


def p(s: str = "") -> None:
    lines.append(s)
    print(s)


def main() -> None:
    p("STEP 5 SYNTHETIC CHECKS (actual values)")
    p("=" * 72)

    # --- 1. perfect predictions ---------------------------------------------------------------
    n = 300
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=n)
    proba = np.zeros((n, 3))
    proba[np.arange(n), y] = 1.0
    m = natural_metrics(y, proba)
    p("[1] perfect predictions (n=%d)" % n)
    p(f"    macro-AUROC={m['macro_auroc']:.6f}  macro-F1={m['macro_f1']:.6f}  "
      f"balanced_acc={m['balanced_accuracy']:.6f}  Brier={m['brier']:.6f}")

    # --- 2. random predictions on balanced data ------------------------------------------------
    n = 20000
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=n)
    proba = rng.random((n, 3))
    proba /= proba.sum(axis=1, keepdims=True)
    m = natural_metrics(y, proba)
    p("")
    p(f"[2] random predictions on balanced data (n={n})")
    p(f"    macro-AUROC={m['macro_auroc']:.6f}   (chance 0.5, tolerance 0.03)")

    # --- 3. hand-computed 6-sample example -----------------------------------------------------
    y6 = np.array([0, 1, 2, 0, 1, 2])
    P6 = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.3, 0.5],
        [0.1, 0.8, 0.1],
        [0.5, 0.3, 0.2],
    ])
    m6 = natural_metrics(y6, P6)
    p("")
    p("[3] hand-computed 6-sample example")
    p("    y      = [0, 1, 2, 0, 1, 2]")
    p("    proba  =")
    for row in P6:
        p("             [" + ", ".join(f"{v:.1f}" for v in row) + "]")
    p(f"    argmax = {P6.argmax(axis=1).tolist()}  (classes {list(CLASSES)})")
    p(f"    macro-AUROC        = {m6['macro_auroc']:.7f}   (mean of 0.875, 0.9375, 0.875)")
    p(f"    macro-F1           = {m6['macro_f1']:.7f}   (= 2/3)")
    p(f"    balanced accuracy  = {m6['balanced_accuracy']:.7f}   (= 2/3)")
    p("    per-class F1       = " + ", ".join(
        f"{CLASSES[c]} {v:.4f}" for c, v in m6["f1_per_class"].items()))
    p(f"    multiclass Brier   = {m6['brier']:.7f}   (0.4266667)")
    p(f"    ECE (15 bins)      = {expected_calibration_error(y6, P6):.7f}   (0.3666667)")

    # --- 4. ECE of a perfectly calibrated synthetic sample -------------------------------------
    n = 40000
    rng = np.random.default_rng(7)
    pred = rng.integers(0, 3, size=n)
    conf = rng.uniform(0.4, 1.0, size=n)
    proba = np.full((n, 3), np.nan)
    proba[np.arange(n), pred] = conf
    rest = (1.0 - conf) / 2.0
    for c in range(3):
        col = proba[:, c]
        miss = np.isnan(col)
        col[miss] = rest[miss]
    y = np.where(rng.random(n) < conf, pred, (pred + rng.integers(1, 3, size=n)) % 3)
    ece = expected_calibration_error(y, proba)
    p("")
    p(f"[4] ECE of a perfectly calibrated synthetic sample (n={n})")
    p(f"    ECE = {ece:.6f}   (< 0.02 required)")

    # --- 5. cluster-bootstrap CI width vs analytic ---------------------------------------------
    n_subj = 400
    rng = np.random.default_rng(3)
    groups = np.arange(n_subj)
    q = 0.6
    yv = (rng.random(n_subj) < q).astype(float)
    dummy = np.zeros((n_subj, 2))

    def mean_fn(yy, pp):
        return float(np.asarray(yy).mean())

    ci = cluster_bootstrap(mean_fn, yv, dummy, groups, n=2000, seed=42)
    se = np.sqrt(q * (1 - q) / n_subj)
    analytic = 2 * 1.96 * se
    width = ci.ci_high - ci.ci_low
    p("")
    p(f"[5] cluster-bootstrap CI width vs analytic (p={q}, n_subjects={n_subj}, 2000 resamples)")
    p(f"    bootstrap width={width:.6f}   analytic width={analytic:.6f}   "
      f"relative diff={abs(width - analytic) / analytic * 100:.2f}%")

    # --- 6. PN weights balance classes within each pattern -------------------------------------
    patterns = np.array(["1000"] * 300 + ["1111"] * 300)
    yv = np.array([0] * 240 + [1] * 60 + [2] * 120 + [1] * 180)
    counts = np.array([[240, 0], [60, 180], [0, 120]], dtype=float)
    prior = pi_table_from_counts(counts, ["1000", "1111"], CLASSES, alpha=0.0)
    w = pn_weights(yv, patterns, prior, rho="uniform", wmin=0.0, wmax=1e9)
    p("")
    p("[6] PN weights make classes balanced within each pattern (alpha=0, unclipped)")
    p(f"    mean(w) = {w.mean():.9f}")
    for pat in ("1000", "1111"):
        msk = patterns == pat
        present = [c for c in (0, 1, 2) if (msk & (yv == c)).any()]
        shares = [w[msk & (yv == c)].sum() / w[msk].sum() for c in present]
        p(f"    pattern {pat}: weighted class shares = "
          + ", ".join(f"{CLASSES[c]} {s:.9f}" for c, s in zip(present, shares)))

    # --- 7. pn_balanced_accuracy hand example --------------------------------------------------
    yy = np.array([0, 0, 0, 0, 1, 2])
    ww = np.array([1.0, 1.0, 1.0, 1.0, 4.0, 4.0])
    p("")
    p("[7] pn_balanced_accuracy (weights upweight the rare classes)")
    p(f"    perfect pred -> {pn_balanced_accuracy(yy, yy, ww):.6f}   (1.0)")
    p(f"    all-MCI pred -> {pn_balanced_accuracy(yy, np.full_like(yy, 1), ww):.6f}   (1/3)")

    # --- 8. Holm and shared-resample bootstrap -------------------------------------------------
    adj = holm([0.01, 0.04, 0.03, 0.5])
    p("")
    p(f"[8] holm([0.01, 0.04, 0.03, 0.5]) = {np.round(adj, 4).tolist()}")

    rng = np.random.default_rng(9)
    groups = np.arange(300)
    yv = rng.integers(0, 3, size=300)
    pv = rng.random((300, 3))
    pv /= pv.sum(1, keepdims=True)
    multi = cluster_bootstrap_multi(
        {"mean": mean_fn, "auroc": lambda a, b: natural_metrics(a, b)["macro_auroc"]},
        yv, pv, groups, n=300, seed=5)
    single = cluster_bootstrap(mean_fn, yv, pv, groups, n=300, seed=5)
    p(f"    cluster_bootstrap_multi matches cluster_bootstrap: point "
      f"{multi['mean'].point:.9f} vs {single.point:.9f}; "
      f"CI [{multi['mean'].ci_low:.6f}, {multi['mean'].ci_high:.6f}] vs "
      f"[{single.ci_low:.6f}, {single.ci_high:.6f}]")

    p("")
    p("Brier helper check: multiclass_brier([0,1,2],[one-hot]) = "
      f"{multiclass_brier(np.array([0, 1, 2]), np.eye(3)):.6f}   (0.0)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[wrote {OUT}]")


if __name__ == "__main__":
    main()
