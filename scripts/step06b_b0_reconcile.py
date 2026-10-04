#!/usr/bin/env python3
"""Step 6b (D.10) — reconcile the B0 macro-AUROC gap (Step 6: 0.602 vs Stage 3: 0.618).

Runs the pattern-only shortcut classifier (Stage 3's ``shortcut_benchmark`` definition: a multinomial
logistic regression on the one-hot pattern, macro-OvR AUROC on pooled out-of-fold predictions) under
four conditions that cross the two candidate explanations:

  (a) Stage 3's OWN partition = StratifiedGroupKFold(5, seed 0) stratified on the per-episode y;
  (b) the FROZEN Step 3 splits (S1 seed 0, folds 0-4);
each with (1) the full training split, and (2) the 10 % validation-subject holdout removed from
training (subject-grouped, stratified on the subject-level modal label, seed 42 — exactly Step 3/4).

Nothing in either pipeline is modified; this only re-runs the Stage 3 benchmark function and its
holdout variant and reports the factorial.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from adni.baselines import pattern_onehot
from adni.splits import _subject_label  # modal-diagnosis subject label, same as Step 3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EPISODES = PROJECT_ROOT / "docs" / "episodes.csv"
DEFAULT_SPLITS = PROJECT_ROOT / "results" / "step03" / "splits.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "step06b"
STAGE3_SCRIPT = PROJECT_ROOT / "scripts" / "adni_stage3_profile.py"
CLASSES = ["CN", "MCI", "Dementia"]
VAL_FRACTION = 0.10
VAL_SEED = 42


def _load_stage3():
    spec = importlib.util.spec_from_file_location("adni_stage3_profile", STAGE3_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _benchmark(df: pd.DataFrame, fold_iter, drop_holdout: bool, sub_label: pd.Series) -> dict:
    """Stage 3's shortcut_benchmark internals, with an optional training-subject holdout.

    ``fold_iter`` yields (train_pos, test_pos) positional index arrays. When ``drop_holdout`` is set,
    10 % of the training *subjects* are removed (stratified on the subject label, seed 42) before the
    classifier is fitted — this reproduces the Step 3/4 validation holdout.
    """
    y = df.DIAGNOSIS.map({c: i for i, c in enumerate(CLASSES)}).values
    X = pd.get_dummies(df[["pattern"]].astype(str)).values.astype(float)
    P = np.zeros((len(df), 3))
    rng_sizes = []
    for tr, te in fold_iter:
        if drop_holdout:
            tr_rids = df.RID.values[tr]
            # subject-level label for the training subjects present in this fold
            labels = sub_label.reindex(pd.unique(tr_rids)).to_numpy()
            uniq = pd.unique(tr_rids)
            keep, _ = train_test_split(uniq, test_size=VAL_FRACTION, random_state=VAL_SEED,
                                       stratify=labels)
            rng_sizes.append((len(uniq), len(keep)))
            tr = tr[np.isin(df.RID.values[tr], keep)]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
        P[te] = clf.predict_proba(X[te])
    return {
        "macro_auroc": float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y, P.argmax(1))),
        "n_train_subject_sizes": rng_sizes[:1],
    }


def _benchmark_bits(df: pd.DataFrame, fold_iter, seed: int) -> dict:
    """Step 6's B0 definition: multinomial logistic regression on the four availability bits."""
    y = df.DIAGNOSIS.map({c: i for i, c in enumerate(CLASSES)}).values
    Xb = pattern_onehot(df).to_numpy().astype(float)          # 4 binary columns
    P = np.zeros((len(df), 3))
    for tr, te in fold_iter:
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed).fit(Xb[tr], y[tr])
        P[te] = clf.predict_proba(Xb[te])
    return {"macro_auroc": float(roc_auc_score(y, P, multi_class="ovr", average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, P.argmax(1)))}


def _frozen_folds(blob: dict, pos_of: pd.Series, seed: int, with_val: bool):
    out = []
    for fold in range(5):
        node = blob["s1"][f"seed_{seed}"][str(fold)]["episode_ids"]
        tr = pos_of.loc[node["train"]].to_numpy()
        if with_val:
            tr = np.concatenate([tr, pos_of.loc[node["val"]].to_numpy()])
        out.append((tr, pos_of.loc[node["test"]].to_numpy()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconcile the B0 vs Stage 3 macro-AUROC gap.")
    ap.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    ap.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stage3-seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.episodes, dtype={"pattern": str}).reset_index(drop=True)
    # episode_id is unique but rows are not guaranteed to be in episode_id order.
    pos_of = pd.Series(df.index.values, index=df["episode_id"].values)
    sub_label = _subject_label(df.set_index("episode_id"))

    # ---- (a1) unmodified Stage 3 function, its own partition --------------------------------
    stage3 = _load_stage3()
    s3 = stage3.shortcut_benchmark(df, {"pattern only": ["pattern"]}, seed=a.stage3_seed, folds=5)
    a1_auroc, a1_bacc = [float(v) for v in s3["pattern only"]]

    # ---- (a) Stage 3's own partition folds (reproduce for the holdout variant) ---------------
    y = df.DIAGNOSIS.map({c: i for i, c in enumerate(CLASSES)}).values
    sk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=a.stage3_seed)
    own_folds = list(sk.split(np.zeros(len(df)), y, df.RID.values))
    a2 = _benchmark(df, own_folds, drop_holdout=True, sub_label=sub_label)

    # ---- (b) frozen Step 3 splits (S1 seed 0) -----------------------------------------------
    # b1 fits on the frozen `train` ids only (the 10 % val holdout already removed -> Step 6);
    # b2 fits on frozen `train` + `val` ids (no holdout), same folds.
    blob = json.loads(Path(a.splits).read_text())
    frozen_train_only, frozen_train_plus_val = [], []
    for fold in range(5):
        node = blob["s1"][f"seed_{a.stage3_seed}"][str(fold)]["episode_ids"]
        test = pos_of.loc[node["test"]].to_numpy()
        tr = pos_of.loc[node["train"]].to_numpy()
        va = pos_of.loc[node["val"]].to_numpy()
        frozen_train_only.append((tr, test))
        frozen_train_plus_val.append((np.concatenate([tr, va]), test))
    b1 = _benchmark(df, frozen_train_only, drop_holdout=False, sub_label=sub_label)
    b2 = _benchmark(df, frozen_train_plus_val, drop_holdout=False, sub_label=sub_label)

    # ---- (c) Step 6's exact B0 definition (4-bit encoding) on the frozen splits ----------------
    c1 = _benchmark_bits(df, frozen_train_only, a.stage3_seed)              # train only, seed 0
    c2 = _benchmark_bits(df, frozen_train_plus_val, a.stage3_seed)          # train+val, seed 0
    c4 = _benchmark_bits(df, own_folds, a.stage3_seed)                      # own partition, seed 0
    c3_pts = [_benchmark_bits(df, _frozen_folds(blob, pos_of, s, with_val=False), s)
              for s in (0, 1, 2)]
    c3 = {"macro_auroc": float(np.mean([p["macro_auroc"] for p in c3_pts])),
          "balanced_accuracy": float(np.mean([p["balanced_accuracy"] for p in c3_pts])),
          "per_seed_auroc": [p["macro_auroc"] for p in c3_pts]}

    # ---- Step 6 reference (3-seed mean of per-seed pooled OOF macro-AUROC) -------------------
    try:
        step6 = json.loads((PROJECT_ROOT / "results" / "step06" / "summary.json").read_text())
        step6_auroc = float(step6["arms"]["b0"]["natural"]["macro_auroc"]["point"])
    except Exception:
        step6_auroc = float("nan")

    out = {
        "stage3_seed": a.stage3_seed,
        "a1_own_partition_full_train": {"macro_auroc": a1_auroc, "balanced_accuracy": a1_bacc,
                                        "description": "Stage 3 shortcut_benchmark (unmodified)"},
        "a2_own_partition_with_val_holdout": a2,
        "b1_frozen_splits_train_only": b1,
        "b2_frozen_splits_train_plus_val": b2,
        "c1_frozen_4bit_train_only_seed0": c1,
        "c2_frozen_4bit_train_plus_val_seed0": c2,
        "c3_frozen_4bit_3seed_mean": c3,
        "c4_own_partition_4bit_seed0": c4,
        "step6_b0_3seed_mean_macro_auroc": step6_auroc,
    }
    (a.out / "b0_reconcile.json").write_text(json.dumps(out, indent=2))

    lines = ["B0 reconciliation (pattern-only logistic regression, macro-OvR AUROC on pooled OOF)",
             f"stage3 seed = {a.stage3_seed}", ""]
    lines.append(f"{'run':52s} {'macro-AUROC':>11} {'BAcc':>7}")
    lines.append(f"{'(a1) Stage 3 own partition, full train':52s} {a1_auroc:11.4f} {a1_bacc:7.4f}")
    lines.append(f"{'(a2) Stage 3 own partition, -10% val holdout':52s} "
                 f"{a2['macro_auroc']:11.4f} {a2['balanced_accuracy']:7.4f}")
    lines.append(f"{'(b1) frozen Step 3 splits, train only (Step 6)':52s} "
                 f"{b1['macro_auroc']:11.4f} {b1['balanced_accuracy']:7.4f}")
    lines.append(f"{'(b2) frozen Step 3 splits, train+val (no holdout)':52s} "
                 f"{b2['macro_auroc']:11.4f} {b2['balanced_accuracy']:7.4f}")
    lines.append(f"{'Step 6 B0 (3-seed mean, frozen splits)':52s} {step6_auroc:11.4f}")
    lines.append("")
    lines.append("Encoding isolation (Step 6's exact 4-bit B0 definition):")
    lines.append(f"{'(c1) frozen splits, 4-bit, train only, seed 0':52s} "
                 f"{c1['macro_auroc']:11.4f} {c1['balanced_accuracy']:7.4f}")
    lines.append(f"{'(c2) frozen splits, 4-bit, train+val, seed 0':52s} "
                 f"{c2['macro_auroc']:11.4f} {c2['balanced_accuracy']:7.4f}")
    lines.append(f"{'(c3) frozen splits, 4-bit, 3-seed mean':52s} "
                 f"{c3['macro_auroc']:11.4f} {c3['balanced_accuracy']:7.4f}")
    lines.append(f"{'(c4) own partition, 4-bit, seed 0':52s} {c4['macro_auroc']:11.4f} "
                 f"{c4['balanced_accuracy']:7.4f}")
    text = "\n".join(lines)
    print(text)
    (a.out / "b0_reconcile.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n[wrote {a.out / 'b0_reconcile.json'}]")


if __name__ == "__main__":
    main()
