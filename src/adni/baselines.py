"""Non-neural reference baselines B0, B1, B2 (PIPELINE_v2 §9).

* **B0** — multinomial logistic regression on the four availability bits (the pattern) only. This is
  the shortcut lower bound (H1): it sees *which scans exist* and nothing else.
* **B1** — LightGBM on the concatenation of all four blocks using the RAW features (NaN left as NaN,
  trees split on missing natively; no z-scoring) plus the four block mask bits.
* **B2** — LightGBM on a single block, trained and evaluated only on episodes where that modality is
  present.

All models are fitted on a Step 3 training split and early-stop on the corresponding validation split;
test predictions are out-of-fold. No label ever enters a feature matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .metrics import CLASSES

MODALITIES: Sequence[str] = ("mri", "fdg", "amy", "tau")

# Fixed, small LightGBM config for B1/B2 (playbook Step 6).
B1_PARAMS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "class_weight": None,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}
B1_NUM_ROUNDS = 1000
B1_EARLY_STOPPING = 50


# --------------------------------------------------------------------------- data loading
def load_feature_spec(path: Path) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())


def load_blocks(blocks_dir: Path) -> dict[str, pd.DataFrame]:
    blocks_dir = Path(blocks_dir)
    return {k: pd.read_csv(blocks_dir / f"{k}.csv").set_index("episode_id") for k in MODALITIES}


def pattern_onehot(ep: pd.DataFrame) -> pd.DataFrame:
    """Four binary availability bits, one per modality (equivalent to the 15-pattern one-hot)."""
    pat = np.asarray(ep["pattern"].values, dtype=str)
    return pd.DataFrame(
        {f"mask_{k}": np.array([int(p[i]) for p in pat], dtype=np.int8) for i, k in enumerate(MODALITIES)},
        index=ep.index,
    )


def build_raw_matrix(
    ep: pd.DataFrame,
    blocks: dict[str, pd.DataFrame],
    feature_spec: dict[str, list[str]],
) -> pd.DataFrame:
    """Concatenated RAW block features (block-prefixed) + the four mask bits, indexed by episode_id.

    Absent modalities leave their feature columns NaN (their block table has no row); LightGBM splits
    on missing values natively. amy and tau share SUVR names, so every column is prefixed with its
    block name.
    """
    parts: dict[str, pd.Series] = {}
    for k in MODALITIES:
        blk = blocks[k]
        for c in feature_spec[k]:
            parts[f"{k}_{c}"] = blk[c]
    X = pd.DataFrame(parts).reindex(ep.index)
    masks = pattern_onehot(ep)
    for c in masks.columns:
        X[c] = masks[c].to_numpy()
    return X


def training_prior_table(ep: pd.DataFrame, train_ids: np.ndarray, alpha: float = 0.5) -> pd.DataFrame:
    """P(c|m) fitted on the TRAINING episodes only (Dirichlet-smoothed)."""
    from .metrics import pi_table_from_counts
    from .priors import crosstab

    sub = ep.loc[train_ids]
    diag = pd.Series([CLASSES[int(c)] for c in sub["y"]], index=sub.index)
    counts, pats = crosstab(diag, sub["pattern"])
    return pi_table_from_counts(counts, pats, CLASSES, alpha=alpha)


# --------------------------------------------------------------------------- models
def _y_of(ep: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """Label vector aligned to *ids*, looked up by episode_id (never by position)."""
    return ep["y"].loc[list(np.asarray(ids))].to_numpy()


def _lgb_params(seed: int, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    params = dict(B1_PARAMS)
    if overrides:
        params.update(overrides)
    params.update(seed=int(seed), data_random_seed=int(seed),
                  feature_fraction_seed=int(seed), bagging_seed=int(seed))
    return params


def fit_predict_b1(
    X: pd.DataFrame,
    ep: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    seed: int = 0,
    params_override: dict[str, Any] | None = None,
    num_rounds: int = B1_NUM_ROUNDS,
    early_stopping: int = B1_EARLY_STOPPING,
) -> tuple[np.ndarray, lgb.Booster]:
    """LightGBM on all blocks + mask bits. Returns (test_proba, booster)."""
    params = _lgb_params(seed, params_override)
    dtrain = lgb.Dataset(X.loc[train_ids], label=_y_of(ep, train_ids))
    dval = lgb.Dataset(X.loc[val_ids], label=_y_of(ep, val_ids), reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=num_rounds, valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    return booster.predict(X.loc[test_ids]), booster


def fit_predict_b2(
    X: pd.DataFrame,
    ep: pd.DataFrame,
    present_ids: np.ndarray,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    test_ids: np.ndarray,
    seed: int = 0,
    params_override: dict[str, Any] | None = None,
    num_rounds: int = B1_NUM_ROUNDS,
    early_stopping: int = B1_EARLY_STOPPING,
) -> tuple[np.ndarray, np.ndarray, lgb.Booster]:
    """LightGBM on a single block, restricted to episodes where the modality is present.

    ``present_ids`` is the set of episode_ids whose modality mask is 1. Returns
    (test_proba, test_ids_present, booster).
    """
    pres = set(np.asarray(present_ids).tolist())
    tr = np.array([i for i in train_ids if i in pres], dtype=np.int64)
    va = np.array([i for i in val_ids if i in pres], dtype=np.int64)
    te = np.array([i for i in test_ids if i in pres], dtype=np.int64)
    params = _lgb_params(seed, params_override)
    dtrain = lgb.Dataset(X.loc[tr], label=_y_of(ep, tr))
    dval = lgb.Dataset(X.loc[va], label=_y_of(ep, va), reference=dtrain)
    booster = lgb.train(
        params, dtrain, num_boost_round=num_rounds, valid_sets=[dval],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )
    return booster.predict(X.loc[te]), te, booster


def fit_predict_b0(
    ep: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Multinomial logistic regression on the four availability bits."""
    X = pattern_onehot(ep)
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    clf.fit(X.loc[train_ids], _y_of(ep, train_ids))
    return clf.predict_proba(X.loc[test_ids])
