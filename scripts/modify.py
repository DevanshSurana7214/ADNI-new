#!/usr/bin/env python3
"""Modify step06c_baselines.py to add B1-imp+mask arm."""
import re

with open('scripts/step06c_baselines.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add b1_imp_mask to NEW_ARMS
content = content.replace(
    'NEW_ARMS = ["b0_oh", "b0_lin_inter", "b1_blind"]',
    'NEW_ARMS = ["b0_oh", "b0_lin_inter", "b1_blind", "b1_imp_mask"]'
)

# Add _impute_blind_with_mask function
impute_func = '''

def _impute_blind_with_mask(
    X_raw: pd.DataFrame,
    present: dict[str, set],
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    feature_spec: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """B1-imp+mask: same imputation as B1-blind but KEEP mask bits."""
    X = X_raw.copy()
    all_ids = np.concatenate([tr, va, te])
    for k in MODALITIES:
        cols = [f"{k}_{c}" for c in feature_spec[k]]
        donors = np.array([i for i in tr.tolist() if i in present[k]], dtype=np.int64)
        if len(donors) == 0:
            raise RuntimeError(f"no present training donors for block {k}")
        absent = [i for i in all_ids.tolist() if i not in present[k]]
        if not absent:
            continue
        pick = rng.choice(donors, size=len(absent), replace=True)
        X.loc[absent, cols] = X_raw.loc[pick, cols].to_numpy()
    return X
'''
content = content.replace('def _load_s6b():', impute_func + '\ndef _load_s6b():')

# Add b1_imp_mask to training loop
old_block = '''            timing["b1_blind"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b1_blind"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                         **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})
        print(f"[seed {seed}] b0_oh / b0_lin_inter / b1_blind done", flush=True)'''

new_block = '''            timing["b1_blind"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b1_blind"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                         **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})
            # B1-imp+mask
            t0 = time.time()
            rng2 = np.random.default_rng(1000 * (seed + 1) + fold)
            Xb_mask = _impute_blind_with_mask(X_raw, present, tr, va, te, spec, rng2)
            p, _ = fit_predict_b1(Xb_mask, ep, tr, va, te, seed=seed, params_override=cfg,
                                  num_rounds=num_rounds, early_stopping=early_stopping)
            timing["b1_imp_mask"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b1_imp_mask"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                            **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})
        print(f"[seed {seed}] b0_oh / b0_lin_inter / b1_blind / b1_imp_mask done", flush=True)'''

content = content.replace(old_block, new_block)

# Update arms_order
content = content.replace('"b1_blind",\r\n                  "b2_mri"', '"b1_blind", "b1_imp_mask",\r\n                  "b2_mri"')

# Add paired diffs
content = content.replace(
    '"b1_i_minus_b1_blind": _paired("b1_i", "b1_blind"),\r\n    }',
    '"b1_i_minus_b1_blind": _paired("b1_i", "b1_blind"),\r\n        "noise_cost": _paired("b1_i", "b1_imp_mask"),\r\n        "mask_info": _paired("b1_imp_mask", "b1_blind"),\r\n    }'
)

# Add b1_imp_mask to bymod
content = content.replace(
    'bymod = {"b1_i": _by_modality("b1_i"), "b1_blind": _by_modality("b1_blind")}',
    'bymod = {"b1_i": _by_modality("b1_i"), "b1_blind": _by_modality("b1_blind"), "b1_imp_mask": _by_modality("b1_imp_mask")}'
)

with open('scripts/step06c_baselines.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done")
