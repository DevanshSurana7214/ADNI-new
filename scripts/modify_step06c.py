#!/usr/bin/env python3
"""Modify step06c_baselines.py to add B1-imp+mask arm and decomposition analysis."""
import re

with open('scripts/step06c_baselines.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add b1_imp_mask to NEW_ARMS
content = content.replace(
    'NEW_ARMS = ["b0_oh", "b0_lin_inter", "b1_blind"]',
    'NEW_ARMS = ["b0_oh", "b0_lin_inter", "b1_blind", "b1_imp_mask"]'
)

# 2. Add _impute_blind_with_mask function after _impute_blind
impute_with_mask = '''

def _impute_blind_with_mask(
    X_raw: pd.DataFrame,
    present: dict[str, set],
    tr: np.ndarray,
    va: np.ndarray,
    te: np.ndarray,
    feature_spec: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """B1-imp+mask matrix: same donor imputation as B1-blind, but KEEP mask bits."""
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
# Insert before def _load_s6b
content = content.replace('def _load_s6b():', impute_with_mask + '\ndef _load_s6b():')

# 3. Add b1_imp_mask to the training loop
old_b1_blind_block = '''            t0 = time.time()
            rng = np.random.default_rng(1000 * (seed + 1) + fold)
            Xb = _impute_blind(X_raw, present, tr, va, te, spec, rng)
            p, _ = fit_predict_b1(Xb, ep, tr, va, te, seed=seed, params_override=cfg,
                                  num_rounds=num_rounds, early_stopping=early_stopping)
            timing["b1_blind"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b1_blind"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                         **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})
        print(f"[seed {seed}] b0_oh / b0_lin_inter / b1_blind done", flush=True)'''

new_b1_blind_block = '''            t0 = time.time()
            rng = np.random.default_rng(1000 * (seed + 1) + fold)
            Xb = _impute_blind(X_raw, present, tr, va, te, spec, rng)
            p, _ = fit_predict_b1(Xb, ep, tr, va, te, seed=seed, params_override=cfg,
                                  num_rounds=num_rounds, early_stopping=early_stopping)
            timing["b1_blind"] += time.time() - t0
            for i, eid in enumerate(te):
                rows["b1_blind"].append({"episode_id": int(eid), "fold": fold, "seed": seed,
                                         **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})

            # B1-imp+mask: same imputation but keep mask bits
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

content = content.replace(old_b1_blind_block, new_b1_blind_block)

# 4. Update arms_order
content = content.replace(
    '"b1_blind",\r\n                  "b2_mri"',
    '"b1_blind", "b1_imp_mask",\r\n                  "b2_mri"'
)

# 5. Add paired diffs for decomposition
old_diffs = '''        "b1_i_minus_b1_blind": _paired("b1_i", "b1_blind"),
    }'''

new_diffs = '''        "b1_i_minus_b1_blind": _paired("b1_i", "b1_blind"),
        # Decomposition (item 3)
        "noise_cost_b1_minus_b1imp_mask": _paired("b1_i", "b1_imp_mask"),
        "mask_info_b1imp_mask_minus_b1blind": _paired("b1_imp_mask", "b1_blind"),
        "total_b1_minus_b1blind": _paired("b1_i", "b1_blind"),
    }'''

content = content.replace(old_diffs, new_diffs)

# 6. Add b1_imp_mask to by-modality
content = content.replace(
    'bymod = {"b1_i": _by_modality("b1_i"), "b1_blind": _by_modality("b1_blind")}',
    'bymod = {"b1_i": _by_modality("b1_i"), "b1_blind": _by_modality("b1_blind"),\n             "b1_imp_mask": _by_modality("b1_imp_mask")}'
)

# 7. Update by-modality markdown table header
content = content.replace(
    '"# Step 6c - B1 vs B1-blind by number of observed modalities (seed-averaged)"',
    '"# Step 6d - B1 vs B1-imp+mask vs B1-blind by number of observed modalities (seed-averaged)"'
)

content = content.replace(
    '"| n_mod | n_ep | B1 macro-AUROC | B1-blind macro-AUROC | B1 PN-BAcc | B1-blind PN-BAcc |"',
    '"| n_mod | n_ep | B1 macro-AUROC | B1-imp+mask macro-AUROC | B1-blind macro-AUROC | B1 PN-BAcc | B1-imp+mask PN-BAcc | B1-blind PN-BAcc |"'
)

content = content.replace(
    '"|-------|------|----------------|----------------------|-----------|------------------|"',
    '"|-------|------|---------------|----------------------|---------------------|-----------|-------------------|------------------|"'
)

# 8. Update the loop that builds the table
content = content.replace(
    'x, z = bymod["b1_i"][k], bymod["b1_blind"][k]\r\n        bm.append(f"| {nm} | {x[\'n\']} | {x[\'macro_auroc\']:.4f} | {z[\'macro_auroc\']:.4f} | "\r\n                  f"{x[\'pn_bacc_uniform\']:.4f} | {z[\'pn_bacc_uniform\']:.4f} |")',
    'x, y, z = bymod["b1_i"][k], bymod["b1_imp_mask"][k], bymod["b1_blind"][k]\r\n        bm.append(f"| {nm} | {x[\'n\']} | {x[\'macro_auroc\']:.4f} | {y[\'macro_auroc\']:.4f} | {z[\'macro_auroc\']:.4f} | "\r\n                  f"{x[\'pn_bacc_uniform\']:.4f} | {y[\'pn_bacc_uniform\']:.4f} | {z[\'pn_bacc_uniform\']:.4f} |")'
)

# 9. Update console output
content = content.replace(
    'print("\\\\n=== B1 vs B1-blind (uniform CI, seed-averaged) ===")',
    'print("\\\\n=== B1 vs B1-blind vs B1-imp+mask (uniform CI, seed-averaged) ===")'
)

content = content.replace(
    'for arm in ("b1_i", "b1_blind"):',
    'for arm in ("b1_i", "b1_imp_mask", "b1_blind"):'
)

content = content.replace(
    'print(f"  {arm:9s} macro-AUROC',
    'print(f"  {arm:12s} macro-AUROC'
)

# Add decomposition output
old_console_end = '''    d = diffs["b1_i_minus_b1_blind"]["macro_auroc"]
    print(f"  B1 - B1-blind macro-AUROC {d[\'diff\']:+.4f} [{d[\'ci_low\']:+.4f}, {d[\'ci_high\']:+.4f}]")
    print(f"[wrote outputs under {a.out}]")'''

new_console_end = '''    d_noise = diffs["noise_cost_b1_minus_b1imp_mask"]["macro_auroc"]
    d_mask = diffs["mask_info_b1imp_mask_minus_b1blind"]["macro_auroc"]
    d_total = diffs["total_b1_minus_b1blind"]["macro_auroc"]
    print(f"  noise cost (B1 - B1-imp+mask) macro-AUROC {d_noise[\'diff\']:+.4f} [{d_noise[\'ci_low\']:+.4f}, {d_noise[\'ci_high\']:+.4f}]")
    print(f"  mask info (B1-imp+mask - B1-blind) macro-AUROC {d_mask[\'diff\']:+.4f} [{d_mask[\'ci_low\']:+.4f}, {d_mask[\'ci_high\']:+.4f}]")
    print(f"  total (B1 - B1-blind) macro-AUROC {d_total[\'diff\']:+.4f} [{d_total[\'ci_low\']:+.4f}, {d_total[\'ci_high\']:+.4f}]")
    print(f"[wrote outputs under {a.out}]")'''

content = content.replace(old_console_end, new_console_end)

with open('scripts/step06c_baselines.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Modified step06c_baselines.py successfully")
