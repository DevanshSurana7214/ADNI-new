import importlib.util, json, time
from pathlib import Path
import numpy as np, pandas as pd, yaml
from adni.baselines import MODALITIES, build_raw_matrix, fit_predict_b1, load_blocks, load_feature_spec, training_prior_table
from adni.metrics import paired_cluster_bootstrap, pn_balanced_accuracy, pn_weights, natural_metrics

ROOT = Path('.').resolve()
ep = pd.read_csv(ROOT/'docs/episodes.csv', dtype={'pattern': str}).set_index('episode_id')
fspec = load_feature_spec(ROOT/'results/step01/stage1/feature_spec.json')
blocks = load_blocks(ROOT/'results/step01/stage1/blocks')
blob = json.loads((ROOT/'results/step03/splits.json').read_text())
cfg = yaml.safe_load((ROOT/'configs/b1.yaml').read_text())
cfg.pop('num_rounds', None); cfg.pop('early_stopping', None)
NR = 1000; ES = 50

X_raw = build_raw_matrix(ep, blocks, fspec)
present = {k: set(blocks[k].index.tolist()) for k in MODALITIES}
PROBA_COLS = ['p_CN', 'p_MCI', 'p_Dementia']

spec = importlib.util.spec_from_file_location('s6b', ROOT/'scripts/step06b_baselines.py')
s6b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6b)

def impute_with_mask(X_raw, present, tr, va, te, fs, rng):
    X = X_raw.copy()
    all_ids = np.concatenate([tr, va, te])
    for k in MODALITIES:
        cols = [f'{k}_{c}' for c in fs[k]]
        donors = np.array([i for i in tr.tolist() if i in present[k]], dtype=np.int64)
        absent = [i for i in all_ids.tolist() if i not in present[k]]
        if absent:
            pick = rng.choice(donors, size=len(absent), replace=True)
            X.loc[absent, cols] = X_raw.loc[pick, cols].to_numpy()
    return X

def wmap_for(ep, tr, te):
    prior = training_prior_table(ep, tr)
    ytr = ep['y'].loc[list(tr)].to_numpy()
    pc = np.bincount(ytr, minlength=3).astype(float)
    pc = pc / pc.sum()
    yy = ep['y'].loc[list(te)].to_numpy()
    pp = ep['pattern'].loc[list(te)].to_numpy()
    return pd.DataFrame({'episode_id': np.asarray(te),
        'w_uniform': pn_weights(yy, pp, prior, rho='uniform'),
        'w_prior': pn_weights(yy, pp, prior, rho=pc)})

rows, wmap = [], {}
for seed in [0, 1, 2]:
    for fold in [0, 1, 2, 3, 4]:
        node = blob['s1'][f'seed_{seed}'][str(fold)]['episode_ids']
        tr = np.asarray(node['train'], dtype=np.int64)
        va = np.asarray(node['val'], dtype=np.int64)
        te = np.asarray(node['test'], dtype=np.int64)
        wmap[(seed, fold)] = wmap_for(ep, tr, te)
        t0 = time.time()
        rng = np.random.default_rng(1000 * (seed + 1) + fold)
        Xb = impute_with_mask(X_raw, present, tr, va, te, fspec, rng)
        p, _ = fit_predict_b1(Xb, ep, tr, va, te, seed=seed, params_override=cfg,
                              num_rounds=NR, early_stopping=ES)
        for i, eid in enumerate(te):
            rows.append({'episode_id': int(eid), 'fold': fold, 'seed': seed,
                        **{c: float(p[i, j]) for j, c in enumerate(PROBA_COLS)}})
    print(f'seed {seed} done', flush=True)

oof_imp = pd.DataFrame(rows)
OUT = ROOT/'results/step06d'
OUT.mkdir(parents=True, exist_ok=True)
oof_imp.to_parquet(OUT/'oof_b1_imp_mask.parquet', index=False)

def with_w(df):
    parts = []
    for (s, f), w in wmap.items():
        sub = df[(df['seed'] == s) & (df['fold'] == f)]
        if len(sub): parts.append(sub.merge(w, on='episode_id', how='left'))
    out = pd.concat(parts, ignore_index=True)
    assert out['w_uniform'].notna().all()
    return out

oof_b1 = pd.read_parquet(ROOT/'results/step06b/oof_b1_i.parquet')
oof_blind = pd.read_parquet(ROOT/'results/step06c/oof_b1_blind.parquet')
frames = {'b1_i': with_w(oof_b1), 'b1_blind': with_w(oof_blind), 'b1_imp_mask': with_w(oof_imp)}

uniform = {arm: s6b._uniform_table(frames[arm], ep, 2000, boot_seed=0)[0] for arm in frames}

def paired(a, b):
    ga, gb = s6b._uniform_geps(frames[a], ep), s6b._uniform_geps(frames[b], ep)
    y = ep['y'].loc[ga.index].to_numpy()
    groups = ep['RID'].loc[ga.index].to_numpy()
    pa = np.column_stack([ga[PROBA_COLS].to_numpy(), ga['w_u'].to_numpy()])
    pb = np.column_stack([gb[PROBA_COLS].to_numpy(), gb['w_u'].to_numpy()])
    mf = s6b.metric_family()
    out = {}
    for k, fn in [('macro_auroc', mf['macro_auroc']), ('pn_bacc_uniform', mf['pn_bacc_uniform']),
                  ('macro_f1', mf['macro_f1']), ('balanced_accuracy', mf['balanced_accuracy'])]:
        ci = paired_cluster_bootstrap(lambda yy, pp, _f=fn: _f(yy, pp), y, pa, pb, groups, n=2000, seed=0)
        out[k] = {'diff': ci.point, 'ci_low': ci.ci_low, 'ci_high': ci.ci_high, 'n_valid': ci.n_valid}
    return out

diffs = {'noise_cost': paired('b1_i', 'b1_imp_mask'),
         'mask_info': paired('b1_imp_mask', 'b1_blind'),
         'total': paired('b1_i', 'b1_blind')}

def bymod(arm):
    g = s6b._uniform_geps(frames[arm], ep)
    nm = ep['n_mod'].loc[g.index].to_numpy()
    y = ep['y'].loc[g.index].to_numpy()
    proba = g[PROBA_COLS].to_numpy()
    w = g['w_u'].to_numpy()
    out = {}
    for n in (1, 2, 3, 4):
        sel = nm == n
        if sel.sum() == 0: continue
        m = natural_metrics(y[sel], proba[sel])
        out[f'n_mod={n}'] = {'n': int(sel.sum()), 'macro_auroc': float(m['macro_auroc']),
            'macro_f1': float(m['macro_f1']),
            'pn_bacc_uniform': float(pn_balanced_accuracy(y[sel], proba[sel].argmax(axis=1), w[sel]))}
    return out

bm = {arm: bymod(arm) for arm in frames}

md = ['# Step 6d - B1 variants decomposition (seed-averaged, 2000 resamples)', '',
      '## Uniform CI for B1 variants', '',
      '| arm | macro-AUROC | macro-F1 | balanced acc | PN-BAcc (uniform) |',
      '|-----|-------------|----------|--------------|-------------------|']
for arm in ['b1_i', 'b1_imp_mask', 'b1_blind']:
    t = uniform[arm]
    md.append(f'| {arm} | {t["macro_auroc"]["point"]:.4f} [{t["macro_auroc"]["ci_low"]:.4f}, {t["macro_auroc"]["ci_high"]:.4f}] | '
              f'{t["macro_f1"]["point"]:.4f} [{t["macro_f1"]["ci_low"]:.4f}, {t["macro_f1"]["ci_high"]:.4f}] | '
              f'{t["balanced_accuracy"]["point"]:.4f} [{t["balanced_accuracy"]["ci_low"]:.4f}, {t["balanced_accuracy"]["ci_high"]:.4f}] | '
              f'{t["pn_bacc_uniform"]["point"]:.4f} [{t["pn_bacc_uniform"]["ci_low"]:.4f}, {t["pn_bacc_uniform"]["ci_high"]:.4f}] |')
md += ['', '## Decomposition of B1 - B1-blind difference', '',
       ' noise cost = B1 - B1-imp+mask (donor-value noise + routing)',
       ' mask information = B1-imp+mask - B1-blind (knowledge of which scans exist, holding noise constant)',
       ' total = B1 - B1-blind (same as Step 6c; upper bound on availability benefit)', '',
       '| Component | macro-AUROC diff | macro-F1 diff | balanced acc diff | PN-BAcc diff |',
       '|-----------|------------------|---------------|-------------------|--------------|']
for name, key in [('noise cost (B1 - B1-imp+mask)', 'noise_cost'),
                   ('mask information (B1-imp+mask - B1-blind)', 'mask_info'),
                   ('total (B1 - B1-blind)', 'total')]:
    d = diffs[key]
    md.append(f'| {name} | {d["macro_auroc"]["diff"]:+.4f} [{d["macro_auroc"]["ci_low"]:+.4f}, {d["macro_auroc"]["ci_high"]:+.4f}] | '
              f'{d["macro_f1"]["diff"]:+.4f} [{d["macro_f1"]["ci_low"]:+.4f}, {d["macro_f1"]["ci_high"]:+.4f}] | '
              f'{d["balanced_accuracy"]["diff"]:+.4f} [{d["balanced_accuracy"]["ci_low"]:+.4f}, {d["balanced_accuracy"]["ci_high"]:+.4f}] | '
              f'{d["pn_bacc_uniform"]["diff"]:+.4f} [{d["pn_bacc_uniform"]["ci_low"]:+.4f}, {d["pn_bacc_uniform"]["ci_high"]:+.4f}] |')
md += ['', '## By number of observed modalities', '',
       '| n_mod | n_ep | B1 macro-AUROC | B1-imp+mask macro-AUROC | B1-blind macro-AUROC |',
       '|-------|------|----------------|----------------------|---------------------|']
for n in (1, 2, 3, 4):
    k = f'n_mod={n}'
    if k not in bm['b1_i']: continue
    x, y, z = bm['b1_i'][k], bm['b1_imp_mask'][k], bm['b1_blind'][k]
    md.append(f'| {n} | {x["n"]} | {x["macro_auroc"]:.4f} | {y["macro_auroc"]:.4f} | {z["macro_auroc"]:.4f} |')

(OUT/'b1_decomposition.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
with open(OUT/'b1_decomposition.json', 'w') as f:
    json.dump({'uniform': uniform, 'diffs': diffs, 'by_modality': bm}, f, indent=2)

print('\n=== Results ===')
for arm in ['b1_i', 'b1_imp_mask', 'b1_blind']:
    t = uniform[arm]
    print(f'{arm:12s} AUROC {t["macro_auroc"]["point"]:.4f} [{t["macro_auroc"]["ci_low"]:.4f}, {t["macro_auroc"]["ci_high"]:.4f}]')
for name, key in [('noise cost', 'noise_cost'), ('mask info', 'mask_info'), ('total', 'total')]:
    d = diffs[key]['macro_auroc']
    print(f'{name}: {d["diff"]:+.4f} [{d["ci_low"]:+.4f}, {d["ci_high"]:+.4f}]')
print(f'[wrote {OUT}]')
