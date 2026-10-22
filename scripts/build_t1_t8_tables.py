#!/usr/bin/env python3
"""Build T1-T8 CSV tables with 95% subject-cluster bootstrap CIs from v6 saved predictions.
Run folder: results/improvement6/v6_full_lgbm_grid/preds (dial LGBM), and any natural/phase/MLP preds present.
"""
import importlib.util, json, os, sys, warnings
from pathlib import Path
from collections import defaultdict as dd
import numpy as np, pandas as pd
import scipy.stats as sp
warnings.filterwarnings("ignore")

HERE=Path(__file__).resolve(); R6=HERE.parent.parent
PREDS=R6/"results/improvement6/v6_full_lgbm_grid/preds"
OUT=R6/"results/improvement6/v6_final_tables"; os.makedirs(OUT, exist_ok=True)
MODS=("mri","fdg","amy","tau"); CLASSES=("CN","MCI","Dementia")
SYS=str(R6/"src")
if SYS not in sys.path: sys.path.insert(0,SYS)
from preprocess import MODALITIES
from store import pack_path, load_pack, pack_to_df
import arms

RNG=np.random.default_rng(20240530)

def metrics_row(df, min_cell_n=20):
    if df is None or len(df)==0: return dict(balanced_accuracy=np.nan,macro_f1=np.nan,
        recall_CN=np.nan,recall_MCI=np.nan,recall_Dementia=np.nan,collapsed=True,n=0)
    y=df['y'].to_numpy(dtype=int)
    P=np.column_stack([df[f'p_{c}'].to_numpy(float) for c in CLASSES])
    pred=P.argmax(axis=1)
    rc=[float((pred[y==i]==i).mean()) if (y==i).any() else 0.0 for i in range(3)]
    balacc=float(np.mean(rc)); macrof1=float(sp.f1_score(y,pred,average='macro',labels=[0,1,2],zero_division=0))
    collapsed=any(r<0.05 for r in rc)
    return dict(balanced_accuracy=balacc,macro_f1=macrof1,recall_CN=rc[0],recall_MCI=rc[1],
                recall_Dementia=rc[2],collapsed=bool(collapsed),n=len(df))

def cell_rows(df):
    y=df['y'].to_numpy(int); pat=df['pattern'].to_numpy(str)
    P=np.column_stack([df[f'p_{c}'].to_numpy(float) for c in CLASSES])
    pred=P.argmax(axis=1)
    rows=[]
    for pid in sorted(set(pat)):
        msk=(pat==pid)
        for cid in range(3):
            cname=CLASSES[cid]; cm=msk & (y==cid); n=int(cm.sum())
            recall=float((pred[cm]==cid).mean()) if n>0 else np.nan
            P_class_pattern=float(cm.mean()) if n>0 else np.nan
            rows.append(dict(class=cname,pattern=pid,n=n,recall=recall,P_class_pattern=P_class_pattern))
    return pd.DataFrame(rows)

def mean_ci(x,n=2000,alpha=0.05):
    x=np.asarray(x,dtype=float); x=x[~np.isnan(x)]
    if len(x)==0: return (np.nan,)*3
    if len(x)==1: return (x[0],x[0],x[0])
    bs=RNG.choice(x,size=(n,),replace=True)
    est=x.mean(); bs=np.sort(bs)
    return est, bs[int(alpha/2*n)], bs[int((1-alpha/2)*n)]

def paired_mean_ci(a,b,n=2000,alpha=0.05):
    a=np.asarray(a,float); b=np.asarray(b,float)
    assert len(a)==len(b)
    d=b-a; est=d.mean()
    if len(d)==1: return (est,est,est)
    bs=RNG.choice(d,size=(n,),replace=True); bs=np.sort(bs)
    return est, bs[int(alpha/2*n)], bs[int((1-alpha/2)*n)]

def ols_coef_ci(xdf,dep,preds):
    y=xdf[dep].to_numpy(float)
    X=xdf[preds].to_numpy(float)
    X=np.column_stack([np.ones(len(X)),X]) if X.ndim==1 else np.column_stack([np.ones(len(X)),X])
    coefs=np.linalg.lstsq(X,y,rcond=None)[0]
    N=len(y); idx=np.arange(N); bs=np.zeros((2000,len(coefs)))
    for i in range(2000):
        s=RNG.choice(idx,size=N,replace=True)
        try: bs[i]=np.linalg.lstsq(X[s],y[s],rcond=None)[0]
        except: bs[i]=np.nan
    lo=np.nanpercentile(bs,2.5,axis=0); hi=np.nanpercentile(bs,97.5,axis=0)
    return coefs,lo,hi

def parse_stem(s):
    parts=s.split('_')
    d=dict(tag=None,coh=None,model=None,arm=None,test=None,draw=None,lam=None,keep=None)
    if s.startswith('dial'):
        d['tag']='dial'
        for i,p in enumerate(parts):
            if p.startswith('l'):
                try: d['lam']=float(p[1:])
                except: pass
            if p.startswith('k'):
                try: d['keep']=float(p[1:])
                except: pass
        for idx in range(len(parts)-2,0,-1):
            if parts[idx] in ('LGBM','MLP') and parts[idx-1].startswith('k'):
                d['model']=parts[idx]; d['arm']=parts[idx+1]; break
        for t in ('T_indep','T_matched','T_full'):
            if t in s: d['test']=t; break
    elif s.startswith('natural'):
        d['tag']='natural'
        for co in ('BASELINE','FULL'): 
            if co in parts: d['coh']=co; break
        for md in ('LGBM','MLP','M0'):
            if md in parts: d['model']=md; break
        for ar in ('M1','M2','M3','M4','M4c'):
            if ar in parts: d['arm']=ar; break
        for tt in ('T_indep','T_matched','T_full','natural'):
            if tt in s: d['test']=tt; break
    elif 'phase' in s or s.startswith('ph'):
        d['tag']='phase'
        for p in parts:
            if p in ('ADNI1','ADNIGO','ADNI2','ADNI3','ADNI4'): d['coh']=p; break
        for md in ('LGBM','MLP','M0'):
            if md in parts: d['model']=md; break
        for ar in ('M1','M2','M3','M4','M4c'):
            if ar in parts: d['arm']=ar; break
    return d

print("scanning preds...", flush=True)
stems={}
if PREDS.is_dir():
    for f in PREDS.glob("*.npz"):
        stems[f.stem]=f
print("packs:",len(stems), flush=True)

# Build index
idx=dd(list)  # key -> list of (d, test, df, metrics)
for stem,pth in stems.items():
    d=parse_stem(stem)
    if d['tag'] is None: continue
    tests_to_try=[]
    if d['tag']=='dial':
        tests_to_try=['T_indep','T_matched']
    elif d['tag']=='natural':
        tests_to_try=['natural','T_matched','T_indep','T_full']
    elif d['tag']=='phase':
        tests_to_try=['t_test','T_indep','T_matched','T_full']
    for test in tests_to_try:
        df=load_pack_df_safe(PREDS, stem, test)
        if df is not None and len(df)>0:
            idx[(d['tag'],d['coh'] or '',d['model'] or '',d['arm'] or '',d['lam'],d['keep'])].append((d,test,df))
print("indexed.")

# ===== T1 =====
t1=dd(list)
for (tag,coh,model,arm,lam,keep),recs in idx.items():
    if tag!='natural' or model not in ('LGBM','MLP') or coh not in ('BASELINE','FULL'): continue
    for d,test,df in recs:
        m=metrics_row(df)
        seed=df['seed'].iloc[0] if 'seed' in df else None
        fold=df['fold'].iloc[0] if 'fold' in df else None
        t1[(coh,model,arm)].append((seed,fold,m))
T1=None
if t1:
    rows=[]
    for (coh,model,arm),recs in t1.items():
        by_key=dd(list)
        for seed,fold,m in recs:
            by_key[(seed,fold)].append(m)
        per_fold=[]
        for k,v in by_key.items():
            if len(v)==1: per_fold.append(v[0])
            else: per_fold.append({mk:np.mean([m[mk] for m in v]) for mk in v[0]})
        row=dict(cohort=coh,model=model,arm=arm,n_folds_run=len(per_fold),
                 n_seeds=len({s for s,_ in by_key}),collapsed=any(m['collapsed'] for m in per_fold))
        for mk in ['balanced_accuracy','macro_f1','recall_CN','recall_MCI','recall_Dementia']:
            vals=[m[mk] for m in per_fold]
            es,lo,hi=mean_ci(vals)
            row[f'{mk}_mean']=es; row[f'{mk}_ci_low']=lo; row[f'{mk}_ci_high']=hi
        rows.append(row)
    T1=pd.DataFrame(rows)
    cols=['cohort','model','arm','n_folds_run','n_seeds','collapsed']+[c for m in ['balanced_accuracy','macro_f1','recall_CN','recall_MCI','recall_Dementia'] for c in [f'{m}_mean',f'{m}_ci_low',f'{m}_ci_high']]
    T1=T1[cols]; T1.to_csv(OUT/"T1_natural_metrics.csv",index=False)
    print("T1 rows:",len(T1), flush=True)
else:
    print("T1 NOT produced: no natural packs", flush=True)

# ===== T2 =====
t2_rows=[]
pf_nat=dd(lambda: dd(dict))  # (coh,model,arm)->(seed,fold)->metrics
for (tag,coh,model,arm,lam,keep),recs in idx.items():
    if tag!='natural': continue
    for d,test,df in recs:
        m=metrics_row(df)
        seed=df['seed'].iloc[0] if 'seed' in df else None
        fold=df['fold'].iloc[0] if 'fold' in df else None
        pf_nat[(coh,model,arm)][(seed,fold)]=m
pairs=[('M2','M1'),('M3','M1'),('M4','M1'),('M4c','M1'),('M4','M4c')]
for (coh,model),arms_d in pf_nat.items():
    for a_t,a_b in pairs:
        if a_t in arms_d and a_b in arms_d:
            common=[k for k in arms_d[a_b] if k in arms_d[a_t]]
            if len(common)<3: continue
            for met in ['balanced_accuracy','macro_f1']:
                b=[arms_d[a_b][k][met] for k in common]
                t=[arms_d[a_t][k][met] for k in common]
                est,lo,hi=paired_mean_ci(np.array(b),np.array(t))
                t2_rows.append(dict(cohort=coh,model=model,comparison=f'{a_t}-{a_b}',metric=met,
                                    diff=est,ci_low=lo,ci_high=hi,n_pairs=len(common)))
T2=pd.DataFrame(t2_rows)
if len(T2)>0: T2.to_csv(OUT/"T2_natural_paired_diffs.csv",index=False); print("T2 rows:",len(T2),flush=True)
else: print("T2 NOT produced", flush=True)

# ===== T3 =====
for model in ('LGBM','MLP'):
    cells=[]
    for (tag,coh,mdl,arm,lam,keep),recs in idx.items():
        if tag!='natural' or mdl!=model: continue
        for d,test,df in recs:
            cells.append(cell_rows(df))
    if cells:
        pool=pd.concat(cells,ignore_index=True)
        pool=pool[pool['n']>=20].copy()
        pool.to_csv(OUT/f"T3_cells_{model}.csv",index=False)
        sub=pool.dropna(subset=['recall','P_class_pattern'])
        if len(sub)>=3:
            rho,pn,n=sp.stats.spearmanr(sub['recall'],sub['P_class_pattern'])
            bs=[]
            idx_l=list(range(len(sub)))
            for _ in range(2000):
                s=RNG.choice(idx_l,size=len(sub),replace=True)
                rho_b,_=sp.stats.spearmanr(sub.iloc[s]['recall'],sub.iloc[s]['P_class_pattern'])
                bs.append(rho_b)
            bs=np.sort(bs); lo=bs[50]; hi=bs[1950]
            pd.DataFrame([{'model':model,'rho':rho,'rho_ci_low':lo,'rho_ci_high':hi,'n_cells':len(sub)}]).to_csv(OUT/f"T3_spearman_{model}.csv",index=False)
            print(f"T3 {model}: rho={rho:.3f} CI[{lo:.3f},{hi:.3f}] cells={len(sub)}",flush=True)
        else:
            print(f"T3 {model}: insufficient cells (n={len(sub)})",flush=True)
    else:
        print(f"T3 {model}: no data",flush=True)

# ===== T4 =====
t4_rows=[]
for (tag,coh,model,arm,lam,keep),recs in idx.items():
    if tag!='dial' or model!='LGBM' or arm not in ('M1','M2','M4','M4c') or lam is None or keep is None: continue
    ind_recs=[r for r in recs if r[1]=='T_indep']
    mat_recs=[r for r in recs if r[1]=='T_matched']
    # group by seed/fold
    def agg(recs):
        by=dd(list)
        for d,test,df in recs:
            m=metrics_row(df)
            seed=df['seed'].iloc[0]; fold=df['fold'].iloc[0]
            by[(seed,fold)].append(m)
        pf=[]
        for k,v in by.items():
            if len(v)==1: pf.append(v[0])
            else: pf.append({mk:np.mean([m[mk] for m in v]) for mk in v[0]})
        return pf
    pf_ind=agg(ind_recs); pf_mat=agg(mat_recs)
    row=dict(lambda_=lam,keep_frac=keep,arm=arm)
    for cond,pf in [('T_indep',pf_ind),('T_matched',pf_mat)]:
        if not pf:
            for mk in ['balanced_accuracy','macro_f1','recall_CN','recall_MCI','recall_Dementia']:
                row[f'{cond}_{mk}_mean']=np.nan
            row[f'{cond}_collapsed']=True; continue
        for mk in ['balanced_accuracy','macro_f1','recall_CN','recall_MCI','recall_Dementia']:
            es,lo,hi=mean_ci([m[mk] for m in pf])
            row[f'{cond}_{mk}_mean']=es; row[f'{cond}_{mk}_ci_low']=lo; row[f'{cond}_{mk}_ci_high']=hi
        row[f'{cond}_collapsed']=any(m['collapsed'] for m in pf)
    if pf_ind and pf_mat and len(pf_ind)==len(pf_mat):
        for mk in ['balanced_accuracy','recall_CN','recall_MCI','recall_Dementia']:
            vi=[m[mk] for m in pf_ind]; vm=[m[mk] for m in pf_mat]
            if len(vi)>=3:
                g,gl,gh=paired_mean_ci(np.array(vi),np.array(vm))
                row[f'gap_{mk}']=g; row[f'gap_{mk}_ci_low']=gl; row[f'gap_{mk}_ci_high']=gh
            else:
                row[f'gap_{mk}']=np.nan
    else:
        row['gap_balanced_accuracy']=np.nan
    t4_rows.append(row)
T4=pd.DataFrame(t4_rows)
if len(T4)>0: T4.to_csv(OUT/"T4_dial_lgbm_metrics.csv",index=False); print("T4 rows:",len(T4),flush=True)
else: print("T4 NOT produced", flush=True)

# ===== T5 =====
t5_rows=[]
for arm in ('M1','M2','M4','M4c'):
    agg=[]
    for lam in sorted(set(k[3] for k in idx if k[0]=='dial' and k[2]=='LGBM' and k[4]==arm and k[5] is not None)):
        for keep in sorted(set(k[4] for k in idx if k[0]=='dial' and k[2]=='LGBM' and k[5]==keep)):
            recs_ind=[r for (t,c,m,a,l,kk),r in idx.items() if t=='dial' and m=='LGBM' and a==arm and lam==l and keep==kk and r[1]=='T_indep']
            recs_mat=[r for (t,c,m,a,l,kk),r in idx.items() if t=='dial' and m=='LGBM' and a==arm and lam==l and keep==kk and r[1]=='T_matched']
            pf_ind=[]
            by=dd(list)
            for d,test,df in recs_ind:
                m=metrics_row(df)
                seed=df['seed'].iloc[0] if 'seed' in df else None; fold=df['fold'].iloc[0] if 'fold' in df else None
                by[(seed,fold)].append(m)
            for k,v in by.items():
                pf_ind.append(v[0] if len(v)==1 else {mk:np.mean([m[mk] for m in v]) for mk in v[0]})
            pf_mat=[]
            by2=dd(list)
            for d,test,df in recs_mat:
                m=metrics_row(df)
                seed=df['seed'].iloc[0] if 'seed' in df else None; fold=df['fold'].iloc[0] if 'fold' in df else None
                by2[(seed,fold)].append(m)
            for k,v in by2.items():
                pf_mat.append(v[0] if len(v)==1 else {mk:np.mean([m[mk] for m in v]) for mk in v[0]})
            if not pf_ind: continue
            ind_ba=mean_ci([m['balanced_accuracy'] for m in pf_ind])[0]
            gap=np.nan
            if pf_mat and len(pf_mat)==len(pf_ind):
                mat_ba=mean_ci([m['balanced_accuracy'] for m in pf_mat])[0]; gap=mat_ba-ind_ba
            agg.append(dict(lambda_=lam,skew=1-keep,bal_ind=ind_ba,gap=gap))
    if len(agg)<4: continue
    xdf=pd.DataFrame(agg)
    for dep,col in [('bal_ind','balanced_accuracy_T_indep'),('gap','gap_balanced_accuracy')]:
        est,lo,hi=ols_coef_ci(xdf,col,['lambda_','skew'])
        t5_rows.append(dict(arm=arm,dep=col,term='lambda',coef=est[1],ci_low=lo[1],ci_high=hi[1]))
        t5_rows.append(dict(arm=arm,dep=col,term='skew',coef=est[2],ci_low=lo[2],ci_high=hi[2]))
        xdf['lam_skew']=xdf['lambda_']*xdf['skew']
        est2,lo2,hi2=ols_coef_ci(xdf,col,['lambda_','skew','lam_skew'])
        inter_mask=[i for i,p in enumerate(['Intercept','lambda_','skew','lam_skew']) if p=='lam_skew']
        if inter_mask:
            t5_rows.append(dict(arm=arm,dep=col,term='lambda:skew',coef=est2[inter_mask[0]],ci_low=lo2[inter_mask[0]],ci_high=hi2[inter_mask[0]]))
T5=pd.DataFrame(t5_rows)
if len(T5)>0: T5.to_csv(OUT/"T5_dial_ols.csv",index=False); print("T5 rows:",len(T5),flush=True)
else: print("T5 NOT produced", flush=True)

# ===== T6 =====
t6_rows=[]
spec_points=[(1.0,0.5),(2.0,0.5),(1.5,0.25)]
for lam,keep in spec_points:
    for a_t,a_b in [('M2','M1'),('M4','M1'),('M4c','M1'),('M4','M4c')]:
        for test in ('T_matched','T_indep'):
            # gather per seed/fold
            pf_b=dd(list); pf_t=dd(list)
            for (t,c,m,a,l,kk),recs in idx.items():
                if t=='dial' and m=='LGBM' and kk==keep and lam==l and a in (a_b,a_t):
                    for d,test_name,df in recs:
                        if test_name==test:
                            m2=metrics_row(df)
                            seed=df['seed'].iloc[0] if 'seed' in df else None
                            fold=df['fold'].iloc[0] if 'fold' in df else None
                            if a==a_b: pf_b[(seed,fold)].append(m2)
                            else: pf_t[(seed,fold)].append(m2)
            common=[k for k in pf_b if k in pf_t]
            if len(common)<3: continue
            for met in ['balanced_accuracy','recall_CN','recall_MCI','recall_Dementia','macro_f1']:
                b=np.array([np.mean([m[met] for m in pf_b[k]]) if len(pf_b[k])>0 else np.nan for k in common])
                t=np.array([np.mean([m[met] for m in pf_t[k]]) if len(pf_t[k])>0 else np.nan for k in common])
                b=b[~np.isnan(b)]; t=t[~np.isnan(t)]
                if len(b)<3: continue
                est,lo,hi=paired_mean_ci(b,t)
                t6_rows.append(dict(lambda_=lam,keep_frac=keep,arm_t=a_t,arm_b=a_b,
                                    test=test,metric=met,diff=est,ci_low=lo,ci_high=hi,n_pairs=len(common)))
T6=pd.DataFrame(t6_rows)
if len(T6)>0: T6.to_csv(OUT/"T6_dial_paired_diffs.csv",index=False); print("T6 rows:",len(T6),flush=True)
else: print("T6 NOT produced", flush=True)

# ===== T7 =====
phase_pack_keys=[(d,pth) for stem,pth in stems.items() for d in [parse_stem(stem)] if d['tag']=='phase']
print("phase packs:",len(phase_pack_keys),flush=True)
if not phase_pack_keys:
    print("T7 NOT produced: no phase-held-out packs; see note",flush=True)
    with open(OUT/"T7_NOTE.txt","w") as f: f.write("T7 not produced: phase-held-out preds not found in v6_full_lgbm_grid/preds. Awaiting dedicated phase run or explicit preds location.\n")
else:
    # absolute metrics M1
    m1=dd(list)
    arms_present=set()
    for d,pth in phase_pack_keys:
        for test in ('t_test','T_indep','T_matched','T_full'):
            df=load_pack_df_safe(PREDS, stem_for(d), test)
            if df is not None and len(df)>0:
                m=metrics_row(df)
                arms_present.add(d['arm'])
                m1[(d['arm'],test)].append(m)
    # diffs
    t7_rows=[]
    for a in arms_present:
        for met in ['balanced_accuracy','macro_f1','recall_CN','recall_MCI','recall_Dementia']:
            if (a,met) in ...: pass
    # placeholder: write shift tables using ep
    # Wait we need ep for shift. Load ep.
    ep=load_ep()
    train_epi=ep.df[ep.df['COLPROT'].isin(['ADNI1','ADNIGO','ADNI2'])]
    test_epi=ep.df[ep.df['COLPROT'] in ['ADNI3','ADNI4'])]
    prior_train={CLASSES[i]:float((train_epi['y']==i).mean()) for i in range(3)}
    prior_test={CLASSES[i]:float((test_epi['y']==i).mean()) for i in range(3)}
    avail_train={k:float(train_epi[f'm_{k}'].mean()) for k in MODS}
    avail_test={k:float(test_epi[f'm_{k}'].mean()) for k in MODS}
    shift=pd.DataFrame([{**prior_train,'prior_test_'+k:prior_test[k] for k in CLASSES},
                        {**{f'avail_train_{k}':avail_train[k] for k in MODS},
                         **{f'avail_test_{k}':avail_test[k] for k in MODS}}])
    shift.to_csv(OUT/"T7_class_avail_shift.csv",index=False)
    print("T7 shift tables written; absolute/diffs need phase predictions",flush=True)

# ===== T8 =====
t8_rows=[]
for lam in (0.0,1.0):
    for keep in (1.0,0.5):
        for arm in ('M1','M2','M4','M4c'):
            recs=[r for (t,c,m,a,l,kk),r in idx.items() if t=='dial' and m=='LGBM' and a==arm and lam==l and keep==kk and r[1]=='T_matched']
            pf=[]
            by=dd(list)
            for d,test,df in recs:
                m=metrics_row(df)
                seed=df['seed'].iloc[0] if 'seed' in df else None; fold=df['fold'].iloc[0] if 'fold' in df else None
                by[(seed,fold)].append(m)
            for k,v in by.items():
                pf.append(v[0] if len(v)==1 else {mk:np.mean([m[mk] for m in v]) for mk in v[0]})
            if not pf: continue
            est,lo,hi=mean_ci([m['balanced_accuracy'] for m in pf])
            collapsed=any(m['collapsed'] for m in pf)
            t8_rows.append(dict(lambda_=lam,keep_frac=keep,arm=arm,balanced_accuracy_mean=est,
                                balanced_accuracy_ci_low=lo,balanced_accuracy_ci_high=hi,collapsed=collapsed,n=len(pf)))
T8=pd.DataFrame(t8_rows)
if len(T8)>0: T8.to_csv(OUT/"T8_mlp_robustness.csv",index=False); print("T8 rows:",len(T8),flush=True)
else: print("T8 NOT produced", flush=True)

print("FINISHED",flush=True)
