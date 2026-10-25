#!/usr/bin/env python3
"""Optical-only diagnostics used in the paper."""
from pathlib import Path
import argparse, json
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

ORDINARY4=["SN IIb","SN Ib","SN Ic","SN Ic-BL"]
MAIN6=ORDINARY4+["SN Ibn","SLSN-I"]

def knn_purity(X,y,k=10,n_perm=10000,seed=20261024):
    ind=NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(return_distance=False)[:,1:]
    obs=float(np.mean(y[ind]==y[:,None]))
    counts=np.array(list(Counter(y).values())); n=len(y)
    expected=float(np.sum(counts*(counts-1))/(n*(n-1)))
    rng=np.random.default_rng(seed); code=np.unique(y,return_inverse=True)[1]
    null=np.empty(n_perm,float)
    for start in range(0,n_perm,500):
        stop=min(start+500,n_perm)
        yp=np.stack([rng.permutation(code) for _ in range(stop-start)])
        null[start:stop]=np.mean(yp[:,ind]==yp[:,:,None],axis=(1,2))
    p=float((1+np.sum(null>=obs))/(n_perm+1))
    return obs,expected,null,p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--n-perm',type=int,default=10000); ap.add_argument('--seed',type=int,default=20261024); a=ap.parse_args()
    root=Path(a.root).resolve(); parent=pd.read_csv(root/'data/optical_reference.csv'); ice=pd.read_csv(root/'data/search_sample.csv')
    summary={}; ordinary_null=None
    for label,classes in [('ordinary4',ORDINARY4),('main6',MAIN6)]:
        d=parent[parent.type.isin(classes)].copy(); y=d.type.to_numpy(); X2=d[['PC1','PC2']].to_numpy(); X3=d[['PC1','PC2','PC3']].to_numpy()
        summary[f'{label}_N']=int(len(d)); summary[f'{label}_silhouette_PC12']=float(silhouette_score(X2,y)); summary[f'{label}_silhouette_PC123']=float(silhouette_score(X3,y))
        obs,exp,null,p=knn_purity(X2,y,10,a.n_perm,a.seed+len(classes)); summary[f'{label}_knn10_same_subtype_purity']=obs; summary[f'{label}_knn10_random_expectation']=exp; summary[f'{label}_knn10_permutation_p']=p
        if label=='ordinary4': ordinary_null=null
    corr=[]
    for pc in ['PC1','PC2','PC3']:
        for var in ['redshift_num','peakmag','dec_deg','peak_mjd']:
            r,p=spearmanr(ice[pc],ice[var],nan_policy='omit'); corr.append({'pc':pc,'variable':var,'spearman_rho':float(r),'p_two_sided':float(p)})
    pd.DataFrame({'null_purity':ordinary_null}).to_csv(root/'results/optical_purity_null.csv',index=False)
    pd.DataFrame(corr).to_csv(root/'results/selection_correlations.csv',index=False)
    (root/'results/optical_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
