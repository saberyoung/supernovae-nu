#!/usr/bin/env python3
"""Generate the next missing detector-level Monte Carlo chunk using the published seeds.

This reproduces exactly the 2000 calibration and 10000 global seed ensembles
used in the paper. Run repeatedly; completed chunks are validated and skipped.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

# Exact seed sets used by the paper.
CAL = [("cal",0,200,810000)] + [("cal",i+1,200,820000+200*i) for i in range(9)]
GLOBAL = [("global",0,500,910000)] + [("global",i+1,200,920000+200*i) for i in range(47)] + [("global",48,100,929400)]
SCHEDULE = CAL + GLOBAL

def blocks(n=241,w=6):
    e=np.rint(np.linspace(0,n,w+1)).astype(int); return [(int(a),int(b)) for a,b in zip(e[:-1],e[1:]) if b>a]

def valid(path,a,b,seed,ntr):
    if not path.exists(): return False
    try:
        x=np.load(path,allow_pickle=True)
        return x['q_bg'].shape==(ntr,b-a) and np.array_equal(x['trial_seed'],seed+np.arange(ntr)) and np.array_equal(x['source_index'],np.arange(a,b))
    except Exception: return False

def merged_valid(path,ntr,seed):
    if not path.exists(): return False
    try:
        x=np.load(path,allow_pickle=True)
        return x['q_bg'].shape==(ntr,241) and np.array_equal(x['trial_seed'],seed+np.arange(ntr)) and np.array_equal(x['source_index'],np.arange(241))
    except Exception: return False

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--icetracks-dir',required=True)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--batch',type=int,default=1)
    ap.add_argument('--status',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    catalog=root/'data/search_sample.csv'
    worker=root/'compute/joint_background.py'
    merger=root/'compute/merge_source_blocks.py'
    outroot=root/'work/chunks'
    ic=Path(args.icetracks_dir).expanduser().resolve()
    if len(pd.read_csv(catalog))!=241: raise RuntimeError('Frozen catalog must contain 241 sources.')
    def merged_path(ph,idx,ntr): return outroot/ph/f'{ph}_chunk_{idx:02d}_{ntr}trials_ALL241.npz'
    def status():
        return [(ph,idx,ntr,seed,merged_valid(merged_path(ph,idx,ntr),ntr,seed)) for ph,idx,ntr,seed in SCHEDULE]
    st=status()
    print('chunk status:',sum(x[4] for x in st),'/',len(st))
    for x in st: print(f'{x[0]:6s} {x[1]:02d}:', 'DONE' if x[4] else 'pending', f'seeds {x[3]}-{x[3]+x[2]-1}')
    if args.status: return
    for _ in range(args.batch):
        target=next((x for x in status() if not x[4]),None)
        if target is None: print('All chunks complete.'); return
        ph,idx,ntr,seed,_=target; d=outroot/ph/f'chunk_{idx:02d}'; d.mkdir(parents=True,exist_ok=True)
        bs=blocks(241,args.workers); jobs=[]; blockfiles=[]
        for a,b in bs:
            f=d/f'block_{a:03d}_{b:03d}.npz'; blockfiles.append(f)
            if valid(f,a,b,seed,ntr): continue
            cmd=[sys.executable,'-u',str(worker),str(catalog),'--icetracks-dir',str(ic),'--start',str(a),'--stop',str(b),'--ntrials',str(ntr),'--seed-base',str(seed),'--output',str(f),'--progress-every','40']
            jobs.append((a,b,cmd))
        print(f'Running {ph} chunk {idx:02d}, {ntr} trials, seed {seed}')
        with cf.ThreadPoolExecutor(max_workers=max(1,min(args.workers,len(jobs)))) as ex:
            futs={ex.submit(subprocess.run,c,check=True): (a,b) for a,b,c in jobs}
            for f in cf.as_completed(futs): print(' completed block',futs[f]); f.result()
        for (a,b),f in zip(bs,blockfiles):
            if not valid(f,a,b,seed,ntr): raise RuntimeError(f'Invalid block {f}')
        dest=merged_path(ph,idx,ntr); dest.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([sys.executable,str(merger),*[str(x) for x in blockfiles],'--output',str(dest)],check=True)
        if not merged_valid(dest,ntr,seed): raise RuntimeError(f'Invalid merged chunk {dest}')
        print('Saved',dest)
if __name__=='__main__': main()
