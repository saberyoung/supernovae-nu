#!/usr/bin/env python3
"""Merge published-seed chunks into the final 2000/10000 cache and compute the primary statistic."""
from pathlib import Path
import subprocess, sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]
CH=ROOT/'work/chunks'; OUT=ROOT/'work/cache'; OUT.mkdir(parents=True,exist_ok=True)
concat=ROOT/'compute/concat_trials.py'
calib=ROOT/'compute/calibrate_scores.py'
cat=ROOT/'data/search_sample.csv'
CAL=[CH/'cal/cal_chunk_00_200trials_ALL241.npz']+[CH/f'cal/cal_chunk_{i:02d}_200trials_ALL241.npz' for i in range(1,10)]
GLOB=[CH/'global/global_chunk_00_500trials_ALL241.npz']+[CH/f'global/global_chunk_{i:02d}_200trials_ALL241.npz' for i in range(1,48)]+[CH/'global/global_chunk_48_100trials_ALL241.npz']
for p in CAL+GLOB:
    if not p.exists(): raise FileNotFoundError(p)
cal=OUT/'cal_ALL_241src_2000trials.npz'; glob=OUT/'global_ALL_241src_10000trials.npz'
subprocess.run([sys.executable,str(concat),*[str(p) for p in CAL],'--output',str(cal)],check=True)
subprocess.run([sys.executable,str(concat),*[str(p) for p in GLOB],'--output',str(glob)],check=True)
x=np.load(cal,allow_pickle=True); y=np.load(glob,allow_pickle=True)
assert x['q_bg'].shape==(2000,241) and y['q_bg'].shape==(10000,241)
assert np.intersect1d(x['trial_seed'],y['trial_seed']).size==0
prefix=OUT/'primary'
subprocess.run([sys.executable,str(calib),str(cat),str(cal),str(cal),str(glob),'--k','10','--output-prefix',str(prefix)],check=True)
print('Merged and calibrated final publication cache.')
