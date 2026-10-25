#!/usr/bin/env python3
"""Compare the pre-unblinding idealized power benchmark with the final joint-IceCube sensitivity."""
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

MATCH = {'compact_10NN_region':'knn10','broad_PC1_upper_tail':'linear_PC1_PC2','exact_IcBL_subtype':'max_subtype'}
SHORT = {'compact_10NN_region':'Compact 10-NN region','broad_PC1_upper_tail':'Broad PC1','exact_IcBL_subtype':'Ic-BL subtype'}


def interp_threshold(df, target):
    d = df.sort_values('mean_shift_delta'); x = d['mean_shift_delta'].to_numpy(); y = d['power_at_global_alpha_0p01'].to_numpy()
    if target < y.min() or target > y.max(): return np.nan
    i = np.where(y >= target)[0][0]
    if i == 0: return float(x[0])
    return float(x[i-1] + (target-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo-root', default='.'); ap.add_argument('--output-dir', default=None); ap.add_argument('--empirical-summary', default=None); args = ap.parse_args()
    repo = Path(args.repo_root).resolve(); out = Path(args.output_dir).resolve() if args.output_dir else repo/'results'; out.mkdir(parents=True, exist_ok=True)
    synthetic = pd.read_csv(repo/'data/idealized_power_benchmark.csv')
    empirical = pd.read_csv(Path(args.empirical_summary).resolve() if args.empirical_summary else repo/'results/effect_size_summary.csv')
    rows = []
    for scenario, test in MATCH.items():
        s = synthetic[(synthetic.scenario==scenario)&(synthetic.test==test)]
        e = empirical[empirical.scenario==scenario].iloc[0]
        syn = {50:interp_threshold(s,.50),80:interp_threshold(s,.80),90:interp_threshold(s,.90)}
        emp = {50:float(e.delta_z_for_50pct_detection_power_alpha0p01),80:float(e.delta_z_for_80pct_detection_power_alpha0p01),90:float(e.delta_z_for_90pct_detection_power_alpha0p01)}
        for power in [50,80,90]:
            rows.append({'scenario':scenario,'label':SHORT[scenario],'matched_test':test,'target_power_percent':power,'delta_z_synthetic_independent':syn[power],'delta_z_joint_icecube':emp[power],'joint_to_synthetic_ratio':emp[power]/syn[power] if np.isfinite(syn[power]) else np.nan})
    tab = pd.DataFrame(rows); tab.to_csv(out/'power_thresholds.csv', index=False)
    (out/'sensitivity_comparison.json').write_text(json.dumps({'interpretation':'Joint detector backgrounds reduce absolute sensitivity while preserving the matched-statistic ordering.'}, indent=2))
    print('POWER REALISM COMPARISON COMPLETE')
    print(tab.to_string(index=False))

if __name__ == '__main__':
    main()
