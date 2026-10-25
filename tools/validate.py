#!/usr/bin/env python3
from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ref=pd.read_csv(ROOT/'data/optical_reference.csv'); cat=pd.read_csv(ROOT/'data/search_sample.csv')
assert len(ref)==384 and len(cat)==241
s=json.loads((ROOT/'results/primary_summary.json').read_text())
assert abs(s['T_observed']-2.0396643815301907)<1e-10
assert abs(s['global_p']-0.5393460653934606)<1e-12
sec=pd.read_csv(ROOT/'results/secondary_tests.csv')
assert np.isclose(float(sec.loc[sec.test=='linear_PC1_PC2_gradient','global_p'].iloc[0]),0.8523147685231477)
assert np.isclose(float(sec.loc[sec.test=='max_predefined_subtype','global_p'].iloc[0]),0.3257674232576742)
print('Validation passed: sample sizes and headline statistics match the paper.')
