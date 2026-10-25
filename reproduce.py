#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
steps=[['tools/validate.py'],['analysis/optical_diagnostics.py','--root','.'],['figures/make_all.py']]
for step in steps:
    print('[reproduce]',' '.join(step),flush=True); subprocess.run([sys.executable,*step],cwd=ROOT,check=True)
print('Reproduction complete. Figures are in figures/output/.')
