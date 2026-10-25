#!/usr/bin/env python3
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for name in ['figure1.py','figure2.py','figure3.py','figure4.py','supplementary.py']:
    print(f'[figures] {name}',flush=True); subprocess.run([sys.executable,str(ROOT/'figures'/name)],cwd=ROOT,check=True)
