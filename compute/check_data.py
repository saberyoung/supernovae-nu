#!/usr/bin/env python3
from pathlib import Path
import argparse

REQ = [
    "events/IC86_I_exp.csv","events/IC86_II_exp.csv","events/IC86_III_exp.csv",
    "events/IC86_IV_exp.csv","events/IC86_V_exp.csv","events/IC86_VI_exp.csv",
    "events/IC86_VII_exp.csv","events/IC86_VIII_exp.csv","events/IC86_IX_exp.csv",
    "events/IC86_X_exp.csv","events/IC86_XI_exp.csv",
    "uptime/IC86_I_exp.csv","uptime/IC86_II_exp.csv","uptime/IC86_III_exp.csv",
    "uptime/IC86_IV_exp.csv","uptime/IC86_V_exp.csv","uptime/IC86_VI_exp.csv",
    "uptime/IC86_VII_exp.csv","uptime/IC86_VIII_exp.csv","uptime/IC86_IX_exp.csv",
    "uptime/IC86_X_exp.csv","uptime/IC86_XI_exp.csv",
    "irfs/IC86_effectiveArea.csv","irfs/IC86_smearing.csv",
]

def find_root(p):
    p=Path(p).expanduser().resolve()
    candidates=[p] + ([x for x in p.iterdir() if x.is_dir()] if p.is_dir() else [])
    for c in candidates:
        if (c/"events").is_dir() and (c/"uptime").is_dir() and (c/"irfs").is_dir():
            return c
    return None

ap=argparse.ArgumentParser()
ap.add_argument("icetracks_dir")
a=ap.parse_args()
root=find_root(a.icetracks_dir)
if root is None:
    raise SystemExit("ERROR: cannot find events/, uptime/, irfs/")
print("IceTracks root:", root)
missing=[x for x in REQ if not (root/x).exists()]
if missing:
    print("Missing files:")
    for x in missing: print("  ",x)
    raise SystemExit(2)
print("LAYOUT OK: all required IC86 files found.")
