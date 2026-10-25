#!/usr/bin/env python3
"""
Concatenate complete-source joint-background NPZ files along the trial axis.

Use after each trial chunk has first been merged horizontally across all source
blocks with merge_source_blocks.py.

Requirements:
- identical source_index / ZTFID ordering
- identical q_obs and best_sigma_obs
- disjoint trial_seed arrays
"""

from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pieces = [np.load(p, allow_pickle=True) for p in args.chunks]
    ref = pieces[0]

    all_seeds = []
    all_qbg = []

    for p in pieces:
        if not np.array_equal(p["source_index"], ref["source_index"]):
            raise ValueError("Source index/order differs across trial chunks.")
        if not np.array_equal(p["ZTFID"], ref["ZTFID"]):
            raise ValueError("ZTFID order differs across trial chunks.")
        if not np.allclose(p["q_obs"], ref["q_obs"], rtol=1e-8, atol=1e-10):
            raise ValueError("Observed q vectors differ across trial chunks.")
        if not np.array_equal(p["best_sigma_obs"], ref["best_sigma_obs"]):
            raise ValueError("Observed best-sigma vectors differ across chunks.")

        all_seeds.append(p["trial_seed"])
        all_qbg.append(p["q_bg"])

    trial_seed = np.concatenate(all_seeds)
    q_bg = np.concatenate(all_qbg, axis=0)

    if len(np.unique(trial_seed)) != len(trial_seed):
        raise ValueError("Duplicate trial seeds across chunks.")

    order = np.argsort(trial_seed)
    trial_seed = trial_seed[order]
    q_bg = q_bg[order]

    np.savez_compressed(
        args.output,
        source_index=ref["source_index"],
        ZTFID=ref["ZTFID"],
        trial_seed=trial_seed,
        q_obs=ref["q_obs"],
        best_sigma_obs=ref["best_sigma_obs"],
        q_bg=q_bg,
    )
    print("Sources:", q_bg.shape[1])
    print("Trials:", q_bg.shape[0])
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
