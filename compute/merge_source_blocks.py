#!/usr/bin/env python3
"""
Merge block-wise outputs from the shared-background worker.

All block files must use the same trial seeds and together cover each source
index exactly once. The merged output is sorted into catalog/source-index order.
"""

from __future__ import annotations
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pieces = [np.load(p, allow_pickle=True) for p in args.blocks]
    ref_seed = pieces[0]["trial_seed"]

    for p in pieces[1:]:
        if not np.array_equal(p["trial_seed"], ref_seed):
            raise ValueError("Block trial seeds differ; cannot form joint realizations.")

    source_index = np.concatenate([p["source_index"] for p in pieces])
    ztfid = np.concatenate([p["ZTFID"] for p in pieces])
    q_obs = np.concatenate([p["q_obs"] for p in pieces])
    best_sigma_obs = np.concatenate([p["best_sigma_obs"] for p in pieces])
    q_bg = np.concatenate([p["q_bg"] for p in pieces], axis=1)

    if len(np.unique(source_index)) != len(source_index):
        raise ValueError("Duplicate source indices across blocks.")

    order = np.argsort(source_index)
    source_index = source_index[order]
    ztfid = ztfid[order]
    q_obs = q_obs[order]
    best_sigma_obs = best_sigma_obs[order]
    q_bg = q_bg[:, order]

    expected = np.arange(source_index.min(), source_index.max() + 1)
    if not np.array_equal(source_index, expected):
        raise ValueError(
            "Source blocks do not form one contiguous complete index range."
        )

    np.savez_compressed(
        args.output,
        source_index=source_index,
        ZTFID=ztfid,
        trial_seed=ref_seed,
        q_obs=q_obs,
        best_sigma_obs=best_sigma_obs,
        q_bg=q_bg,
    )
    print("Merged", len(args.blocks), "blocks")
    print("Sources:", len(source_index), "Trials:", len(ref_seed))
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
