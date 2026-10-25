#!/usr/bin/env python3
"""
Fast exact calibration of the frozen k=10 phenotype statistic.

The empirical source-wise transform is evaluated with sorted calibration arrays and
numpy.searchsorted instead of comparing every global trial against every
calibration trial.

For source j:
    p = [1 + #{q_cal >= q}] / (B_cal + 1)

searchsorted(side="left") exactly reproduces the >= tie convention.

No science/statistic changes.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm


def empirical_pz_vector(q, qcal):
    q = np.asarray(q, float)
    qcal = np.asarray(qcal, float)
    B = qcal.shape[0]
    p = np.empty(len(q), float)

    for j in range(len(q)):
        s = np.sort(qcal[:, j])
        n_lt = np.searchsorted(
            s, q[j], side="left"
        )
        n_ge = B - n_lt
        p[j] = (1.0 + n_ge) / (B + 1.0)

    lo = 0.5 / (B + 1.0)
    p = np.clip(p, lo, 1.0-lo)
    return norm.isf(p), p


def empirical_z_matrix(qmat, qcal):
    qmat = np.asarray(qmat, float)
    qcal = np.asarray(qcal, float)
    B = qcal.shape[0]
    p = np.empty_like(qmat, dtype=float)

    for j in range(qmat.shape[1]):
        s = np.sort(qcal[:, j])
        n_lt = np.searchsorted(
            s, qmat[:, j], side="left"
        )
        n_ge = B - n_lt
        p[:, j] = (1.0 + n_ge) / (B + 1.0)

    lo = 0.5 / (B + 1.0)
    p = np.clip(p, lo, 1.0-lo)
    return norm.isf(p)


def knn_matrix(x, k=10):
    d = cdist(x, x)
    nbr = np.argsort(d, axis=1)[:, 1:k+1]
    A = np.zeros((len(x), len(x)), dtype=float)
    for i, inds in enumerate(nbr):
        A[i, inds] = 1.0 / np.sqrt(k)
    return A, nbr


def global_stat(zmat, A):
    if zmat.ndim == 1:
        local = A @ zmat
        return float(local.max()), local
    local = zmat @ A.T
    return local.max(axis=1), local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("observed_npz")
    ap.add_argument("calibration_npz")
    ap.add_argument("global_npz")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--output-prefix",
        default="primary",
    )
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    obs = np.load(args.observed_npz, allow_pickle=True)
    cal = np.load(args.calibration_npz, allow_pickle=True)
    glob = np.load(args.global_npz, allow_pickle=True)

    q_obs = np.asarray(obs["q_obs"], float)
    q_cal = np.asarray(cal["q_bg"], float)
    q_glob = np.asarray(glob["q_bg"], float)

    if q_obs.shape[0] != len(cat):
        raise ValueError(
            "Observed source count does not match catalog."
        )
    if (
        q_cal.shape[1] != len(cat)
        or q_glob.shape[1] != len(cat)
    ):
        raise ValueError(
            "Background matrix source count does not match catalog."
        )

    z_obs, p_obs = empirical_pz_vector(
        q_obs, q_cal
    )
    z_glob = empirical_z_matrix(
        q_glob, q_cal
    )

    A, nbr = knn_matrix(
        cat[["PC1", "PC2"]].to_numpy(float),
        k=args.k,
    )
    T_obs, local_obs = global_stat(z_obs, A)
    T_glob, _ = global_stat(z_glob, A)

    p_global = (
        1.0 + np.sum(T_glob >= T_obs)
    ) / (len(T_glob) + 1.0)
    i_hot = int(np.argmax(local_obs))

    nodes = cat.copy()
    nodes["q_obs"] = q_obs
    nodes["p_local_calibrated"] = p_obs
    nodes["z_local_calibrated"] = z_obs
    nodes["phenotype_local_score"] = local_obs
    nodes.to_csv(
        args.output_prefix + "_nodes.csv",
        index=False,
    )

    members = nodes.iloc[nbr[i_hot]].copy()
    members["neighbor_rank"] = np.arange(
        1, len(members)+1
    )
    members.to_csv(
        args.output_prefix + "_hotspot_members.csv",
        index=False,
    )

    np.savez_compressed(
        args.output_prefix + "_Tnull.npz",
        T_global=T_glob,
        T_observed=np.asarray([T_obs]),
    )

    summary = {
        "N_sources": int(len(cat)),
        "B_calibration": int(q_cal.shape[0]),
        "B_global": int(q_glob.shape[0]),
        "k": int(args.k),
        "T_observed": float(T_obs),
        "global_p": float(p_global),
        "hotspot_center_ZTFID": str(
            cat.iloc[i_hot]["ZTFID"]
        ),
        "hotspot_center_type": str(
            cat.iloc[i_hot]["type"]
        ),
        "best_individual_ZTFID": str(
            cat.iloc[int(np.argmax(z_obs))]["ZTFID"]
        ),
        "best_individual_z": float(
            np.max(z_obs)
        ),
    }

    with open(
        args.output_prefix + "_summary.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
