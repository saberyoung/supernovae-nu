#!/usr/bin/env python3
"""
Empirical joint-null injection/recovery benchmark

Purpose
-------
Use the completed IceTracks publication calibration/global-null matrices
to repeat the pre-unblinding score-space injection benchmark while preserving:

  - source-specific IceCube calibration,
  - shared detector-level background covariance across all 241 sources,
  - declination/time/exposure structure,
  - finite-calibration discreteness.

No SkyLLH fitting is performed.

Signal is injected as an additive shift Delta-z in the calibrated source-level
neutrino evidence.  This is therefore a METHOD / EFFECT-SIZE benchmark, NOT yet
an astrophysical flux or expected-event sensitivity.

Only signal scenarios whose definitions were explicitly preserved in the
pre-unblinding benchmark metadata are regenerated here:

1. compact_10NN_region
   Add the same positive Delta-z to the 10 frozen optical nearest neighbours
   of a random phenotype-space center (the center itself is excluded).

2. broad_PC1_upper_tail
   Add Delta-z to the upper 20% of sources in frozen PC1.

3. exact_IcBL_subtype
   Add Delta-z to all secure SN Ic-BL sources.

The old "smooth_local_region" result remains available in the frozen
pre-unblinding CSV, but its exact Gaussian bandwidth was not preserved in the
metadata.  We intentionally DO NOT reconstruct/tune that definition after
unblinding.

Tests
-----
A. frozen primary k=10 phenotype hotspot
B. frozen linear PC1-PC2 gradient
C. frozen max-over-predefined-subtypes statistic

Outputs include:
- empirical global-null thresholds at alpha=0.01 and alpha=0.05,
- power curves versus Delta-z,
- observed-statistic consistency curves,
- approximate 90% score-space upper effect size for the best-matched test
  under each explicitly frozen injection scenario.

Development caveat
------------------
B_global=500 gives only coarse tail resolution, especially at alpha=0.01.
The same script can later be rerun with the publication 10,000-trial global
matrix without changing any definitions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import norm


MAX_SUBTYPE_CLASSES = [
    "SN IIb", "SN Ib", "SN Ic", "SN Ic-BL", "SN Ibn", "SLSN-I"
]


def empirical_z(q, qcal):
    """Exact empirical >= calibration using source-wise sorting."""
    q = np.asarray(q, float)
    qcal = np.asarray(qcal, float)
    B = qcal.shape[0]
    p = np.empty(len(q), float)
    for j in range(len(q)):
        s = np.sort(qcal[:, j])
        n_lt = np.searchsorted(s, q[j], side="left")
        p[j] = (1.0 + B - n_lt) / (B + 1.0)
    lo = 0.5 / (B + 1.0)
    p = np.clip(p, lo, 1.0-lo)
    return norm.isf(p), p


def empirical_z_matrix(qmat, qcal):
    """Exact matrix transform optimized for publication-size ensembles."""
    qmat = np.asarray(qmat, float)
    qcal = np.asarray(qcal, float)
    B = qcal.shape[0]
    p = np.empty_like(qmat, dtype=float)
    for j in range(qmat.shape[1]):
        s = np.sort(qcal[:, j])
        n_lt = np.searchsorted(s, qmat[:, j], side="left")
        p[:, j] = (1.0 + B - n_lt) / (B + 1.0)
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


def phenotype_stat(z, A):
    if z.ndim == 1:
        local = A @ z
        return float(local.max()), local
    local = z @ A.T
    return local.max(axis=1), local


def gradient_basis(cat):
    X = cat[["PC1", "PC2"]].to_numpy(float)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Q, _ = np.linalg.qr(X, mode="reduced")
    return Q


def gradient_stat(z, Q):
    if z.ndim == 1:
        proj = Q.T @ z
        return float(np.sum(proj * proj))
    proj = z @ Q
    return np.sum(proj * proj, axis=1)


def subtype_masks(cat):
    t = cat["type"].astype(str).to_numpy()
    return {
        cls: (t == cls)
        for cls in MAX_SUBTYPE_CLASSES
    }


def max_subtype_stat(z, masks):
    vals = []
    for cls in MAX_SUBTYPE_CLASSES:
        m = masks[cls]
        n = int(np.sum(m))
        if z.ndim == 1:
            vals.append(np.sum(z[m]) / np.sqrt(n))
        else:
            vals.append(np.sum(z[:, m], axis=1) / np.sqrt(n))
    if z.ndim == 1:
        return float(np.max(vals))
    return np.max(np.column_stack(vals), axis=1)


def threshold_higher(null, alpha):
    return float(np.quantile(null, 1.0-alpha, method="higher"))


def achieved_tail(null, threshold):
    return float(np.mean(null >= threshold))


def interpolation_crossing(x, y, target):
    """
    Return first linearly interpolated x where monotonically-trending y crosses
    target.  Works for increasing power or decreasing consistency curves.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    for i in range(len(x)-1):
        y1, y2 = y[i], y[i+1]
        if (y1-target) == 0:
            return float(x[i])
        if (y1-target) * (y2-target) <= 0 and y1 != y2:
            return float(
                x[i] + (target-y1)*(x[i+1]-x[i])/(y2-y1)
            )
    if y[-1] == target:
        return float(x[-1])
    return np.nan


def build_injected_z(
    z_base,
    scenario,
    delta,
    rng,
    nbr,
    cat,
):
    z = z_base.copy()
    ntrial, nsrc = z.shape

    if delta == 0:
        return z

    if scenario == "compact_10NN_region":
        centers = rng.integers(0, nsrc, size=ntrial)
        for r, c in enumerate(centers):
            z[r, nbr[c]] += delta
        return z

    if scenario == "broad_PC1_upper_tail":
        cutoff = np.quantile(cat["PC1"].to_numpy(float), 0.80)
        mask = cat["PC1"].to_numpy(float) >= cutoff
        z[:, mask] += delta
        return z

    if scenario == "exact_IcBL_subtype":
        mask = cat["type"].astype(str).to_numpy() == "SN Ic-BL"
        z[:, mask] += delta
        return z

    raise ValueError(scenario)


def stat_triplet(z, A, Q, masks):
    tph, _ = phenotype_stat(z, A)
    tgrad = gradient_stat(z, Q)
    tsub = max_subtype_stat(z, masks)
    return {
        "knn10": tph,
        "linear_PC1_PC2": tgrad,
        "max_subtype": tsub,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/search_sample.csv",
    )
    ap.add_argument(
        "--cal",
        default=(
            "work/cache/"
            "cal_ALL_241src_2000trials.npz"
        ),
    )
    ap.add_argument(
        "--global-null",
        default=(
            "work/cache/"
            "global_ALL_241src_10000trials.npz"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default="work/power",
    )
    ap.add_argument(
        "--n-injection-trials",
        type=int,
        default=20000,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=161001,
    )
    ap.add_argument(
        "--deltas",
        default="0,0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3",
    )
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    cal = np.load(args.cal, allow_pickle=True)
    glob = np.load(args.global_null, allow_pickle=True)

    q_obs = np.asarray(cal["q_obs"], float)
    q_cal = np.asarray(cal["q_bg"], float)
    q_glob = np.asarray(glob["q_bg"], float)

    if len(cat) != 241:
        raise ValueError(f"Expected 241 SESNe; got {len(cat)}")
    if q_cal.shape[1] != len(cat) or q_glob.shape[1] != len(cat):
        raise ValueError("Matrix/catalog source count mismatch")

    z_obs, _ = empirical_z(q_obs, q_cal)
    z_glob = empirical_z_matrix(q_glob, q_cal)

    A, nbr = knn_matrix(cat[["PC1","PC2"]].to_numpy(float), k=10)
    Q = gradient_basis(cat)
    masks = subtype_masks(cat)

    obs_stats = stat_triplet(z_obs, A, Q, masks)
    null_stats = stat_triplet(z_glob, A, Q, masks)

    thresholds = {}
    for test, arr in null_stats.items():
        thresholds[test] = {}
        for alpha in [0.01, 0.05]:
            th = threshold_higher(arr, alpha)
            thresholds[test][str(alpha)] = {
                "threshold": th,
                "achieved_empirical_tail_fraction": achieved_tail(arr, th),
            }

    deltas = np.array(
        [float(x) for x in args.deltas.split(",")],
        dtype=float,
    )

    scenarios = [
        "compact_10NN_region",
        "broad_PC1_upper_tail",
        "exact_IcBL_subtype",
    ]
    tests = ["knn10", "linear_PC1_PC2", "max_subtype"]

    best_test = {
        "compact_10NN_region": "knn10",
        "broad_PC1_upper_tail": "linear_PC1_PC2",
        "exact_IcBL_subtype": "max_subtype",
    }

    rng_master = np.random.default_rng(args.seed)

    rows = []

    # Use the actual joint-null z vectors as baseline and sample rows with
    # replacement for 20k injection trials at each scenario/delta.
    for scenario in scenarios:
        for delta in deltas:
            # Separate reproducible seed for every scenario/delta.
            local_seed = int(
                rng_master.integers(0, 2**31-1)
            )
            rng = np.random.default_rng(local_seed)

            draw = rng.integers(
                0, z_glob.shape[0],
                size=args.n_injection_trials,
            )
            z_base = z_glob[draw].copy()
            z_inj = build_injected_z(
                z_base,
                scenario=scenario,
                delta=float(delta),
                rng=rng,
                nbr=nbr,
                cat=cat,
            )

            stats = stat_triplet(
                z_inj, A, Q, masks
            )

            for test in tests:
                arr = np.asarray(stats[test], float)

                for alpha in [0.01, 0.05]:
                    threshold = thresholds[test][str(alpha)]["threshold"]
                    power = float(np.mean(arr >= threshold))

                    rows.append({
                        "scenario": scenario,
                        "delta_z": float(delta),
                        "test": test,
                        "alpha": alpha,
                        "power": power,
                        "n_injection_trials": int(args.n_injection_trials),
                        "seed": local_seed,
                        "observed_statistic": float(obs_stats[test]),
                        "null_threshold": float(threshold),
                        "prob_stat_le_observed": float(
                            np.mean(arr <= obs_stats[test])
                        ),
                    })

    res = pd.DataFrame(rows)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(outdir / "empirical_jointnull_injection_power.csv", index=False)

    # Compact score-space effect-size summary.  For each scenario use its
    # pre-specified best-matched test and estimate the Delta-z at which only
    # 10% of injected trials remain at or below the actually observed statistic.
    effect_rows = []
    for scenario in scenarios:
        test = best_test[scenario]
        s = res[
            (res["scenario"] == scenario)
            & (res["test"] == test)
            & (res["alpha"] == 0.01)
        ].sort_values("delta_z")

        delta90 = interpolation_crossing(
            s["delta_z"].to_numpy(float),
            s["prob_stat_le_observed"].to_numpy(float),
            0.10,
        )
        delta50power = interpolation_crossing(
            s["delta_z"].to_numpy(float),
            s["power"].to_numpy(float),
            0.50,
        )
        delta80power = interpolation_crossing(
            s["delta_z"].to_numpy(float),
            s["power"].to_numpy(float),
            0.80,
        )
        delta90power = interpolation_crossing(
            s["delta_z"].to_numpy(float),
            s["power"].to_numpy(float),
            0.90,
        )

        effect_rows.append({
            "scenario": scenario,
            "best_matched_test": test,
            "observed_statistic": float(obs_stats[test]),
            "delta_z_approx_90pct_upper_effect_size": delta90,
            "delta_z_for_50pct_detection_power_alpha0p01": delta50power,
            "delta_z_for_80pct_detection_power_alpha0p01": delta80power,
            "delta_z_for_90pct_detection_power_alpha0p01": delta90power,
        })

    effect = pd.DataFrame(effect_rows)
    effect.to_csv(outdir / "empirical_effect_size_summary.csv", index=False)

    summary = {
        "N_sources": int(len(cat)),
        "B_calibration": int(q_cal.shape[0]),
        "B_global": int(q_glob.shape[0]),
        "n_injection_trials_per_scenario_per_delta": int(args.n_injection_trials),
        "deltas_z": deltas.tolist(),
        "observed_statistics": {
            k: float(v) for k,v in obs_stats.items()
        },
        "null_thresholds": thresholds,
        "scenarios_regenerated": scenarios,
        "scenario_not_regenerated": {
            "smooth_local_region": (
                "Exact Gaussian bandwidth was not preserved in frozen metadata; "
                "definition intentionally not reconstructed after unblinding."
            )
        },
        "interpretation": (
            "Additive calibrated-z method/effect-size benchmark using actual "
            "joint detector-level null covariance; not detector-level flux sensitivity."
        ),
    }
    (outdir / "empirical_injection_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    labels = {
        "knn10": "Phenotype kNN (k=10)",
        "linear_PC1_PC2": "Linear PC1-PC2 trend",
        "max_subtype": "Max spectroscopic subtype",
    }
    titles = {
        "compact_10NN_region": "Compact 10-NN optical-region injection",
        "broad_PC1_upper_tail": "Broad PC1-organized injection",
        "exact_IcBL_subtype": "Ic-BL subtype-aligned injection",
    }

    # One distinct plot per scenario (no subplots).
    for scenario in scenarios:
        s0 = res[
            (res["scenario"] == scenario)
            & (res["alpha"] == 0.01)
        ]
        fig, ax = plt.subplots(figsize=(7.2,5.2))
        for test in tests:
            s = s0[s0["test"] == test].sort_values("delta_z")
            ax.plot(
                s["delta_z"],
                s["power"],
                marker="o",
                label=labels[test],
            )
        ax.axhline(0.5, linewidth=1, linestyle="--")
        ax.axhline(0.8, linewidth=1, linestyle=":")
        ax.set_ylim(0,1.04)
        ax.set_xlabel("Injected additive shift in calibrated neutrino evidence, Δz")
        ax.set_ylabel("Detection power at empirical global α=0.01")
        ax.set_title(titles[scenario])
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            outdir / f"power_{scenario}_alpha0p01.png",
            dpi=300,
        )
        fig.savefig(
            outdir / f"power_{scenario}_alpha0p01.pdf"
        )
        plt.close(fig)

    print("\nOBSERVED STATISTICS")
    print(json.dumps(
        {k: float(v) for k,v in obs_stats.items()},
        indent=2,
    ))

    print("\nEMPIRICAL NULL THRESHOLDS")
    print(json.dumps(thresholds, indent=2))

    print("\nEFFECT-SIZE / POWER SUMMARY")
    print(effect.to_string(index=False))

    print("\nWROTE")
    for p in sorted(outdir.iterdir()):
        print(" ", p)


if __name__ == "__main__":
    main()
