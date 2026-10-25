#!/usr/bin/env python3
"""
Analyze the completed 2000+10000 BTS x IceTracks publication run.

This script does NOT rerun SkyLLH and does NOT alter the frozen source scores.
It uses:
  - the 2000-trial source calibration matrix,
  - the 10000-trial independent joint global-null matrix,
  - the frozen optical phenotype catalog.

Outputs:
  1) primary hotspot/member diagnostics
  2) Figure 2: neutrino-painted PC1-PC2 plane
  3) Figure 3: joint-background T_local null distribution
  4) frozen secondary global PC-gradient test
  5) frozen spectroscopic subtype stack tests
  6) same-matrix robustness tests:
       bright, r-only, endpoint buffer, exclude SLSN-I,
       PC1-PC2-PC3 geometry, k=5/15/20

The 300-d time branch and matched-Ia negative control require additional
SkyLLH source-score matrices and are intentionally NOT computed here.
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


# The max-subtype synthetic benchmark was frozen on these six adequately
# populated classes. SN Icn (N=2) is still reported individually but is not
# included in the confirmatory max-over-subtypes statistic.
MAX_SUBTYPE_CLASSES = [
    "SN IIb", "SN Ib", "SN Ic", "SN Ic-BL", "SN Ibn", "SLSN-I"
]
ALL_SUBTYPE_CLASSES = MAX_SUBTYPE_CLASSES + ["SN Icn"]


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
    p = np.clip(p, lo, 1.0 - lo)
    return norm.isf(p), p


def empirical_z_matrix(qmat, qcal):
    """Exact matrix transform, optimized for 10k x 241 publication nulls."""
    qmat = np.asarray(qmat, float)
    qcal = np.asarray(qcal, float)
    B = qcal.shape[0]
    p = np.empty_like(qmat, dtype=float)
    for j in range(qmat.shape[1]):
        s = np.sort(qcal[:, j])
        n_lt = np.searchsorted(s, qmat[:, j], side="left")
        p[:, j] = (1.0 + B - n_lt) / (B + 1.0)
    lo = 0.5 / (B + 1.0)
    p = np.clip(p, lo, 1.0 - lo)
    return norm.isf(p)


def p_empirical_ge(null, obs):
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= obs)) / (len(null) + 1.0))


def knn_neighbors(x, k):
    x = np.asarray(x, float)
    if len(x) <= k:
        raise ValueError(f"k={k} requires >{k} sources; got {len(x)}")
    d = cdist(x, x)
    nbr = np.argsort(d, axis=1)[:, 1:k+1]
    return nbr, d


def knn_stats(z, x, k):
    nbr, d = knn_neighbors(x, k)
    if z.ndim == 1:
        local = np.array([
            np.sum(z[inds]) / np.sqrt(k) for inds in nbr
        ])
        return float(np.max(local)), local, nbr, d

    local = np.empty((z.shape[0], len(x)), dtype=float)
    for i, inds in enumerate(nbr):
        local[:, i] = np.sum(z[:, inds], axis=1) / np.sqrt(k)
    return np.max(local, axis=1), local, nbr, d


def standardized_design(df, cols):
    X = df[cols].to_numpy(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    Xs = (X - mu) / sd
    return Xs, mu, sd


def gradient_basis(df):
    # Frozen definition:
    # T_grad = ||Q^T z||^2, Q spanning centered/standardized PC1 and PC2.
    Xs, mu, sd = standardized_design(df, ["PC1", "PC2"])
    Q, _ = np.linalg.qr(Xs, mode="reduced")
    return Q, Xs, mu, sd


def gradient_stat(z, Q):
    if z.ndim == 1:
        proj = Q.T @ z
        return float(np.sum(proj * proj)), proj
    proj = z @ Q
    return np.sum(proj * proj, axis=1), proj


def subtype_scores(z, types, classes):
    types = np.asarray(types)
    scores = {}
    for cls in classes:
        m = types == cls
        n = int(np.sum(m))
        if n == 0:
            scores[cls] = np.nan
            continue
        if z.ndim == 1:
            scores[cls] = float(np.sum(z[m]) / np.sqrt(n))
        else:
            scores[cls] = np.sum(z[:, m], axis=1) / np.sqrt(n)
    return scores


def compute_knn_branch(name, cat, z_obs, z_glob, mask, dims, k):
    sub = cat.loc[mask].copy().reset_index()
    zo = z_obs[np.asarray(mask)]
    zg = z_glob[:, np.asarray(mask)]
    x = sub[dims].to_numpy(float)

    Tobs, local, nbr, dist = knn_stats(zo, x, k)
    Tnull, _, _, _ = knn_stats(zg, x, k)
    p = p_empirical_ge(Tnull, Tobs)
    ihot = int(np.argmax(local))

    return {
        "branch": name,
        "N": int(len(sub)),
        "dims": "+".join(dims),
        "k": int(k),
        "T_observed": float(Tobs),
        "global_p": float(p),
        "hotspot_center_ZTFID": str(sub.iloc[ihot]["ZTFID"]),
        "hotspot_center_type": str(sub.iloc[ihot]["type"]),
    }


def save_primary_hotspot(outdir, cat, q_obs, z_obs, p_obs, z_glob, k=10):
    x = cat[["PC1", "PC2"]].to_numpy(float)
    Tobs, local, nbr, dist = knn_stats(z_obs, x, k)
    Tnull, _, _, _ = knn_stats(z_glob, x, k)
    pglobal = p_empirical_ge(Tnull, Tobs)
    ihot = int(np.argmax(local))
    ibest = int(np.argmax(z_obs))

    nodes = cat.copy()
    nodes["q_obs"] = q_obs
    nodes["p_local_calibrated"] = p_obs
    nodes["z_local_calibrated"] = z_obs
    nodes["phenotype_local_score"] = local
    nodes["is_primary_hotspot_center"] = False
    nodes.loc[ihot, "is_primary_hotspot_center"] = True
    nodes["is_best_individual"] = False
    nodes.loc[ibest, "is_best_individual"] = True
    nodes.to_csv(outdir / "primary_nodes_enriched.csv", index=False)

    inds = nbr[ihot]
    members = nodes.iloc[inds].copy()
    members["distance_from_center_PC12"] = dist[ihot, inds]
    members["contribution_to_T"] = (
        members["z_local_calibrated"].to_numpy(float) / np.sqrt(k)
    )
    members["neighbor_rank_by_distance"] = np.arange(1, len(members)+1)
    members["contribution_rank"] = (
        members["contribution_to_T"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    members = members.sort_values(
        ["contribution_rank", "distance_from_center_PC12"]
    )
    members.to_csv(outdir / "primary_hotspot_members_enriched.csv", index=False)

    center = nodes.iloc[ihot]
    summary = {
        "N_sources": int(len(cat)),
        "k": int(k),
        "T_observed": float(Tobs),
        "global_p": float(pglobal),
        "hotspot_center_ZTFID": str(center["ZTFID"]),
        "hotspot_center_IAUID": str(center["IAUID"]),
        "hotspot_center_type": str(center["type"]),
        "hotspot_center_PC1": float(center["PC1"]),
        "hotspot_center_PC2": float(center["PC2"]),
        "best_individual_ZTFID": str(nodes.iloc[ibest]["ZTFID"]),
        "best_individual_IAUID": str(nodes.iloc[ibest]["IAUID"]),
        "best_individual_type": str(nodes.iloc[ibest]["type"]),
        "best_individual_z": float(z_obs[ibest]),
        "best_individual_p_calibrated": float(p_obs[ibest]),
        "hotspot_member_z_sum": float(members["z_local_calibrated"].sum()),
        "hotspot_member_mean_z": float(members["z_local_calibrated"].mean()),
        "hotspot_member_positive_z_count": int(
            np.sum(members["z_local_calibrated"] > 0)
        ),
        "hotspot_subtype_counts": {
            str(k): int(v)
            for k, v in members["type"].value_counts().to_dict().items()
        },
    }
    (outdir / "primary_summary_detailed.json").write_text(
        json.dumps(summary, indent=2)
    )

    return summary, nodes, members, Tnull


def make_figure2(outdir, nodes, members, summary):
    fig, ax = plt.subplots(figsize=(8, 6.6))

    sc = ax.scatter(
        nodes["PC1"], nodes["PC2"],
        c=nodes["z_local_calibrated"],
        s=36,
        alpha=0.82,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Calibrated neutrino evidence $z_i$")

    # Hotspot neighbours: larger open markers (no explicit color choices).
    ax.scatter(
        members["PC1"], members["PC2"],
        s=130,
        facecolors="none",
        linewidths=1.3,
        label="Primary k=10 neighbourhood",
    )

    center = nodes.loc[nodes["is_primary_hotspot_center"]].iloc[0]
    best = nodes.loc[nodes["is_best_individual"]].iloc[0]

    ax.scatter(
        [center["PC1"]], [center["PC2"]],
        marker="X", s=170,
        label=f"Hotspot center: {center['ZTFID']}",
    )
    ax.scatter(
        [best["PC1"]], [best["PC2"]],
        marker="*", s=220,
        label=f"Best individual: {best['ZTFID']}",
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "BTS stripped-envelope transient phenotype space\n"
        f"$T_{{pheno}}$={summary['T_observed']:.3f}, "
        f"publication global $p$={summary['global_p']:.3f}"
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / "Figure2_neutrino_painted_PC12.png", dpi=250)
    fig.savefig(outdir / "Figure2_neutrino_painted_PC12.pdf")
    plt.close(fig)


def make_figure3(outdir, Tnull, Tobs, pglobal):
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.hist(Tnull, bins="auto", alpha=0.8)
    ax.axvline(Tobs, linewidth=2, label=f"Observed T={Tobs:.3f}")
    ax.set_xlabel("$T_{pheno}$")
    ax.set_ylabel("Joint-background realizations")
    ax.set_title(
        "Global null distribution of the frozen phenotype statistic\n"
        f"publication global $p$={pglobal:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "Figure3_Tlocal_global_null.png", dpi=250)
    fig.savefig(outdir / "Figure3_Tlocal_global_null.pdf")
    plt.close(fig)


def secondary_tests(outdir, cat, z_obs, z_glob, pca_model_path=None):
    rows = []

    # 1) Global linear PC1-PC2 gradient.
    Q, Xs, pc_mu, pc_sd = gradient_basis(cat)
    Tgrad_obs, proj_obs = gradient_stat(z_obs, Q)
    Tgrad_null, _ = gradient_stat(z_glob, Q)
    pgrad = p_empirical_ge(Tgrad_null, Tgrad_obs)

    # OLS slopes are only for reporting direction; significance uses T_grad.
    beta, *_ = np.linalg.lstsq(Xs, z_obs, rcond=None)

    grad_row = {
        "test": "linear_PC1_PC2_gradient",
        "N": int(len(cat)),
        "statistic": float(Tgrad_obs),
        "global_p": float(pgrad),
        "beta_PC1_standardized": float(beta[0]),
        "beta_PC2_standardized": float(beta[1]),
    }

    # Translate the observed gradient direction back into standardized
    # original optical features if the frozen PCA model is supplied.
    if pca_model_path and Path(pca_model_path).exists():
        model = json.loads(Path(pca_model_path).read_text())
        comps = np.asarray(model["pca_components"], float)
        # z ~ beta1*PC1_std + beta2*PC2_std. PC coordinates are rescaled here
        # only to describe direction; the formal T_grad above is unchanged.
        direction_pc = beta / pc_sd
        direction_orig_std = (
            direction_pc[0] * comps[0] +
            direction_pc[1] * comps[1]
        )
        for feat, val in zip(
            ["M_peak_std", "log_t_half_rise_std", "log_t_half_fade_std"],
            direction_orig_std
        ):
            grad_row["direction_" + feat] = float(val)

    rows.append(grad_row)

    # 2) Spectroscopic subtype stacks.
    sobs = subtype_scores(
        z_obs, cat["type"].to_numpy(), ALL_SUBTYPE_CLASSES
    )
    snull = subtype_scores(
        z_glob, cat["type"].to_numpy(), ALL_SUBTYPE_CLASSES
    )

    for cls in ALL_SUBTYPE_CLASSES:
        n = int(np.sum(cat["type"].to_numpy() == cls))
        obs = sobs[cls]
        null = snull[cls]
        p = p_empirical_ge(null, obs)
        rows.append({
            "test": f"subtype_{cls}",
            "N": n,
            "statistic": float(obs),
            "global_p": float(p),
            "note": (
                "Reported individually; N=2 and excluded from frozen "
                "max-subtype synthetic benchmark"
                if cls == "SN Icn" else ""
            ),
        })

    # Frozen confirmatory max-over-subtypes statistic used in the synthetic
    # benchmark (six adequately populated classes).
    max_obs = max(float(sobs[c]) for c in MAX_SUBTYPE_CLASSES)
    max_null = np.max(
        np.column_stack([snull[c] for c in MAX_SUBTYPE_CLASSES]),
        axis=1,
    )
    rows.append({
        "test": "max_predefined_subtype",
        "N": int(len(cat)),
        "statistic": float(max_obs),
        "global_p": float(p_empirical_ge(max_null, max_obs)),
        "winning_subtype": str(max(
            MAX_SUBTYPE_CLASSES, key=lambda c: sobs[c]
        )),
        "classes_in_max": ",".join(MAX_SUBTYPE_CLASSES),
    })

    sec = pd.DataFrame(rows)
    sec.to_csv(outdir / "secondary_tests.csv", index=False)

    detail = {
        "linear_PC_gradient": grad_row,
        "subtype_scores": {
            cls: {
                "N": int(np.sum(cat["type"] == cls)),
                "score": float(sobs[cls]),
                "p": float(p_empirical_ge(snull[cls], sobs[cls])),
            }
            for cls in ALL_SUBTYPE_CLASSES
        },
        "max_predefined_subtype": {
            "classes": MAX_SUBTYPE_CLASSES,
            "statistic": float(max_obs),
            "winning_subtype": str(max(
                MAX_SUBTYPE_CLASSES, key=lambda c: sobs[c]
            )),
            "global_p": float(p_empirical_ge(max_null, max_obs)),
        },
    }
    (outdir / "secondary_tests.json").write_text(
        json.dumps(detail, indent=2)
    )

    return sec


def robustness_tests(outdir, cat, z_obs, z_glob):
    branches = []

    full = np.ones(len(cat), dtype=bool)

    # Reproduce primary for reference.
    branches.append(compute_knn_branch(
        "primary_PC12_k10", cat, z_obs, z_glob,
        full, ["PC1", "PC2"], 10
    ))

    # Alternate neighbourhood scales.
    for k in [5, 15, 20]:
        branches.append(compute_knn_branch(
            f"PC12_k{k}", cat, z_obs, z_glob,
            full, ["PC1", "PC2"], k
        ))

    # Three-PC geometry.
    branches.append(compute_knn_branch(
        "PC123_k10", cat, z_obs, z_glob,
        full, ["PC1", "PC2", "PC3"], 10
    ))

    # Predefined sample cuts. The kNN graph is rebuilt inside each fixed cut.
    sample_defs = {
        "mpeak_le_18p5_PC12_k10":
            cat["gold_mpeak18p5"].astype(bool).to_numpy(),
        "r_peak_only_PC12_k10":
            cat["r_peak_only"].astype(bool).to_numpy(),
        "endpoint_buffer100_PC12_k10":
            cat["endpoint_buffer100"].astype(bool).to_numpy(),
        "exclude_SLSNI_PC12_k10":
            (cat["type"] != "SLSN-I").to_numpy(),
    }

    for name, mask in sample_defs.items():
        branches.append(compute_knn_branch(
            name, cat, z_obs, z_glob,
            mask, ["PC1", "PC2"], 10
        ))

    rob = pd.DataFrame(branches)
    rob.to_csv(outdir / "robustness_same_matrix.csv", index=False)
    return rob


def print_compact(summary, members, secondary, robust):
    print("\n================ PRIMARY ================")
    print(json.dumps(summary, indent=2))

    cols = [
        "ZTFID", "IAUID", "type",
        "z_local_calibrated", "p_local_calibrated",
        "contribution_to_T",
        "peakabs_num", "rise_num", "fade_num",
        "PC1", "PC2",
    ]
    cols = [c for c in cols if c in members.columns]
    print("\nPRIMARY HOTSPOT MEMBERS")
    print(members[cols].to_string(index=False))

    print("\n================ SECONDARY ================")
    display = [
        c for c in
        ["test", "N", "statistic", "global_p",
         "winning_subtype", "beta_PC1_standardized",
         "beta_PC2_standardized"]
        if c in secondary.columns
    ]
    print(secondary[display].to_string(index=False))

    print("\n================ ROBUSTNESS ================")
    print(robust.to_string(index=False))


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
        "--pca-model",
        default="config/pca_model.json",
    )
    ap.add_argument(
        "--output-dir",
        default="work/analysis",
    )
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog)
    cal = np.load(args.cal, allow_pickle=True)
    glob = np.load(args.global_null, allow_pickle=True)

    q_obs = np.asarray(cal["q_obs"], float)
    q_cal = np.asarray(cal["q_bg"], float)
    q_glob = np.asarray(glob["q_bg"], float)

    if len(cat) != len(q_obs):
        raise ValueError("Catalog/source-score length mismatch.")
    if q_cal.shape[1] != len(cat) or q_glob.shape[1] != len(cat):
        raise ValueError("Background matrix/source count mismatch.")

    # Ensure the merged files really follow catalog order.
    expected = np.arange(len(cat), dtype=np.int64)
    for name, f in [("cal", cal), ("global", glob)]:
        if "source_index" in f.files and not np.array_equal(
            f["source_index"], expected
        ):
            raise ValueError(f"{name} source order != catalog order")

    z_obs, p_obs = empirical_z(q_obs, q_cal)
    z_glob = empirical_z_matrix(q_glob, q_cal)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary, nodes, members, Tnull = save_primary_hotspot(
        outdir, cat, q_obs, z_obs, p_obs, z_glob, k=10
    )

    np.savez_compressed(
        outdir / "primary_null.npz",
        T_global=Tnull,
        T_observed=np.asarray([summary["T_observed"]]),
    )

    make_figure2(outdir, nodes, members, summary)
    make_figure3(
        outdir, Tnull,
        summary["T_observed"], summary["global_p"]
    )

    secondary = secondary_tests(
        outdir, cat, z_obs, z_glob,
        pca_model_path=args.pca_model,
    )
    robust = robustness_tests(
        outdir, cat, z_obs, z_glob
    )

    print_compact(
        summary, members, secondary, robust
    )

    print("\nWROTE")
    for p in sorted(outdir.iterdir()):
        print(" ", p)


if __name__ == "__main__":
    main()
