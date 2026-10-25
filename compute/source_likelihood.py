#!/usr/bin/env python3
"""
BTS phenotype × IceTracks-DR2 source-level neutrino scores (v1).

Frozen discovery design
-----------------------
Optical phenotype is built independently from neutrino information.
For every source we use the same time-kernel bank:
    Gaussian centered on BTS peak_mjd with sigma = 10, 30, 100 d.

The per-source statistic is:
    T_i = max(TS_10, TS_30, TS_100)

Its local background p-value is calibrated with pseudo-experiments that repeat
the same max-over-three procedure, so the time-scale trials factor is included.

Dataset choice
--------------
The current SkyLLH v26.1 time-dependent public-data analysis supports a single
dataset at a time. Since BTS begins in 2018, all sources in this project lie
within the full-detector IC86 era. We therefore use the combined
IceTracks-DR2 dataset `IC86_I-XI`, covering the relevant 2011-2022 IC86 data.

Requirements
------------
Python >= 3.11
skyllh == 26.1.0 (or API-compatible newer release)
numpy, pandas, scipy

SkyLLH automatically downloads IceTracks-DR2 from Harvard Dataverse by default.

Example
-------
python compute/source_likelihood.py \
    search_sample.csv \
    --output sesn_icetracks_scores_dev.csv \
    --ntrials 200 --max-sources 5

Production runs should use substantially more background trials.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

import skyllh
from skyllh.analyses.i3.publicdata_ps.time_dependent_ps import (
    change_signal_time_pdf_of_llhratio_function,
    create_analysis,
)
from skyllh.core.config import Config
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import PointLikeSource
from icetracks import create_ic86_dataset


PRIMARY_SIGMAS_D = (10.0, 30.0, 100.0)


def maximize_current_kernel(ana, rss: RandomStateService):
    """Maximize the current fixed-time-PDF likelihood."""
    log_lambda, fitparam_values, status = ana.llhratio.maximize(rss=rss)
    ts = ana.calculate_test_statistic(
        log_lambda=log_lambda,
        fitparam_values=fitparam_values,
    )
    ns_fit = float(fitparam_values[0])
    gamma_fit = float(fitparam_values[1]) if len(fitparam_values) > 1 else np.nan
    return float(ts), ns_fit, gamma_fit, status


def score_initialized_trial(
    ana,
    peak_mjd: float,
    rss: RandomStateService,
    sigmas=PRIMARY_SIGMAS_D,
):
    """Evaluate the same initialized trial under every frozen time kernel."""
    per_kernel = {}
    best = None

    for sigma_d in sigmas:
        change_signal_time_pdf_of_llhratio_function(
            ana=ana,
            gauss={"mu": float(peak_mjd), "sigma": float(sigma_d)},
        )
        ts, ns_fit, gamma_fit, status = maximize_current_kernel(ana, rss)
        per_kernel[float(sigma_d)] = {
            "ts": ts,
            "ns_fit": ns_fit,
            "gamma_fit": gamma_fit,
        }
        if best is None or ts > best["ts"]:
            best = {
                "sigma_d": float(sigma_d),
                "ts": ts,
                "ns_fit": ns_fit,
                "gamma_fit": gamma_fit,
            }

    assert best is not None
    return best, per_kernel


def initialize_observed_data(ana, rss: RandomStateService):
    """
    Initialize SkyLLH with the real experimental data.

    ana.unblind() also performs one fit for the currently configured kernel.
    We ignore that fit here and re-evaluate the complete frozen kernel bank.
    """
    ana.unblind(minimizer_rss=rss)


def initialize_background_trial(ana, rss: RandomStateService):
    """
    Generate one background-only pseudo-experiment using SkyLLH's configured
    IceCube time/RA scrambling and initialize the trial.
    """
    n_sig, n_events_list, events_list = ana.generate_pseudo_data(
        rss=rss,
        mean_n_sig=0.0,
    )
    ana.initialize_trial(events_list, n_events_list)
    return int(n_sig)


def score_one_source(
    row: pd.Series,
    datasets,
    cfg: Config,
    ntrials: int,
    seed: int,
):
    ra_rad = np.deg2rad(float(row["ra_deg"]))
    dec_rad = np.deg2rad(float(row["dec_deg"]))
    peak_mjd = float(row["peak_mjd"])

    source = PointLikeSource(
        name=str(row["ZTFID"]),
        ra=ra_rad,
        dec=dec_rad,
    )

    # Initial sigma=30 d only initializes the analysis. The actual score is
    # max over the frozen 10/30/100 d bank below.
    ana = create_analysis(
        cfg=cfg,
        datasets=datasets,
        source=source,
        gauss={"mu": peak_mjd, "sigma": 30.0},
        gamma_seed=2.5,
        gamma_min=1.0,
        gamma_max=4.0,
        evt_sel_delta_angle_deg=10.0,
    )

    obs_rss = RandomStateService(seed=seed)
    initialize_observed_data(ana, obs_rss)
    obs_best, obs_each = score_initialized_trial(
        ana=ana,
        peak_mjd=peak_mjd,
        rss=obs_rss,
    )

    bg_max_ts = np.empty(ntrials, dtype=float)
    trial_rss = RandomStateService(seed=seed + 10_000_000)

    for k in range(ntrials):
        initialize_background_trial(ana, trial_rss)
        best, _ = score_initialized_trial(
            ana=ana,
            peak_mjd=peak_mjd,
            rss=trial_rss,
        )
        bg_max_ts[k] = best["ts"]

    # Finite-sample empirical p-value with +1 correction.
    p_local = (1.0 + np.sum(bg_max_ts >= obs_best["ts"])) / (ntrials + 1.0)
    z_local = float(norm.isf(p_local))

    result = {
        "ZTFID": row["ZTFID"],
        "IAUID": row.get("IAUID", ""),
        "type": row.get("type", ""),
        "ra_deg": float(row["ra_deg"]),
        "dec_deg": float(row["dec_deg"]),
        "peak_mjd": peak_mjd,
        "PC1": row.get("PC1", np.nan),
        "PC2": row.get("PC2", np.nan),
        "PC3": row.get("PC3", np.nan),
        "ts_10d": obs_each[10.0]["ts"],
        "ts_30d": obs_each[30.0]["ts"],
        "ts_100d": obs_each[100.0]["ts"],
        "ts_max": obs_best["ts"],
        "best_sigma_d": obs_best["sigma_d"],
        "ns_fit_best": obs_best["ns_fit"],
        "gamma_fit_best": obs_best["gamma_fit"],
        "n_bg_trials": int(ntrials),
        "p_local": float(p_local),
        "z_local": z_local,
        "bg_ts_median": float(np.median(bg_max_ts)),
        "bg_ts_90pct": float(np.quantile(bg_max_ts, 0.90)),
        "bg_ts_99pct": float(np.quantile(bg_max_ts, 0.99)),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", help="Frozen BTS phenotype/control CSV")
    parser.add_argument("--icetracks-dir", required=True,
                        help="Extracted IceTracks-DR2 directory containing events/, uptime/, irfs/")
    parser.add_argument("--output", default="source_scores.csv")
    parser.add_argument("--ntrials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--max-sources", type=int, default=None)
    args = parser.parse_args()

    cat = pd.read_csv(args.catalog)
    if args.max_sources is not None:
        cat = cat.iloc[: args.max_sources].copy()

    cfg = Config()

    # BTS is entirely in the IC86 era. The v26.1 time-dependent public-data
    # analysis supports one dataset, so use the combined IC86_I-XI release.
    datasets = create_ic86_dataset(cfg=cfg, icetracks_dir=args.icetracks_dir)

    results = []
    for i, row in cat.iterrows():
        print(f"[{len(results)+1}/{len(cat)}] {row['ZTFID']} {row.get('type','')}")
        res = score_one_source(
            row=row,
            datasets=datasets,
            cfg=cfg,
            ntrials=args.ntrials,
            seed=args.seed + 100003 * len(results),
        )
        results.append(res)
        pd.DataFrame(results).to_csv(args.output, index=False)

    out = pd.DataFrame(results)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
