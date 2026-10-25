#!/usr/bin/env python3
"""
Validate the SkyLLH v26.1.0 / IceTracks-DR2 installation against the
official TXS 0506+056 time-dependent tutorial before unblinding BTS SNe.

Official tutorial reference (v26.1.0, IC86_I-XI):
    best TS       ~ 15.4799
    best EM mu    ~ 56966.1 MJD
    best EM sigma ~ 28.3169 d
    best ns       ~ 7.12294
    best gamma    ~ 2.21386

The assertions use loose tolerances because minimizer/library versions may
produce small numerical differences. This is an implementation validation,
not a scientific re-analysis of TXS 0506+056.
"""

import numpy as np
import skyllh

from skyllh.analyses.i3.publicdata_ps.time_dependent_ps import (
    TXS_0506_PLUS056_ALERT_TIME,
    TXS_0506_PLUS056_SOURCE,
    create_analysis,
    unblind_single_flare,
)
from skyllh.core.config import Config
from icetracks import create_ic86_dataset


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--icetracks-dir', required=True)
    args = ap.parse_args()
    cfg = Config()
    datasets = create_ic86_dataset(cfg=cfg, icetracks_dir=args.icetracks_dir)

    ana = create_analysis(
        cfg=cfg,
        datasets=datasets,
        source=TXS_0506_PLUS056_SOURCE,
        refplflux_gamma=2.0,
        gauss={"mu": 57000.0, "sigma": 65.0},
    )

    best_ts, best_em_result, best_fitparam_values = unblind_single_flare(
        ana=ana,
        remove_time=TXS_0506_PLUS056_ALERT_TIME,
    )

    got = {
        "TS": float(best_ts),
        "mu": float(best_em_result["mu"]),
        "sigma": float(best_em_result["sigma"]),
        "ns": float(best_fitparam_values[0]),
        "gamma": float(best_fitparam_values[1]),
    }

    ref = {
        "TS": 15.4799,
        "mu": 56966.1,
        "sigma": 28.3169,
        "ns": 7.12294,
        "gamma": 2.21386,
    }

    print("Observed validation result:")
    for key in got:
        print(f"  {key:>6s}: {got[key]:.6g}   (tutorial: {ref[key]:.6g})")

    # Deliberately broad tolerances: enough to catch a broken dataset/API,
    # but not reject harmless numerical/minimizer differences.
    checks = {
        "TS": abs(got["TS"] - ref["TS"]) < 1.0,
        "mu": abs(got["mu"] - ref["mu"]) < 5.0,
        "sigma": abs(got["sigma"] - ref["sigma"]) < 5.0,
        "ns": abs(got["ns"] - ref["ns"]) < 1.5,
        "gamma": abs(got["gamma"] - ref["gamma"]) < 0.3,
    }

    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(
            "SkyLLH/IceTracks validation failed for: " + ", ".join(failed)
        )

    print("\nVALIDATION PASSED: environment is consistent with the official tutorial.")


if __name__ == "__main__":
    main()
