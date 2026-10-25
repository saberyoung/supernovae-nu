#!/usr/bin/env python3
"""
Validate that a fixed RandomStateService seed produces the same background-only
IceTracks pseudo-data for different source analyses.

This is required before using block-wise joint-background trials. With mean_n_sig=0,
the generated detector background must be source-independent. The source-specific
spatial event selection is applied only after the common pseudo-data are generated.
"""

import numpy as np
import skyllh

from skyllh.analyses.i3.publicdata_ps.time_dependent_ps import create_analysis
from skyllh.core.config import Config
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import PointLikeSource
from icetracks import create_ic86_dataset


def make_analysis(cfg, datasets, name, ra_deg, dec_deg):
    src = PointLikeSource(
        name=name,
        ra=np.deg2rad(ra_deg),
        dec=np.deg2rad(dec_deg),
    )
    return create_analysis(
        cfg=cfg,
        datasets=datasets,
        source=src,
        gauss={"mu": 59000.0, "sigma": 30.0},
        gamma_seed=2.5,
        gamma_min=1.0,
        gamma_max=4.0,
    )


def compare_event_arrays(a, b):
    if len(a) != len(b):
        return False, f"dataset-list lengths differ: {len(a)} vs {len(b)}"
    for ds_idx, (x, y) in enumerate(zip(a, b, strict=True)):
        if len(x) != len(y):
            return False, f"dataset {ds_idx} event counts differ: {len(x)} vs {len(y)}"
        # Compare the fields that define the scrambled realization.
        for field in ("time", "ra", "dec", "log_energy", "ang_err"):
            if field not in x.field_name_list or field not in y.field_name_list:
                continue
            xv = x[field]
            yv = y[field]
            if not np.array_equal(xv, yv):
                if not np.allclose(xv, yv, rtol=0, atol=0):
                    return False, f"dataset {ds_idx} field {field} differs"
    return True, "identical"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--icetracks-dir', required=True)
    args = ap.parse_args()
    cfg = Config()
    datasets = create_ic86_dataset(cfg=cfg, icetracks_dir=args.icetracks_dir)

    ana1 = make_analysis(cfg, datasets, "test_north", 10.0, 30.0)
    ana2 = make_analysis(cfg, datasets, "test_south", 220.0, -20.0)

    seed = 881726
    out = []
    for ana in (ana1, ana2):
        rss = RandomStateService(seed=seed)
        n_sig, n_events_list, events_list = ana.generate_pseudo_data(
            rss=rss,
            mean_n_sig=0.0,
        )
        out.append((n_sig, n_events_list, events_list))

    if out[0][0] != 0 or out[1][0] != 0:
        raise RuntimeError("Unexpected injected signal in background-only validation.")

    if out[0][1] != out[1][1]:
        raise RuntimeError("Background event-count lists differ between source analyses.")

    ok, message = compare_event_arrays(out[0][2], out[1][2])
    if not ok:
        raise RuntimeError("Shared-scramble validation FAILED: " + message)

    print("SHARED-SCRAMBLE VALIDATION PASSED:", message)
    print("A common trial seed can be reused block-wise across all BTS sources.")


if __name__ == "__main__":
    main()
