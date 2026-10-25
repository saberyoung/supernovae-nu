#!/usr/bin/env python3
"""
TS_300-only exact-cache worker for the pre-defined long-timescale robustness.

This script computes ONLY the 300 d Gaussian-kernel TS using the exact full
SkyLLH likelihood. It relies on the already-validated compute engine:
    code/joint_background.py

The result is later combined elementwise with the saved primary statistic:
    q_primary = max(TS_10, TS_30, TS_100)

to form the robustness statistic:
    q_4kernel = max(q_primary, TS_300)

A  identity benchmark on five BTS sources plus one shared background
trial found exact equality (max abs diff = 0.0) between standalone TS300 and
TS300 evaluated after the 10/30/100 fits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from skyllh.core.config import Config
from skyllh.core.random import RandomStateService

# This script lives in code/, alongside these already-installed helpers.
from icetracks import create_ic86_dataset
from source_likelihood import score_initialized_trial
from joint_background import (
    build_once,
    EnergyResponseCache,
    switch_source_exact,
)

SIGMAS = (300.0,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--icetracks-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--ntrials", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()

    cat_all = pd.read_csv(args.catalog)
    stop = len(cat_all) if args.stop is None else min(args.stop, len(cat_all))
    cat = cat_all.iloc[args.start:stop].copy().reset_index(drop=False)
    if len(cat) == 0:
        raise ValueError("Empty source block.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    cfg = Config()
    datasets = create_ic86_dataset(
        cfg=cfg,
        icetracks_dir=args.icetracks_dir,
    )

    t0 = time.time()
    ana = build_once(cfg, datasets, cat.iloc[0])
    print(
        f"build_once={time.time()-t0:.2f}s "
        f"block={args.start}:{stop} Nsrc={len(cat)} "
        f"Ntrials={args.ntrials} seed_base={args.seed_base}",
        flush=True,
    )

    ecache = EnergyResponseCache(ana, cfg)
    ecache.register_current(
        np.deg2rad(float(cat.iloc[0]["dec_deg"]))
    )

    real_events = [data.exp for data in ana.data_list]
    real_n_events = [len(x) for x in real_events]

    ns = len(cat)
    ts300_obs = np.empty(ns, dtype=float)

    # Observed pass also pre-populates response-bin cache.
    for j, (_, row) in enumerate(cat.iterrows()):
        switch_source_exact(ana, row, ecache)
        ana.initialize_trial(real_events, real_n_events)

        fit_rss = RandomStateService(
            seed=31000000 + int(row["index"])
        )
        best, _ = score_initialized_trial(
            ana=ana,
            peak_mjd=float(row["peak_mjd"]),
            rss=fit_rss,
            sigmas=SIGMAS,
        )
        ts300_obs[j] = float(best["ts"])

    print(
        f"observed_done response_bins={len(ecache.cache)} "
        f"elapsed={time.time()-t_all:.1f}s",
        flush=True,
    )

    seeds = args.seed_base + np.arange(
        args.ntrials, dtype=np.int64
    )
    ts300_bg = np.empty((args.ntrials, ns), dtype=float)

    t_bg = time.time()

    for b, seed in enumerate(seeds):
        tb = time.time()

        gen_rss = RandomStateService(seed=int(seed))
        n_sig, n_events_list, events_list = ana.generate_pseudo_data(
            rss=gen_rss,
            mean_n_sig=0.0,
        )
        if int(n_sig) != 0:
            raise RuntimeError("Unexpected signal in background trial.")

        for j, (_, row) in enumerate(cat.iterrows()):
            _, built = switch_source_exact(ana, row, ecache)
            if built:
                raise RuntimeError(
                    "New response bin appeared after observed pre-cache."
                )

            ana.initialize_trial(events_list, n_events_list)

            fit_rss = RandomStateService(
                seed=int(
                    seed
                    + 1000003 * (int(row["index"]) + 1)
                )
            )
            best, _ = score_initialized_trial(
                ana=ana,
                peak_mjd=float(row["peak_mjd"]),
                rss=fit_rss,
                sigmas=SIGMAS,
            )
            ts300_bg[b, j] = float(best["ts"])

        if (
            b == 0
            or (b + 1) % max(1, args.progress_every) == 0
            or b + 1 == args.ntrials
        ):
            elapsed = time.time() - t_bg
            mean = elapsed / (b + 1)
            eta = mean * (args.ntrials - b - 1)
            print(
                f"trial {b+1}/{args.ntrials} "
                f"last={time.time()-tb:.2f}s "
                f"mean={mean:.2f}s/trial "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

    # q_obs/q_bg names allow the existing validated merge/concat utilities
    # to be reused unchanged.
    np.savez_compressed(
        out,
        source_index=cat["index"].to_numpy(np.int64),
        ZTFID=cat["ZTFID"].astype(str).to_numpy(),
        trial_seed=seeds,
        q_obs=ts300_obs,
        best_sigma_obs=np.full(ns, 300.0),
        q_bg=ts300_bg,
    )

    print(
        f"Wrote {out}\nTOTAL={time.time()-t_all:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
