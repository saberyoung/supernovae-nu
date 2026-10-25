#!/usr/bin/env python3
"""
Exact fixed-gamma joint-background worker for BTS x IceTracks.

Robustness branches:
    gamma = 2.0
    gamma = 2.5

For each source:
    q_i(gamma) = max(TS_10, TS_30, TS_100)

Gamma is a TRUE fixed SkyLLH parameter. Only n_s is floated.

Implementation validation
-------------------------
For both gamma=2.0 and gamma=2.5, a dedicated five-source A/B benchmark found:
  - gamma absent from floating fit vector;
  - observed q_i: exact equality between independent per-source construction
    and source-reuse implementation;
  - shared-background q_i: exact equality;
  - best time kernel: exact equality.

This worker uses one analysis per horizontal source block plus the validated
declination-response cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

import numpy as np
import pandas as pd

import skyllh.analyses.i3.publicdata_ps.time_dependent_ps as tdps
from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff
from skyllh.analyses.i3.publicdata_ps.signalpdf import PDSignalEnergyPDFSet
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import PDSmearingMatrix
from skyllh.core.config import Config
from skyllh.core.llhratio import (
    MultiDatasetTCLLHRatio,
    ZeroSigH0SingleDatasetTCLLHRatio,
)
from skyllh.core.parameters import Parameter as SkyParameter
from skyllh.core.random import RandomStateService
from skyllh.core.source_model import PointLikeSource

from icetracks import create_ic86_dataset
from source_likelihood import score_initialized_trial

SIGMAS = (10.0, 30.0, 100.0)


def source_from_row(row):
    return PointLikeSource(
        name=str(row["ZTFID"]),
        ra=np.deg2rad(float(row["ra_deg"])),
        dec=np.deg2rad(float(row["dec_deg"])),
    )


def create_fixed_gamma_analysis(cfg, datasets, row, gamma_fixed):
    """
    Build the public time-dependent analysis with gamma truly fixed.

    SkyLLH's create_analysis constructs Parameter objects through the module
    symbol tdps.Parameter. We temporarily substitute a factory that makes
    gamma fixed while leaving n_s unchanged.
    """
    original_parameter = tdps.Parameter

    def parameter_factory(
        name,
        initial,
        valmin=None,
        valmax=None,
        isfixed=None,
    ):
        if name == "gamma":
            return SkyParameter(
                name="gamma",
                initial=float(gamma_fixed),
                isfixed=True,
            )
        return SkyParameter(
            name=name,
            initial=initial,
            valmin=valmin,
            valmax=valmax,
            isfixed=isfixed,
        )

    tdps.Parameter = parameter_factory
    try:
        ana = tdps.create_analysis(
            cfg=cfg,
            datasets=datasets,
            source=source_from_row(row),
            gauss={
                "mu": float(row["peak_mjd"]),
                "sigma": 30.0,
            },
            refplflux_gamma=float(gamma_fixed),
            gamma_seed=float(gamma_fixed),
            gamma_min=1.0,
            gamma_max=4.0,
            evt_sel_delta_angle_deg=10.0,
        )
    finally:
        tdps.Parameter = original_parameter

    return ana


def get_energy_ratio(ana):
    llh = cast(MultiDatasetTCLLHRatio, ana.llhratio)
    one = cast(
        ZeroSigH0SingleDatasetTCLLHRatio,
        llh.llhratio_list[0],
    )
    return one.pdfratio.pdfratio1.pdfratio2


class FixedGammaEnergyResponseCache:
    def __init__(self, ana, cfg, gamma_fixed):
        self.cfg = cfg
        self.gamma_fixed = float(gamma_fixed)
        self.ds = ana.dataset_list[0]

        self.sm = PDSmearingMatrix(
            pathfilenames=self.ds.get_abs_pathfilename_list(
                self.ds.get_aux_data_definition(
                    "smearing_datafile"
                )
            )
        )
        self.aeff = PDAeff(
            pathfilenames=self.ds.get_abs_pathfilename_list(
                self.ds.get_aux_data_definition(
                    "eff_area_datafile"
                )
            )
        )

        self.fluxmodel = ana.shg_mgr.shg_list[0].fluxmodel

        param_gamma = SkyParameter(
            name="gamma",
            initial=self.gamma_fixed,
            isfixed=True,
        )
        self.gamma_grid = param_gamma.as_linear_grid(delta=0.1)

        self.energy_ratio = get_energy_ratio(ana)
        self.cache = {}

    def key_for_dec(self, dec):
        sm_idx = int(self.sm.get_true_dec_idx(float(dec)))
        aeff_idx = int(
            np.digitize(
                float(dec),
                self.aeff.decnu_binedges,
            ) - 1
        )
        return (sm_idx, aeff_idx)

    def register_current(self, dec):
        key = self.key_for_dec(dec)
        self.cache[key] = self.energy_ratio.sig_pdf_set
        return key

    def _invalidate(self):
        er = self.energy_ratio
        er._interpolmethod = er.interpolmethod_cls(
            func=er._get_ratio_values,
            param_grid_set=er.sig_pdf_set.param_grid_set,
        )
        er._cache_tdm_trial_data_state_id = None
        er._cache_fitparams_hash = None
        er._cache_ratio = None
        er._cache_grads = None

    def activate_for_dec(self, dec):
        key = self.key_for_dec(dec)
        built = False

        if key not in self.cache:
            self.cache[key] = PDSignalEnergyPDFSet(
                cfg=self.cfg,
                ds=self.ds,
                src_dec=float(dec),
                fluxmodel=self.fluxmodel,
                param_grid_set=self.gamma_grid,
                sm=self.sm,
            )
            built = True

        self.energy_ratio.sig_pdf_set = self.cache[key]
        self._invalidate()

        return key, built


def switch_source_exact(ana, row, cache):
    src = source_from_row(row)
    ana.change_source(
        src,
        update_detsigyield_service=True,
    )
    return cache.activate_for_dec(src.dec)


def assert_gamma_not_floating(each):
    for sigma in SIGMAS:
        gf = each[sigma]["gamma_fit"]
        if not np.isnan(gf):
            raise RuntimeError(
                "Gamma unexpectedly appears among floating parameters: "
                f"sigma={sigma}, gamma_fit={gf}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--icetracks-dir", required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--ntrials", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--progress-every", type=int, default=20)
    args = ap.parse_args()

    if args.gamma not in (2.0, 2.5):
        raise ValueError("Production gamma must be 2.0 or 2.5.")

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
    ana = create_fixed_gamma_analysis(
        cfg=cfg,
        datasets=datasets,
        row=cat.iloc[0],
        gamma_fixed=args.gamma,
    )
    print(
        f"build_once={time.time()-t0:.2f}s "
        f"gamma={args.gamma:g} "
        f"block={args.start}:{stop} "
        f"Nsrc={len(cat)} Ntrials={args.ntrials} "
        f"seed_base={args.seed_base}",
        flush=True,
    )

    cache = FixedGammaEnergyResponseCache(
        ana=ana,
        cfg=cfg,
        gamma_fixed=args.gamma,
    )
    cache.register_current(
        np.deg2rad(float(cat.iloc[0]["dec_deg"]))
    )

    real_events = [data.exp for data in ana.data_list]
    real_n_events = [len(x) for x in real_events]

    q_obs = np.empty(len(cat), dtype=float)
    best_sigma_obs = np.empty(len(cat), dtype=float)

    # Observed pass; also fills detector-response cache.
    for j, (_, row) in enumerate(cat.iterrows()):
        switch_source_exact(ana, row, cache)
        ana.initialize_trial(real_events, real_n_events)

        fit_rss = RandomStateService(
            seed=31000000 + int(row["index"])
        )
        best, each = score_initialized_trial(
            ana=ana,
            peak_mjd=float(row["peak_mjd"]),
            rss=fit_rss,
            sigmas=SIGMAS,
        )
        assert_gamma_not_floating(each)

        q_obs[j] = float(best["ts"])
        best_sigma_obs[j] = float(best["sigma_d"])

    print(
        f"observed_done gamma={args.gamma:g} "
        f"response_bins={len(cache.cache)} "
        f"elapsed={time.time()-t_all:.1f}s",
        flush=True,
    )

    seeds = args.seed_base + np.arange(
        args.ntrials, dtype=np.int64
    )
    q_bg = np.empty(
        (args.ntrials, len(cat)),
        dtype=float,
    )

    t_bg = time.time()

    for b, seed in enumerate(seeds):
        tb = time.time()

        # One shared detector-level realization for all sources.
        gen_rss = RandomStateService(seed=int(seed))
        n_sig, n_events_list, events_list = ana.generate_pseudo_data(
            rss=gen_rss,
            mean_n_sig=0.0,
        )
        if int(n_sig) != 0:
            raise RuntimeError("Unexpected signal in background trial.")

        for j, (_, row) in enumerate(cat.iterrows()):
            _, built = switch_source_exact(
                ana, row, cache
            )
            if built:
                raise RuntimeError(
                    "New detector-response bin appeared after "
                    "observed pre-cache."
                )

            ana.initialize_trial(
                events_list,
                n_events_list,
            )

            fit_rss = RandomStateService(
                seed=int(
                    seed
                    + 1000003
                    * (int(row["index"]) + 1)
                )
            )
            best, each = score_initialized_trial(
                ana=ana,
                peak_mjd=float(row["peak_mjd"]),
                rss=fit_rss,
                sigmas=SIGMAS,
            )
            assert_gamma_not_floating(each)

            q_bg[b, j] = float(best["ts"])

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
                f"gamma={args.gamma:g} "
                f"last={time.time()-tb:.2f}s "
                f"mean={mean:.2f}s/trial "
                f"ETA={eta/60:.1f}min",
                flush=True,
            )

    np.savez_compressed(
        out,
        source_index=cat["index"].to_numpy(np.int64),
        ZTFID=cat["ZTFID"].astype(str).to_numpy(),
        trial_seed=seeds,
        gamma_fixed=np.asarray([args.gamma]),
        q_obs=q_obs,
        best_sigma_obs=best_sigma_obs,
        q_bg=q_bg,
    )

    print(
        f"Wrote {out}\n"
        f"TOTAL={time.time()-t_all:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
