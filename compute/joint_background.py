#!/usr/bin/env python3
"""
Exact-cache full-SkyLLH joint background worker (compute engine ).

SCIENCE FREEZE
--------------
This does NOT change the frozen  statistic:
    q_i = max(TS_10d, TS_30d, TS_100d)
with the full SkyLLH likelihood, n_s fitted, gamma profiled on [1,4].

COMPUTE OPTIMIZATION
--------------------
One SkyLLH analysis is built per source block. Sources are switched with
Analysis.change_source(), and the declination-dependent public-data signal
energy PDF is cached by its exact detector-response bin
(smearing_dec_bin, aeff_dec_bin).

This implementation has been A/B validated against the original "one full
analysis per source" runner:
- observed q_i: exact equality
- best sigma: exact equality
- shared-background q_i: exact equality

Horizontal source blocks MUST use the same seed range.
Vertical trial chunks MUST use disjoint seed ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast
import time

import numpy as np
import pandas as pd

from skyllh.analyses.i3.publicdata_ps.aeff import PDAeff
from skyllh.analyses.i3.publicdata_ps.signalpdf import PDSignalEnergyPDFSet
from skyllh.analyses.i3.publicdata_ps.smearing_matrix import PDSmearingMatrix
from skyllh.analyses.i3.publicdata_ps.time_dependent_ps import create_analysis
from skyllh.core.config import Config
from skyllh.core.llhratio import (
    MultiDatasetTCLLHRatio,
    ZeroSigH0SingleDatasetTCLLHRatio,
)
from skyllh.core.parameters import Parameter
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


def build_once(cfg, datasets, row):
    return create_analysis(
        cfg=cfg,
        datasets=datasets,
        source=source_from_row(row),
        gauss={"mu": float(row["peak_mjd"]), "sigma": 30.0},
        gamma_seed=2.5,
        gamma_min=1.0,
        gamma_max=4.0,
        evt_sel_delta_angle_deg=10.0,
    )


def get_energy_ratio(ana):
    llh = cast(MultiDatasetTCLLHRatio, ana.llhratio)
    one = cast(
        ZeroSigH0SingleDatasetTCLLHRatio,
        llh.llhratio_list[0],
    )
    # public time-dependent construction:
    # ((spatial ratio * energy ratio) * time ratio)
    return one.pdfratio.pdfratio1.pdfratio2


class EnergyResponseCache:
    def __init__(self, ana, cfg):
        self.ana = ana
        self.cfg = cfg
        self.ds = ana.dataset_list[0]

        self.sm = PDSmearingMatrix(
            pathfilenames=self.ds.get_abs_pathfilename_list(
                self.ds.get_aux_data_definition("smearing_datafile")
            )
        )
        self.aeff = PDAeff(
            pathfilenames=self.ds.get_abs_pathfilename_list(
                self.ds.get_aux_data_definition("eff_area_datafile")
            )
        )
        self.fluxmodel = ana.shg_mgr.shg_list[0].fluxmodel

        param_gamma = Parameter(
            name="gamma",
            initial=2.5,
            valmin=1.0,
            valmax=4.0,
        )
        self.gamma_grid = param_gamma.as_linear_grid(delta=0.1)
        self.energy_ratio = get_energy_ratio(ana)
        self.cache = {}

    def key_for_dec(self, dec):
        sm_idx = int(self.sm.get_true_dec_idx(float(dec)))
        aeff_idx = int(
            np.digitize(float(dec), self.aeff.decnu_binedges) - 1
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


def switch_source_exact(ana, row, ecache):
    src = source_from_row(row)
    ana.change_source(src, update_detsigyield_service=True)
    return ecache.activate_for_dec(src.dec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog")
    ap.add_argument("--icetracks-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--ntrials", type=int, required=True)
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--progress-every", type=int, default=10)
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
    first_dec = np.deg2rad(float(cat.iloc[0]["dec_deg"]))
    ecache.register_current(first_dec)

    real_events = [data.exp for data in ana.data_list]
    real_n_events = [len(x) for x in real_events]

    # Observed source scores. This pass also pre-populates every detector
    # response bin required by this source block.
    q_obs = np.empty(len(cat), dtype=float)
    best_sigma_obs = np.empty(len(cat), dtype=float)

    for j, (_, row) in enumerate(cat.iterrows()):
        _, built = switch_source_exact(ana, row, ecache)
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
        q_obs[j] = best["ts"]
        best_sigma_obs[j] = best["sigma_d"]

    print(
        f"observed_done response_bins={len(ecache.cache)} "
        f"elapsed={time.time()-t_all:.1f}s",
        flush=True,
    )

    q_bg = np.empty((args.ntrials, len(cat)), dtype=float)
    trial_seeds = args.seed_base + np.arange(
        args.ntrials, dtype=np.int64
    )

    t_bg = time.time()

    for b, seed in enumerate(trial_seeds):
        tb = time.time()

        # One shared detector-level background realization. mean_n_sig=0,
        # so generation is source-independent.
        gen_rss = RandomStateService(seed=int(seed))
        n_sig, n_events_list, events_list = ana.generate_pseudo_data(
            rss=gen_rss,
            mean_n_sig=0.0,
        )
        if int(n_sig) != 0:
            raise RuntimeError(
                "Background trial unexpectedly contains signal."
            )

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
                    + 1000003
                    * (int(row["index"]) + 1)
                )
            )
            best, _ = score_initialized_trial(
                ana=ana,
                peak_mjd=float(row["peak_mjd"]),
                rss=fit_rss,
                sigmas=SIGMAS,
            )
            q_bg[b, j] = best["ts"]

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

    np.savez_compressed(
        out,
        source_index=cat["index"].to_numpy(np.int64),
        ZTFID=cat["ZTFID"].astype(str).to_numpy(),
        trial_seed=trial_seeds,
        q_obs=q_obs,
        best_sigma_obs=best_sigma_obs,
        q_bg=q_bg,
    )

    print(
        f"Wrote {out}\nTOTAL={time.time()-t_all:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
