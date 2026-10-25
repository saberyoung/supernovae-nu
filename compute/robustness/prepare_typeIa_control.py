#!/usr/bin/env python3
"""
Prepare the pre-frozen matched Type Ia negative-control catalog.

The 241 Type Ia controls were matched BEFORE formal IceTracks unblinding on:
  - declination
  - peak epoch
  - log10(redshift)

They are NOT re-matched here.

For the phenotype geometry, each Ia is projected through the SAME frozen
SESN optical representation:
  [M_peak, log10(t_half,rise), log10(t_half,fade)]
    -> frozen SESN scaler
    -> frozen SESN PCA basis

No PCA refit is performed on the Ia sample.

This makes the negative control ask whether the same analysis machinery and
IceCube exposure structure can generate an apparent phenotype hotspot in an
unrelated transient population.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="data/typeIa_control.csv",
    )
    ap.add_argument(
        "--pca-model",
        default="config/pca_model.json",
    )
    ap.add_argument(
        "--output",
        default="data/typeIa_control.csv",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    model = json.loads(Path(args.pca_model).read_text())

    required = [
        "ZTFID", "IAUID", "type", "ra_deg", "dec_deg", "peak_mjd",
        "peakabs_num", "rise_num", "fade_num", "redshift_num",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if len(df) != 241:
        raise ValueError(f"Expected 241 matched Ia controls; got {len(df)}")

    # Recompute logs for consistency even if already present in the file.
    if np.any(df["rise_num"].to_numpy(float) <= 0):
        raise ValueError("Non-positive rise_num in control sample")
    if np.any(df["fade_num"].to_numpy(float) <= 0):
        raise ValueError("Non-positive fade_num in control sample")

    df["log10_rise"] = np.log10(df["rise_num"].to_numpy(float))
    df["log10_fade"] = np.log10(df["fade_num"].to_numpy(float))

    features = model["features"]
    if features != ["peakabs_num", "log10_rise", "log10_fade"]:
        raise ValueError(f"Unexpected frozen feature order: {features}")

    X = df[features].to_numpy(float)
    mu = np.asarray(model["standard_scaler_mean"], float)
    scale = np.asarray(model["standard_scaler_scale"], float)
    comps = np.asarray(model["pca_components"], float)

    Xstd = (X - mu) / scale
    pcs = Xstd @ comps.T

    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]
    df["PC3"] = pcs[:, 2]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {out}")
    print(f"N={len(df)}")
    print("Projection uses frozen SESN scaler/PCA; no Ia refit.")
    print("\nPC ranges:")
    for c in ["PC1", "PC2", "PC3"]:
        print(
            f" {c}: min={df[c].min():.3f}, "
            f"median={df[c].median():.3f}, max={df[c].max():.3f}"
        )


if __name__ == "__main__":
    main()
