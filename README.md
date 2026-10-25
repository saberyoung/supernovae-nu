# supernovae-nu

Reproduction code for **Searching for High-Energy Neutrinos in Stripped-Envelope Supernova Light-Curve Space**.

The repository contains the frozen optical inputs, compact numerical products from the IceCube Monte Carlo analysis, the analysis code, and scripts used to make the paper figures. The public IceCube IceTracks-DR2 event and detector-response files are not redistributed.

## Quick start

```bash
conda env create -f environment.yml
conda activate bts-icetracks
python reproduce.py
```

This checks the headline values, reruns the optical-only diagnostics, and regenerates Figures 1-4 and the appendix sensitivity figure in `figures/output/`.

Expected population-test probabilities are:

- localized optical-space search: `p = 0.539346`
- continuous PC1-PC2 gradient: `p = 0.852315`
- maximum predefined subtype: `p = 0.325767`

## What the files do

- `data/` - frozen BTS optical samples and pre-unblinding optical benchmarks.
- `results/` - compact numerical products used by the paper figures and tables.
- `analysis/optical_diagnostics.py` - subtype overlap, nearest-neighbor purity, and selection-coordinate correlations.
- `analysis/population_tests.py` - localized, gradient, and subtype population statistics.
- `analysis/injection_power.py` - joint-background score-space injection/recovery calculation.
- `analysis/compare_sensitivity.py` - idealized versus joint-background sensitivity comparison.
- `figures/` - scripts for Figures 1-4 and the appendix sensitivity figure.
- `compute/` - optional SkyLLH/IceTracks scripts for regenerating detector-level Monte Carlo from the official IceTracks-DR2 release.
- `tools/validate.py` - checks sample sizes and headline paper values.

## Detector-level rerun (optional)

Download IceTracks-DR2 from <https://doi.org/10.7910/DVN/MMIIZA>. The scripts in `compute/` implement the shared-background source calculation used in the paper. Start by checking the data layout and the public TXS 0506+056 benchmark:

```bash
python compute/check_data.py /path/to/dataverse_files
python compute/validate_txs.py --icetracks-dir /path/to/dataverse_files
python compute/validate_joint_background.py --icetracks-dir /path/to/dataverse_files
```

The exact Monte Carlo trial counts and seed ranges are recorded in `config/mc_schedule.json`. To rebuild the primary detector-level ensemble and derived power products from the official release:

```bash
python compute/generate_mc.py --icetracks-dir /path/to/dataverse_files --batch 59
python compute/merge_mc.py
python analysis/population_tests.py
python analysis/injection_power.py
python analysis/compare_sensitivity.py --empirical-summary work/power/empirical_effect_size_summary.csv
```

The detector-level rebuild is expensive; it is not required for the quick reproduction above.

## License

Analysis software is released under the BSD 3-Clause License. External IceCube and ZTF/BTS data remain subject to their original release and citation terms.
