from pathlib import Path
import skyllh

REQUIRED_RELATIVE = [
    "events/IC86_I_exp.csv",
    "events/IC86_II_exp.csv",
    "events/IC86_III_exp.csv",
    "events/IC86_IV_exp.csv",
    "events/IC86_V_exp.csv",
    "events/IC86_VI_exp.csv",
    "events/IC86_VII_exp.csv",
    "events/IC86_VIII_exp.csv",
    "events/IC86_IX_exp.csv",
    "events/IC86_X_exp.csv",
    "events/IC86_XI_exp.csv",
    "uptime/IC86_I_exp.csv",
    "uptime/IC86_II_exp.csv",
    "uptime/IC86_III_exp.csv",
    "uptime/IC86_IV_exp.csv",
    "uptime/IC86_V_exp.csv",
    "uptime/IC86_VI_exp.csv",
    "uptime/IC86_VII_exp.csv",
    "uptime/IC86_VIII_exp.csv",
    "uptime/IC86_IX_exp.csv",
    "uptime/IC86_X_exp.csv",
    "uptime/IC86_XI_exp.csv",
    "irfs/IC86_effectiveArea.csv",
    "irfs/IC86_smearing.csv",
]

def normalize_icetracks_dir(path):
    """Return the directory that directly contains events/, uptime/, irfs/."""
    p = Path(path).expanduser().resolve()
    candidates = [p]
    if p.is_dir():
        # Common extra wrapper directory from ZIP extraction.
        candidates += [x for x in p.iterdir() if x.is_dir()]
    for c in candidates:
        if (c/"events").is_dir() and (c/"uptime").is_dir() and (c/"irfs").is_dir():
            return c
    raise FileNotFoundError(
        f"Could not find events/, uptime/, irfs/ under {p}. "
        "Point --icetracks-dir to the extracted IceTracks-DR2 folder."
    )

def validate_layout(path):
    root = normalize_icetracks_dir(path)
    missing = [rel for rel in REQUIRED_RELATIVE if not (root/rel).exists()]
    return root, missing

def create_ic86_dataset(cfg, icetracks_dir):
    """Create local IceTracks-DR2 IC86_I-XI without downloading."""
    root, missing = validate_layout(icetracks_dir)
    if missing:
        msg = "\n".join("  - " + x for x in missing)
        raise FileNotFoundError(
            "Required local IceTracks files are missing:\n" + msg
        )
    # SkyLLH path = base_path / sub_path_fmt / relative filename
    return skyllh.create_datasets(
        "IceTracks-DR2",
        cfg=cfg,
        names=["IC86_I-XI"],
        base_path=str(root.parent),
        sub_path_fmt=root.name,
    )
