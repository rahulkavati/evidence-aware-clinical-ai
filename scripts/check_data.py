"""Check that local data matches the split manifests behind every result.

Usage:  python scripts/check_data.py
Exit code 0 means the processed parquet files (which all experiments use) are
consistent with outputs/splits. Missing raw PSV files are reported as a
warning because they are only needed to rebuild the parquet files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "src/data/raw/physionet2019"
SPLITS = {"train": "train", "val": "val", "test": "test"}


def raw_ids() -> set[str]:
    ids = set()
    for folder, hospital in (("training_setA", "A"), ("training_setB", "B")):
        path = RAW / folder
        if path.exists():
            ids |= {f"{hospital}_{file.stem}" for file in path.glob("*.psv")}
    return ids


def main() -> int:
    ok = True
    available_raw = raw_ids()
    print(f"Raw PSV files present: {len(available_raw):,} of 40,336")
    seen: set[str] = set()
    for name in SPLITS:
        manifest = set(pd.read_csv(ROOT / f"outputs/splits/{name}.csv")["patient_id"].astype(str))
        processed = set(
            pd.read_parquet(ROOT / f"data/processed/{name}.parquet", columns=["patient_id"])["patient_id"].astype(str)
        )
        overlap = seen & manifest
        seen |= manifest
        status = "OK" if processed == manifest and not overlap else "MISMATCH"
        ok &= status == "OK"
        print(
            f"{name:>5}: manifest {len(manifest):,} patients | parquet {len(processed):,} | "
            f"raw files {len(manifest & available_raw):,} | {status}"
            + (f" | {len(overlap)} patients overlap another split" if overlap else "")
        )
    if len(available_raw) < 40_336:
        print(
            "WARNING: raw files are incomplete. Experiments still run from data/processed, "
            "but rebuilding parquet files needs the full dataset (scripts/download_physionet2019.sh)."
        )
    print("Processed data consistent with splits." if ok else "Processed data does NOT match splits.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
