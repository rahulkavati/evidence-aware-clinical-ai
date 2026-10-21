
"""Build leakage-safe ML datasets from PhysioNet 2019 patient splits."""

from pathlib import Path
import argparse
import json

import pandas as pd

from src.data.build_temporal_features import build_temporal_features


SPLIT_DIR = Path("outputs/splits")
OUTPUT_DIR = Path("data/processed")

SPLITS = ("train", "val", "test")


def build_split(split_name: str) -> dict:
    split_file = SPLIT_DIR / f"{split_name}.csv"

    if not split_file.exists():
        raise FileNotFoundError(split_file)

    patients = pd.read_csv(split_file)

    required = {"patient_id", "file_path", "hospital"}
    if not required.issubset(patients.columns):
        raise ValueError("Split CSV missing required columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"{split_name}.parquet"

    chunks = []
    total_rows = 0

    for patient in patients.itertuples(index=False):
        path = Path(patient.file_path)

        if not path.exists():
            raise FileNotFoundError(path)

        frame = pd.read_csv(path, sep="|")
        features = build_temporal_features(frame)

        features.insert(0, "patient_id", patient.patient_id)
        features.insert(1, "hospital", patient.hospital)

        chunks.append(features)
        total_rows += len(features)

    dataset = pd.concat(chunks, ignore_index=True)
    dataset.to_parquet(output_file, index=False)

    return {
        "split": split_name,
        "patients": len(patients),
        "hourly_rows": total_rows,
        "positive_hourly_labels": int(
            (dataset["SepsisLabel"] == 1).sum()
        ),
        "feature_columns": len(dataset.columns) - 3,
        "output": str(output_file),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=(*SPLITS, "all"),
        default="all",
    )
    args = parser.parse_args()

    selected = SPLITS if args.split == "all" else (args.split,)

    results = []

    for split_name in selected:
        print(f"Building {split_name} dataset...")
        result = build_split(split_name)
        results.append(result)
        print(json.dumps(result, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_DIR / "build_summary.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
