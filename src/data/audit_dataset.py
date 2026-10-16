
from pathlib import Path
import json

import numpy as np
import pandas as pd

DATA_DIR = Path("src/data/raw/physionet2019/")
OUTPUT_DIR = Path("outputs/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_COLUMNS = 41  # 40 features + SepsisLabel
HOSPITAL_FOLDERS = ["training_setA", "training_setB"]


def audit_dataset():
    patient_records = []
    missing_counts = None
    total_rows = 0

    for hospital in HOSPITAL_FOLDERS:
        folder = DATA_DIR / hospital

        if not folder.exists():
            raise FileNotFoundError(f"Missing dataset folder: {folder}")

        files = sorted(folder.glob("*.psv"))
        print(f"{hospital}: {len(files)} patient files")

        for file in files:
            df = pd.read_csv(file, sep="|")

            if df.shape[1] != EXPECTED_COLUMNS:
                raise ValueError(
                    f"{file.name}: expected {EXPECTED_COLUMNS} columns, "
                    f"found {df.shape[1]}"
                )

            if "SepsisLabel" not in df.columns:
                raise ValueError(f"Missing SepsisLabel: {file.name}")

            if missing_counts is None:
                missing_counts = pd.Series(
                    0, index=df.columns, dtype="int64"
                )

            missing_counts += df.isna().sum()
            total_rows += len(df)

            labels = df["SepsisLabel"]

            patient_records.append({
                "hospital": hospital,
                "patient_id": file.stem,
                "hours": len(df),
                "sepsis_positive": int((labels == 1).any()),
                "positive_hours": int((labels == 1).sum()),
            })

    patients = pd.DataFrame(patient_records)

    summary = {
        "total_patients": int(len(patients)),
        "total_hourly_observations": int(total_rows),
        "sepsis_positive_patients": int(
            patients["sepsis_positive"].sum()
        ),
        "sepsis_negative_patients": int(
            (patients["sepsis_positive"] == 0).sum()
        ),
        "patient_sepsis_prevalence": float(
            patients["sepsis_positive"].mean()
        ),
        "median_hours_per_patient": float(
            patients["hours"].median()
        ),
        "hospital_counts": {
            key: int(value)
            for key, value in patients["hospital"].value_counts().items()
        },
    }

    missingness = (
        missing_counts / total_rows * 100
    ).sort_values(ascending=False)

    patients.to_csv(
        OUTPUT_DIR / "patient_summary.csv", index=False
    )

    missingness.rename("missing_percentage").to_csv(
        OUTPUT_DIR / "feature_missingness.csv"
    )

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDataset Audit Summary")
    print(json.dumps(summary, indent=2))

    print("\nTop 10 most missing variables:")
    print(missingness.head(10).round(2))

    return summary


if __name__ == "__main__":
    audit_dataset()
