
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("src/data/raw/physionet2019")
OUTPUT_DIR = Path("outputs/splits")
RANDOM_STATE = 42
EXPECTED_PATIENTS = 40_336  # PhysioNet 2019 training sets A (20,336) + B (20,000)


def collect_patients():
    records = []

    for hospital_dir, hospital in [
        ("training_setA", "A"),
        ("training_setB", "B"),
    ]:
        folder = DATA_DIR / hospital_dir

        if not folder.exists():
            raise FileNotFoundError(folder)

        for path in sorted(folder.glob("*.psv")):
            df = pd.read_csv(
                path,
                sep="|",
                usecols=["SepsisLabel"],
            )

            records.append({
                "patient_id": f"{hospital}_{path.stem}",
                "hospital": hospital,
                "sepsis_positive": int(
                    (df["SepsisLabel"] == 1).any()
                ),
                "file_path": str(path),
            })

    return pd.DataFrame(records)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Create patient-level splits.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing split manifests (they define every reported result).",
    )
    args = parser.parse_args(argv)
    if (OUTPUT_DIR / "train.csv").exists() and not args.overwrite:
        raise SystemExit(
            "outputs/splits already exists and defines all reported results; "
            "pass --overwrite only if you intend to recreate it."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patients = collect_patients()
    if len(patients) != EXPECTED_PATIENTS:
        raise SystemExit(
            f"Found {len(patients)} raw patient files, expected {EXPECTED_PATIENTS}. "
            "Download the complete dataset (scripts/download_physionet2019.sh) first."
        )

    patients["stratum"] = (
        patients["hospital"]
        + "_"
        + patients["sepsis_positive"].astype(str)
    )

    train, remaining = train_test_split(
        patients,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=patients["stratum"],
    )

    val, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remaining["stratum"],
    )

    for name, split in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        split.drop(columns=["stratum"]).to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
        )

    summary = {}

    for name, split in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        summary[name] = {
            "patients": len(split),
            "sepsis_positive": int(
                split["sepsis_positive"].sum()
            ),
            "prevalence": float(
                split["sepsis_positive"].mean()
            ),
            "hospital_counts": (
                split["hospital"].value_counts().to_dict()
            ),
        }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
