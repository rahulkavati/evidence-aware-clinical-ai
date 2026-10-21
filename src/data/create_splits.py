
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("src/data/raw/physionet2019")
OUTPUT_DIR = Path("outputs/splits")
RANDOM_STATE = 42


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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patients = collect_patients()

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
