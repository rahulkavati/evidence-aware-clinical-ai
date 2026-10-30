"""Create immutable, patient-disjoint G1 cohorts from the existing train manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED = {"patient_id", "hospital", "sepsis_positive"}


def fingerprint(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def validate_manifest(frame, name):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    if frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        raise ValueError(f"{name} has null or duplicate patient IDs")
    if not frame.sepsis_positive.isin([0, 1]).all():
        raise ValueError(f"{name} has nonbinary sepsis_positive")
    if frame.hospital.isna().any():
        raise ValueError(f"{name} has missing hospital")


def make_cohorts(train, val, test, seed=42):
    for name, frame in (("train", train), ("val", val), ("test", test)):
        validate_manifest(frame, name)
    sets = [set(f.patient_id.astype(str)) for f in (train, val, test)]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Original train/val/test manifests overlap")
    strata = train.hospital.astype(str) + "_" + train.sepsis_positive.astype(str)
    fit_idx, other_idx = train_test_split(
        train.index.to_numpy(), test_size=0.30, random_state=seed, stratify=strata
    )
    other = train.loc[other_idx]
    other_strata = other.hospital.astype(str) + "_" + other.sepsis_positive.astype(str)
    cal_idx, policy_idx = train_test_split(
        other.index.to_numpy(), test_size=0.50, random_state=seed + 1,
        stratify=other_strata
    )
    cohorts = {
        "predictor_fit": train.loc[fit_idx].sort_values("patient_id").reset_index(drop=True),
        "calibration": train.loc[cal_idx].sort_values("patient_id").reset_index(drop=True),
        "policy_fit": train.loc[policy_idx].sort_values("patient_id").reset_index(drop=True),
    }
    groups = [set(f.patient_id.astype(str)) for f in cohorts.values()]
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("G1 cohort overlap")
    if set.union(*groups) != sets[0]:
        raise AssertionError("G1 cohorts do not cover original training cohort")
    if any(group & (sets[1] | sets[2]) for group in groups):
        raise AssertionError("G1 cohort contains validation or test patient")
    return cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=Path("outputs/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/g1_cohorts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifests = {name: pd.read_csv(args.splits_dir / f"{name}.csv") for name in ("train", "val", "test")}
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output_dir}")
    cohorts = make_cohorts(**manifests, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in cohorts.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    summary = {
        "seed": args.seed,
        "source_train_sha256_patient_ids": fingerprint(manifests["train"].patient_id),
        "source_val_sha256_patient_ids": fingerprint(manifests["val"].patient_id),
        "source_test_sha256_patient_ids": fingerprint(manifests["test"].patient_id),
        "cohorts": {name: {
            "patients": len(frame),
            "sepsis_positive": int(frame.sepsis_positive.sum()),
            "hospital_counts": {str(k): int(v) for k, v in frame.hospital.value_counts().items()},
            "sha256_patient_ids": fingerprint(frame.patient_id),
        } for name, frame in cohorts.items()},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
