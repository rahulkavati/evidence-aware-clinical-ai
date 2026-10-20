"""Audit historical measurement availability for PhysioNet 2019."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("src/data/raw/physionet2019")
DEFAULT_OUTPUT_DIR = Path("outputs/acquisition_feasibility")
DEFAULT_REPORT = Path("docs/acquisition_feasibility_audit.md")
DATASET_VERSION = "1.0.0"
CHECKPOINTS = (6, 12, 24)
HOSPITALS = {"training_setA": "Hospital A", "training_setB": "Hospital B"}

VITAL_FEATURES = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
)
LAB_FEATURES = (
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
)
OTHER_FEATURES = ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime")
REQUIRED_COLUMNS = (*VITAL_FEATURES, *LAB_FEATURES, *OTHER_FEATURES, "ICULOS", "SepsisLabel")


def checkpoint_patient_metrics(frame: pd.DataFrame, checkpoint: int) -> dict[str, Any]:
    """Summarize one patient's available features using rows through checkpoint.

    Eligibility requires an observation at the checkpoint itself. The frame is
    truncated before eligibility and feature availability are evaluated, so a
    later row cannot make a patient eligible or affect checkpoint features.
    """
    available = frame.loc[frame["ICULOS"] <= checkpoint]
    eligible = bool((available["ICULOS"] == checkpoint).any())
    if not eligible:
        return {"eligible": False}

    lab_values = available.loc[:, LAB_FEATURES].apply(pd.to_numeric, errors="coerce")
    vital_values = available.loc[:, VITAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )
    lab_valid = lab_values.notna() & np.isfinite(lab_values)
    vital_valid = vital_values.notna() & np.isfinite(vital_values)
    observed_labs = lab_valid.any(axis=0)
    observed_vitals = vital_valid.any(axis=0)

    return {
        "eligible": True,
        "observed_lab_features": int(observed_labs.sum()),
        "observed_vital_features": int(observed_vitals.sum()),
        "lab_feature_flags": observed_labs.to_dict(),
        "lab_maskable_cells_by_feature": lab_valid.sum(axis=0).astype(int).to_dict(),
        "lab_maskable_cells": int(lab_valid.to_numpy().sum()),
    }


def read_patient_file(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="|")
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")

    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["ICULOS"].isna().any():
        raise ValueError(f"{path}: ICULOS contains missing or non-numeric values")
    if ((frame["ICULOS"] < 1) | (frame["ICULOS"] % 1 != 0)).any():
        raise ValueError(f"{path}: ICULOS must contain positive whole hours")
    if frame["ICULOS"].duplicated().any():
        raise ValueError(f"{path}: ICULOS contains duplicate hourly observations")
    if frame["SepsisLabel"].isna().any() or not frame["SepsisLabel"].isin([0, 1]).all():
        raise ValueError(f"{path}: SepsisLabel must contain only 0 or 1")
    return frame


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _cohort_stats(
    patients: list[dict[str, Any]], checkpoint: int, full_positive_count: int
) -> dict[str, Any]:
    eligible = [
        patient
        for patient in patients
        if patient["checkpoints"][checkpoint]["eligible"]
    ]
    lab_distribution = {str(count): 0 for count in range(len(LAB_FEATURES) + 1)}
    vital_distribution = {str(count): 0 for count in range(len(VITAL_FEATURES) + 1)}
    for patient in eligible:
        metrics = patient["checkpoints"][checkpoint]
        lab_distribution[str(metrics["observed_lab_features"])] += 1
        vital_distribution[str(metrics["observed_vital_features"])] += 1

    one_plus = sum(
        patient["checkpoints"][checkpoint]["observed_lab_features"] >= 1
        for patient in eligible
    )
    three_plus = sum(
        patient["checkpoints"][checkpoint]["observed_lab_features"] >= 3
        for patient in eligible
    )
    positive_eligible = sum(patient["sepsis_positive"] for patient in eligible)
    return {
        "checkpoint_iculos_hour": checkpoint,
        "patients_with_observations_through_checkpoint": len(eligible),
        "patients_with_1plus_observed_laboratory_features": one_plus,
        "percentage_with_1plus_observed_laboratory_features": _percent(
            one_plus, len(eligible)
        ),
        "patients_with_3plus_observed_laboratory_features": three_plus,
        "percentage_with_3plus_observed_laboratory_features": _percent(
            three_plus, len(eligible)
        ),
        "distinct_laboratory_feature_distribution": lab_distribution,
        "distinct_vital_sign_feature_distribution": vital_distribution,
        "sepsis_positive_patients_among_eligible": positive_eligible,
        "full_record_sepsis_positive_patient_count": full_positive_count,
        "positive_case_retention_percentage": _percent(
            positive_eligible, full_positive_count
        ),
        "maskable_laboratory_observation_cells": sum(
            patient["checkpoints"][checkpoint]["lab_maskable_cells"]
            for patient in eligible
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty aggregate: {path}")
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _build_outputs(patient_records: list[dict[str, Any]]) -> dict[str, Any]:
    total_positive = sum(patient["sepsis_positive"] for patient in patient_records)
    summary: dict[str, Any] = {
        "dataset": "PhysioNet/Computing in Cardiology Challenge 2019",
        "dataset_version": DATASET_VERSION,
        "patient_count": len(patient_records),
        "full_record_sepsis_positive_patient_count": total_positive,
        "hospital_patient_counts": {
            label: sum(patient["hospital"] == label for patient in patient_records)
            for label in HOSPITALS.values()
        },
        "time_index_convention": (
            "Each row is an hourly observation indexed by ICULOS, with the first "
            "hour numbered 1. A checkpoint t includes rows with ICULOS <= t, "
            "inclusive. Eligibility requires a row at ICULOS == t, determined "
            "from the truncated rows only."
        ),
        "feature_groups": {
            "vital_sign_features": list(VITAL_FEATURES),
            "laboratory_features": list(LAB_FEATURES),
        },
        "measurement_definitions": {
            "historically_observed": "A finite, non-null feature value in a patient row at or before the checkpoint.",
            "deliberately_maskable": "A historically observed laboratory cell that can be hidden in a retrospective simulation; this does not imply that an unobserved test was obtainable.",
            "unobserved_unknown": "A laboratory feature with no finite, non-null value by the checkpoint; its value is unknown, not a candidate measurement assumed to be available.",
            "sepsis_status": "Retrospective cohort characterization: any SepsisLabel=1 anywhere in the full patient record. It is not used for checkpoint eligibility or checkpoint feature state.",
        },
        "checkpoints": {},
    }
    availability_rows: list[dict[str, Any]] = []
    laboratory_rows: list[dict[str, Any]] = []
    hospital_rows: list[dict[str, Any]] = []

    scopes = [("All", patient_records)] + [
        (label, [p for p in patient_records if p["hospital"] == label])
        for label in HOSPITALS.values()
    ]
    for checkpoint in CHECKPOINTS:
        overall = _cohort_stats(patient_records, checkpoint, total_positive)
        summary["checkpoints"][str(checkpoint)] = {
            "overall": overall,
            "by_hospital": {},
        }
        for hospital in HOSPITALS.values():
            hospital_patients = [p for p in patient_records if p["hospital"] == hospital]
            hospital_positive = sum(p["sepsis_positive"] for p in hospital_patients)
            stats = _cohort_stats(hospital_patients, checkpoint, hospital_positive)
            summary["checkpoints"][str(checkpoint)]["by_hospital"][hospital] = stats
            hospital_row: dict[str, Any] = {"hospital": hospital, **stats}
            for count, value in stats["distinct_laboratory_feature_distribution"].items():
                hospital_row[f"patients_with_{count}_distinct_laboratory_features"] = value
            for count, value in stats["distinct_vital_sign_feature_distribution"].items():
                hospital_row[f"patients_with_{count}_distinct_vital_sign_features"] = value
            hospital_row.pop("distinct_laboratory_feature_distribution")
            hospital_row.pop("distinct_vital_sign_feature_distribution")
            hospital_rows.append(hospital_row)

        for scope, scoped_patients in scopes:
            stats = _cohort_stats(
                scoped_patients,
                checkpoint,
                sum(p["sepsis_positive"] for p in scoped_patients),
            )
            row: dict[str, Any] = {"hospital": scope, **stats}
            for count, value in stats["distinct_laboratory_feature_distribution"].items():
                row[f"patients_with_{count}_distinct_laboratory_features"] = value
            for count, value in stats["distinct_vital_sign_feature_distribution"].items():
                row[f"patients_with_{count}_distinct_vital_sign_features"] = value
            row.pop("distinct_laboratory_feature_distribution")
            row.pop("distinct_vital_sign_feature_distribution")
            availability_rows.append(row)

            eligible = [
                p
                for p in scoped_patients
                if p["checkpoints"][checkpoint]["eligible"]
            ]
            for feature in LAB_FEATURES:
                available_patients = sum(
                    p["checkpoints"][checkpoint]["lab_feature_flags"][feature]
                    for p in eligible
                )
                laboratory_rows.append(
                    {
                        "checkpoint_iculos_hour": checkpoint,
                        "hospital": scope,
                        "laboratory_feature": feature,
                        "eligible_patients": len(eligible),
                        "patients_with_historical_observation": available_patients,
                        "availability_percentage": _percent(
                            available_patients, len(eligible)
                        ),
                        "patients_with_unobserved_unknown_value": len(eligible)
                        - available_patients,
                        "maskable_non_null_observation_cells": sum(
                            p["checkpoints"][checkpoint]["lab_maskable_cells_by_feature"][feature]
                            for p in eligible
                        ),
                    }
                )

    return {
        "summary": summary,
        "availability_rows": availability_rows,
        "laboratory_rows": laboratory_rows,
        "hospital_rows": hospital_rows,
    }


def _markdown_report(summary: dict[str, Any], laboratory_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Measurement-Acquisition Feasibility Audit",
        "",
        f"**Dataset:** {summary['dataset']}, version {summary['dataset_version']}<br>",
        "**Source:** Official PhysioNet Challenge 2019 training sets A and B<br>",
        f"**Patients audited:** {summary['patient_count']:,}<br>",
        f"**Full-record sepsis-positive patients:** {summary['full_record_sepsis_positive_patient_count']:,}",
        "",
        "## Time Convention and Definitions",
        "",
        summary["time_index_convention"],
        "",
        "The 8 vital-sign features are analyzed separately from the 26 laboratory/test features. At each checkpoint, feature availability is evaluated only using the patient's rows at or before that checkpoint. A patient without a row exactly at the checkpoint is not in that checkpoint's eligible cohort; a later row cannot confer eligibility.",
        "",
        "A historically observed measurement is a finite, non-null value in an eligible patient's history through the checkpoint. Each such laboratory cell is retrospectively maskable in a simulation. A feature with no such value is unobserved and has an unknown value; this audit does not assume it could have been ordered or obtained on demand. Full-record sepsis status is used only to describe retrospective positive-case retention, not to determine eligibility or acquisition state.",
        "",
        "## Checkpoint Results",
        "",
        "### Overall cohort",
        "",
        "| ICU hour | Eligible patients | 1+ labs, n (%) | 3+ labs, n (%) | Positive cases retained, n (%) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for hour in CHECKPOINTS:
        stats = summary["checkpoints"][str(hour)]["overall"]
        lines.append(
            f"| {hour} | {stats['patients_with_observations_through_checkpoint']:,} "
            f"| {stats['patients_with_1plus_observed_laboratory_features']:,} "
            f"({stats['percentage_with_1plus_observed_laboratory_features']:.2f}%) "
            f"| {stats['patients_with_3plus_observed_laboratory_features']:,} "
            f"({stats['percentage_with_3plus_observed_laboratory_features']:.2f}%) "
            f"| {stats['sepsis_positive_patients_among_eligible']:,}/"
            f"{stats['full_record_sepsis_positive_patient_count']:,} "
            f"({stats['positive_case_retention_percentage']:.2f}%) |"
        )

    lines.extend(
        [
            "",
            "### By hospital",
            "",
            "| ICU hour | Hospital | Eligible patients | 1+ labs, n (%) | 3+ labs, n (%) | Positive cases retained, n (%) |",
            "|---:|:---|---:|---:|---:|---:|",
        ]
    )
    for hour in CHECKPOINTS:
        for hospital in HOSPITALS.values():
            stats = summary["checkpoints"][str(hour)]["by_hospital"][hospital]
            lines.append(
                f"| {hour} | {hospital} | {stats['patients_with_observations_through_checkpoint']:,} "
                f"| {stats['patients_with_1plus_observed_laboratory_features']:,} ({stats['percentage_with_1plus_observed_laboratory_features']:.2f}%) "
                f"| {stats['patients_with_3plus_observed_laboratory_features']:,} ({stats['percentage_with_3plus_observed_laboratory_features']:.2f}%) "
                f"| {stats['sepsis_positive_patients_among_eligible']:,}/{stats['full_record_sepsis_positive_patient_count']:,} ({stats['positive_case_retention_percentage']:.2f}%) |"
            )

    lines.extend(
        [
            "",
            "### Laboratory feature availability",
            "",
            "Patient-level availability among eligible patients, pooled across hospitals:",
            "",
            "| Laboratory feature | ICU hour 6 | ICU hour 12 | ICU hour 24 |",
            "|:---|---:|---:|---:|",
        ]
    )
    pooled = {
        (row["checkpoint_iculos_hour"], row["laboratory_feature"]): row
        for row in laboratory_rows
        if row["hospital"] == "All"
    }
    for feature in LAB_FEATURES:
        cells = []
        for hour in CHECKPOINTS:
            row = pooled[(hour, feature)]
            cells.append(
                f"{row['patients_with_historical_observation']:,}/"
                f"{row['eligible_patients']:,} ({row['availability_percentage']:.2f}%)"
            )
        lines.append(f"| {feature} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Distribution of distinct observed laboratory features",
            "",
            "Eligible patient counts by the number of distinct laboratory/test features seen by each checkpoint:",
            "",
            "| Distinct laboratory features | ICU hour 6 | ICU hour 12 | ICU hour 24 |",
            "|---:|---:|---:|---:|",
        ]
    )
    for feature_count in range(len(LAB_FEATURES) + 1):
        counts = [
            summary["checkpoints"][str(hour)]["overall"][
                "distinct_laboratory_feature_distribution"
            ][str(feature_count)]
            for hour in CHECKPOINTS
        ]
        lines.append(
            f"| {feature_count} | "
            + " | ".join(f"{count:,}" for count in counts)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This is a retrospective audit of the PhysioNet 2019 Challenge training sets, not a prospective or deployment study.",
            "- The hourly files do not establish whether a non-null value represents a newly performed test, a carried-forward value, or a clinically orderable measurement at that time. Cell counts therefore describe maskable recorded entries, not verified laboratory orders or independent assays.",
            "- Missing values are unknown. The audit cannot estimate the value, cost, delay, or feasibility of acquiring an unrecorded test, and it does not treat unobserved tests as available acquisition candidates.",
            "- Recorded availability reflects historical clinical practice and selection, which may differ by patient, hospital, and severity; it is not a randomized measure of test utility.",
            "- Sepsis-positive status uses any positive label in a patient's full record. Its use here is limited to retrospective cohort characterization and positive-case retention; future labels must remain excluded from policy state.",
            "- This feasibility audit does not train a model or evaluate predictive benefit from acquiring measurements.",
            "",
            "## Interpretation and Recommendation",
            "",
            "The proportion with at least one or three historically observed laboratory features quantifies how often a retrospective masking experiment can hide an existing lab entry at each checkpoint. It does not quantify how often a new measurement could have been acquired. Availability should be treated as patient- and time-dependent, and conclusions should not be generalized to on-demand clinical testing.",
            "",
            "Next, run controlled retrospective masking experiments only on measurements actually present by the checkpoint. Compare no acquisition with policies that reveal eligible hidden historical entries under explicit budgets; keep never-observed tests unavailable or unknown. Use patient-level train/validation/test splits, prevent full-record labels from entering policy state, and report performance separately by hospital and checkpoint. Any simulated delay, cost, or acquisition success assumption should be clearly labeled and tested in sensitivity analyses.",
            "",
            "Aggregate machine-readable outputs are in `outputs/acquisition_feasibility/`. No patient-level output is produced.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    report_path = Path(report_path)
    patient_records: list[dict[str, Any]] = []

    for folder_name, hospital in HOSPITALS.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing hospital dataset folder: {folder}")
        files = sorted(folder.glob("*.psv"))
        if not files:
            raise FileNotFoundError(f"No .psv patient files found in: {folder}")
        print(f"{hospital}: reading {len(files):,} patient files")
        for path in files:
            frame = read_patient_file(path)
            patient_records.append(
                {
                    "hospital": hospital,
                    "sepsis_positive": bool((frame["SepsisLabel"] == 1).any()),
                    "checkpoints": {
                        checkpoint: checkpoint_patient_metrics(frame, checkpoint)
                        for checkpoint in CHECKPOINTS
                    },
                }
            )

    outputs = _build_outputs(patient_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(outputs["summary"], output, indent=2)
        output.write("\n")
    _write_csv(output_dir / "availability_by_hour.csv", outputs["availability_rows"])
    _write_csv(output_dir / "lab_availability.csv", outputs["laboratory_rows"])
    _write_csv(output_dir / "hospital_comparison.csv", outputs["hospital_rows"])
    report_path.write_text(
        _markdown_report(outputs["summary"], outputs["laboratory_rows"]),
        encoding="utf-8",
    )
    return outputs["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    arguments = parser.parse_args()
    summary = run_audit(arguments.data_dir, arguments.output_dir, arguments.report)
    print(json.dumps(summary["checkpoints"], indent=2))


if __name__ == "__main__":
    main()