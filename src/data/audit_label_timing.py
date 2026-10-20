"""Audit retrospective first-positive SepsisLabel timing by ICU checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.audit_acquisition_feasibility import (
	CHECKPOINTS,
	DATASET_VERSION,
	DEFAULT_DATA_DIR,
	HOSPITALS,
	read_patient_file,
)


DEFAULT_OUTPUT_DIR = Path("outputs/label_timing")
DEFAULT_REPORT = Path("docs/label_timing_audit.md")
DEFAULT_FEASIBILITY_SUMMARY = Path("outputs/acquisition_feasibility/summary.json")


def first_positive_label_hour(frame: pd.DataFrame) -> int | None:
	"""Return the earliest ICU hour labeled positive, or None if never positive."""
	positive_hours = frame.loc[frame["SepsisLabel"] == 1, "ICULOS"]
	if positive_hours.empty:
		return None
	return int(positive_hours.min())


def checkpoint_label_metrics(
	frame: pd.DataFrame, checkpoint: int, first_positive_hour: int | None = None
) -> dict[str, Any]:
	"""Classify checkpoint eligibility and retrospective label timing.

	Full-record labels are used only for retrospective cohort characterization.
	They must not be provided to model state or an acquisition policy.
	"""
	eligible = bool((frame["ICULOS"] == checkpoint).any())
	if first_positive_hour is None:
		first_positive_hour = first_positive_label_hour(frame)
	if not eligible:
		return {
			"eligible": False,
			"first_positive_label_at_or_before_checkpoint": False,
			"first_positive_label_after_checkpoint": False,
			"no_positive_label": False,
		}

	at_or_before = first_positive_hour is not None and first_positive_hour <= checkpoint
	after = first_positive_hour is not None and first_positive_hour > checkpoint
	return {
		"eligible": True,
		"first_positive_label_at_or_before_checkpoint": at_or_before,
		"first_positive_label_after_checkpoint": after,
		"no_positive_label": first_positive_hour is None,
	}


def _percentage(count: int, denominator: int) -> float:
	return round(100.0 * count / denominator, 4) if denominator else 0.0


def _cohort_stats(patients: list[dict[str, Any]], checkpoint: int) -> dict[str, Any]:
	eligible = [
		patient
		for patient in patients
		if patient["checkpoints"][checkpoint]["eligible"]
	]
	at_or_before = sum(
		patient["checkpoints"][checkpoint]["first_positive_label_at_or_before_checkpoint"]
		for patient in eligible
	)
	after = sum(
		patient["checkpoints"][checkpoint]["first_positive_label_after_checkpoint"]
		for patient in eligible
	)
	no_positive = sum(
		patient["checkpoints"][checkpoint]["no_positive_label"]
		for patient in eligible
	)
	if at_or_before + after + no_positive != len(eligible):
		raise AssertionError("Checkpoint label timing groups do not partition eligibility")

	return {
		"checkpoint_iculos_hour": checkpoint,
		"eligible_patients": len(eligible),
		"eligible_with_first_positive_label_at_or_before_checkpoint": at_or_before,
		"percentage_with_first_positive_label_at_or_before_checkpoint": _percentage(
			at_or_before, len(eligible)
		),
		"eligible_with_first_positive_label_after_checkpoint": after,
		"percentage_with_first_positive_label_after_checkpoint": _percentage(
			after, len(eligible)
		),
		"pre_positive_label_eligible_patients": after,
		"pre_positive_label_eligibility_percentage": _percentage(after, len(eligible)),
		"eligible_with_no_positive_label": no_positive,
		"percentage_with_no_positive_label": _percentage(no_positive, len(eligible)),
	}


def _verify_acquisition_eligibility(
	summary: dict[str, Any], feasibility_path: Path
) -> dict[str, Any]:
	if not feasibility_path.is_file():
		return {"status": "skipped", "reason": f"Reference summary not found: {feasibility_path}"}

	feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
	comparisons = []
	for checkpoint in CHECKPOINTS:
		timing_checkpoint = summary["checkpoints"][str(checkpoint)]
		feasibility_checkpoint = feasibility["checkpoints"][str(checkpoint)]
		pairs = [("All", timing_checkpoint["overall"], feasibility_checkpoint["overall"])]
		pairs.extend(
			(
				hospital,
				timing_checkpoint["by_hospital"][hospital],
				feasibility_checkpoint["by_hospital"][hospital],
			)
			for hospital in HOSPITALS.values()
		)
		for hospital, timing_stats, feasibility_stats in pairs:
			timing_count = timing_stats["eligible_patients"]
			feasibility_count = feasibility_stats[
				"patients_with_observations_through_checkpoint"
			]
			if timing_count != feasibility_count:
				raise ValueError(
					f"Eligibility mismatch at ICU hour {checkpoint} for {hospital}: "
					f"label timing={timing_count}, acquisition audit={feasibility_count}"
				)
			timing_positive_count = (
				timing_stats["eligible_with_first_positive_label_at_or_before_checkpoint"]
				+ timing_stats["eligible_with_first_positive_label_after_checkpoint"]
			)
			feasibility_positive_count = feasibility_stats[
				"sepsis_positive_patients_among_eligible"
			]
			if timing_positive_count != feasibility_positive_count:
				raise ValueError(
					f"Positive-label count mismatch at ICU hour {checkpoint} for {hospital}: "
					f"label timing={timing_positive_count}, acquisition audit={feasibility_positive_count}"
				)
			comparisons.append(
				{
					"checkpoint_iculos_hour": checkpoint,
					"hospital": hospital,
					"eligible_patients": timing_count,
					"positive_labeled_eligible_patients": timing_positive_count,
					"matches_acquisition_audit": True,
				}
			)
	return {"status": "passed", "comparisons": comparisons}


def _build_summary(patients: list[dict[str, Any]]) -> dict[str, Any]:
	summary: dict[str, Any] = {
		"dataset": "PhysioNet/Computing in Cardiology Challenge 2019",
		"dataset_version": DATASET_VERSION,
		"patient_count": len(patients),
		"hospital_patient_counts": {
			hospital: sum(patient["hospital"] == hospital for patient in patients)
			for hospital in HOSPITALS.values()
		},
		"first_positive_label_definition": (
			"The minimum ICULOS with SepsisLabel=1 in the full patient record; null if no row is positive."
		),
		"time_index_convention": (
			"Rows represent hourly observations indexed by ICULOS, with hour 1 as the first row. "
			"A checkpoint t is eligible only when a row exists exactly at ICULOS=t. "
			"Label timing is compared to t inclusively: at or before means <=t; after means >t."
		),
		"outcome_label_caveat": (
			"First positive SepsisLabel timing is the first positive Challenge label, not verified true clinical onset. "
			"Full-record labels are used only for retrospective cohort analysis and are not model input or acquisition-policy information."
		),
		"checkpoints": {},
	}
	for checkpoint in CHECKPOINTS:
		hospital_patients = {
			hospital: [p for p in patients if p["hospital"] == hospital]
			for hospital in HOSPITALS.values()
		}
		summary["checkpoints"][str(checkpoint)] = {
			"overall": _cohort_stats(patients, checkpoint),
			"by_hospital": {
				hospital: _cohort_stats(group, checkpoint)
				for hospital, group in hospital_patients.items()
			},
		}
	return summary


def _markdown_report(summary: dict[str, Any], verification: dict[str, Any]) -> str:
	lines = [
		"# First-Positive Label Timing Audit",
		"",
		f"**Dataset:** {summary['dataset']}, version {summary['dataset_version']}<br>",
		f"**Patients audited:** {summary['patient_count']:,}<br>",
		"**Hospitals:** Hospital A and Hospital B",
		"",
		"## Definitions",
		"",
		summary["time_index_convention"],
		"",
		"The first positive label is the minimum `ICULOS` whose `SepsisLabel` equals 1 anywhere in that patient's complete record. Patients are eligible at a checkpoint only if their record contains a row exactly at that hour. Within eligible patients, the three reported groups are mutually exclusive and exhaustive: first positive label at or before the checkpoint, first positive label after it, and no positive label in the full record.",
		"",
		"The ‘pre-positive-label eligible’ group means eligible patients whose first positive Challenge label occurs strictly after the checkpoint. It is a retrospective label-timing characterization, not evidence that the patient had no clinical sepsis at that checkpoint.",
		"",
		f"**Important outcome caveat:** {summary['outcome_label_caveat']}",
		"",
		"## Overall Results",
		"",
		"Counts are among checkpoint-eligible patients; percentages use eligible patients as the denominator.",
		"",
		"| ICU hour | Eligible | First positive at/before, n (%) | First positive after, n (%) | No positive label, n (%) | Pre-positive-label eligible, n (%) |",
		"|---:|---:|---:|---:|---:|---:|",
	]
	for checkpoint in CHECKPOINTS:
		stats = summary["checkpoints"][str(checkpoint)]["overall"]
		lines.append(
			f"| {checkpoint} | {stats['eligible_patients']:,} "
			f"| {stats['eligible_with_first_positive_label_at_or_before_checkpoint']:,} ({stats['percentage_with_first_positive_label_at_or_before_checkpoint']:.2f}%) "
			f"| {stats['eligible_with_first_positive_label_after_checkpoint']:,} ({stats['percentage_with_first_positive_label_after_checkpoint']:.2f}%) "
			f"| {stats['eligible_with_no_positive_label']:,} ({stats['percentage_with_no_positive_label']:.2f}%) "
			f"| {stats['pre_positive_label_eligible_patients']:,} ({stats['pre_positive_label_eligibility_percentage']:.2f}%) |"
		)

	lines.extend(
		[
			"",
			"## Hospital Results",
			"",
			"| ICU hour | Hospital | Eligible | First positive at/before, n (%) | First positive after, n (%) | No positive label, n (%) | Pre-positive-label eligible, n (%) |",
			"|---:|:---|---:|---:|---:|---:|---:|",
		]
	)
	for checkpoint in CHECKPOINTS:
		for hospital in HOSPITALS.values():
			stats = summary["checkpoints"][str(checkpoint)]["by_hospital"][hospital]
			lines.append(
				f"| {checkpoint} | {hospital} | {stats['eligible_patients']:,} "
				f"| {stats['eligible_with_first_positive_label_at_or_before_checkpoint']:,} ({stats['percentage_with_first_positive_label_at_or_before_checkpoint']:.2f}%) "
				f"| {stats['eligible_with_first_positive_label_after_checkpoint']:,} ({stats['percentage_with_first_positive_label_after_checkpoint']:.2f}%) "
				f"| {stats['eligible_with_no_positive_label']:,} ({stats['percentage_with_no_positive_label']:.2f}%) "
				f"| {stats['pre_positive_label_eligible_patients']:,} ({stats['pre_positive_label_eligibility_percentage']:.2f}%) |"
			)

	lines.extend(["", "## Cross-Audit Validation", ""])
	if verification["status"] == "passed":
		lines.append(
			"Eligible patient counts and the number of full-record positive-labeled eligible patients "
			"match `outputs/acquisition_feasibility/summary.json` for the overall cohort and both "
			"hospitals at all three checkpoints."
		)
	else:
		lines.append(f"Eligibility comparison was skipped: {verification['reason']}")

	lines.extend(
		[
			"",
			"## Limitations and Interpretation",
			"",
			"- `SepsisLabel` timing is a dataset label convention. The first positive hourly label is not necessarily the true biological or clinical onset time.",
			"- Labels later in the patient record are intentionally used here only to characterize historical timing and pre-positive-label cohort size. They must never be exposed as model features, patient state, or acquisition-policy information at an earlier checkpoint.",
			"- Patients without a row exactly at a checkpoint are excluded from that checkpoint's eligible denominator, even if records exist both before and after it.",
			"- The report is descriptive and does not train a prediction model or establish prospective performance.",
			"",
			"A future prediction experiment may use this audit to describe how much eligible data precedes the first positive Challenge label. It must preserve the distinction between that label and clinical onset and must enforce temporal separation in every feature and policy input.",
			"",
			"Aggregate output: `outputs/label_timing/summary.json`. No patient-level output is written.",
			"",
		]
	)
	return "\n".join(lines)


def run_audit(
	data_dir: Path = DEFAULT_DATA_DIR,
	output_dir: Path = DEFAULT_OUTPUT_DIR,
	report_path: Path = DEFAULT_REPORT,
	feasibility_summary_path: Path = DEFAULT_FEASIBILITY_SUMMARY,
) -> dict[str, Any]:
	patients: list[dict[str, Any]] = []
	for folder_name, hospital in HOSPITALS.items():
		folder = Path(data_dir) / folder_name
		if not folder.is_dir():
			raise FileNotFoundError(f"Missing hospital dataset folder: {folder}")
		patient_files = sorted(folder.glob("*.psv"))
		if not patient_files:
			raise FileNotFoundError(f"No .psv patient files found in: {folder}")
		print(f"{hospital}: reading {len(patient_files):,} patient files")
		for path in patient_files:
			frame = read_patient_file(path)
			first_positive_hour = first_positive_label_hour(frame)
			patients.append(
				{
					"hospital": hospital,
					"first_positive_label_iculos": first_positive_hour,
					"checkpoints": {
						checkpoint: checkpoint_label_metrics(
							frame, checkpoint, first_positive_hour
						)
						for checkpoint in CHECKPOINTS
					},
				}
			)

	summary = _build_summary(patients)
	verification = _verify_acquisition_eligibility(
		summary, Path(feasibility_summary_path)
	)
	summary["acquisition_audit_eligibility_verification"] = verification

	output_dir = Path(output_dir)
	report_path = Path(report_path)
	output_dir.mkdir(parents=True, exist_ok=True)
	report_path.parent.mkdir(parents=True, exist_ok=True)
	with (output_dir / "summary.json").open("w", encoding="utf-8") as output:
		json.dump(summary, output, indent=2)
		output.write("\n")
	report_path.write_text(_markdown_report(summary, verification), encoding="utf-8")
	return summary


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
	parser.add_argument(
		"--feasibility-summary", type=Path, default=DEFAULT_FEASIBILITY_SUMMARY
	)
	arguments = parser.parse_args()
	summary = run_audit(
		arguments.data_dir,
		arguments.output_dir,
		arguments.report,
		arguments.feasibility_summary,
	)
	print(json.dumps(summary["checkpoints"], indent=2))


if __name__ == "__main__":
	main()
