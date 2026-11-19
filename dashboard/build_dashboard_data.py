"""Export real results and held-out patient replays for the local dashboard.

Usage (from the repository root):
    PYTHONPATH=. python dashboard/build_dashboard_data.py

Reads only committed result files and the frozen model's cached test-set
outputs (run ``make final-test`` first if outputs/final_evaluation is empty),
and writes ``dashboard/data.js``. Nothing here retrains or re-tunes anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.data.clinical_lab_groups import CLINICAL_LAB_GROUPS

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "experiments/results/final/final_test_evaluation.json"
CACHE = ROOT / "outputs/final_evaluation/test_predictions.npz"
TEST = ROOT / "data/processed/test.parquet"
SPLITS = ROOT / "outputs/splits/summary.json"
MISSINGNESS = ROOT / "outputs/dataset_audit/feature_missingness.csv"
DATASET_SUMMARY = ROOT / "outputs/dataset_audit/summary.json"
OUTPUT = ROOT / "dashboard/data.js"
GROUP_ORDER = list(CLINICAL_LAB_GROUPS)
GROUP_LABELS = {
    "blood_gas": "Blood gas",
    "inflammation_perfusion": "Inflammation",
    "organ_function": "Organ function",
    "hematology_coagulation": "Hematology",
}
SHOWCASE_LABS = {
    "blood_gas": ["pH", "PaCO2", "HCO3"],
    "inflammation_perfusion": ["Lactate", "WBC"],
    "organ_function": ["Creatinine", "BUN", "Bilirubin_total"],
    "hematology_coagulation": ["Platelets", "Hgb"],
}
VITALS = {"HR": "Heart rate", "MAP": "Mean arterial pressure", "Resp": "Respiratory rate", "Temp": "Temperature", "O2Sat": "SpO2"}
ONSET_OFFSET = 6
PROJECT = {
    "title": "Knowing When the Evidence Is Not Enough",
    "subtitle": "Selective Prediction and Laboratory Acquisition for Early Sepsis Warning",
    "author": "Rahul Kavati",
    "degree": "Master of Science in Computer Science",
    "university": "California State University, Dominguez Hills",
    "type": "Master's Project",
    "term": "Fall 2026",
    "repository": "https://github.com/rahulkavati/evidence-aware-clinical-ai",
}
MAX_HOURS = 72


def r(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def downsample(points: list[dict[str, float]], keep: int = 160) -> list[dict[str, float]]:
    if len(points) <= keep:
        return points
    index = np.unique(np.linspace(0, len(points) - 1, keep).round().astype(int))
    return [points[i] for i in index]


def roc_points(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return downsample([{"x": r(a), "y": r(b)} for a, b in zip(fpr, tpr)], 180)


def calibration_bins(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    """Reliability points on fixed probability bins (dense near the low base rate)."""
    edges = np.array([0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5, 1.0001])
    which = np.searchsorted(edges, probabilities, side="right") - 1
    result = []
    for b in range(len(edges) - 1):
        mask = which == b
        if mask.sum() >= 50:
            result.append({"predicted": r(probabilities[mask].mean(), 5), "observed": r(labels[mask].mean(), 5), "rows": int(mask.sum())})
    return result


def risk_coverage(labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float]]:
    confidence = np.maximum(probabilities, 1 - probabilities)
    order = np.argsort(-confidence, kind="stable")
    errors = ((probabilities[order] >= 0.5) != labels[order].astype(bool)).astype(float)
    positives = labels[order].astype(float)
    count = np.arange(1, len(order) + 1)
    risk = np.cumsum(errors) / count
    escalated_capture = 1 - np.cumsum(positives) / positives.sum()
    points = []
    for q in np.linspace(0.5, 1.0, 101):
        i = min(len(order) - 1, int(q * len(order)) - 1)
        points.append({
            "coverage": r(count[i] / len(order)),
            "risk": r(risk[i], 5),
            "threshold": r(confidence[order][i], 5),
            "escalated_positive_capture": r(escalated_capture[i]),
        })
    return points


def select_patients(frame: pd.DataFrame, cache: dict[str, np.ndarray], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    tau = protocol["abstention_threshold"]
    alert = protocol["alert_thresholds"]["labs_hidden_and_policies"]
    frame = frame.assign(_row=np.arange(len(frame)))
    hidden = cache["labs_hidden"]
    fixed = cache["fixed_order"]
    rows = []
    for patient_id, patient in frame.groupby("patient_id", sort=True):
        patient = patient.sort_values("ICULOS")
        idx = patient["_row"].to_numpy()
        hours = patient["ICULOS"].to_numpy()
        labels = patient["SepsisLabel"].to_numpy(dtype=int)
        septic = bool(labels.any())
        onset = float(hours[int(np.argmax(labels))] + ONSET_OFFSET) if septic else None
        alarms = hidden[idx] >= alert
        early = alarms & (hours < onset) if septic else np.zeros(len(idx), bool)
        lead = float(onset - hours[np.flatnonzero(early)[0]]) if early.any() else None
        abstained = np.maximum(hidden[idx], 1 - hidden[idx]) < tau
        rescued = abstained & (np.maximum(fixed[idx], 1 - fixed[idx]) >= tau) & ((fixed[idx] >= 0.5) == labels.astype(bool))
        rows.append({
            "patient_id": patient_id, "hours": len(idx), "septic": septic, "lead": lead,
            "alarm_hours": int(alarms.sum()), "first_alarm_hour": float(hours[alarms][0]) if alarms.any() else None,
            "rescued": int(rescued.sum()), "escalations": int(abstained.sum()),
        })
    meta = pd.DataFrame(rows)
    meta = meta[(meta.hours >= 24) & (meta.hours <= MAX_HOURS)]
    rng = np.random.default_rng(42)

    def pick(subset: pd.DataFrame, n: int, sort: str | None = None) -> list[str]:
        if subset.empty:
            return []
        if sort:
            subset = subset.sort_values(sort)
            positions = np.linspace(0, len(subset) - 1, min(n, len(subset))).round().astype(int)
            return subset.iloc[positions]["patient_id"].tolist()
        return subset.sample(min(n, len(subset)), random_state=int(rng.integers(1e9)))["patient_id"].tolist()

    curated = [
        ("Early warning", "Septic; first alarm 8–20 h before clinical onset.",
         pick(meta[meta.septic & meta.lead.between(8, 20) & (meta.first_alarm_hour >= 4)], 1, "lead")),
        ("Quiet night", "Not septic; the model never alarms.",
         pick(meta[~meta.septic & (meta.alarm_hours == 0) & (meta.escalations == 0)], 1)),
        ("Lab changed the decision", "An ordered lab group turned an escalation into a confident, correct call.",
         pick(meta[meta.rescued >= 2], 1, "rescued")),
        ("False alarm", "Not septic, yet the model alarms repeatedly.",
         pick(meta[~meta.septic & (meta.alarm_hours >= 4)], 1)),
        ("Missed case", "Septic, but no alarm before clinical onset.",
         pick(meta[meta.septic & meta.lead.isna()], 1)),
    ]
    used = {pid for _, _, ids in curated for pid in ids}
    rest = meta[~meta.patient_id.isin(used)]
    extra = pick(rest[rest.septic], 8) + pick(rest[~rest.septic], 8)

    selected = [(label, why, pid) for label, why, ids in curated for pid in ids]
    selected += [("Random patient", "Drawn at random from the held-out set.", pid) for pid in extra]
    patients = []
    for label, why, pid in selected:
        patient = frame[frame.patient_id == pid].sort_values("ICULOS")
        idx = patient["_row"].to_numpy()
        labels = patient["SepsisLabel"].to_numpy(dtype=int)
        hours = patient["ICULOS"].astype(int).tolist()
        record = {
            "id": pid, "case": label, "why": why,
            "hospital": str(patient["hospital"].iloc[0]),
            "age": r(patient["Age"].iloc[0], 0),
            "gender": "female" if patient["Gender"].iloc[0] == 0 else "male",
            "hours": hours,
            "septic": bool(labels.any()),
            "onset_hour": (hours[int(np.argmax(labels))] + ONSET_OFFSET) if labels.any() else None,
            "p": {
                "labs_hidden": [r(v) for v in cache["labs_hidden"][idx]],
                "as_recorded": [r(v) for v in cache["as_recorded"][idx]],
                **{f"group_{g}": [r(v) for v in cache[f"group_{g}"][idx]] for g in GROUP_ORDER},
            },
            "policy_choice": {
                "patient_specific": [GROUP_ORDER[s] if s >= 0 else None for s in cache["patient_specific_selected"][idx]],
            },
            "available": {g: [bool(any(patient[f"{lab}_observed"].iloc[i] == 1 for lab in labs)) for i in range(len(idx))]
                          for g, labs in CLINICAL_LAB_GROUPS.items()},
            "labs": {g: {lab: {"value": [r(v, 2) if pd.notna(v) else None for v in patient[f"{lab}_last"]],
                               "age": [r(v, 0) if pd.notna(v) else None for v in patient[f"{lab}_age"]]}
                         for lab in SHOWCASE_LABS[g]} for g in GROUP_ORDER},
            "vitals": {name: [r(v, 1) if pd.notna(v) else None for v in patient[f"{name}_last"]] for name in VITALS},
        }
        patients.append(record)
    return patients


def build() -> dict[str, Any]:
    if not CACHE.exists():
        raise SystemExit("Missing outputs/final_evaluation/test_predictions.npz; run `make final-test` first.")
    result = json.loads(FINAL.read_text())
    protocol, evaluation = result["protocol"], result["evaluation"]
    with np.load(CACHE) as data:
        cache = {k: data[k] for k in data.files}
    frame = pd.read_parquet(TEST).reset_index(drop=True)
    if len(frame) != len(cache["labs_hidden"]):
        raise SystemExit("Cached predictions do not align with the test split")
    labels = frame["SepsisLabel"].astype(int).to_numpy()

    systems = {}
    for name, s in evaluation["systems"].items():
        u = s["early_warning_utility"]
        systems[name] = {
            "auroc": r(s["discrimination"]["auroc"]), "auprc": r(s["discrimination"]["auprc"]),
            "brier": r(s["discrimination"]["brier"], 5),
            "utility": r(u["normalized_utility"]), "utility_ci": [r(v) for v in u["ci95"]],
            "utility_delta": r(u.get("difference_vs_reference")),
            "utility_delta_ci": [r(v) for v in u["difference_ci95"]] if "difference_ci95" in u else None,
            "coverage": r(s["selective"]["coverage"]), "risk": r(s["selective"]["risk"], 5),
            "escalated_positive_capture": r(s["selective"]["abstained_positive_capture"]),
            "warned": r(s["alert_patient_level"]["septic_warning_rate"]),
            "lead_median": r(s["alert_patient_level"]["median_lead_time_hours"], 1),
            "false_alarm_patients": r(s["alert_patient_level"]["non_septic_flag_rate"]),
            "escalated_before_onset": r(s["escalation_patient_level"]["septic_warning_rate"]),
            "eligible_rows": s.get("eligible_rows"),
        }
    bootstrap = {
        name: {
            metric: {"est": r(b["observed"][metric], 6), "ci": [r(v, 6) for v in b["confidence_intervals_95"][metric]]}
            for metric in ("auroc", "auprc", "brier_improvement")
        }
        for name, b in result["paired_bootstrap_vs_labs_hidden"].items()
    }
    missing = pd.read_csv(MISSINGNESS, index_col=0)["missing_percentage"]
    labs_and_vitals = [c for c in missing.index if c not in {"Unit1", "Unit2", "HospAdmTime", "Gender", "Age", "ICULOS", "SepsisLabel"}]
    return {
        "project": PROJECT,
        "generated_from": {"final_report": str(FINAL.relative_to(ROOT)), "predictions": str(CACHE.relative_to(ROOT))},
        "dataset": json.loads(DATASET_SUMMARY.read_text()) if DATASET_SUMMARY.exists() else {},
        "splits": json.loads(SPLITS.read_text()),
        "missingness": [{"feature": f, "missing": r(missing[f], 2)} for f in labs_and_vitals],
        "lab_groups": {g: {"label": GROUP_LABELS[g], "labs": list(CLINICAL_LAB_GROUPS[g])} for g in GROUP_ORDER},
        "protocol": {
            "abstention_threshold": protocol["abstention_threshold"],
            "alert_thresholds": protocol["alert_thresholds"],
            "frozen_at": protocol["frozen_at"],
            "artifact_sha256": {k: v[:12] for k, v in protocol["artifact_sha256"].items()},
            "utility_curves_validation": protocol["alert_threshold_curves"],
        },
        "test": {
            "patients": evaluation["patients"], "rows": evaluation["rows"], "positive_rows": evaluation["positive_rows"],
            "systems": systems, "bootstrap": bootstrap,
            "always_alert_utility": r(evaluation["references"]["always_alert_utility"]),
            "predict_negative_risk": r(evaluation["references"]["selective_trivial_baselines"]["predict_negative_everywhere"]["risk"], 5),
            "wait_recheck": evaluation["wait_recheck"], "horizons": evaluation["horizons"],
            "roc": {"labs_hidden": roc_points(labels, cache["labs_hidden"]), "as_recorded": roc_points(labels, cache["as_recorded"])},
            "calibration": {"labs_hidden": calibration_bins(labels, cache["labs_hidden"]), "as_recorded": calibration_bins(labels, cache["as_recorded"])},
            "risk_coverage": risk_coverage(labels, cache["labs_hidden"]),
        },
        "ledger": result.get("test_access_ledger", {}).get("accesses", []),
        "patients": select_patients(frame, cache, protocol),
    }


def main() -> None:
    print(f"{PROJECT['type']}: {PROJECT['title']}")
    print(f"{PROJECT['author']} | {PROJECT['degree']} | {PROJECT['university']} | {PROJECT['term']}")
    print("Building dashboard data from the frozen final results...")
    data = build()
    OUTPUT.write_text("window.DASH = " + json.dumps(data, separators=(",", ":"), default=float) + ";\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.1f} MB) with {len(data['patients'])} patients")


if __name__ == "__main__":
    main()
