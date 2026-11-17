"""Early-warning evaluation: alert thresholds, utility intervals, lead time.

The official PhysioNet 2019 utility rewards binary hourly alerts. Converting a
calibrated probability into an alert at a fixed 0.5 cut-off almost never fires
when the hourly positive rate is about 2%, which scores close to the
"never alert" reference (normalized utility 0). The alert threshold is
therefore an operating point that must be selected on validation data and then
frozen before any held-out evaluation, exactly like the abstention threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.physionet_utility import compute_prediction_utility, score_frame

CLINICAL_ONSET_OFFSET_HOURS = 6
"""PhysioNet 2019 labels turn positive six hours before clinical onset."""


def _ordered(frame: pd.DataFrame, values: np.ndarray, name: str) -> pd.DataFrame:
    required = {"patient_id", "ICULOS", "SepsisLabel"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    values = np.asarray(values)
    if len(values) != len(frame):
        raise ValueError(f"{name} must align with frame rows")
    return (
        frame[["patient_id", "ICULOS", "SepsisLabel"]]
        .assign(**{name: values})
        .sort_values(["patient_id", "ICULOS"], kind="stable")
    )


def candidate_alert_thresholds(probabilities: np.ndarray, count: int = 40) -> np.ndarray:
    """Return a deterministic grid of candidate thresholds from score quantiles."""
    p = np.asarray(probabilities, dtype=float)
    if len(p) == 0 or count < 2:
        raise ValueError("Need probabilities and at least two candidates")
    quantiles = np.r_[np.linspace(0.50, 0.99, count - 5), 0.993, 0.995, 0.997, 0.998, 0.999]
    return np.unique(np.quantile(p, quantiles))


def select_alert_threshold(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    candidates: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Choose the alert threshold that maximizes validation normalized utility.

    Must only be called on validation data. Ties keep the higher threshold,
    which produces fewer alerts for the same utility.
    """
    p = np.asarray(probabilities, dtype=float)
    grid = candidate_alert_thresholds(p) if candidates is None else np.asarray(candidates, dtype=float)
    curve = []
    for threshold in grid:
        alerts = (p >= threshold).astype(int)
        score = score_frame(frame, alerts)
        curve.append({
            "threshold": float(threshold),
            "normalized_utility": float(score["normalized_utility"]),
            "alert_rate": float(alerts.mean()),
        })
    best = max(curve, key=lambda item: (item["normalized_utility"], item["threshold"]))
    return {"threshold": best["threshold"], "selected": best, "curve": curve}


def patient_utility_components(frame: pd.DataFrame, alerts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-patient observed, best, and inaction utilities (official scoring)."""
    ordered = _ordered(frame, np.asarray(alerts, dtype=int), "_alert")
    observed, best, inaction, ids = [], [], [], []
    for patient_id, patient in ordered.groupby("patient_id", sort=False):
        y = patient["SepsisLabel"].astype(int).to_numpy()
        a = patient["_alert"].to_numpy(dtype=int)
        optimal = np.zeros(len(y), dtype=int)
        if y.any():
            t_sepsis = int(np.argmax(y)) + CLINICAL_ONSET_OFFSET_HOURS
            optimal[max(0, t_sepsis - 12): min(t_sepsis + 3 + 1, len(y))] = 1
        observed.append(compute_prediction_utility(y, a))
        best.append(compute_prediction_utility(y, optimal))
        inaction.append(compute_prediction_utility(y, np.zeros(len(y), dtype=int)))
        ids.append(patient_id)
    return {
        "patient_id": np.asarray(ids),
        "observed": np.asarray(observed, dtype=float),
        "best": np.asarray(best, dtype=float),
        "inaction": np.asarray(inaction, dtype=float),
    }


def _normalized(components: dict[str, np.ndarray], weights: np.ndarray | None = None) -> float:
    w = np.ones(len(components["observed"])) if weights is None else weights
    observed = float(np.dot(w, components["observed"]))
    best = float(np.dot(w, components["best"]))
    inaction = float(np.dot(w, components["inaction"]))
    if np.isclose(best, inaction):
        return float("nan")  # resample without any septic patient
    return (observed - inaction) / (best - inaction)


def utility_with_interval(
    frame: pd.DataFrame,
    alerts: np.ndarray,
    *,
    reference_alerts: np.ndarray | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Normalized utility with a patient-cluster bootstrap 95% interval.

    With ``reference_alerts`` the paired difference (alerts minus reference)
    is also reported, resampling the same patients for both systems.
    """
    components = patient_utility_components(frame, alerts)
    reference = None if reference_alerts is None else patient_utility_components(frame, reference_alerts)
    if reference is not None and not np.array_equal(reference["patient_id"], components["patient_id"]):
        raise AssertionError("Paired utility components are misaligned")
    rng = np.random.default_rng(seed)
    n = len(components["observed"])
    values, differences = [], []
    for _ in range(n_resamples):
        weights = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        value = _normalized(components, weights)
        values.append(value)
        if reference is not None:
            differences.append(value - _normalized(reference, weights))
    values_array = np.asarray(values, dtype=float)
    valid = np.isfinite(values_array)
    if not valid.any():
        raise ValueError("No bootstrap replicate contained a septic patient")
    result: dict[str, Any] = {
        "normalized_utility": _normalized(components),
        "ci95": [
            float(np.percentile(values_array[valid], 2.5)),
            float(np.percentile(values_array[valid], 97.5)),
        ],
        "valid_replicates": int(valid.sum()),
        "n_resamples": n_resamples,
        "seed": seed,
        "patients": n,
    }
    if reference is not None:
        result["difference_vs_reference"] = _normalized(components) - _normalized(reference)
        diffs = np.asarray(differences, dtype=float)[valid]
        result["difference_ci95"] = [
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
        ]
    return result


def patient_level_warning(frame: pd.DataFrame, flags: np.ndarray) -> dict[str, Any]:
    """Patient-level view of any hourly flag (an alert or an escalation).

    For septic patients, clinical onset is the first positive label hour plus
    six hours. A patient counts as warned when a flag occurs strictly before
    clinical onset; lead time is onset minus the first such flag hour.
    """
    ordered = _ordered(frame, np.asarray(flags, dtype=bool), "_flag")
    septic = warned = 0
    non_septic = non_septic_flagged = 0
    lead_times: list[float] = []
    for _, patient in ordered.groupby("patient_id", sort=False):
        y = patient["SepsisLabel"].astype(int).to_numpy()
        hours = patient["ICULOS"].to_numpy(dtype=float)
        flagged = patient["_flag"].to_numpy(dtype=bool)
        if y.any():
            septic += 1
            onset = hours[int(np.argmax(y))] + CLINICAL_ONSET_OFFSET_HOURS
            early = flagged & (hours < onset)
            if early.any():
                warned += 1
                lead_times.append(float(onset - hours[np.flatnonzero(early)[0]]))
        else:
            non_septic += 1
            non_septic_flagged += int(flagged.any())
    return {
        "septic_patients": septic,
        "septic_patients_warned_before_onset": warned,
        "septic_warning_rate": float(warned / septic) if septic else None,
        "median_lead_time_hours": float(np.median(lead_times)) if lead_times else None,
        "non_septic_patients": non_septic,
        "non_septic_patients_ever_flagged": non_septic_flagged,
        "non_septic_flag_rate": float(non_septic_flagged / non_septic) if non_septic else None,
        "hourly_flag_rate": float(np.mean(flags)),
    }
