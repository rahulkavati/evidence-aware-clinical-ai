import numpy as np
import pandas as pd
import pytest

from src.evaluation.early_warning import (
    candidate_alert_thresholds,
    patient_level_warning,
    patient_utility_components,
    select_alert_threshold,
    utility_with_interval,
)
from src.evaluation.physionet_utility import score_frame


def _cohort():
    rows = []
    # Septic patient: label turns positive at hour 10 (clinical onset hour 16).
    for hour in range(1, 21):
        rows.append({"patient_id": "s1", "ICULOS": hour, "SepsisLabel": int(hour >= 10)})
    for patient in ("n1", "n2"):
        for hour in range(1, 21):
            rows.append({"patient_id": patient, "ICULOS": hour, "SepsisLabel": 0})
    return pd.DataFrame(rows)


def test_utility_components_reproduce_official_normalization():
    frame = _cohort()
    rng = np.random.default_rng(0)
    alerts = rng.integers(0, 2, size=len(frame))
    parts = patient_utility_components(frame, alerts)
    manual = (parts["observed"].sum() - parts["inaction"].sum()) / (
        parts["best"].sum() - parts["inaction"].sum()
    )
    assert manual == pytest.approx(score_frame(frame, alerts)["normalized_utility"])


def test_select_alert_threshold_prefers_informative_scores():
    frame = _cohort()
    probability = np.where(
        (frame.patient_id == "s1") & (frame.ICULOS >= 6), 0.9, 0.01
    )
    selected = select_alert_threshold(frame, probability, candidates=[0.005, 0.5, 0.95])
    assert selected["threshold"] == 0.5
    assert selected["selected"]["normalized_utility"] > 0.5


def test_never_alerting_scores_zero_and_interval_contains_estimate():
    frame = _cohort()
    result = utility_with_interval(frame, np.zeros(len(frame)), n_resamples=50)
    assert result["normalized_utility"] == pytest.approx(0.0)
    low, high = result["ci95"]
    assert low <= result["normalized_utility"] <= high


def test_paired_difference_is_zero_for_identical_systems():
    frame = _cohort()
    alerts = (frame.ICULOS % 3 == 0).astype(int).to_numpy()
    result = utility_with_interval(frame, alerts, reference_alerts=alerts, n_resamples=20)
    assert result["difference_vs_reference"] == pytest.approx(0.0)
    assert result["difference_ci95"] == [pytest.approx(0.0), pytest.approx(0.0)]


def test_patient_level_warning_lead_time_and_false_flags():
    frame = _cohort()
    flags = ((frame.patient_id == "s1") & (frame.ICULOS == 8)) | (
        (frame.patient_id == "n1") & (frame.ICULOS == 2)
    )
    summary = patient_level_warning(frame, flags.to_numpy())
    assert summary["septic_patients_warned_before_onset"] == 1
    assert summary["median_lead_time_hours"] == 8.0  # onset 16 minus flag 8
    assert summary["non_septic_patients_ever_flagged"] == 1
    assert summary["non_septic_flag_rate"] == 0.5


def test_flag_after_onset_is_not_a_warning():
    frame = _cohort()
    flags = (frame.patient_id == "s1") & (frame.ICULOS == 17)
    assert patient_level_warning(frame, flags.to_numpy())["septic_patients_warned_before_onset"] == 0


def test_candidate_grid_is_sorted_and_unique():
    grid = candidate_alert_thresholds(np.linspace(0, 1, 1000))
    assert np.all(np.diff(grid) > 0)
