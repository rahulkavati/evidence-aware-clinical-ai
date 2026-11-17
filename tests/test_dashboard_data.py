import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/final_evaluation/test_predictions.npz"


@pytest.mark.skipif(not CACHE.exists(), reason="run `make final-test` to create the cached predictions")
def test_dashboard_numbers_match_final_report():
    from dashboard.build_dashboard_data import build

    data = build()
    final = json.loads((ROOT / "experiments/results/final/final_test_evaluation.json").read_text())
    for name, system in final["evaluation"]["systems"].items():
        assert data["test"]["systems"][name]["utility"] == pytest.approx(
            system["early_warning_utility"]["normalized_utility"], abs=1e-6
        )
        assert data["test"]["systems"][name]["auroc"] == pytest.approx(system["discrimination"]["auroc"], abs=1e-6)
    assert data["protocol"]["abstention_threshold"] == final["protocol"]["abstention_threshold"]
    cases = {p["case"] for p in data["patients"]}
    assert {"Early warning", "Quiet night", "False alarm", "Missed case"} <= cases
    for patient in data["patients"]:
        n = len(patient["hours"])
        assert all(len(v) == n for v in patient["p"].values())
        assert all(len(v) == n for v in patient["available"].values())
