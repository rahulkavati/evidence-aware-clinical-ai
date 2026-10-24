
"""Run a retrospective acquisition demo on one validation patient."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.simulation.acquisition_simulator import AcquisitionSimulator
from src.acquisition.baseline_policies import fixed_order_policy


SPLIT_FILE = Path("outputs/splits/val.csv")
MODEL_FILE = Path("outputs/models/xgboost.joblib")
CALIBRATOR_FILE = Path(
    "outputs/models/xgboost_sigmoid_calibrator.joblib"
)
CHECKPOINT = 12
HIDDEN_LABS = ["Lactate", "WBC", "Creatinine", "Platelets"]


def main():
    patients = pd.read_csv(SPLIT_FILE)
    model = joblib.load(MODEL_FILE)
    calibrator = joblib.load(CALIBRATOR_FILE)

    for patient in patients.itertuples(index=False):
        frame = pd.read_csv(patient.file_path, sep="|")

        if CHECKPOINT not in frame["ICULOS"].values:
            continue

        simulator = AcquisitionSimulator(
            patient=frame,
            checkpoint=CHECKPOINT,
            model=model,
            calibrator=calibrator,
            hidden_features=HIDDEN_LABS,
        )

        eligible = simulator.eligible_features()

        if not eligible:
            continue

        initial_risk = simulator.predict_risk()

        selected = fixed_order_policy(eligible)
        result = simulator.acquire(selected)

        # Do not display patient IDs or raw clinical measurements.
        output = {
            "checkpoint": CHECKPOINT,
            "eligible_hidden_feature_count": len(eligible),
            "selected_feature": selected,
            "initial_calibrated_risk": initial_risk,
            "updated_calibrated_risk": result.risk_after,
            "risk_change": result.risk_after - initial_risk,
            "acquired_count": result.acquired_count,
        }

        print(json.dumps(output, indent=2))
        return

    raise RuntimeError(
        "No eligible validation patient found at the checkpoint"
    )


if __name__ == "__main__":
    main()
