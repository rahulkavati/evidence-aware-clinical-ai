import pandas as pd

from experiments.run_all_lab_training_evaluation import ALL_LABS, mask_patient_consistently


def test_all_lab_masking_is_patient_consistent():
    data = {"patient_id": ["a", "a", "b"]}
    for lab in ALL_LABS:
        for suffix in ("last", "observed", "age"):
            data[f"{lab}_{suffix}"] = [1.0, 1.0, 1.0]
    frame = pd.DataFrame(data)
    features = [column for column in frame.columns if column != "patient_id"]
    masked = mask_patient_consistently(frame, features, seed=42)
    for lab in ALL_LABS:
        if masked.loc[0, f"{lab}_observed"] == 0:
            assert masked.loc[1, f"{lab}_observed"] == 0
