import numpy as np
import pandas as pd

from experiments.run_all_lab_training_evaluation import ALL_LABS, mask_patient_consistently
from src.data.g1_masking import sample_hidden_matrix


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


def test_all_lab_masking_samples_uniform_hidden_counts():
    matrix = sample_hidden_matrix(270, len(ALL_LABS), np.random.default_rng(42))
    hidden_counts = matrix.sum(axis=1)
    assert set(hidden_counts) == set(range(len(ALL_LABS) + 1))


def test_all_lab_masking_uses_one_state_per_patient():
    data = {"patient_id": ["a", "a", "b", "b"]}
    for lab in ALL_LABS:
        for suffix in ("last", "observed", "age"):
            data[f"{lab}_{suffix}"] = [1.0] * 4
    frame = pd.DataFrame(data)
    features = [column for column in frame.columns if column != "patient_id"]
    masked = mask_patient_consistently(frame, features, seed=7)
    for lab in ALL_LABS:
        assert masked.loc[0, f"{lab}_observed"] == masked.loc[1, f"{lab}_observed"]
        assert masked.loc[2, f"{lab}_observed"] == masked.loc[3, f"{lab}_observed"]
