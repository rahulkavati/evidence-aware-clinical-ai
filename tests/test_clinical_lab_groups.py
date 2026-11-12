import pandas as pd
import pytest

from src.data.build_temporal_features import LAB_FEATURES
from src.data.clinical_lab_groups import (
    CLINICAL_LAB_GROUPS,
    apply_hidden_groups,
    labs_for_groups,
    validate_lab_groups,
)


def test_groups_partition_all_26_laboratory_features():
    validate_lab_groups()
    assert sum(len(labs) for labs in CLINICAL_LAB_GROUPS.values()) == 26
    assert set(labs_for_groups(CLINICAL_LAB_GROUPS)) == set(LAB_FEATURES)


def test_group_expansion_is_stable_and_rejects_unknown_group():
    assert labs_for_groups(["hematology_coagulation"]) == (
        "Hct", "Hgb", "PTT", "Platelets"
    )
    with pytest.raises(KeyError):
        labs_for_groups(["not_a_real_group"])


def test_group_masking_hides_each_lab_triplet():
    frame = pd.DataFrame({
        "Hct_last": [10.0], "Hct_observed": [1], "Hct_age": [0.0],
        "Hgb_last": [11.0], "Hgb_observed": [1], "Hgb_age": [1.0],
        "PTT_last": [30.0], "PTT_observed": [1], "PTT_age": [2.0],
        "Platelets_last": [100.0], "Platelets_observed": [1], "Platelets_age": [3.0],
    })
    masked = apply_hidden_groups(frame, ["hematology_coagulation"])
    assert masked.filter(like="_observed").to_numpy().tolist() == [[0, 0, 0, 0]]
    assert masked.filter(like="_last").isna().all().all()
    assert masked.filter(like="_age").isna().all().all()
