def test_complete_runner_imports():
    from experiments.run_complete_next_phase import GROUPS, GROUP_ORDER
    assert len(GROUPS) == 4
    assert set(GROUP_ORDER) == set(GROUPS)


def test_shortcut_ablation_variants_remove_only_declared_features():
    from experiments.run_shortcut_ablations import feature_variants

    columns = ["patient_id", "hospital", "SepsisLabel", "HR_last", "ICULOS", "HospAdmTime", "Unit1", "Unit2"]
    variants = feature_variants(columns)
    assert "ICULOS" in variants["none"]
    assert "ICULOS" not in variants["time"]
    assert "Unit1" not in variants["unit"]
    assert "ICULOS" not in variants["time_and_unit"]
    assert "HR_last" in variants["time_and_unit"]


def test_oracle_prefers_group_that_is_retained_and_correct():
    import numpy as np
    import pandas as pd
    from experiments.run_complete_next_phase import GROUPS, GROUP_ORDER, policy_probabilities

    columns = {}
    for labs in GROUPS.values():
        for lab in labs:
            columns[f"{lab}_observed"] = [1, 1]
    frame = pd.DataFrame(columns)
    frame["SepsisLabel"] = [0, 0]
    base = np.array([0.2, 0.2])
    groups = {name: np.array([0.2, 0.2]) for name in GROUP_ORDER}
    last = GROUP_ORDER[-1]
    groups[last] = np.array([0.01, 0.3])  # only row 0 becomes confidently correct
    values, eligible = policy_probabilities(
        frame, base, groups, "oracle_upper_bound", np.array([True, True]), threshold=0.97
    )
    assert eligible == 2
    assert values[0] == 0.01
    fixed, _ = policy_probabilities(
        frame, base, groups, "fixed_order", np.array([True, True]), threshold=0.97
    )
    assert fixed[0] == 0.2
