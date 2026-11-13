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
