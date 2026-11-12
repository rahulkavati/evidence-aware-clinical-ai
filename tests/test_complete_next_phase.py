def test_complete_runner_imports():
    from experiments.run_complete_next_phase import GROUPS, GROUP_ORDER
    assert len(GROUPS) == 4
    assert set(GROUP_ORDER) == set(GROUPS)
