import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_held_out_split_requires_locked_flag():
    result = subprocess.run(
        [sys.executable, "experiments/run_final_evaluation.py", "--stage", "predict", "--split", "test"],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    assert result.returncode != 0
    assert "--locked-test" in result.stderr


def test_verify_protocol_rejects_changed_artifacts(tmp_path, monkeypatch):
    import experiments.run_final_evaluation as final

    for name in ("model", "calibrator", "selector"):
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(final, "MODEL_PATH", tmp_path / "model")
    monkeypatch.setattr(final, "CALIBRATOR_PATH", tmp_path / "calibrator")
    monkeypatch.setattr(final, "SELECTOR_PATH", tmp_path / "selector")
    protocol = {"artifact_sha256": {n: final.sha256(tmp_path / n) for n in ("model", "calibrator", "selector")}}
    final.verify_protocol(protocol)  # unchanged files pass
    (tmp_path / "model").write_bytes(b"retrained")
    with pytest.raises(RuntimeError, match="changed after the protocol was frozen"):
        final.verify_protocol(protocol)


def test_test_access_is_recorded_once_per_protocol(tmp_path, monkeypatch):
    import experiments.run_final_evaluation as final

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"accesses": [{"index": 1}]}))
    monkeypatch.setattr(final, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(final, "LEDGER_PATH", ledger_path)
    protocol = {"abstention_threshold": 0.97, "alert_thresholds": {"as_recorded": 0.02}}
    final.record_test_access(protocol)
    final.record_test_access(protocol)
    accesses = json.loads(ledger_path.read_text())["accesses"]
    assert len(accesses) == 2
    assert accesses[-1]["index"] == 2


def test_create_splits_refuses_to_overwrite(monkeypatch, tmp_path):
    import src.data.create_splits as splits

    (tmp_path / "train.csv").write_text("patient_id\n")
    monkeypatch.setattr(splits, "OUTPUT_DIR", tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        splits.main([])
