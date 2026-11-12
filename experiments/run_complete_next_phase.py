"""Run the remaining next-phase evaluations in one reproducible report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from src.data.build_temporal_features import LAB_FEATURES
from src.data.clinical_lab_groups import CLINICAL_LAB_GROUPS
from src.data.g1_masking import apply_hidden_set
from src.data.multihorizon_labels import add_horizon_targets
from src.evaluation.abstention_acquisition import decision_metrics
from src.evaluation.selective_prediction import confidence
from src.models.sequence_baselines import append_missingness_channels
from src.narrative.structured_summary import render_decision_summary


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("ALL_LAB_MODEL_DIR", "/private/tmp/evidence-aware-next-phase-all-labs"))
MODEL_PATH = MODEL_DIR / "all_lab_mask_aware_xgboost.joblib"
CALIBRATOR_PATH = MODEL_DIR / "all_lab_sigmoid_calibrator.joblib"
THRESHOLD = 0.9743948720817948
ALL_LABS = tuple(LAB_FEATURES)
GROUPS = dict(CLINICAL_LAB_GROUPS)
GROUP_ORDER = tuple(GROUPS)


def predict(model, calibrator, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame[list(model.feature_names_in_)])[:, 1]
    from src.uncertainty.platt import apply_platt
    return apply_platt(calibrator, raw)


def group_probabilities(model, calibrator, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    base = predict(model, calibrator, apply_hidden_set(frame, ALL_LABS))
    outputs = {}
    for name, labs in GROUPS.items():
        hidden = [lab for lab in ALL_LABS if lab not in labs]
        outputs[name] = predict(model, calibrator, apply_hidden_set(frame, hidden))
    return base, outputs


def observed_groups(frame: pd.DataFrame) -> np.ndarray:
    result = np.zeros((len(frame), len(GROUP_ORDER)), dtype=bool)
    for index, name in enumerate(GROUP_ORDER):
        result[:, index] = np.column_stack([
            frame[f"{lab}_observed"].fillna(0).to_numpy() == 1 for lab in GROUPS[name]
        ]).any(axis=1)
    return result


def policy_probabilities(frame, base, groups, policy):
    eligible = observed_groups(frame)
    names = list(GROUP_ORDER)
    selected = np.full(len(frame), -1, dtype=int)
    if policy == "fixed_order":
        for j in range(len(names)):
            selected[(selected < 0) & eligible[:, j]] = j
    elif policy == "random":
        rng = np.random.default_rng(42)
        for i in range(len(frame)):
            choices = np.flatnonzero(eligible[i])
            if len(choices):
                selected[i] = rng.choice(choices)
    elif policy in {"patient_specific", "oracle"}:
        stacked = np.column_stack([groups[name] for name in names])
        for i in range(len(frame)):
            choices = np.flatnonzero(eligible[i])
            if len(choices):
                if policy == "patient_specific":
                    score = np.where(
                        (stacked[i, choices] >= THRESHOLD) != (base[i] >= THRESHOLD),
                        np.abs(stacked[i, choices] - base[i]),
                        0.0,
                    )
                    if not np.any(score):
                        score = np.abs(stacked[i, choices] - base[i])
                    selected[i] = choices[int(np.argmax(score))]
                else:
                    label = int(frame.iloc[i]["SepsisLabel"])
                    errors = ((stacked[i, choices] >= 0.5) != bool(label)).astype(int)
                    selected[i] = choices[int(np.argmin(errors))]
    result = base.copy()
    for j, name in enumerate(names):
        mask = selected == j
        result[mask] = groups[name][mask]
    return result, int((selected >= 0).sum())


def metrics(labels, probabilities):
    retained = confidence(probabilities) >= THRESHOLD
    return decision_metrics(labels, probabilities, retained)


def run_group_policies(model, calibrator, frame):
    labels = frame["SepsisLabel"].astype(int).to_numpy()
    base, groups = group_probabilities(model, calibrator, frame)
    output = {"all_labs_hidden": metrics(labels, base)}
    for policy in ("fixed_order", "random", "patient_specific", "oracle"):
        values, eligible = policy_probabilities(frame, base, groups, policy)
        output[policy] = {"metrics": metrics(labels, values), "eligible_rows": eligible}
    return output


def run_wait_recheck(model, calibrator, frame):
    ordered = frame.sort_values(["patient_id", "ICULOS"]).copy()
    next_frame = ordered.groupby("patient_id", sort=False).shift(-1)
    valid = next_frame["ICULOS"].notna().to_numpy()
    current = ordered.loc[valid].copy()
    future = next_frame.loc[valid].copy()
    future["patient_id"] = current["patient_id"].to_numpy()
    labels = future["SepsisLabel"].astype(int).to_numpy()
    probabilities = predict(model, calibrator, future)
    return {"rows": int(valid.sum()), "next_observation_metrics": metrics(labels, probabilities)}


def run_horizons(model, calibrator, frame):
    data = add_horizon_targets(frame, (6, 12, 24))
    state = apply_hidden_set(data, ALL_LABS)
    probabilities = predict(model, calibrator, state)
    result = {}
    for horizon in (6, 12, 24):
        labels = data[f"SepsisWithin{horizon}h"].to_numpy()
        result[str(horizon)] = {
            "positive_rate": float(labels.mean()),
            "auroc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, probabilities)) if labels.min() != labels.max() else None,
        }
    return result


def sequence_tensor(frame, features, max_patients=None):
    """Build one terminal eight-hour sequence per patient."""
    samples, labels = [], []
    for _, patient in frame.sort_values("ICULOS").groupby("patient_id", sort=False):
        values = patient[features].to_numpy(dtype=np.float32)
        values = append_missingness_channels(values[None, :, :])[0]
        values = values[-8:]
        if len(values) < 8:
            values = np.pad(values, ((8 - len(values), 0), (0, 0)))
        samples.append(values)
        labels.append(int(patient.iloc[-1]["SepsisLabel"]))
        if max_patients and len(samples) >= max_patients:
            break
    return torch.tensor(np.asarray(samples)), torch.tensor(np.asarray(labels), dtype=torch.float32)


class _GRU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.rnn = nn.GRU(features, 32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.head(hidden[-1]).squeeze(-1)


class _TCN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv = nn.Conv1d(features, 32, 3, padding=2)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        y = self.conv(x.transpose(1, 2))[:, :, :-2]
        return self.head(y[:, :, -1]).squeeze(-1)


def run_sequence_models(model, train, test):
    features = list(model.feature_names_in_)
    train_x, train_y = sequence_tensor(train, features, max_patients=1000)
    test_x, test_y = sequence_tensor(test, features, max_patients=1000)
    result = {}
    for name, cls in (("GRU", _GRU), ("TCN", _TCN)):
        torch.manual_seed(42)
        net = cls(train_x.shape[-1])
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for _ in range(3):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(net(train_x), train_y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            probabilities = torch.sigmoid(net(test_x)).numpy()
        result[name] = {
            "train_patients": int(len(train_y)),
            "test_patients": int(len(test_y)),
            "positive_rate": float(test_y.mean()),
            "auroc": float(roc_auc_score(test_y, probabilities)) if len(np.unique(test_y)) == 2 else None,
            "auprc": float(average_precision_score(test_y, probabilities)) if test_y.sum() else None,
            "brier_score": float(np.mean((probabilities - test_y.numpy()) ** 2)),
        }
    return result


def main():
    model = joblib.load(MODEL_PATH)
    calibrator = joblib.load(CALIBRATOR_PATH)
    test = pd.read_parquet(ROOT / "data/processed/test.parquet")
    train = pd.read_parquet(ROOT / "data/processed/train.parquet")
    result = {
        "threshold": THRESHOLD,
        "patients": int(test.patient_id.nunique()),
        "rows": int(len(test)),
        "group_policies": run_group_policies(model, calibrator, test),
        "wait_recheck": run_wait_recheck(model, calibrator, test),
        "multi_horizon_descriptives": run_horizons(model, calibrator, test),
        "sequence_models": run_sequence_models(model, train, test),
        "narrative_example": render_decision_summary({
            "action": "abstain", "probability": 0.12, "confidence": 0.88,
            "detail": "Structured output only; no language-layer decision was made.",
        }),
    }
    output = Path(os.environ.get("NEXT_PHASE_COMPLETE_OUTPUT", "/private/tmp/evidence-aware-complete-next-phase.json"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
