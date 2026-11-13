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
from src.data.multihorizon_labels import add_horizon_targets, future_onset_rows
from src.evaluation.abstention_acquisition import (
    decision_metrics,
    evaluate_trivial_baselines,
    risk_coverage_curve,
)
from src.evaluation.selective_prediction import confidence
from src.evaluation.wait_recheck import wait_recheck_candidates
from src.evaluation.paired_bootstrap import paired_cluster_bootstrap_rows
from src.evaluation.physionet_utility import score_frame
from src.models.sequence_baselines import append_missingness_channels
from src.narrative.structured_summary import render_decision_summary


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("NEXT_PHASE_CONFIG", ROOT / "configs/next_phase.json"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
MODEL_DIR = Path(os.environ.get("ALL_LAB_MODEL_DIR", ROOT / CONFIG.get("model_dir", "outputs/next_phase_all_labs")))
MODEL_PATH = MODEL_DIR / "all_lab_mask_aware_xgboost.joblib"
CALIBRATOR_PATH = MODEL_DIR / "all_lab_sigmoid_calibrator.joblib"
THRESHOLD = float(os.environ.get("NEXT_PHASE_THRESHOLD", CONFIG.get("threshold", 0.9669101582337782)))
ALL_LABS = tuple(LAB_FEATURES)
GROUPS = dict(CLINICAL_LAB_GROUPS)
GROUP_ORDER = tuple(GROUPS)
BOOTSTRAP_RESAMPLES = int(os.environ.get("NEXT_PHASE_BOOTSTRAP_RESAMPLES", CONFIG.get("bootstrap_resamples", 200)))


def predict(model, calibrator, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame[list(model.feature_names_in_)])[:, 1]
    from src.uncertainty.platt import apply_platt
    if not isinstance(calibrator, dict):
        return apply_platt(calibrator, raw)
    hidden_count = np.column_stack([
        frame[f"{lab}_observed"].fillna(0).to_numpy() == 0 for lab in ALL_LABS
    ]).sum(axis=1)
    result = np.empty(len(frame), dtype=float)
    for name, (low, high) in {
        "all_visible": (0, 0),
        "lightly_hidden": (1, 8),
        "moderately_hidden": (9, 17),
        "mostly_hidden": (18, 25),
        "all_hidden": (26, 26),
    }.items():
        rows = (hidden_count >= low) & (hidden_count <= high)
        if rows.any():
            result[rows] = apply_platt(calibrator[name], raw[rows])
    return result


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


def policy_probabilities(frame, base, groups, policy, abstained):
    """Apply one acquisition policy only to baseline-abstained rows.

    ``oracle_upper_bound`` is intentionally an evaluation ceiling: it may use
    the observed label and post-acquisition probability, but is never a
    deployable recommendation policy. A deployable patient-specific policy
    requires a separate model of possible test results and their likelihoods;
    the old implementation incorrectly used hidden results directly.
    """
    eligible = observed_groups(frame)
    names = list(GROUP_ORDER)
    selected = np.full(len(frame), -1, dtype=int)
    abstained = np.asarray(abstained, dtype=bool)
    if len(abstained) != len(frame):
        raise ValueError("abstained mask must align with frame")
    eligible &= abstained[:, None]
    if policy == "fixed_order":
        for j in range(len(names)):
            selected[(selected < 0) & eligible[:, j]] = j
    elif policy == "random":
        rng = np.random.default_rng(42)
        for i in range(len(frame)):
            choices = np.flatnonzero(eligible[i])
            if len(choices):
                selected[i] = rng.choice(choices)
    elif policy == "oracle_upper_bound":
        stacked = np.column_stack([groups[name] for name in names])
        for i in range(len(frame)):
            choices = np.flatnonzero(eligible[i])
            if len(choices):
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
    baseline_retained = confidence(base) >= THRESHOLD
    baseline_abstained = ~baseline_retained
    output = {
        "all_labs_hidden": {
            "metrics": metrics(labels, base),
            "physionet_utility": score_frame(frame, (base >= 0.5).astype(int)),
        },
        "baselines": evaluate_trivial_baselines(labels, base),
        "risk_coverage_curve": risk_coverage_curve(labels, base),
        "policy_scope": {
            "baseline_abstained_rows": int(baseline_abstained.sum()),
            "policy_rows_are_restricted_to_abstentions": True,
        },
    }
    for policy in ("fixed_order", "random", "oracle_upper_bound"):
        values, eligible = policy_probabilities(
            frame, base, groups, policy, baseline_abstained
        )
        baseline_records = [
            {"patient_id": patient, "label": int(label), "probability": float(probability)}
            for patient, label, probability in zip(frame["patient_id"], labels, base)
        ]
        policy_records = [
            {"patient_id": patient, "label": int(label), "probability": float(probability)}
            for patient, label, probability in zip(frame["patient_id"], labels, values)
        ]
        output[policy] = {
            "metrics": metrics(labels, values),
            "physionet_utility": score_frame(frame, (values >= 0.5).astype(int)),
            "eligible_rows": eligible,
            "paired_cluster_bootstrap_vs_all_labs_hidden": paired_cluster_bootstrap_rows(
                baseline_records,
                policy_records,
                n_resamples=BOOTSTRAP_RESAMPLES,
            ),
        }
    output["patient_specific"] = {
        "status": "not_evaluated",
        "reason": (
            "A valid pre-acquisition patient-specific policy requires predicted "
            "test-result likelihoods; observed post-acquisition probabilities "
            "are not used as a recommendation proxy."
        ),
    }
    return output


def run_wait_recheck(model, calibrator, frame):
    base = predict(model, calibrator, apply_hidden_set(frame, ALL_LABS))
    baseline_abstained = confidence(base) < THRESHOLD
    current_index, future_index = wait_recheck_candidates(frame, baseline_abstained)
    if len(current_index) == 0:
        return {
            "baseline_abstained_rows": int(baseline_abstained.sum()),
            "waited_rows": 0,
            "rows_with_future_observation": 0,
            "no_future_observation_rows": int(baseline_abstained.sum()),
            "delay_hours": None,
            "next_observation_metrics": None,
        }
    current = frame.iloc[current_index]
    future = frame.iloc[future_index]
    labels = future["SepsisLabel"].astype(int).to_numpy()
    probabilities = predict(model, calibrator, apply_hidden_set(future, ALL_LABS))
    retained_after_wait = confidence(probabilities) >= THRESHOLD
    delays = future["ICULOS"].to_numpy() - current["ICULOS"].to_numpy()
    return {
        "baseline_abstained_rows": int(baseline_abstained.sum()),
        "waited_rows": int(len(current_index)),
        "rows_with_future_observation": int(len(current_index)),
        "no_future_observation_rows": int(baseline_abstained.sum() - len(current_index)),
        "delay_hours": {
            "mean": float(np.mean(delays)),
            "median": float(np.median(delays)),
            "p95": float(np.percentile(delays, 95)),
        },
        "unresolved_after_wait": int((~retained_after_wait).sum()),
        "unresolved_after_wait_rate": float((~retained_after_wait).mean()),
        "next_observation_metrics": decision_metrics(
            labels, probabilities, retained_after_wait
        ),
    }


def run_horizons(model, calibrator, frame):
    data = future_onset_rows(add_horizon_targets(frame, (6, 12, 24)))
    state = apply_hidden_set(data, ALL_LABS)
    probabilities = predict(model, calibrator, state)
    result = {}
    for horizon in (6, 12, 24):
        labels = data[f"SepsisWithin{horizon}h"].to_numpy()
        result[str(horizon)] = {
            "evaluation": "currently negative rows only",
            "model_target": "current SepsisLabel; descriptive horizon association, not horizon-trained",
            "rows": int(len(labels)),
            "positive_rate": float(labels.mean()),
            "auroc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, probabilities)) if labels.min() != labels.max() else None,
            "auprc": float(average_precision_score(labels, probabilities)) if labels.sum() else None,
            "brier_score": float(np.mean((probabilities - labels) ** 2)),
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
    output = Path(os.environ.get(
        "NEXT_PHASE_COMPLETE_OUTPUT",
        ROOT / CONFIG.get("complete_output", "experiments/results/next_phase/complete_next_phase.json"),
    ))
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
