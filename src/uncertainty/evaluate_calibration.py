"""Evaluate calibration and selective prediction on the validation split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

DATA_PATH = Path("data/processed/val.parquet")
MODEL_DIR = Path("outputs/models")
RESULTS_DIR = Path("experiments/results/calibration")
EXCLUDED_FEATURES = {"patient_id", "hospital", "SepsisLabel"}
N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
MODEL_FILES = {
    "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    "xgboost": MODEL_DIR / "xgboost.joblib",
}


def align_features(
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | np.ndarray,
) -> pd.DataFrame:
    """Return predictors in saved-model order, rejecting missing or excluded names."""
    ordered_names = list(feature_names)
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("Model feature names must be unique")
    if EXCLUDED_FEATURES.intersection(ordered_names):
        raise ValueError("Patient identifiers, hospital, and label cannot be predictors")

    missing = [name for name in ordered_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Validation dataset is missing model features: {missing}")
    return frame.loc[:, ordered_names]


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    n_bins: int = N_BINS,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute equal-width ECE; probability 1.0 belongs to the final bin."""
    labels = np.asarray(y_true)
    probs = np.asarray(probabilities, dtype=float)
    _validate_binary_probabilities(labels, probs)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum(np.digitize(probs, edges[1:-1]), n_bins - 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0

    for index in range(n_bins):
        selected = bin_indices == index
        count = int(selected.sum())
        bin_result: dict[str, Any] = {
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_predicted_probability": None,
            "observed_frequency": None,
        }
        if count:
            mean_probability = float(probs[selected].mean())
            observed_frequency = float(labels[selected].mean())
            ece += count / len(labels) * abs(mean_probability - observed_frequency)
            bin_result["mean_predicted_probability"] = mean_probability
            bin_result["observed_frequency"] = observed_frequency
        bins.append(bin_result)

    return float(ece), bins


def selective_risk_curve(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Return risk and coverage retaining predictions from most to least certain.

    Risk is the fraction of incorrect binary predictions using a 0.5 threshold.
    Uncertainty is 1 - abs(2p - 1), so probabilities nearest 0.5 are least certain.
    Tied uncertainty values enter the retained set together.
    """
    labels = np.asarray(y_true)
    probs = np.asarray(probabilities, dtype=float)
    _validate_binary_probabilities(labels, probs)

    errors = (probs >= 0.5) != labels.astype(bool)
    uncertainty = 1.0 - np.abs(2.0 * probs - 1.0)
    _, inverse = np.unique(uncertainty, return_inverse=True)
    group_counts = np.bincount(inverse)
    group_errors = np.bincount(inverse, weights=errors.astype(float))
    retained_counts = np.cumsum(group_counts)
    retained_errors = np.cumsum(group_errors)

    coverage = retained_counts / len(labels)
    risk = retained_errors / retained_counts
    return coverage, risk


def _validate_binary_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    if labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional")
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must have equal, nonzero length")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("Probabilities must be within [0, 1]")


def _model_feature_names(model: Any, name: str) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None and hasattr(model, "get_booster"):
        feature_names = model.get_booster().feature_names
    if feature_names is None:
        raise ValueError(f"Saved {name} model does not expose its training feature names")
    return list(feature_names)


def _evaluate_model(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ece, bins = expected_calibration_error(labels, probabilities, N_BINS)
    metrics = {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "ece": ece,
    }
    return metrics, bins


def _plot_reliability(
    bin_results: dict[str, list[dict[str, Any]]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    for name, bins in bin_results.items():
        populated = [item for item in bins if item["count"]]
        ax.plot(
            [item["mean_predicted_probability"] for item in populated],
            [item["observed_frequency"] for item in populated],
            marker="o",
            label=name.replace("_", " ").title(),
        )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive frequency",
        title="Validation reliability diagrams",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "reliability_diagrams.png", dpi=160)
    plt.close(fig)


def _plot_probability_histograms(
    probability_results: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, len(probability_results), figsize=(12, 4), sharey=True)
    for ax, (name, probabilities) in zip(axes, probability_results.items()):
        ax.hist(probabilities, bins=BIN_EDGES, color="#4472C4", edgecolor="white")
        ax.set(
            xlabel="Predicted probability",
            title=name.replace("_", " ").title(),
            xlim=(0, 1),
        )
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Hourly prediction count")
    fig.suptitle("Validation probability distributions")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "probability_histograms.png", dpi=160)
    plt.close(fig)


def _plot_selective_risk(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (coverage, risk) in curves.items():
        ax.plot(
            coverage,
            risk,
            label=name.replace("_", " ").title(),
        )
    ax.set(
        xlabel="Coverage (fraction of hourly predictions retained)",
        ylabel="Classification error among retained predictions",
        title="Selective risk versus coverage (validation)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "selective_risk_coverage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_parquet(DATA_PATH)
    labels = validation["SepsisLabel"].to_numpy(dtype=int)
    models = {name: joblib.load(path) for name, path in MODEL_FILES.items()}

    feature_names = {
        name: _model_feature_names(model, name)
        for name, model in models.items()
    }
    reference_features = feature_names["logistic_regression"]
    for name, model_features in feature_names.items():
        if model_features != reference_features:
            raise ValueError(f"Saved {name} model feature order differs from Logistic Regression")
    predictors = align_features(validation, reference_features)

    metric_results: dict[str, Any] = {}
    bin_results: dict[str, list[dict[str, Any]]] = {}
    probability_results: dict[str, np.ndarray] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, model in models.items():
        probabilities = model.predict_proba(predictors)[:, 1]
        metrics, bins = _evaluate_model(labels, probabilities)
        coverage, risk = selective_risk_curve(labels, probabilities)
        probability_results[name] = probabilities
        bin_results[name] = bins
        curves[name] = (coverage, risk)
        metric_results[name] = {
            **metrics,
            "selective_risk_at_coverage": {
                str(target): {
                    "actual_coverage": float(coverage[np.searchsorted(coverage, target)]),
                    "risk": float(risk[np.searchsorted(coverage, target)]),
                }
                for target in (0.5, 0.8, 0.9, 1.0)
            },
        }

    _plot_reliability(bin_results)
    _plot_probability_histograms(probability_results)
    _plot_selective_risk(curves)

    aggregate = {
        "dataset": "data/processed/val.parquet",
        "hourly_predictions": int(len(labels)),
        "predictor_count": len(reference_features),
        "excluded_predictors": sorted(EXCLUDED_FEATURES),
        "ece": {
            "definition": "Sum over equal-width bins of bin fraction times absolute observed-frequency minus mean-probability difference.",
            "bin_count": N_BINS,
            "bin_edges": BIN_EDGES.tolist(),
            "models": bin_results,
        },
        "selective_evaluation": {
            "error_metric": "0/1 classification error at probability threshold 0.5",
            "uncertainty_score": "1 - abs(2 * probability - 1); lower values are retained first",
            "coverage_definition": "fraction of hourly predictions retained",
            "interpretation": "This probability-based uncertainty proxy is not a measure of clinical evidence sufficiency.",
        },
        "models": metric_results,
    }
    with (RESULTS_DIR / "aggregate_metrics.json").open("w") as output:
        json.dump(aggregate, output, indent=2)
    print(json.dumps(metric_results, indent=2))


if __name__ == "__main__":
    main()