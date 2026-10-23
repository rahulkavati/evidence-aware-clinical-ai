"""Evaluate saved XGBoost calibration methods on held-out validation patients."""

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

from src.uncertainty.calibrate_models import (
    RANDOM_STATE,
    expected_calibration_error,
    split_patient_ids,
)

DATA_PATH = Path("data/processed/val.parquet")
MODEL_DIR = Path("outputs/models")
RESULTS_DIR = Path("experiments/results/calibration")
FIGURES_DIR = RESULTS_DIR / "figures"
COMPARISON_PATH = RESULTS_DIR / "calibration_comparison.json"
EXCLUDED_FEATURES = {"patient_id", "hospital", "SepsisLabel"}
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 42
HIGH_RISK_RANGES = ((0.1, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0))
SMALL_SAMPLE_ROWS = 100
SMALL_SAMPLE_PATIENTS = 30


def align_features(
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...] | np.ndarray,
) -> pd.DataFrame:
    """Select model predictors in their saved order, excluding metadata and labels."""
    ordered_names = list(feature_names)
    if len(ordered_names) != len(set(ordered_names)):
        raise ValueError("Model feature names must be unique")
    excluded = EXCLUDED_FEATURES.intersection(ordered_names)
    if excluded:
        raise ValueError(f"Excluded columns found in model features: {sorted(excluded)}")
    missing = [name for name in ordered_names if name not in frame.columns]
    if missing:
        raise ValueError(f"Validation dataset is missing model features: {missing}")
    return frame.loc[:, ordered_names]


def validate_probabilities(probabilities: np.ndarray, method: str) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1:
        raise ValueError(f"{method} probabilities must be one-dimensional")
    if not np.isfinite(probs).all():
        raise ValueError(f"{method} probabilities must be finite")
    if ((probs < 0.0) | (probs > 1.0)).any():
        raise ValueError(f"{method} probabilities must be within [0, 1]")
    return probs


def apply_calibrators(
    raw_probabilities: np.ndarray,
    sigmoid_calibrator: Any,
    isotonic_calibrator: Any,
) -> dict[str, np.ndarray]:
    raw = validate_probabilities(raw_probabilities, "Raw")
    epsilon = 1e-7
    clipped = np.clip(raw, epsilon, 1.0 - epsilon)
    logits = np.log(clipped / (1.0 - clipped))
    calibrated = {
        "raw": raw,
        "sigmoid": np.asarray(
            sigmoid_calibrator.predict_proba(logits.reshape(-1, 1))[:, 1],
            dtype=float,
        ),
        "isotonic": np.asarray(isotonic_calibrator.predict(raw), dtype=float),
    }
    return {
        name: validate_probabilities(probabilities, name.title())
        for name, probabilities in calibrated.items()
    }


def hospital_row_indices(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return row indices for the two cohort hospitals."""
    hospitals = frame["hospital"]
    observed = set(hospitals.dropna().unique())
    if observed != {"A", "B"}:
        raise ValueError(f"Expected hospitals A and B, found {sorted(observed)}")
    return {
        hospital: np.flatnonzero(hospitals.to_numpy() == hospital)
        for hospital in ("A", "B")
    }


def evaluate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=int)
    probs = validate_probabilities(probabilities, "Evaluation")
    if y.ndim != 1 or len(y) == 0 or len(y) != len(probs):
        raise ValueError("Labels and probabilities must have equal nonzero length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must contain only 0 and 1")
    return {
        "observations": int(len(y)),
        "positive_labels": int(y.sum()),
        "auroc": float(roc_auc_score(y, probs)),
        "auprc": float(average_precision_score(y, probs)),
        "brier_score": float(brier_score_loss(y, probs)),
        "ece_10_bins": expected_calibration_error(y, probs, bins=10),
    }


def reliability_bins(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int | None]]:
    y = np.asarray(labels, dtype=int)
    probs = validate_probabilities(probabilities, "Reliability")
    if len(y) == 0 or len(y) != len(probs):
        raise ValueError("Labels and probabilities must have equal nonzero length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, n_bins - 1)
    result: list[dict[str, float | int | None]] = []
    for index in range(n_bins):
        selected = indices == index
        count = int(selected.sum())
        result.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_predicted_probability": (
                    float(probs[selected].mean()) if count else None
                ),
                "observed_frequency": float(y[selected].mean()) if count else None,
            }
        )
    return result


def high_risk_diagnostics(
    frame: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patient_ids = frame["patient_id"].to_numpy()
    for lower, upper in HIGH_RISK_RANGES:
        selected = (probabilities >= lower) & (
            (probabilities <= upper) if upper == 1.0 else (probabilities < upper)
        )
        count = int(selected.sum())
        unique_patients = int(pd.Series(patient_ids[selected]).nunique())
        rows.append(
            {
                "scope": scope,
                "method": method,
                "range": f"[{lower:.1f}, {upper:.1f}{']' if upper == 1.0 else ')'}",
                "count": count,
                "mean_predicted_probability": (
                    float(probabilities[selected].mean()) if count else None
                ),
                "observed_positive_frequency": (
                    float(np.asarray(labels)[selected].mean()) if count else None
                ),
                "unique_patients": unique_patients,
                "small_sample": (
                    count < SMALL_SAMPLE_ROWS
                    or unique_patients < SMALL_SAMPLE_PATIENTS
                ),
                "small_sample_threshold": (
                    f"fewer than {SMALL_SAMPLE_ROWS} hourly rows or "
                    f"{SMALL_SAMPLE_PATIENTS} patients"
                ),
            }
        )
    return rows


def bootstrap_row_weights(
    patient_codes: np.ndarray,
    sampled_patient_indices: np.ndarray,
    patient_count: int,
) -> np.ndarray:
    codes = np.asarray(patient_codes, dtype=int)
    sampled = np.asarray(sampled_patient_indices, dtype=int)
    if patient_count < 1 or ((codes < 0) | (codes >= patient_count)).any():
        raise ValueError("Patient codes must reference the available patient cohort")
    if ((sampled < 0) | (sampled >= patient_count)).any():
        raise ValueError("Sampled patient indices must reference the cohort")
    multiplicities = np.bincount(sampled, minlength=patient_count)
    return multiplicities[codes]


def _weighted_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(labels, dtype=float)
    probs = np.asarray(probabilities, dtype=float)
    weights = np.asarray(weights, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, bins - 1)
    total_weight = weights.sum()
    ece = 0.0
    for index in range(bins):
        selected = (indices == index) & (weights > 0)
        bin_weight = weights[selected].sum()
        if bin_weight:
            observed = np.average(y[selected], weights=weights[selected])
            mean_probability = np.average(probs[selected], weights=weights[selected])
            ece += bin_weight / total_weight * abs(observed - mean_probability)
    return float(ece)


def patient_cluster_bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    patient_codes: np.ndarray,
    patient_count: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    random_state: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float]]:
    """Percentile intervals from patient-cluster resampling using row multiplicities."""
    y = np.asarray(labels, dtype=int)
    probs = validate_probabilities(probabilities, "Bootstrap")
    codes = np.asarray(patient_codes, dtype=int)
    if len(y) != len(probs) or len(y) != len(codes) or not len(y):
        raise ValueError("Labels, probabilities, and patient codes must align")
    if patient_count < 1 or ((codes < 0) | (codes >= patient_count)).any():
        raise ValueError("Patient codes must reference the available patient cohort")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    rng = np.random.default_rng(random_state)
    brier_values = np.empty(n_resamples, dtype=float)
    ece_values = np.empty(n_resamples, dtype=float)
    for iteration in range(n_resamples):
        sampled = rng.integers(0, patient_count, size=patient_count)
        weights = bootstrap_row_weights(codes, sampled, patient_count)
        brier_values[iteration] = np.average((y - probs) ** 2, weights=weights)
        ece_values[iteration] = _weighted_ece(y, probs, weights)

    return {
        "brier_score": {
            "lower_95": float(np.percentile(brier_values, 2.5)),
            "upper_95": float(np.percentile(brier_values, 97.5)),
        },
        "ece_10_bins": {
            "lower_95": float(np.percentile(ece_values, 2.5)),
            "upper_95": float(np.percentile(ece_values, 97.5)),
        },
    }


def _make_scope_result(
    frame: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    row_indices: np.ndarray,
    scope: str,
    n_resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scoped_frame = frame.iloc[row_indices]
    labels = scoped_frame["SepsisLabel"].to_numpy(dtype=int)
    patient_codes, patient_names = pd.factorize(
        scoped_frame["patient_id"], sort=False
    )
    if len(patient_names) < 1:
        raise ValueError(f"No observations available for {scope}")
    results: dict[str, Any] = {}
    high_risk_rows: list[dict[str, Any]] = []
    for method, all_probabilities in probabilities.items():
        probs = all_probabilities[row_indices]
        metrics = evaluate_metrics(labels, probs)
        metrics["bootstrap_95_percentile_ci"] = patient_cluster_bootstrap_ci(
            labels,
            probs,
            patient_codes,
            len(patient_names),
            n_resamples=n_resamples,
            random_state=BOOTSTRAP_SEED,
        )
        results[method] = metrics
        high_risk_rows.extend(
            high_risk_diagnostics(scoped_frame, labels, probs, method, scope)
        )
    return results, high_risk_rows


def _plot_reliability(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    rows: np.ndarray,
    scope: str,
    filename: str,
    zoom: bool = False,
) -> None:
    fig = plt.figure(figsize=(10, 7))
    grid = fig.add_gridspec(2, 1, height_ratios=(4, 1), hspace=0.28)
    ax = fig.add_subplot(grid[0])
    count_ax = fig.add_subplot(grid[1])
    count_ax.axis("off")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    cutoff = 0.2 if zoom else 1.0
    count_bins: dict[str, list[dict[str, float | int | None]]] = {}
    for method, all_probs in probabilities.items():
        probs = all_probs[rows]
        scoped_labels = labels[rows]
        bins = reliability_bins(scoped_labels, probs)
        count_bins[method] = bins
        populated = [
            item
            for item in bins
            if item["count"] and item["mean_predicted_probability"] <= cutoff
        ]
        x_values = [item["mean_predicted_probability"] for item in populated]
        y_values = [item["observed_frequency"] for item in populated]
        ax.plot(x_values, y_values, marker="o", label=method.title())
    visible_bins = [
        item
        for item in count_bins["raw"]
        if item["lower"] < cutoff
    ]
    table_rows = []
    for method, bins in count_bins.items():
        counts = []
        for item in bins[: len(visible_bins)]:
            count = int(item["count"])
            counts.append(f"{count:,}" + ("*" if count < SMALL_SAMPLE_ROWS else ""))
        table_rows.append([method.title(), *counts])
    bin_headers = [
        f"[{item['lower']:.1f},{item['upper']:.1f}{')' if item['upper'] < 1 else ']'}"
        for item in visible_bins
    ]
    count_ax.table(
        cellText=table_rows,
        colLabels=["Method / probability bin", *bin_headers],
        loc="center",
        cellLoc="center",
    )
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive frequency",
        xlim=(0, cutoff),
        ylim=(0, cutoff),
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.text(
        0.5,
        0.015,
        "*Table counts are hourly observations; * denotes fewer than 100 and may be statistically unreliable.",
        ha="center",
        fontsize=8,
    )
    fig.suptitle(f"Reliability diagram — {scope}")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.92, bottom=0.08, hspace=0.2)
    fig.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close(fig)


def _comparison_check(
    results: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    differences: dict[str, dict[str, float]] = {}
    for method in ("raw", "sigmoid", "isotonic"):
        observed = results["overall"][method]
        expected = comparison[method]
        differences[method] = {}
        for metric in ("auroc", "auprc", "brier_score", "ece_10_bins"):
            difference = abs(observed[metric] - expected[metric])
            differences[method][metric] = float(difference)
            if not np.isclose(observed[metric], expected[metric], rtol=1e-6, atol=1e-6):
                raise AssertionError(
                    f"{method} {metric} does not match calibration_comparison.json: "
                    f"{observed[metric]} != {expected[metric]} "
                    f"(absolute difference {difference})"
                )
    return {
        "status": "passed",
        "absolute_differences": differences,
        "tolerance": {"relative": 1e-6, "absolute": 1e-6},
    }


def _write_report(results: dict[str, Any]) -> None:
    lines = [
        "# Calibration robustness evaluation",
        "",
        "Validation-only evaluation on the patient-disjoint selection subset. "
        "No test-set records were loaded.",
        "",
        f"- Calibration patients: {results['split']['calibration_patients']:,}",
        f"- Selection patients: {results['split']['selection_patients']:,}",
        f"- Selection hourly observations: {results['split']['selection_hourly_rows']:,}",
        f"- Patient-cluster bootstrap: {results['bootstrap']['resamples']} resamples "
        f"(seed {results['bootstrap']['random_state']}); 95% percentile intervals.",
        f"- Existing calibration metrics consistency check: "
        f"{results['comparison_check']['status']} "
        f"(maximum absolute difference "
        f"{max(max(metrics.values()) for metrics in results['comparison_check']['absolute_differences'].values()):.2e}).",
        f"- Small-sample flag: fewer than {SMALL_SAMPLE_ROWS} rows or "
        f"{SMALL_SAMPLE_PATIENTS} patients in a high-risk range.",
        "",
        "## Overall and hospital metrics",
        "",
        "| Scope | Method | N | Positives | AUROC | AUPRC | Brier | ECE (10 bins) | "
        "Brier 95% CI | ECE 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("overall", "hospital_A", "hospital_B"):
        for method, metric in results["metrics"][scope].items():
            brier_ci = metric["bootstrap_95_percentile_ci"]["brier_score"]
            ece_ci = metric["bootstrap_95_percentile_ci"]["ece_10_bins"]
            lines.append(
                f"| {scope} | {method} | {metric['observations']:,} | "
                f"{metric['positive_labels']:,} | {metric['auroc']:.6f} | "
                f"{metric['auprc']:.6f} | {metric['brier_score']:.6f} | "
                f"{metric['ece_10_bins']:.6f} | "
                f"[{brier_ci['lower_95']:.6f}, {brier_ci['upper_95']:.6f}] | "
                f"[{ece_ci['lower_95']:.6f}, {ece_ci['upper_95']:.6f}] |"
            )
    lines.extend(
        [
            "",
            "## High-risk probability diagnostics",
            "",
            "Ranges are [0.1, 0.2), [0.2, 0.5), [0.5, 0.8), and [0.8, 1.0]. "
            "Empty ranges are reported with unavailable means/frequencies. "
            "A flagged row indicates a small sample under the thresholds above; "
            "bin-level rates should not be treated as precise.",
            "",
            "| Scope | Method | Probability range | Count | Mean probability | "
            "Observed positive frequency | Unique patients | Small sample? |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in results["high_risk_diagnostics"]:
        mean_probability = item["mean_predicted_probability"]
        observed_rate = item["observed_positive_frequency"]
        lines.append(
            f"| {item['scope']} | {item['method']} | {item['range']} | "
            f"{item['count']:,} | "
            f"{'—' if mean_probability is None else f'{mean_probability:.6f}'} | "
            f"{'—' if observed_rate is None else f'{observed_rate:.6f}'} | "
            f"{item['unique_patients']:,} | {'Yes' if item['small_sample'] else 'No'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "Calibration should not be compared using overall ECE alone. Review the "
            "Brier score, discrimination, hospital-specific results, high-risk "
            "ranges, sample counts, and cluster-bootstrap intervals together. "
            "The calibration subset and selection subset contain different patients; "
            "these results do not establish performance on an external cohort. "
            "Bootstrap intervals resample patients, while all hourly observations "
            "within a selected patient remain together.",
            "",
        ]
    )
    (RESULTS_DIR / "robustness_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_parquet(DATA_PATH)
    patient_labels = validation.groupby("patient_id")["SepsisLabel"].max()
    calibration_ids, selection_ids = split_patient_ids(
        patient_labels,
        random_state=RANDOM_STATE,
    )
    if len(calibration_ids) != 3025 or len(selection_ids) != 3025:
        raise AssertionError(
            "Expected 3,025 calibration and 3,025 selection patients for this dataset; "
            f"got {len(calibration_ids)} and {len(selection_ids)}"
        )
    calibration_set = set(calibration_ids)
    selection_set = set(selection_ids)
    if calibration_set.intersection(selection_set):
        raise AssertionError("Calibration and selection patients overlap")

    selection_mask = validation["patient_id"].isin(selection_set).to_numpy()
    selection_frame = validation.loc[selection_mask].reset_index(drop=True)
    if set(selection_frame["patient_id"].unique()) != selection_set:
        raise AssertionError("Selection partition did not include its exact patient set")
    labels = selection_frame["SepsisLabel"].to_numpy(dtype=int)

    model = joblib.load(MODEL_DIR / "xgboost.joblib")
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError("Saved XGBoost model does not expose its training feature names")
    predictors = align_features(selection_frame, feature_names)
    raw_probabilities = model.predict_proba(predictors)[:, 1]
    sigmoid_calibrator = joblib.load(
        MODEL_DIR / "xgboost_sigmoid_calibrator.joblib"
    )
    isotonic_calibrator = joblib.load(
        MODEL_DIR / "xgboost_isotonic_calibrator.joblib"
    )
    probabilities = apply_calibrators(
        raw_probabilities,
        sigmoid_calibrator,
        isotonic_calibrator,
    )

    overall_rows = np.arange(len(selection_frame))
    hospital_rows = hospital_row_indices(selection_frame)
    metrics: dict[str, Any] = {}
    high_risk_rows: list[dict[str, Any]] = []
    for scope, row_indices in (
        ("overall", overall_rows),
        ("hospital_A", hospital_rows["A"]),
        ("hospital_B", hospital_rows["B"]),
    ):
        metrics[scope], scope_high_risk = _make_scope_result(
            selection_frame,
            probabilities,
            row_indices,
            scope,
            BOOTSTRAP_RESAMPLES,
        )
        high_risk_rows.extend(scope_high_risk)

    _plot_reliability(
        labels,
        probabilities,
        overall_rows,
        "Overall, 0–1",
        "reliability_overall.png",
    )
    _plot_reliability(
        labels,
        probabilities,
        overall_rows,
        "Overall, 0–0.2 zoom",
        "reliability_overall_zoom.png",
        zoom=True,
    )
    for hospital, rows in hospital_rows.items():
        _plot_reliability(
            labels,
            probabilities,
            rows,
            f"Hospital {hospital}",
            f"reliability_hospital_{hospital.lower()}.png",
        )

    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    result = {
        "dataset": "data/processed/val.parquet",
        "evaluation_split": "patient-disjoint validation selection subset",
        "predictor_count": len(feature_names),
        "excluded_predictors": sorted(EXCLUDED_FEATURES),
        "split": {
            "random_state": RANDOM_STATE,
            "calibration_patients": len(calibration_ids),
            "selection_patients": len(selection_ids),
            "calibration_hourly_rows": int(
                validation["patient_id"].isin(calibration_set).sum()
            ),
            "selection_hourly_rows": int(len(selection_frame)),
        },
        "bootstrap": {
            "method": "patient-cluster bootstrap; resample patient IDs with replacement and weight all their hourly rows",
            "resamples": BOOTSTRAP_RESAMPLES,
            "random_state": BOOTSTRAP_SEED,
            "interval": "95% percentile",
        },
        "metrics": metrics,
        "high_risk_diagnostics": high_risk_rows,
        "comparison_check": {
            "file": "calibration_comparison.json",
        },
    }
    result["comparison_check"].update(
        _comparison_check(result["metrics"], comparison)
    )
    (RESULTS_DIR / "robustness_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    _write_report(result)
    print(
        json.dumps(
            {
                "split": result["split"],
                "metrics": result["metrics"],
                "comparison_check": result["comparison_check"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()