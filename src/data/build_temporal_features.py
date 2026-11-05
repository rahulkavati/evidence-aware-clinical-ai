
"""Leakage-safe temporal feature engineering for PhysioNet 2019."""

from __future__ import annotations

import numpy as np
import pandas as pd


VITAL_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP",
    "MAP", "DBP", "Resp", "EtCO2",
]

LAB_FEATURES = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
    "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
    "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI",
    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]

DYNAMIC_FEATURES = VITAL_FEATURES + LAB_FEATURES

STATIC_FEATURES = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
]


def build_temporal_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one feature row per ICU hour.

    Uses only current and historical observations.
    Does not use SepsisLabel as a feature.
    """

    required = DYNAMIC_FEATURES + STATIC_FEATURES + [
        "ICULOS", "SepsisLabel"
    ]

    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = frame.copy().sort_values("ICULOS")

    if df["ICULOS"].isna().any():
        raise ValueError("ICULOS cannot contain missing values")

    if df["ICULOS"].duplicated().any():
        raise ValueError("Duplicate ICU hours detected")

    result_data = {}

    hours = df["ICULOS"].to_numpy(dtype=float)

    for feature in DYNAMIC_FEATURES:
        values = pd.to_numeric(
            df[feature], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)

        observed = values.notna()

        last_values = values.ffill()

        last_seen_hours = pd.Series(
            np.where(observed, hours, np.nan),
            index=df.index,
        ).ffill()

        age = df["ICULOS"] - last_seen_hours

        result_data[f"{feature}_last"] = last_values
        result_data[f"{feature}_observed"] = last_seen_hours.notna().astype("int8")
        result_data[f"{feature}_age"] = age

    for feature in STATIC_FEATURES:
        result_data[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    result_data["ICULOS"] = df["ICULOS"]

    # Target retained separately for supervised training.
    result_data["SepsisLabel"] = df["SepsisLabel"]

    return pd.DataFrame(result_data, index=df.index).reset_index(drop=True)


def build_checkpoint_feature_vector(
    frame: pd.DataFrame,
    checkpoint: int,
    feature_names: list[str] | tuple[str, ...],
    hidden_features: set[str] | frozenset[str] = frozenset(),
) -> np.ndarray:
    """Build one ordered model vector directly from history through a checkpoint."""
    hidden = set(hidden_features)
    invalid_hidden = hidden - set(LAB_FEATURES)
    if invalid_hidden:
        raise ValueError(f"Invalid hidden laboratory features: {sorted(invalid_hidden)}")

    required = set(DYNAMIC_FEATURES + STATIC_FEATURES + ["ICULOS", "SepsisLabel"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    history = frame.loc[frame["ICULOS"] <= checkpoint].sort_values(
        "ICULOS",
        kind="mergesort",
    )
    if history["ICULOS"].isna().any():
        raise ValueError("ICULOS cannot contain missing values")
    if history["ICULOS"].duplicated().any():
        raise ValueError("Duplicate ICU hours detected")
    checkpoint_rows = history.loc[history["ICULOS"] == checkpoint]
    if len(checkpoint_rows) != 1:
        raise ValueError("Expected exactly one row at the requested checkpoint")

    hours = history["ICULOS"].to_numpy(dtype=float)
    last_values: dict[str, float] = {}
    last_ages: dict[str, float] = {}
    for feature in DYNAMIC_FEATURES:
        values = pd.to_numeric(history[feature], errors="coerce").to_numpy(
            dtype=float,
            copy=True,
        )
        values[~np.isfinite(values)] = np.nan
        observed_indices = np.flatnonzero(np.isfinite(values))
        if len(observed_indices):
            last_index = int(observed_indices[-1])
            last_values[feature] = float(values[last_index])
            last_ages[feature] = float(checkpoint - hours[last_index])
        else:
            last_values[feature] = np.nan
            last_ages[feature] = np.nan

    current = checkpoint_rows.iloc[0]
    values_by_name: dict[str, float] = {}
    for feature in DYNAMIC_FEATURES:
        is_hidden = feature in hidden
        values_by_name[f"{feature}_last"] = (
            np.nan if is_hidden else last_values[feature]
        )
        values_by_name[f"{feature}_observed"] = (
            0.0 if is_hidden else float(np.isfinite(last_values[feature]))
        )
        values_by_name[f"{feature}_age"] = (
            np.nan if is_hidden else last_ages[feature]
        )
    for feature in STATIC_FEATURES:
        value = pd.to_numeric(current[feature], errors="coerce")
        values_by_name[feature] = float(value) if pd.notna(value) else np.nan
    values_by_name["ICULOS"] = float(checkpoint)

    unknown = [name for name in feature_names if name not in values_by_name]
    if unknown:
        raise ValueError(f"Unknown model features: {unknown}")
    return np.asarray([values_by_name[name] for name in feature_names], dtype=float)


def mask_checkpoint_feature_vector(
    feature_vector: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    hidden_features: set[str] | frozenset[str],
) -> np.ndarray:
    """Mask the saved-model triplets for hidden measurements in a cached vector."""
    hidden = set(hidden_features)
    invalid_hidden = hidden - set(LAB_FEATURES)
    if invalid_hidden:
        raise ValueError(f"Invalid hidden laboratory features: {sorted(invalid_hidden)}")

    names = list(feature_names)
    vector = np.asarray(feature_vector, dtype=float)
    if vector.ndim != 1 or len(vector) != len(names):
        raise ValueError("Feature vector and feature names must be aligned")
    positions = {name: index for index, name in enumerate(names)}
    masked = vector.copy()
    for feature in hidden:
        expected = (
            f"{feature}_last",
            f"{feature}_observed",
            f"{feature}_age",
        )
        absent = [name for name in expected if name not in positions]
        if absent:
            raise ValueError(f"Missing model columns for hidden feature: {absent}")
        masked[positions[expected[0]]] = np.nan
        masked[positions[expected[1]]] = 0.0
        masked[positions[expected[2]]] = np.nan
    return masked
