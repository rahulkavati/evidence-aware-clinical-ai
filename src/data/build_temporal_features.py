
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

    result = pd.DataFrame(index=df.index)

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

        result[f"{feature}_last"] = last_values
        result[f"{feature}_observed"] = (
            last_seen_hours.notna().astype("int8")
        )
        result[f"{feature}_age"] = age

    for feature in STATIC_FEATURES:
        result[feature] = pd.to_numeric(
            df[feature], errors="coerce"
        )

    result["ICULOS"] = df["ICULOS"]

    # Target retained separately for supervised training.
    result["SepsisLabel"] = df["SepsisLabel"]

    return result.reset_index(drop=True)
