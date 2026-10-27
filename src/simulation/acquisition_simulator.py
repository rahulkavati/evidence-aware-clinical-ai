
"""Retrospective clinical information-acquisition simulator."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.data.build_temporal_features import (
    LAB_FEATURES,
    build_checkpoint_feature_vector,
)


@dataclass
class AcquisitionResult:
    feature: str
    risk_before: float
    risk_after: float
    acquired_count: int


class AcquisitionSimulator:
    def __init__(
        self,
        patient: pd.DataFrame,
        checkpoint: int,
        model,
        calibrator=None,
        hidden_features=None,
    ):
        self.checkpoint = checkpoint
        self.model = model
        self.calibrator = calibrator

        # Keep only historical observations.
        self.history = patient.loc[
            patient["ICULOS"] <= checkpoint
        ].copy()

        if not (self.history["ICULOS"] == checkpoint).any():
            raise ValueError("Patient not eligible at checkpoint")

        self.hidden = set(hidden_features or [])

        invalid = self.hidden - set(LAB_FEATURES)
        if invalid:
            raise ValueError(f"Invalid laboratory features: {invalid}")

        self.acquired = []
        self.model_features = list(self.model.feature_names_in_)
        self._feature_vector_cache = {}
        self._risk_cache = {}

        # Ground truth is retained privately by the simulator.
        self.original_history = self.history.copy()

    def visible_history(self):
        """Return history with hidden laboratory features masked."""
        visible = self.original_history.copy()

        for feature in self.hidden:
            visible[feature] = np.nan

        return visible

    def eligible_features(self):
        """Only hidden features with historically observed values."""
        return sorted([
            feature
            for feature in self.hidden
            if self.original_history[feature].notna().any()
        ])

    def predict_risk(self):
        state = frozenset(self.hidden)
        if state in self._risk_cache:
            return self._risk_cache[state]
        if state not in self._feature_vector_cache:
            self._feature_vector_cache[state] = build_checkpoint_feature_vector(
                self.original_history,
                self.checkpoint,
                self.model_features,
                hidden_features=state,
            )
        current = pd.DataFrame(
            [self._feature_vector_cache[state]],
            columns=self.model_features,
        )
        raw = float(self.model.predict_proba(current)[0, 1])

        if self.calibrator is None:
            self._risk_cache[state] = raw
            return raw

        # Sigmoid calibrator fitted on log-odds.
        eps = 1e-7
        clipped = np.clip(raw, eps, 1 - eps)
        logit = np.log(clipped / (1 - clipped))

        calibrated = float(
            self.calibrator.predict_proba(
                np.array([[logit]])
            )[0, 1]
        )
        self._risk_cache[state] = calibrated
        return calibrated

    def acquire(self, feature: str):
        if feature not in self.eligible_features():
            raise ValueError(
                f"Feature is unavailable or already acquired: {feature}"
            )

        before = self.predict_risk()

        self.hidden.remove(feature)
        self.acquired.append(feature)

        after = self.predict_risk()

        return AcquisitionResult(
            feature=feature,
            risk_before=before,
            risk_after=after,
            acquired_count=len(self.acquired),
        )
