
"""Training-derived feature-level value-of-information baseline."""

from dataclasses import dataclass
import numpy as np


@dataclass
class VOIConfig:
    cost_weight: float = 0.0
    shrinkage: float = 100.0
    min_examples: int = 30


class FeatureVOIPolicy:
    def __init__(self, config=None):
        self.config = config or VOIConfig()
        self.values = {}
        self.counts = {}

    def fit(self, acquisition_records):
        """
        Fit from TRAINING-DERIVED acquisition outcomes only.

        Each record:
        {
            "feature": str,
            "loss_before": float,
            "loss_after": float
        }
        """
        grouped = {}

        for record in acquisition_records:
            feature = record["feature"]
            benefit = (
                record["loss_before"]
                - record["loss_after"]
            )

            if not np.isfinite(benefit):
                raise ValueError("Non-finite acquisition benefit")

            grouped.setdefault(feature, []).append(benefit)

        for feature, benefits in grouped.items():
            n = len(benefits)
            self.counts[feature] = n

            if n < self.config.min_examples:
                self.values[feature] = 0.0
                continue

            mean_benefit = float(np.mean(benefits))

            # Shrink noisy feature estimates toward zero.
            self.values[feature] = (
                n / (n + self.config.shrinkage)
            ) * mean_benefit

        return self

    def score(self, feature, cost=1.0):
        benefit = self.values.get(feature, 0.0)
        return benefit - self.config.cost_weight * cost

    def select(self, eligible_features, costs=None):
        """Select the highest positive estimated net benefit."""
        costs = costs or {}

        candidates = sorted(set(eligible_features))

        if not candidates:
            return None

        ranked = [
            (
                self.score(
                    feature,
                    costs.get(feature, 1.0)
                ),
                feature,
            )
            for feature in candidates
        ]

        best_score, best_feature = max(
            ranked,
            key=lambda item: (item[0], item[1])
        )

        if best_score <= 0:
            return None

        return best_feature
