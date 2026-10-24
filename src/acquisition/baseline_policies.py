
"""Baseline information-acquisition policies."""

import random


def random_policy(eligible_features, rng=None):
    if not eligible_features:
        return None

    rng = rng or random.Random(42)
    return rng.choice(list(eligible_features))


def fixed_order_policy(eligible_features, priority=None):
    if not eligible_features:
        return None

    priority = priority or [
        "Lactate",
        "WBC",
        "Creatinine",
        "Platelets",
        "BUN",
        "Glucose",
    ]

    for feature in priority:
        if feature in eligible_features:
            return feature

    return sorted(eligible_features)[0]
