"""Optional GRU/TCN sequence baselines and missingness channels.

PyTorch is intentionally optional: the established XGBoost baseline remains
the default model, while these classes provide a reproducible extension point
for sequence experiments when the dependency is installed.
"""

from __future__ import annotations

import numpy as np


def append_missingness_channels(sequence: np.ndarray) -> np.ndarray:
    """Append one binary channel for every finite input value."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("sequence must have shape (batch, time, features)")
    observed = np.isfinite(values).astype(np.float32)
    filled = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([filled, observed], axis=-1)


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "GRU/TCN models require optional dependency 'torch'; "
            "the XGBoost baseline does not."
        ) from error
    return torch, nn


class GRUSequenceClassifier:
    """Small sequence classifier with a sigmoid output."""

    def __init__(self, input_features: int, hidden_size: int = 64):
        torch, nn = _require_torch()
        self.torch = torch
        self.model = nn.GRU(
            input_size=input_features, hidden_size=hidden_size, batch_first=True
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, sequence):
        _, hidden = self.model(sequence)
        return self.head(hidden[-1]).squeeze(-1)


class CausalTCNClassifier:
    """Compact causal temporal-convolution classifier."""

    def __init__(self, input_features: int, channels: int = 32, kernel_size: int = 3):
        torch, nn = _require_torch()
        self.torch = torch
        padding = kernel_size - 1
        self.conv = nn.Conv1d(input_features, channels, kernel_size, padding=padding)
        self.head = nn.Linear(channels, 1)
        self.padding = padding

    def forward(self, sequence):
        x = sequence.transpose(1, 2)
        x = self.conv(x)
        if self.padding:
            x = x[:, :, :-self.padding]
        return self.head(x[:, :, -1]).squeeze(-1)
