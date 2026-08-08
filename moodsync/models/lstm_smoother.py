"""Bi-LSTM arc smoother + section detector (Module M2, steps 2-3).

The CNN emits noisy per-window (valence, arousal). The Bi-LSTM learns the
temporal dynamics of how emotion actually evolves and outputs a clean arc.
Song sections (verse / chorus / bridge / outro) are labelled from the second
derivative of the smoothed arousal (peaks->chorus, valleys->verse, sharp
rise->buildup, sharp drop->drop/outro).
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class ArcSmoother(nn.Module):
    def __init__(self, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2, hidden_size=hidden, num_layers=layers,
            batch_first=True, bidirectional=True,
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2, 2), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 2) noisy arc -> (B, T, 2) smoothed arc
        h, _ = self.lstm(x)
        return self.head(h)


def detect_sections(arc: np.ndarray) -> List[dict]:
    """Label sections from the smoothed arc.

    arc: (T, 2) valence, arousal in [-1,1]. Returns a list of
    {start, end, label} segments over window indices.
    """
    arc = np.asarray(arc, dtype=np.float32).reshape(-1, 2)
    T = len(arc)
    if T < 2:
        return [{"start": 0, "end": T, "label": "verse"}]
    arousal = arc[:, 1]
    # First and second derivatives.
    d1 = np.gradient(arousal)
    labels = []
    for i in range(T):
        a = arousal[i]
        slope = d1[i]
        if slope > 0.15:
            labels.append("buildup")
        elif slope < -0.15:
            labels.append("drop")
        elif a > 0.3:
            labels.append("chorus")
        elif a < -0.3:
            labels.append("verse")
        else:
            labels.append("bridge")
    # Compress consecutive identical labels into segments.
    segments = []
    start = 0
    for i in range(1, T + 1):
        if i == T or labels[i] != labels[start]:
            segments.append({"start": start, "end": i, "label": labels[start]})
            start = i
    # Mark the final segment as outro if arousal resolves downward.
    if segments and arousal[-1] < 0 and segments[-1]["label"] in ("verse", "drop", "bridge"):
        segments[-1]["label"] = "outro"
    return segments
