"""CNN mood classifier (Module M2, step 1).

Input : mel-spectrogram (1 x n_mels x mel_frames) per 3-second window.
Output: (valence, arousal) in [-1, 1] via two independent heads with tanh.
Loss  : MSE(valence) + MSE(arousal). Target R2 >= 0.65 on held-out data.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MoodCNN(nn.Module):
    """Mood regressor over a (in_channels x n_mels x mel_frames) window.

    `in_channels=2` feeds mel (energy/timbre) and chroma (harmony) together —
    mel alone is close to blind to tonality, which is what valence depends on.
    `dropout` regularises the heads; on DEAM the training loss falls ~12x while
    held-out valence barely moves, so the gap is overfitting, not capacity.
    """

    def __init__(self, n_mels: int = 128, filters=(32, 64, 128, 256),
                 in_channels: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_channels = int(in_channels)
        chans = [self.in_channels, *filters]
        self.features = nn.Sequential(
            *[_conv_block(chans[i], chans[i + 1]) for i in range(len(filters))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pool -> (C,1,1)
        feat = filters[-1]
        self.valence_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Tanh()
        )
        self.arousal_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(feat, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:              # (B, H, W) -> (B, 1, H, W)
            x = x.unsqueeze(1)
        h = self.features(x)
        h = self.pool(h).flatten(1)
        v = self.valence_head(h)
        a = self.arousal_head(h)
        return torch.cat([v, a], dim=1)   # (B, 2)


def mood_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss
    return mse(pred[:, 0], target[:, 0]) + mse(pred[:, 1], target[:, 1])
