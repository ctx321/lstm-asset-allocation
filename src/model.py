"""LSTM return forecaster: input (60, F) -> 7 assets x 21-day return."""

import torch
import torch.nn as nn


class LSTMReturnModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.2,
                 n_assets: int = 7):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_assets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.head(self.drop(self.norm(h)))
