"""Frozen-artifact inference shared by the API and the local backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.allocate import allocate
from src.data import ETF
from src.features import compute_features
from src.model import LSTMReturnModel


class Predictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.scaler = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
        ckpt = torch.load(model_dir / "lstm_model.pt", map_location="cpu")
        self.model = LSTMReturnModel(**ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.x_mean = np.asarray(self.scaler["x_mean"], dtype=np.float64)
        self.x_std = np.asarray(self.scaler["x_std"], dtype=np.float64)
        self.y_mean = np.asarray(self.scaler["y_mean"], dtype=np.float64)
        self.y_std = np.asarray(self.scaler["y_std"], dtype=np.float64)
        self.window = int(self.scaler["window"])

    @torch.no_grad()
    def predict_returns(self, history: pd.DataFrame) -> np.ndarray:
        feats = compute_features(history)
        win = feats.iloc[-self.window:].values.astype(np.float64)
        if win.shape[0] < self.window:
            raise ValueError("not enough history to build the feature window")
        if not np.isfinite(win).all():
            win = np.where(np.isfinite(win), win, self.x_mean)
            if not np.isfinite(win).all():
                raise ValueError("features contain non-finite values")
        x = (win - self.x_mean) / self.x_std
        tensor = torch.from_numpy(x[None, :, :]).float()
        pred_scaled = self.model(tensor).numpy()[0]
        return pred_scaled * self.y_std + self.y_mean

    def predict_weights(self, history: pd.DataFrame) -> dict[str, float]:
        r_hat = self.predict_returns(history)
        vol60 = history[ETF].pct_change().iloc[-60:].std().values
        return allocate(r_hat, vol60)


def spy_timer_weights(history: pd.DataFrame) -> dict[str, float]:
    px = history["SPY"].dropna()
    w = {c: 0.0 for c in ETF}
    w["CASH"] = 0.0
    if len(px) >= 200 and px.iloc[-1] > px.iloc[-200:].mean():
        w["SPY"] = 1.0
    else:
        w["CASH"] = 1.0
    return w


def qqq_weights() -> dict[str, float]:
    """100% QQQ buy-and-hold (a benchmark, not the required LSTM model)."""
    w = {c: 0.0 for c in ETF}
    w["QQQ"] = 1.0
    w["CASH"] = 0.0
    return w


def qqq_gated_weights(r_hat: np.ndarray) -> dict[str, float]:
    """LSTM-gated QQQ: hold QQQ when the model's QQQ forecast is positive,
    otherwise move to cash. Keeps the LSTM in the loop while concentrating on QQQ."""
    w = {c: 0.0 for c in ETF}
    w["CASH"] = 0.0
    if float(r_hat[0]) > 0:
        w["QQQ"] = 1.0
    else:
        w["CASH"] = 1.0
    return w
