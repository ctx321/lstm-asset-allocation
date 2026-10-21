"""Train the LSTM return forecaster and freeze artifacts for the API.

Chronological split with 21-day purge:
  train 2005-2017, validation 2018-2020, local test 2021-2023.
Scalers are fitted on the training samples only.

Usage:
    python training/train.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.data import ETF, load_market_data  # noqa: E402
from src.features import WINDOW, build_windows, compute_features, feature_names  # noqa: E402
from src.model import LSTMReturnModel  # noqa: E402

TRAIN_END = pd.Timestamp("2018-01-01")
VAL_END = pd.Timestamp("2021-01-01")
SEED = 42


def build_dataset(prices: pd.DataFrame):
    feats = compute_features(prices)
    ends = build_windows(feats, WINDOW)
    n = len(prices)

    X, Y, dates, label_end = [], [], [], []
    for i in ends:
        if i + 21 >= n:
            continue
        X.append(feats.iloc[i - WINDOW + 1: i + 1].values.astype(np.float32))
        Y.append((prices[ETF].iloc[i + 21] / prices[ETF].iloc[i] - 1.0).values.astype(np.float32))
        dates.append(prices.index[i])
        label_end.append(prices.index[i + 21])

    dates = pd.DatetimeIndex(dates)
    label_end = pd.DatetimeIndex(label_end)
    X = np.stack(X)
    Y = np.stack(Y)
    return X, Y, dates, label_end


def split_masks(dates, label_end):
    train = (dates < TRAIN_END) & (label_end < TRAIN_END)
    val = (dates >= TRAIN_END) & (dates < VAL_END) & (label_end < VAL_END)
    test = dates >= VAL_END
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    prices = load_market_data()
    X, Y, dates, label_end = build_dataset(prices)
    train_m, val_m, test_m = split_masks(dates, label_end)
    print(f"samples: train={train_m.sum()} val={val_m.sum()} test={test_m.sum()} "
          f"features={X.shape[2]} window={WINDOW}")
    print(f"train dates: {dates[train_m].min().date()} .. {dates[train_m].max().date()}")
    print(f"val dates  : {dates[val_m].min().date()} .. {dates[val_m].max().date()}")
    print(f"test dates : {dates[test_m].min().date()} .. {dates[test_m].max().date()}")

    Xtr = X[train_m]
    x_mean = Xtr.reshape(-1, X.shape[2]).mean(axis=0)
    x_std = Xtr.reshape(-1, X.shape[2]).std(axis=0) + 1e-8

    Ytr = Y[train_m]
    y_mean = Ytr.mean(axis=0)
    y_std = Ytr.std(axis=0) + 1e-8

    def scaled(Xa):
        return (Xa - x_mean) / x_std

    def scaled_y(Ya):
        return (Ya - y_mean) / y_std

    model = LSTMReturnModel(X.shape[2], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.from_numpy(scaled(Xtr)).float().to(device)
    Ytr_t = torch.from_numpy(scaled_y(Ytr)).float().to(device)
    Xva_t = torch.from_numpy(scaled(X[val_m])).float().to(device)
    Yva_t = torch.from_numpy(scaled_y(Y[val_m])).float().to(device)

    n_tr = len(Xtr_t)
    best_val = float("inf")
    best_state = None
    wait = 0
    train_hist, val_hist = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        epoch_loss = 0.0
        for i in range(0, n_tr, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * len(idx)
        train_loss = epoch_loss / n_tr

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xva_t), Yva_t).item())
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stopping at epoch {epoch} (best val {best_val:.5f})")
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(Xa):
        with torch.no_grad():
            out = model(torch.from_numpy(scaled(Xa)).float().to(device)).cpu().numpy()
        return out * y_std + y_mean

    def mse(a, b):
        return float(np.mean((a - b) ** 2))

    results = {}
    for name, mask in (("val", val_m), ("test", test_m)):
        pred = predict(X[mask])
        actual = Y[mask]
        zero = np.zeros_like(actual)
        mean_base = np.tile(y_mean, (len(actual), 1))
        results[name] = {
            "n": int(mask.sum()),
            "mse_lstm": round(mse(pred, actual), 8),
            "mse_zero": round(mse(zero, actual), 8),
            "mse_train_mean": round(mse(mean_base, actual), 8),
            "mae_lstm": round(float(np.mean(np.abs(pred - actual))), 8),
            "mae_zero": round(float(np.mean(np.abs(actual))), 8),
        }
        print(f"{name}: MSE lstm={results[name]['mse_lstm']:.3e} "
              f"zero={results[name]['mse_zero']:.3e} "
              f"train_mean={results[name]['mse_train_mean']:.3e}")

    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": {
            "n_features": int(X.shape[2]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "n_assets": len(ETF),
        },
    }, out / "lstm_model.pt")

    scaler = {
        "features": feature_names(),
        "window": WINDOW,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "seed": SEED,
        "split": {"train_end": str(TRAIN_END.date()), "val_end": str(VAL_END.date())},
    }
    (out / "scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    meta = {
        "device": str(device),
        "epochs_run": len(train_hist),
        "best_val_loss_scaled": round(best_val, 6),
        "train_loss_curve": [round(v, 6) for v in train_hist],
        "val_loss_curve": [round(v, 6) for v in val_hist],
        "metrics": results,
        "samples": {"train": int(train_m.sum()), "val": int(val_m.sum()), "test": int(test_m.sum())},
        "settings": vars(args),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved artifacts to", out)


if __name__ == "__main__":
    main()
