"""Local backtest engine: 21-day rebalancing, T+1 execution, zero costs.

Decisions use only the history up to (and including) the decision date; the
resulting weights are executed at the next trading day's adjusted close.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from src.data import ALL, ETF

STEP = 21
INITIAL = 10_000.0


def backtest(prices: pd.DataFrame, start: int,
             weight_fn: Callable[[pd.DataFrame], dict[str, float]],
             step: int = STEP, initial: float = INITIAL):
    n = len(prices)
    px = prices[ETF].values
    cash = initial
    shares = np.zeros(len(ETF))
    value = np.full(n, np.nan)
    weight_history = []
    turnover = []

    decisions = list(range(start, n - 1, step))
    if not decisions:
        raise ValueError("window too short")

    for j, d in enumerate(decisions):
        e = d + 1
        p = px[e]
        val = cash + shares @ p
        w = weight_fn(prices.iloc[: d + 1])
        wv = np.array([w[c] for c in ETF], dtype=float)
        wc = float(w["CASH"])
        s = wv.sum() + wc
        if s <= 0:
            wv, wc = np.zeros(len(ETF)), 1.0
        else:
            wv, wc = wv / s, wc / s
        new8 = np.append(wv, wc)
        cur = np.append(shares * p / val, cash / val) if val > 0 else np.zeros(len(ALL))
        if j > 0:
            turnover.append(float(np.abs(cur - new8).sum()))
        shares = wv * val / p
        cash = wc * val
        weight_history.append((prices.index[d], new8.copy()))
        nxt = decisions[j + 1] + 1 if j + 1 < len(decisions) else n
        for i in range(e, nxt):
            value[i] = cash + shares @ px[i]

    first = decisions[0] + 1
    value[first - 1] = initial
    wdf = pd.DataFrame({d: w for d, w in weight_history}).T
    wdf.columns = ALL
    series = pd.Series(value[first - 1:], index=prices.index[first - 1:])
    return series, wdf, turnover


def metrics(vals: pd.Series) -> dict:
    ret = vals.pct_change().dropna()
    years = len(vals) / 252
    total = vals.iloc[-1] / vals.iloc[0] - 1
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if vol > 0 else 0.0
    dd = float((1 - vals / vals.cummax()).max())
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": dd,
        "final_value": float(vals.iloc[-1]),
    }
