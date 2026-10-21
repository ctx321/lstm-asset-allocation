"""Task D: local backtest of the LSTM strategy vs QQQ and Equal Weight.

Uses exactly the course conventions: USD 10,000, rebalance every 21 trading
days, decisions after date-t close executed at t+1 close, zero costs, long-only.

Outputs (outputs/):
  api_backtest_metrics.csv / .json   - performance table
  api_backtest.png                   - cumulative portfolio value
  api_backtest_weights.png           - LSTM weights through time
  api_backtest_yearly.csv            - calendar-year returns 2021-2023
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.backtest import backtest, metrics  # noqa: E402
from src.data import ETF, load_market_data  # noqa: E402
from src.inference import Predictor, spy_timer_weights  # noqa: E402

OUT = ROOT / "outputs"
WINDOW_START = "2021-01-04"


def qqq_weights(_):
    w = {c: 0.0 for c in ETF}
    w["QQQ"] = 1.0
    w["CASH"] = 0.0
    return w


def ew_weights(_):
    w = {c: 1.0 / len(ETF) for c in ETF}
    w["CASH"] = 0.0
    return w


def main():
    OUT.mkdir(exist_ok=True)
    prices = load_market_data()
    predictor = Predictor(ROOT / "models")

    def lstm_weights(history: pd.DataFrame):
        return predictor.predict_weights(history.iloc[-252:])

    start = int(np.searchsorted(prices.index, pd.Timestamp(WINDOW_START)))
    strategies = {
        "QQQ Buy&Hold": qqq_weights,
        "Equal Weight(7)": ew_weights,
        "LSTM + Allocation": lstm_weights,
        "SPY 200d Timer": lambda h: spy_timer_weights(h.iloc[-252:]),
    }

    results = {}
    curves = {}
    weights_by_strategy = {}
    for name, fn in strategies.items():
        vals, wdf, turn = backtest(prices, start, fn)
        m = metrics(vals)
        m["turnover_per_year"] = float(np.mean(turn) * 252 / 21) if turn else 0.0
        m["cash_avg"] = float(wdf["CASH"].mean())
        results[name] = m
        curves[name] = vals
        weights_by_strategy[name] = wdf

    print(f"Backtest {curves['QQQ Buy&Hold'].index[0].date()} -> "
          f"{curves['QQQ Buy&Hold'].index[-1].date()} "
          f"({len(curves['QQQ Buy&Hold'])} days)")
    header = f"{'strategy':<20}{'total':>9}{'CAGR':>8}{'vol':>8}{'Sharpe':>8}{'maxDD':>8}{'final$':>11}{'cash':>7}"
    print(header)
    for name in strategies:
        m = results[name]
        print(f"{name:<20}{m['total_return']*100:>8.1f}%{m['cagr']*100:>7.1f}%"
              f"{m['ann_vol']*100:>7.1f}%{m['sharpe']:>8.2f}{m['max_drawdown']*100:>7.1f}%"
              f"{m['final_value']:>11,.0f}{m['cash_avg']*100:>6.0f}%")

    pd.DataFrame(results).T.to_csv(OUT / "api_backtest_metrics.csv")

    # cumulative value
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    for name, vals in curves.items():
        ax.plot(vals.index, vals.values, label=name, lw=1.9)
    ax.set_ylabel("portfolio value (USD)")
    ax.set_title("LSTM API strategy vs benchmarks, 2021–2023 (21-day rebalancing, zero costs)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "api_backtest.png")
    plt.close(fig)

    # weights over time (LSTM strategy)
    w = weights_by_strategy["LSTM + Allocation"]
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    ax.stackplot(w.index, [w[c].values for c in ETF + ["CASH"]],
                 labels=ETF + ["CASH"], alpha=0.9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("portfolio weight")
    ax.set_title("LSTM allocation weights through time, 2021–2023 (21-day rebalancing)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "api_backtest_weights.png")
    plt.close(fig)

    # calendar-year returns
    years = [2021, 2022, 2023]
    yearly = {}
    for name, vals in curves.items():
        row = {}
        for y in years:
            v = vals[vals.index.year == y]
            base_y = vals[vals.index < f"{y}-01-01"]
            base = base_y.iloc[-1] if len(base_y) else vals.iloc[0]
            row[str(y)] = v.iloc[-1] / base - 1
        yearly[name] = row
    ydf = pd.DataFrame(yearly).T * 100
    ydf["total"] = ((1 + ydf[[str(y) for y in years]] / 100).prod(axis=1) - 1) * 100
    ydf.round(2).to_csv(OUT / "api_backtest_yearly.csv")
    print("\nCalendar-year returns (%):")
    print(ydf.round(1).to_string())

    with open(OUT / "api_backtest_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\noutputs written to", OUT)


if __name__ == "__main__":
    main()
