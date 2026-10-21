"""Market data loading for the LSTM asset-allocation project.

The repository ships the released dataset in `data/market_data.csv`.
`MARKET_DATA_CSV` can point at another copy.
"""

import os
from pathlib import Path

import pandas as pd

ETF = ["QQQ", "SPY", "IWM", "IEF", "TLT", "GLD", "SHY"]
ALL = ETF + ["CASH"]

DEFAULT_DATA = (
    r"D:\LSTM_Asset_Allocation_Student_Bundle"
    r"\LSTM_Asset_Allocation_Student_Package\data\market_data.csv"
)


def data_path() -> Path:
    env = os.environ.get("MARKET_DATA_CSV")
    if env:
        return Path(env)
    local = Path(__file__).resolve().parents[1] / "data" / "market_data.csv"
    if local.exists():
        return local
    return Path(DEFAULT_DATA)


def load_market_data(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else data_path()
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    missing = [c for c in ETF + ["VIX", "IRX"] if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return df
