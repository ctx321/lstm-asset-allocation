"""Causal feature engineering shared by training, backtesting and the API.

All features at date t use observations up to and including t only.
The longest lookback is the 12-1 momentum (shift 232), so a 252-row history
is enough to compute the features for the final row.
"""

import numpy as np
import pandas as pd

from src.data import ETF

WINDOW = 60
FEATURE_MAX_LOOKBACK = 232


def feature_names() -> list[str]:
    names: list[str] = []
    for c in ETF:
        names += [f"{c}_{s}" for s in
                  ("ret1", "ret5", "ret20", "mom60", "mom12_1",
                   "vol20", "vol60", "ma20r", "ma60r", "dd60")]
    names += ["vix_log", "vix_chg5", "irx"]
    return names


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the feature frame for every row of df (NaN during warm-up)."""
    out: dict[str, pd.Series] = {}
    rets = df[ETF].pct_change()
    for c in ETF:
        p = df[c]
        r = rets[c]
        out[f"{c}_ret1"] = p.pct_change(1)
        out[f"{c}_ret5"] = p.pct_change(5)
        out[f"{c}_ret20"] = p.pct_change(20)
        out[f"{c}_mom60"] = p.pct_change(60)
        out[f"{c}_mom12_1"] = p.shift(21) / p.shift(FEATURE_MAX_LOOKBACK) - 1.0
        out[f"{c}_vol20"] = r.rolling(20).std()
        out[f"{c}_vol60"] = r.rolling(60).std()
        out[f"{c}_ma20r"] = p / p.rolling(20).mean() - 1.0
        out[f"{c}_ma60r"] = p / p.rolling(60).mean() - 1.0
        out[f"{c}_dd60"] = p / p.rolling(60).max() - 1.0
    out["vix_log"] = np.log(df["VIX"].clip(lower=1e-6))
    out["vix_chg5"] = np.log(df["VIX"].clip(lower=1e-6) /
                             df["VIX"].shift(5).clip(lower=1e-6))
    out["irx"] = df["IRX"].ffill()
    return pd.DataFrame(out, index=df.index)[feature_names()]


def build_windows(features: pd.DataFrame, window: int = WINDOW):
    """Return sample end indices whose full feature window is valid."""
    valid = features.notna().all(axis=1).values
    ends = []
    for i in range(window - 1, len(features)):
        if valid[i - window + 1: i + 1].all():
            ends.append(i)
    return np.asarray(ends, dtype=int)
