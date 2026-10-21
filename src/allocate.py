"""Allocation rule: risk-adjusted scores + cash gate + 40% cap + de-fragment.

Stateless by construction (the API contract gives only a 252-row history, so
no dependence on previous weights / turnover smoothing is possible).
"""

import numpy as np

from src.data import ALL, ETF

VOL_FLOOR = 0.005
CAP = 0.40
MIN_WEIGHT = 0.03


def allocate(r_hat: np.ndarray, vol60: np.ndarray,
             cap: float = CAP, min_weight: float = MIN_WEIGHT) -> dict[str, float]:
    r_hat = np.asarray(r_hat, dtype=float).reshape(-1)
    vol60 = np.asarray(vol60, dtype=float).reshape(-1)
    score = np.maximum(r_hat, 0.0) / np.maximum(vol60, VOL_FLOOR)

    if score.sum() <= 1e-12:
        w = {c: 0.0 for c in ETF}
        w["CASH"] = 1.0
        return w

    w = score / score.sum()
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        free = ~over
        if not free.any():
            break
        free_sum = w[free].sum()
        if free_sum <= 1e-12:
            break
        w[free] += excess * w[free] / free_sum

    w[w < min_weight] = 0.0
    cash = max(0.0, 1.0 - w.sum())
    weights = np.append(w, cash)
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    return {c: float(v) for c, v in zip(ALL, weights)}


def equal_weight() -> dict[str, float]:
    w = {c: 1.0 / len(ETF) for c in ETF}
    w["CASH"] = 0.0
    return w
