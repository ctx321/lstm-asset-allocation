"""FastAPI service implementing the required /predict contract.

Loads frozen artifacts once at startup (PyTorch CPU inference) and returns
portfolio weights computed from the 252-row request history only.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.ui import PAGE
from src.allocate import equal_weight
from src.data import ALL, ETF, load_market_data
from src.inference import Predictor, qqq_gated_weights, qqq_weights, spy_timer_weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

log = logging.getLogger("uvicorn.error")

predictor: Predictor | None = None
market_data: pd.DataFrame | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, market_data
    try:
        predictor = Predictor(MODEL_DIR)
        log.info("frozen LSTM artifacts loaded from %s", MODEL_DIR)
    except Exception as exc:  # pragma: no cover - only if artifacts are missing
        predictor = None
        log.error("could not load model artifacts: %s", exc)
    try:
        market_data = load_market_data()
        log.info("market data loaded: %s .. %s",
                 market_data.index[0].date(), market_data.index[-1].date())
    except Exception as exc:
        market_data = None
        log.error("could not load market data: %s", exc)
    yield


app = FastAPI(title="LSTM Asset Allocation API", version="1.0.0", lifespan=lifespan,
              docs_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "api" / "static")), name="static")


@app.get("/docs", include_in_schema=False)
def custom_docs():
    """Self-hosted Swagger UI (the default /docs loads assets from jsDelivr,
    which is often unreachable from mainland China)."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/static/swagger/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger/swagger-ui.css",
    )


class MarketRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: date
    QQQ: float = Field(gt=0)
    SPY: float = Field(gt=0)
    IWM: float = Field(gt=0)
    IEF: float = Field(gt=0)
    TLT: float = Field(gt=0)
    GLD: float = Field(gt=0)
    SHY: float = Field(gt=0)
    VIX: float = Field(ge=0)
    IRX: Optional[float] = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    as_of_date: date
    history: list[MarketRow] = Field(min_length=252, max_length=252)

    @model_validator(mode="after")
    def verify_chronology(self):
        dates = [row.date for row in self.history]
        if dates != sorted(set(dates)):
            raise ValueError("history must be unique and sorted oldest to newest")
        if self.as_of_date != dates[-1]:
            raise ValueError("as_of_date must equal final history date")
        return self


def history_frame(req: PredictionRequest) -> pd.DataFrame:
    rows = [row.model_dump() for row in req.history]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["IRX"] = pd.to_numeric(df["IRX"], errors="coerce")
    return df[ETF + ["VIX", "IRX"]]


def normalize_weights(w: dict[str, float]) -> dict[str, float]:
    out = {c: max(0.0, float(w.get(c, 0.0))) for c in ALL}
    total = sum(out.values())
    if total <= 0:
        return equal_weight()
    out = {c: v / total for c, v in out.items()}
    return out


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/", response_class=HTMLResponse)
def root():
    return PAGE


@app.get("/predict-ui")
def predict_ui(as_of_date: str | None = None, strategy: str = "lstm"):
    """Visualization helper: build the 252-row history from the released data,
    run the frozen model and return predictions + weights for the dashboard.

    strategy: lstm (default, assignment contract) | qqq_gated | qqq
    """
    if predictor is None or market_data is None:
        raise HTTPException(status_code=503, detail="model or market data not loaded")

    prices = market_data
    if as_of_date:
        try:
            target = pd.Timestamp(as_of_date)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid as_of_date")
        idx = int(prices.index.searchsorted(target, side="right") - 1)
        if idx < 0:
            raise HTTPException(status_code=400, detail="date is before the data start")
    else:
        idx = len(prices) - 1

    if idx < 251:
        raise HTTPException(status_code=400, detail="not enough history before this date")

    hist = prices.iloc[idx - 251: idx + 1]
    actual = hist.index[-1]
    try:
        r_hat = predictor.predict_returns(hist)
        if strategy == "qqq":
            weights = normalize_weights(qqq_weights())
        elif strategy == "qqq_gated":
            weights = normalize_weights(qqq_gated_weights(r_hat))
        else:
            strategy = "lstm"
            weights = normalize_weights(predictor.predict_weights(hist))
    except Exception as exc:
        log.exception("dashboard inference failed: %s", exc)
        r_hat = np.zeros(len(ETF))
        weights = equal_weight()

    return {
        "as_of_date": str(actual.date()),
        "strategy": strategy,
        "min_date": str(prices.index[251].date()),
        "max_date": str(prices.index[-1].date()),
        "predicted_returns": {c: round(float(v), 6) for c, v in zip(ETF, r_hat)},
        "weights": weights,
        "last_close": {c: round(float(hist[c].iloc[-1]), 4) for c in ETF},
        "dates": [str(d.date()) for d in hist.index],
        "history": {c: [round(float(v), 4) for v in hist[c].values] for c in ETF},
    }


@app.get("/predict")
def predict_get():
    return {
        "detail": "This endpoint only accepts POST JSON. Open /docs to try it interactively, "
                  "or run: python tools/api_smoke_test.py --url http://127.0.0.1:8000/predict",
        "request_example": "examples/sample_request_252.json",
    }


@app.post("/predict")
def predict(req: PredictionRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="frozen model artifacts are not loaded")
    df = history_frame(req)
    try:
        weights = predictor.predict_weights(df)
    except Exception as exc:  # deterministic fallback keeps the contract alive
        log.exception("inference failed, falling back to equal weight: %s", exc)
        weights = equal_weight()
    return {"weights": normalize_weights(weights)}


@app.post("/predict-alternative")
def predict_alternative(req: PredictionRequest):
    df = history_frame(req)
    try:
        weights = spy_timer_weights(df)
    except Exception as exc:
        log.exception("alternative strategy failed, falling back to equal weight: %s", exc)
        weights = equal_weight()
    return {"weights": normalize_weights(weights)}


@app.post("/predict-qqq")
def predict_qqq(req: PredictionRequest):
    """100% QQQ buy-and-hold weights (benchmark for the profit comparison).

    Kept separate from /predict so the assignment's LSTM contract stays intact.
    """
    return {"weights": normalize_weights(qqq_weights())}
