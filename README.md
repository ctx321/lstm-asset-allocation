# LSTM for Dynamic Multi-Asset Allocation

Course project: train an LSTM return forecaster on the released market data,
convert forecasts into portfolio weights, backtest against QQQ / Equal Weight,
and serve the frozen strategy through a FastAPI `POST /predict` contract.

- **Permanent demo (static, browser inference):** https://ctx321.github.io/handwriting-demo/stock/
- **Presentation (slides):** https://ctx321.github.io/handwriting-demo/stock/presentation.html
- **Local API dashboard:** http://127.0.0.1:8000/ (after starting the server)

## Project structure

```
├─ src/
│  ├─ data.py            market data loading
│  ├─ features.py        73 causal features, 60-day window
│  ├─ model.py           LSTM(73→32) + LayerNorm + Linear(7)
│  ├─ allocate.py        risk-adjusted scoring + cash gate + 40% cap
│  ├─ inference.py       frozen-artifact inference (API & backtest)
│  └─ backtest.py        21-day rebalance, T+1 execution, zero costs
├─ training/train.py     chronological split, purge, training loop, artifacts
├─ api/app.py            FastAPI: /predict, /predict-alternative (+ demo UI)
├─ api/static/swagger/   self-hosted Swagger UI (works offline)
├─ backtest_local.py     Task D comparison + charts
├─ models/               frozen lstm_model.pt, scaler.json, train_meta.json
├─ data/                 market_data.csv (released data) + README_DATA.md
├─ examples/ tools/ tests/  contract fixtures, smoke test, unit tests
└─ outputs/              backtest metrics, charts, compliance report
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`data/market_data.csv` is included; set `MARKET_DATA_CSV` to use another copy.

## Train (frozen artifacts)

```powershell
python training\train.py --epochs 100
```

Split 2005–2017 / 2018–2020 / 2021–2023, 21-day purge, scalers fitted on the
training window only, seed 42. Saves `models/lstm_model.pt`, `scaler.json`,
`train_meta.json` (loss curves + metrics). Selected configuration: hidden 32,
dropout 0.5, weight decay 5e-3, lr 2e-4, early stopping at epoch 46.

## Local backtest (Task D)

```powershell
python backtest_local.py
```

Outputs in `outputs/`: cumulative value plot, weights-through-time plot,
yearly returns, metrics CSV/JSON. See `outputs/assignment_report.md` for the
requirement-by-requirement compliance report.

## Run the API

```powershell
py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- `GET /` — visual dashboard (date picker → predicted returns, weights, charts)
- `GET /docs` — Swagger UI (self-hosted, works without external CDNs)
- `POST /predict` — strict contract: 252 rows in, 8 weights out
- `POST /predict-alternative` — SPY 200-day MA timer (optional Task C)
- `POST /predict-qqq` — QQQ-only benchmark (comparison only; `/predict` stays LSTM)

Contract checks:

```powershell
python tools\api_smoke_test.py --url http://127.0.0.1:8000/predict
python -m pytest tests -q
```

## Results (public local test 2021–2023)

Prediction quality (average 21-day return MSE):

| Model | Validation | Test |
|---|---|---|
| LSTM | **2.423e-3** | 2.269e-3 |
| Zero forecast | 2.542e-3 | 2.103e-3 |
| Train-mean | 2.432e-3 | 2.122e-3 |

Backtest (same conventions: USD 10,000, 21-day rebalance, T+1, zero costs):

| Strategy | Total | Vol | Sharpe | Max DD |
|---|---|---|---|---|
| QQQ Buy & Hold | +33.8% | 23.8% | 0.53 | -35.1% |
| Equal Weight (7) | +4.3% | 10.6% | 0.18 | -21.7% |
| **LSTM + Allocation** | **+13.9%** | 12.0% | 0.42 | -21.2% |
| SPY 200d Timer | +17.9% | 12.8% | 0.50 | -27.8% |

The LSTM has no test-set predictive edge but its allocation beats Equal Weight;
lower prediction loss does not automatically imply a better portfolio (discussed
in the report). Past performance is not a guarantee.

## Integrity notes

- Only the released 2005-01-03..2023-12-29 dataset is used; no post-cutoff data.
- Chronological splits, 21-day purge at boundaries, scalers frozen from training.
- Inference never trains or updates weights; the model loads once at startup.
- Random seed 42 is recorded in `models/scaler.json` / `train_meta.json`.

*Simulation-only academic project; not financial advice.*
