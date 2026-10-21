# LSTM Asset Allocation — Assignment Compliance Report

This report maps the implementation to the requirements in
`Project_Instructions.pdf` (5 pages; content identical to the bundled
`Project_Instructions.md`) and records the reproducible results.

---

## Task A — Features and LSTM training

**A1. Data exploration.** `static_site/assets/data.png` (ETF growth) and the
traditional-strategy exploration (`../outputs/strategy_metrics.csv`) cover
returns, volatility, drawdowns and regime differences (2022 bond/equity
drawdown, 2021/2023 rallies).

**A2. Features (73 per day, 60-day window, causally computed).**
Per ETF: 1/5/20-day returns; 60-day and 12-1 momentum; 20/60-day rolling
volatility; price/MA20−1 and price/MA60−1; 60-day drawdown. Market: log VIX,
5-day VIX change, IRX (forward-filled). Scalers are fitted on training samples
only and frozen in `models/scaler.json`.

**A3. Target.** `P[t+21]/P[t] - 1` for the 7 ETFs; future returns are never used
in the features. Samples: 2,960 train / 735 validation / 732 test; labels at
split boundaries are purged by 21 trading days.

**A4. LSTM with a hand-written training loop.** `src/model.py` defines
`LSTM(73→32) → LayerNorm → Dropout(0.5) → Linear(32→7)`. `training/train.py`
implements the full loop: forward pass, MSE loss, backpropagation, Adam steps,
per-epoch validation, best-checkpoint saving and early stopping (patience 25).

**A5. Learning-process evaluation.** `models/train_meta.json` stores both loss
curves and metrics; four design configurations were tested (see below).
Combining the train/validation curves shows clear overfitting in the unregularized
configuration, which motivated the stronger regularization.

| # | hidden | dropout | weight decay | lr | stopped | val MSE | test MSE |
|---|---|---|---|---|---|---|---|
| 1 | 64 | 0.2 | 0 | 1e-3 | epoch 16 | 2.561e-3 | 2.385e-3 |
| 2 | 32 | 0.4 | 1e-3 | 3e-4 | epoch 32 | 2.442e-3 | 2.302e-3 |
| 3 | 16 | 0.5 | 1e-2 | 3e-4 | epoch 41 | 2.468e-3 | 2.217e-3 |
| **4** | **32** | **0.5** | **5e-3** | **2e-4** | **epoch 46** | **2.423e-3** | 2.269e-3 |

Selected model = #4 (best validation MSE). Baselines: zero forecast val 2.542e-3 /
test 2.103e-3; train-mean val 2.432e-3 / test 2.122e-3. The model beats both
baselines on validation but not on the 2021–2023 test window — reported honestly.

**A6. Chronological splits.** 2005–2017 train / 2018–2020 validation /
2021–2023 test; no shuffling; 21-day purge; final artifacts frozen
(`models/lstm_model.pt`, `models/scaler.json`).

## Task B — Portfolio allocation

`s_i = max(r̂_i, 0) / max(σ_i, 0.5%)`; cash gate when all scores are zero;
normalize; 40% single-asset cap with redistribution; drop weights < 3% into
cash. Negative forecasts therefore move capital toward cash instead of forcing
a long position. Weights are non-negative and sum to 1 (verified by the
contract tests).

## Task C — Optional non-neural strategy

`POST /predict-alternative`: hold SPY when SPY > its 200-day moving average,
otherwise 100% cash. Same request/response contract as `/predict`.

## Task D — Local backtest (2021–2023, same conventions)

| Strategy | Total | CAGR | Vol | Sharpe | Max DD | Final | Avg cash |
|---|---|---|---|---|---|---|---|
| QQQ Buy & Hold | +33.8% | 10.2% | 23.8% | 0.53 | -35.1% | $13,384 | 0% |
| Equal Weight (7) | +4.3% | 1.4% | 10.6% | 0.18 | -21.7% | $10,427 | 0% |
| **LSTM + Allocation** | **+13.9%** | 4.5% | 12.0% | 0.42 | -21.2% | $11,391 | 1% |
| SPY 200d Timer | +17.9% | 5.7% | 12.8% | 0.50 | -27.8% | $11,793 | 25% |

Calendar-year returns:

| Strategy | 2021 | 2022 | 2023 |
|---|---|---|---|
| QQQ | +28.2% | -32.6% | +54.9% |
| Equal Weight | +8.1% | -17.5% | +16.9% |
| LSTM + Allocation | +19.4% | -14.8% | +11.9% |
| SPY 200d Timer | +29.6% | -22.1% | +16.8% |

Plots: `outputs/api_backtest.png` (cumulative value),
`outputs/api_backtest_weights.png` (weights through time — Task D item 6),
`outputs/api_backtest_yearly.csv`.

**Volatile/declining markets.** In 2022 the LSTM strategy lost 14.8% (QQQ -32.6%,
EW -17.5%) with a maximum drawdown of 21.2% over the window. It improved the
risk profile but did not eliminate losses.

**Lower loss vs better portfolio.** The model has no test-set predictive edge
(test MSE worse than a zero forecast), yet its allocation still beat Equal
Weight and stayed close to the SPY timer. This supports the assignment's point
that lower prediction loss does not automatically imply a better portfolio: the
allocation rule and diversification matter as much as the forecast.

## Task E — Prediction API

`api/app.py` (FastAPI) implements `POST /predict` exactly per the contract:
exactly 252 rows, strictly increasing unique dates, `as_of_date` equals the last
date, IRX nullable, response only `{"weights": {...8 floats}}`, all finite in
[0,1], sum = 1 ± 1e-6. Frozen artifacts are loaded once at startup; no training,
no network, no data after `as_of_date`. `python tools/api_smoke_test.py --url
http://127.0.0.1:8000/predict` passes all three checks; `pytest tests` passes
6/6. Swagger UI is self-hosted so `/docs` works without external CDNs.

Optional additions (not used by `/predict`): `POST /predict-qqq` (QQQ-only
benchmark) and `/predict-ui?strategy=lstm|qqq_gated|qqq` for the visual demo.

## Deliverables

- Training/backtest code: `training/train.py`, `src/*`, `backtest_local.py`
- Frozen model: `models/lstm_model.pt`, `models/scaler.json`, `models/train_meta.json`
- API: `api/app.py`, `requirements.txt`, `README.md`
- Local backtest analysis: `outputs/*` (metrics, charts, yearly table)
- Presentation: `https://ctx321.github.io/handwriting-demo/stock/presentation.html`
- Permanent prediction page: `https://ctx321.github.io/handwriting-demo/stock/`

## Reproduce

```powershell
cd "D:\deep learning\lstm-asset-allocation\api_deploy"
.\.venv\Scripts\Activate.ps1        # or reuse your Python environment
pip install -r requirements.txt
python training\train.py --epochs 100
python backtest_local.py
py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
py tools\api_smoke_test.py --url http://127.0.0.1:8000/predict
```

*Simulation-only academic project; not financial advice.*
