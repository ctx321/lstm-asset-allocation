# Training ablations (21-day return MSE, original units)

Data: 2005-03-01 .. 2023-11-29 samples, 73 features, 60-day window.
Split: train 2006-03 .. 2017-11, val 2018-01 .. 2020-12, test 2021-01 .. 2023-11
(21-day purge at split boundaries). Seed 42.

| # | hidden | dropout | weight decay | lr | batch | stopped at | val MSE | test MSE |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 0.2 | 0 | 1e-3 | 64 | epoch 16 | 2.561e-3 | 2.385e-3 |
| 2 | 32 | 0.4 | 1e-3 | 3e-4 | 128 | epoch 32 | 2.442e-3 | 2.302e-3 |
| 3 | 16 | 0.5 | 1e-2 | 3e-4 | 128 | epoch 41 | 2.468e-3 | 2.217e-3 |
| 4 | 32 | 0.5 | 5e-3 | 2e-4 | 128 | epoch 46 | **2.423e-3** | 2.269e-3 |

Baselines (same rows):
- zero forecast: val 2.542e-3, test 2.103e-3
- train-mean forecast: val 2.432e-3, test 2.122e-3

**Chosen model: #4** (best validation MSE). It beats both naive baselines on the
validation period, but has **no edge on the 2021–2023 test period** — a common
and honestly reported result for 21-day return forecasting. Weight decay and
smaller capacity were essential: config #1 overfits by epoch 10.

# Local backtest (Task D, same conventions as the course)

USD 10,000, 21-day rebalancing, T+1 execution, zero costs, long-only.

| Strategy | Total | CAGR | Vol | Sharpe | Max DD | Final |
|---|---|---|---|---|---|---|
| QQQ Buy & Hold | +33.8% | 10.2% | 23.8% | 0.53 | -35.1% | 13,384 |
| Equal Weight (7) | +4.3% | 1.4% | 10.6% | 0.18 | -21.7% | 10,427 |
| **LSTM + Allocation** | **+13.9%** | 4.5% | 12.0% | 0.42 | -21.2% | 11,391 |
| SPY 200d Timer | +17.9% | 5.7% | 12.8% | 0.50 | -27.8% | 11,793 |

The LSTM strategy clearly beats Equal Weight and is close to the SPY timer,
but does not beat QQQ on total return — consistent with the assignment's note
that lower prediction loss does not automatically imply a better portfolio.
