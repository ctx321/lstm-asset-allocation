# Dataset Guide (Student Release)

**Date range:** 2005-01-03 to 2023-12-29  
**Trading-day observations:** 4,781  
**Investable ETF columns:** QQQ, SPY, IWM, IEF, TLT, GLD, SHY  
**Other observed columns:** VIX, IRX

The CSV files contain **only the public training/development window**. There are no hidden evaluation rows in this package. Every row is aligned to a common trading-day calendar for the seven ETFs.

## Files

- `market_data.csv` — Date, seven ETF adjusted closing prices, VIX, IRX.
- `asset_prices_adjusted.csv` — Date and seven ETF adjusted closing prices.
- `market_context.csv` — Date, VIX, IRX.
- `market_data_public.xlsx` — an optional spreadsheet view of the same student-release observations.

## Units and meaning

- ETF values are split/distribution-adjusted closes; **returns** are computed as `P[t]/P[t-1] - 1`.
- VIX is a volatility index level and **cannot** be traded as an eighth ETF in this project.
- IRX is a 13-week U.S. Treasury bill yield indicator **in percent**, not a tradable ETF or the project's cash yield.
- Cash is a **portfolio weight** with 0% simulated interest; there is no CASH price series.
- Blank IRX values are not zeros. Handle them causally or choose not to use IRX.

## Data quality

- Public release rows: 4,781
- Missing values among the seven investable ETF columns: 0
- Missing VIX values: 0
- Missing IRX observations: 5
- Strictly increasing dates: True
- Student release ends: 2023-12-29

## Source and reproducibility

Teaching snapshot prepared from publicly available daily U.S. market data, including distribution/split-adjusted ETF closing prices and market indicators. The instructor maintains the original provenance record and will make the raw-source links available after the hidden evaluation period. The student file itself contains all data required for the assignment.

The student version is **truncated at the specified public cutoff**. Do not extend it using other sources for this assignment. See `Project_Instructions.docx` for the no-hidden-data rule.

For teaching/research simulation only; not an investment recommendation.
