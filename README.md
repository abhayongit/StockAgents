# StockAgents

Two small Python agents that run one after the other:

1. **Stock volume agent** (`volAgent/stock_volume_agent.py`) downloads hourly volume and peak-minute data for a stock from Yahoo Finance.
2. **Support price predictor** (`forcastAgent/support_price_predictor.py`) reads that data and works out a Bullish, Bearish or Neutral signal and a support price for each hour.

`run_pipeline.py` runs both agents in order.

## Requirements

- Python 3.10 or newer
- An internet connection (the data comes from Yahoo Finance)

## Setup (one time)

```bash
git clone https://github.com/abhayongit/StockAgents.git
cd StockAgents

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python run_pipeline.py NVDA
python run_pipeline.py NVDA AAPL MSFT --days 5 --window 7
python run_pipeline.py HCLTECH.NS --outdir output
```

| Option     | Default     | Meaning                                                |
|------------|-------------|--------------------------------------------------------|
| `symbols`  | (required)  | One or more tickers. Non-US stocks need a suffix, such as `.NS` or `.T`. |
| `--days`   | 5           | Number of trading days to fetch                        |
| `--window` | 7           | Number of hours used for the standard deviation        |
| `--outdir` | `resources` | Folder the CSVs are written to                         |

## Output (in `--outdir`)

| File | Contents |
|------|----------|
| `<SYMBOL>_hourly_volume.csv` | Hourly volume, with estimated bought and sold volume |
| `<SYMBOL>_hourly_volume_daily.csv` | Daily totals |
| `<SYMBOL>_hourly_volume_peak_minutes.csv` | The busiest buy and sell minute in each hour |
| `<SYMBOL>_hourly_volume_peak_minutes_with_support.csv` | The peak-minutes data plus standard deviation and support price |
| `<SYMBOL>_hourly_volume_peak_minutes_with_support_signals.csv` | The signal and how each % figure was worked out |

## Running each agent on its own

```bash
python volAgent/stock_volume_agent.py NVDA --csv
python forcastAgent/support_price_predictor.py NVDA_hourly_volume_peak_minutes.csv
```

## Notes

- Uses the unofficial `yfinance` library. It is meant for personal use, and Yahoo can change or block it without notice.
- Yahoo keeps 1-minute data for only about the last 7 days.
- Bought and sold volumes are estimates, not real order data.
- Not investment advice.
