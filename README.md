# StockAgents

Three small Python agents that run one after the other:

1. **Stock volume agent** (`volAgent/stock_volume_agent.py`) downloads hourly volume and peak-minute data for a stock from Yahoo Finance.
2. **Support price predictor** (`forcastAgent/support_price_predictor.py`) reads that data and works out a Bullish, Bearish or Neutral signal and a support price for each hour.
3. **Technical chart predictor** (`forcastAgent/technical_chart_predictor.py`) reads the last 2 weeks of the daily chart, works out the indicators (moving averages, MACD, RSI, Bollinger bands, ATR and trend), and gives a central estimate with 68% and 90% ranges for the price 3 and 5 trading days ahead. It also backtests its own method on the past ~120 trading days.

`run_pipeline.py` runs all three agents in order.

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
| `<SYMBOL>_technical_forecast.csv` | Central estimate, 68% and 90% ranges and chance of a rise for each of the next 5 trading days |
| `<SYMBOL>_technical_indicators.csv` | Each indicator's value, vote and reading |
| `<SYMBOL>_technical_backtest.csv` | How the method's error compared with a "no change" guess, and how often the real price landed in the ranges |

## Web app

```bash
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501. Type a ticker or company name and click **Run** to see the signals, a price and support chart, and the data tables. You can download every CSV on its own or all of them as a zip. Results are cached for 15 minutes.

### Deploy free on Streamlit Community Cloud

1. Push this repo to GitHub. Streamlit Community Cloud can deploy public repos, and private repos once you've given it access.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose this repo and the `main` branch, and set the main file path to `streamlit_app.py`.
4. Under **Advanced settings**, choose Python 3.12, then click **Deploy**. You get a public `*.streamlit.app` URL to share.

## Running each agent on its own

```bash
python volAgent/stock_volume_agent.py NVDA --csv
python forcastAgent/support_price_predictor.py NVDA_hourly_volume_peak_minutes.csv
python forcastAgent/technical_chart_predictor.py NVDA --csv
```

## Notes

- Uses the unofficial `yfinance` library. It is meant for personal use, and Yahoo can change or block it without notice.
- Yahoo keeps 1-minute data for only about the last 7 days.
- Bought and sold volumes are estimates, not real order data.
- Not investment advice.
