# ScalpEdge

> **⚠️ Risk Warning** — This software is for **educational and research purposes only**.  
> Past backtest performance does **not** guarantee future results.  
> Day trading and options carry significant financial risk.  
> **Never trade with money you cannot afford to lose.**

---

A clean, modular, laptop-friendly Python backtesting framework for **intraday 5-minute candle scalping strategies** on **SPY** and **TSLA** (trivially extensible to any ticker).

Combines:
- Full **Technical Analysis** (20+ indicators, 60+ candlestick patterns)
- **Markov Chain** order-2 direction probability
- **Monte Carlo** random-walk forward probability
- **Black-Scholes** 0DTE option pricing & delta filter
- **RandomForest + LSTM** ML hybrid score
- **Vectorized backtester** with realistic fees, slippage, and full metrics

---

## Setup

### Data Source — Polygon.io (recommended)

ScalpEdge uses **Polygon.io** as the primary market data source.
Sign up for a free key at <https://polygon.io/> and export it before running:

```bash
export POLYGON_API_KEY="your_key_here"
```

Free tier limits observed by the client:

| Limit | Value |
|---|---|
| API calls/minute | 5 |
| Historical minute data | 2 years |
| Coverage | All US stocks |

> **Fallback** — if `POLYGON_API_KEY` is not set, ScalpEdge falls back to
> **yfinance** automatically (limited to ~60 days of intraday history per
> request, auto-chunked for longer ranges).

### Option A — Nix + uv (recommended, fully reproducible)

```bash
# 1. Enter the dev shell (installs Python 3.12 + uv automatically)
nix develop

# 2. Install Python dependencies
uv sync

# 3. (Optional) Install ML dependencies (scikit-learn + PyTorch)
uv sync --extra ml

# 4. Run the full backtest
uv run python main.py
```

### Option B — plain uv (no Nix)

```bash
# Requires Python 3.12+ and uv installed
uv sync                  # core deps only
uv sync --extra ml       # + ML layer (optional)
uv run python main.py
```

### Option C — plain pip

```bash
pip install -e "."       # core deps only
pip install -e ".[ml]"   # + ML layer (optional)
python main.py
```

---

## Project Structure

```
ScalpEdge/
├── main.py                  # Entry point — runs SPY + TSLA backtests
├── pyproject.toml           # Dependencies (uv/pip)
├── flake.nix                # Nix dev shell
├── data/                    # Parquet data files (auto-created, grows over time)
│   ├── SPY.parquet
│   └── TSLA.parquet
└── scalpedge/
    ├── __init__.py
    ├── data.py              # Data management: fetch, store, auto-append (Polygon/yfinance + Parquet)
    ├── ta_indicators.py     # 20+ TA indicators + 60+ candlestick patterns (vectorized)
    ├── probabilities.py     # Monte Carlo + Markov chain (order-2)
    ├── options.py           # Black-Scholes pricing, Greeks, implied vol
    ├── ml.py                # RandomForest + PyTorch LSTM + MLEngine
    ├── backtester.py        # Vectorized backtester + full performance metrics
    └── strategies.py        # TAStrategy (baseline) + HybridStrategy (all layers)
```

---

## Example Output

```
=======================================================
  SPY  |  Strategy: hybrid
=======================================================
  Trades          : 47
  Win Rate        : 58.5%
  Avg Win         : 0.142%
  Avg Loss        : -0.098%
  Expectancy      : 0.042%
  Profit Factor   : 1.39
  Sharpe Ratio    : 1.82
  Max Drawdown    : -4.21%
  Total Return    : 8.73%
  CAGR            : 12.50%
=======================================================
```

An equity curve PNG is also saved to `data/<TICKER>_equity_curve.png`.

---

## Adding a New Ticker

Edit `TICKERS` in `main.py`:

```python
TICKERS: list[str] = ["SPY", "TSLA", "QQQ", "NVDA"]  # any US stock ticker
```

That's it — data management, indicators, and backtest all work automatically.

---

## Adding a New Strategy

1. Subclass `BaseStrategy` in `scalpedge/strategies.py`:

```python
class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Return 1 where you want to enter long, 0 otherwise
        return (df["rsi_14"] < 30).astype(int)
```

2. Use it in `main.py`:

```python
from scalpedge.strategies import MyStrategy
result = MyStrategy().backtest(df, ticker=ticker, **BACKTEST_CONFIG)
print(result.summary())
```

---

## Adding a Custom Rule to HybridStrategy

```python
def my_rule(df: pd.DataFrame) -> pd.Series:
    """Enter only when volume is above 20-bar average."""
    return (df["volume"] > df["volume"].rolling(20).mean()).astype(bool)

strategy = HybridStrategy(extra_rules=[my_rule])
```

---

## Layers Explained

| Layer | Module | What it does |
|---|---|---|
| Data | `data.py` | Fetches 5m OHLCV from Polygon.io (or yfinance fallback), stores in Parquet, auto-appends new bars |
| TA | `ta_indicators.py` | EMA/SMA/RSI/MACD/BB/ATR/OBV/VWAP/Stoch/ADX + 60+ candle patterns |
| Probabilities | `probabilities.py` | Monte Carlo random walk + Markov chain order-2 transition probs |
| Options | `options.py` | Black-Scholes call/put, delta, gamma, vega, theta, rho, implied vol |
| ML | `ml.py` | RandomForest + PyTorch LSTM → combined P(up) score |
| Backtester | `backtester.py` | Vectorized simulation, fee+slippage, equity curve, full metrics |
| Strategies | `strategies.py` | TAStrategy (baseline) + HybridStrategy (all layers) |

---

## Performance Notes

- Data auto-persists and grows each run — your historical dataset gets richer over time.
- ML models are trained on 80% of available data, tested on the remaining 20%.
- LSTM training uses CPU by default; set `device="cuda"` in `HYBRID_CONFIG` for GPU.
- All computations are vectorized; no bar-by-bar Python loops except the strategy engine.

---

## Troubleshooting

### `Error importing numpy: you should not try to import numpy from its source directory`

The virtual environment is out of sync with the current Python interpreter. This typically happens when switching between environments (e.g. entering a Nix devshell after the `.venv` was already created).

**Fix:**

```bash
rm -rf .venv && uv sync && uv run python main.py
```

---

## License

MIT — use at your own risk.
