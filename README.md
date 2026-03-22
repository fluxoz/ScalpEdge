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
- **Bid-ask microstructure features** from NBBO quote data (spread, imbalance)
- **Catalyst suppression filter** to avoid trading around earnings/events
- **Market regime filter** — SPY 5-bar rolling VWAP gates signals for other tickers
- **Real-time WebSocket streaming** via Polygon Stocks Advanced

---

## Setup

### Data Source — Polygon.io (recommended)

ScalpEdge uses **Polygon.io** as the primary market data source.
Sign up for a **Stocks Advanced** key at <https://polygon.io/> and export it before running:

```bash
export POLYGON_API_KEY="your_key_here"
```

Stocks Advanced plan limits observed by the client:

| Limit | Value |
|---|---|
| API calls/minute | Unlimited |
| Historical minute data | 5+ years |
| Tick-level trades & quotes | ✅ (v3/trades, v3/quotes) |
| Market snapshots | ✅ (whole market or per-ticker) |
| WebSocket streaming | ✅ (real-time bars & trades) |
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

# 4. (Optional) Install streaming dependencies (websockets)
uv sync --extra streaming

# 5. (Optional) Install TUI dashboard dependencies
uv sync --extra tui

# 6. Run the full backtest
uv run python main.py
```

### Option B — plain uv (no Nix)

```bash
# Requires Python 3.12+ and uv installed
uv sync                      # core deps only
uv sync --extra ml           # + ML layer (optional)
uv sync --extra streaming    # + WebSocket streaming (optional)
uv sync --extra tui          # + TUI dashboard (optional)
uv run python main.py
```

### Option C — plain pip

```bash
pip install -e "."                        # core deps only
pip install -e ".[ml]"                    # + ML layer (optional)
pip install -e ".[streaming]"             # + WebSocket streaming (optional)
pip install -e ".[tui]"                   # + TUI dashboard (optional)
pip install -e ".[ml,streaming,tui]"      # everything
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
    ├── ta_indicators.py     # 20+ TA indicators + 60+ candlestick patterns + microstructure features
    ├── probabilities.py     # Monte Carlo + Markov chain (order-2)
    ├── options.py           # Black-Scholes pricing, Greeks, implied vol
    ├── ml.py                # RandomForest + PyTorch LSTM + MLEngine
    ├── backtester.py        # Vectorized backtester + full performance metrics
    ├── strategies.py        # TAStrategy (baseline) + HybridStrategy (all layers + catalyst filter)
    ├── live_engine.py       # LiveSignalEngine: bar buffer, indicators, strategy, WebSocket feed
    └── dashboard.py         # Textual TUI dashboard for live signal monitoring
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

## Market Scanner

Use the `scan` sub-command to pull a live snapshot table via Polygon Stocks Advanced:

```bash
# Scan the whole market (sorted by absolute change %):
uv run python main.py scan

# Scan specific tickers:
uv run python main.py scan SPY TSLA AAPL NVDA QQQ

# Top 20 movers:
uv run python main.py scan --top 20
```

Example output:
```
TICKER    LAST PRICE    CHANGE %      DAY VOLUME    PREV CLOSE
--------------------------------------------------------------------
NVDA         878.35      +4.27%    42,103,200         842.75
TSLA         185.50      -2.13%    89,234,100         189.53
SPY          519.12      +0.52%   102,000,000         516.43
```

---

## WebSocket Streaming

Stream real-time bars and trades via Polygon's WebSocket API (Stocks Advanced required):

```python
import asyncio
from scalpedge.data import PolygonStream

async def handle_bar(bar: dict) -> None:
    print(f"[{bar['ticker']}] {bar['datetime']}  O={bar['open']}  C={bar['close']}  V={bar['volume']}")

async def handle_trade(trade: dict) -> None:
    print(f"[{trade['ticker']}] trade @ {trade['price']} x {trade['size']}")

stream = PolygonStream(
    tickers=["SPY", "TSLA"],
    on_bar=handle_bar,
    on_trade=handle_trade,
    subscriptions=["AM.*", "T.*"],   # per-minute bars + trade ticks
)

asyncio.run(stream.run())
```

The stream reconnects automatically with exponential back-off (max 60 s) on disconnect.

---

## Live TUI Dashboard

ScalpEdge ships a full terminal UI powered by [Textual](https://github.com/Textualize/textual) that streams live 5-minute bars from Polygon, runs the full signal engine, and renders everything in real time — charts, indicators, signal log, and regime banner.

### Install the TUI extra

```bash
pip install -e ".[tui]"
# or with uv:
uv sync --extra tui
```

This adds `textual`, `textual-plotext`, and `plotext`.

### Launch

```bash
# Requires POLYGON_API_KEY (Stocks Advanced)
export POLYGON_API_KEY="your_key_here"

# Start with the default tickers:
uv run python main.py live

# Monitor specific tickers:
uv run python main.py live SPY TSLA NVDA AAPL

# Skip the TUI and print signals to stdout instead:
uv run python main.py live SPY TSLA --no-dashboard

# Disable the ML scoring layer (faster start):
uv run python main.py live SPY TSLA --no-ml

# Keep only 200 bars in the rolling per-ticker buffer:
uv run python main.py live SPY --buffer-size 200
```

Press **Ctrl+C** to exit gracefully. While the TUI is running, all logging output is redirected to `scalpedge_live.log` in the working directory.

### Dashboard layout

```
┌─────────────────────────────────────────────────────────────┐
│  ScalpEdge  ──  🟢  BULLISH                 [regime banner] │
├─────────────────────────────────────────────────────────────┤
│  Overview │ SPY │ TSLA │ NVDA │ …            [tabbed panes] │
│                                                             │
│  ┌─ Candlestick + EMA9 ──────────────────────────────────┐  │
│  │  (plotext chart, last 120 bars)                       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─ Volume ──────────┐ ┌─ Indicator stats ───────────────┐  │
│  │  colour-coded bars│ │ RSI · MACD · ADX · VWAP Δ      │  │
│  └───────────────────┘ └────────────────────────────────-┘  │
│  ┌─ RSI ─────────────────────────────────────────────────┐  │
│  │  (with 70 / 30 reference lines)                       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─ MACD histogram ──────────────────────────────────────┐  │
│  │  (green above zero, red below zero)                   │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌─ ADX / DI +/− ────────────────────────────────────────┐  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  HH:MM  🟢 LONG  SPY  $519.12  RSI 58 …    [signal log]    │
│  HH:MM  ⚪ FLAT  TSLA  $185.50 RSI 44 …                    │
├─────────────────────────────────────────────────────────────┤
│  Signals: 3 · Uptime: 00:12:34 · Last bar: 10:35  [footer] │
└─────────────────────────────────────────────────────────────┘
```

| Panel | Description |
|---|---|
| **Regime banner** | SPY market regime (🟢 BULLISH / 🔴 BEARISH / ⚪ NEUTRAL) via two-factor vote (close vs. VWAP + EMA-21) |
| **Overview tab** | Compact grid — price, RSI/MACD/ADX/VWAP Δ, signal badge, OHLCV, per-ticker signal count |
| **Per-ticker tabs** | Candlestick + EMA9, volume bars, RSI, MACD histogram, ADX + DI+/DI− |
| **Signal log** | Scrolling `RichLog` — timestamp · signal · ticker · price · indicator snapshot |
| **Session bar** | Total signals fired, uptime, last bar timestamp |

### Keyboard shortcuts

| Key | Action |
|---|---|
| `/` | Focus the ticker search box |
| `Enter` (in search) | Jump to the tab for the typed symbol (exact → prefix → substring match) |
| `Escape` | Clear the search box |
| `Ctrl+C` | Quit |

### `on_bar_update` callback

If you integrate `LiveSignalEngine` directly instead of using `main.py`, wire the
`on_bar_update` callback to receive every processed bar (not just signal bars):

```python
import asyncio
from scalpedge.live_engine import LiveSignalEngine
from scalpedge.strategies import HybridStrategy

def handle_signal(event) -> None:
    print(event)

def handle_bar(ticker: str, bar: dict, last_signal: int) -> None:
    # fired on every bar — use for chart updates, custom logging, etc.
    print(f"[{ticker}] close={bar['close']:.2f}  signal={last_signal}")

strategy = HybridStrategy(use_ml=False)
engine = LiveSignalEngine(
    tickers=["SPY", "TSLA"],
    strategy=strategy,
    on_signal=handle_signal,
    on_bar_update=handle_bar,   # fires every bar, not just signal bars
)

asyncio.run(engine.run())
```

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

## Catalyst Suppression

Suppress signals around known earnings, splits, or macro events:

```python
import pandas as pd
from scalpedge.strategies import HybridStrategy

strategy = HybridStrategy(
    use_catalyst_filter=True,
    catalyst_dates=[
        pd.Timestamp("2024-01-25", tz="UTC"),   # TSLA Q4 earnings (whole day)
        pd.Timestamp("2024-01-26 14:30", tz="UTC"),  # FOMC announcement (±30 min)
    ],
    catalyst_suppress_bars=6,   # ±6 bars = ±30 min at 5m
)
```

Catalyst dates can also be fetched automatically via `PolygonClient.fetch_events()`.

---

## Market Regime Filter (SPY Benchmark)

The **market regime filter** tracks SPY's intraday trend using a rolling
5-bar VWAP and gates entry signals for other tickers accordingly.

* **Bullish regime** (SPY close > 5-bar VWAP) → long entries allowed.
* **Bearish regime** (SPY close < 5-bar VWAP) → long entries suppressed.
* **Neutral** (insufficient history, first ≤ 4 bars) → entries allowed.

### Standalone regime computation

```python
from scalpedge.ta_indicators import compute_market_regime

# spy_df: a standard OHLCV DataFrame with high/low/close/volume columns
regime = compute_market_regime(spy_df, lookback=5)
# regime is a pd.Series of integers: 1 (bullish), -1 (bearish), 0 (neutral)
```

### Enable in HybridStrategy

```python
import pandas as pd
from scalpedge.data import DataManager
from scalpedge.strategies import HybridStrategy

dm = DataManager()
spy_df = dm.load("SPY")   # load SPY benchmark data

strategy = HybridStrategy(
    use_regime_filter=True,
    spy_df=spy_df,
    regime_lookback=5,      # 5-bar rolling VWAP (default)
)
signals = strategy.generate_signals(tsla_df)
```

When `main.py` is run with multiple tickers, SPY is loaded once and
automatically passed as the regime benchmark for all non-SPY tickers:

```bash
uv run python main.py backtest TSLA AAPL NVDA
```

---

## VWAP-Anchored Mean Reversion Strategy

`VWAPMeanReversionStrategy` is a dedicated intraday mean-reversion approach
that enters long when price is near the session VWAP with optional RSI,
volume, and candlestick confirmation.  ATR-based stop-loss and take-profit
exits fire automatically.

Works especially well for **liquid index ETFs** (SPY, QQQ, IVV) at the
5-minute bar level.

### Entry conditions

All of the following must be satisfied at the entry bar:

| # | Condition | Default |
|---|---|---|
| 1 | `\|price_vs_vwap\|` ≤ `vwap_proximity_pct` | ±0.1 % |
| 2 | `rsi_min` ≤ RSI_14 ≤ `rsi_max` | 30 – 60 |
| 3 | Volume ≥ `volume_factor` × rolling-average volume | 1.0 × |
| 4 | Bullish candlestick pattern present (`pat_bull_signal == 1`) | required |

### Quick start

```python
from scalpedge.ta_indicators import add_all_indicators
from scalpedge.strategies import VWAPMeanReversionStrategy

df = add_all_indicators(raw_df)          # adds vwap, rsi_14, atr_14, pat_bull_signal …

strategy = VWAPMeanReversionStrategy(
    vwap_proximity_pct=0.1,  # enter within ±0.1 % of VWAP
    rsi_min=30.0,            # avoid deeply oversold traps
    rsi_max=60.0,            # avoid overbought entries
    volume_factor=1.0,       # require at-or-above-average volume
    require_bull_candle=True,
    atr_sl_mult=1.0,         # stop loss = entry − 1.0 × ATR_14
    atr_tp_mult=1.5,         # take profit = entry + 1.5 × ATR_14
    hold_bars=6,             # time exit after 30 min (6 × 5 min)
)

result = strategy.backtest(df, ticker="SPY", fee_pct=0.005, slippage_pct=0.005)
print(result.summary())
```

### Combining with the market regime filter

For extra robustness you can chain `VWAPMeanReversionStrategy` with the
market regime filter via `HybridStrategy`'s `extra_rules` mechanism:

```python
from scalpedge.strategies import VWAPMeanReversionStrategy, HybridStrategy

vwap_strat = VWAPMeanReversionStrategy(vwap_proximity_pct=0.15)

strategy = HybridStrategy(
    use_ml=False,
    use_markov=False,
    use_mc=False,
    use_bs=False,
    use_regime_filter=True,
    spy_df=spy_df,
    extra_rules=[vwap_strat.generate_signals],
)
signals = strategy.generate_signals(df)
```

---

## Layers Explained

| Layer | Module | What it does |
|---|---|---|
| Data | `data.py` | Fetches 5m OHLCV from Polygon.io Stocks Advanced (or yfinance fallback), stores in Parquet, auto-appends new bars; also exposes tick trades, NBBO quotes, snapshots, news, events, and WebSocket streaming |
| TA | `ta_indicators.py` | EMA/SMA/RSI/MACD/BB/ATR/OBV/VWAP/Stoch/ADX + Volume Profile/POC + 60+ candle patterns; also computes bid-ask microstructure features (spread, imbalance) when quote data is available |
| Probabilities | `probabilities.py` | Monte Carlo random walk + Markov chain order-2 transition probs |
| Options | `options.py` | Black-Scholes call/put, delta, gamma, vega, theta, rho, implied vol |
| ML | `ml.py` | RandomForest + PyTorch LSTM → combined P(up) score; microstructure features (`spread_pct`, `bid_ask_imbalance`, `imbalance_ma_10`) included when available |
| Backtester | `backtester.py` | Vectorized simulation, fee+slippage, equity curve, full metrics |
| Strategies | `strategies.py` | TAStrategy (baseline) + HybridStrategy (all layers + optional catalyst suppression & market regime filters) + VWAPMeanReversionStrategy (VWAP-anchored mean reversion with ATR exits) |
| Live Engine | `live_engine.py` | Connects to Polygon WebSocket, maintains rolling bar buffer per ticker, computes indicators on every bar, fires `on_signal` / `on_bar_update` callbacks |
| Dashboard | `dashboard.py` | Textual TUI — charts, regime banner, signal log, ticker search; launched automatically by `main.py live` when `textual` is installed |

---

## Volume Profile & Point of Control (POC)

`add_all_indicators()` now computes a **rolling intraday volume profile** for
each calendar session and exposes four signal columns:

| Column | Description |
|---|---|
| `poc_price` | Price level with the most cumulative volume traded in the session so far (no look-ahead) |
| `poc_proximity_pct` | `(close − POC) / POC × 100` — negative = below POC, positive = above POC |
| `poc_above` | `1` when `close > poc_price`, else `0` |
| `poc_below` | `1` when `close < poc_price`, else `0` |

These features are appended alongside all other indicators and are immediately
usable as signal inputs in your custom strategy.

### Visualization

```python
import matplotlib.pyplot as plt
import pandas as pd
from scalpedge.ta_indicators import add_all_indicators, plot_volume_profile

df = pd.read_parquet("data/SPY_5m.parquet")
df = add_all_indicators(df)

fig = plot_volume_profile(df, session_date="2024-01-02")
plt.show()
# or: fig.savefig("volume_profile_2024-01-02.png", dpi=150)
```

`plot_volume_profile` renders two side-by-side panels:
* **Left** — close-price line with a dashed line marking the session POC.
* **Right** — horizontal volume-at-price histogram (the profile itself).

The function can be called for any session date present in the DataFrame and
returns a `matplotlib.figure.Figure` so you can embed it in notebooks or save
to disk.

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
