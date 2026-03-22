"""ScalpEdge — main entry point.

Runs full backtests on SPY and TSLA (and any additional tickers you add)
using the hybrid strategy that combines:
  * TA indicators (EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, patterns)
  * Markov chain order-2 direction probability
  * Monte Carlo random-walk probability
  * Black-Scholes 0DTE delta filter
  * RandomForest + LSTM ML score  (requires ``pip install scalpedge[ml]``)

Usage
-----
    # Run full backtests (default behaviour):
    uv run python main.py

    # Fetch historical data for one or more tickers:
    uv run python main.py fetch SPY TSLA
    uv run python main.py fetch AAPL --interval 1d --start 2023-01-01 --end 2023-12-31
    uv run python main.py fetch SPY --output-dir /tmp/mydata
    uv run python main.py fetch SPY --years 10          # ~10 years of 5-min bars (chunked)

    # Market scanner (requires POLYGON_API_KEY with Stocks Advanced):
    uv run python main.py scan
    uv run python main.py scan SPY TSLA AAPL NVDA QQQ
    uv run python main.py scan --top 20                 # top 20 by absolute change %

    # Live signal engine + TUI dashboard (requires POLYGON_API_KEY):
    uv run python main.py live SPY TSLA NVDA            # TUI dashboard (default when textual installed)
    uv run python main.py live --no-dashboard           # plain stdout fallback (no TUI)
    uv run python main.py live SPY --no-ml              # disable ML scoring layer
    uv run python main.py live SPY --buffer-size 200    # keep 200 bars in rolling buffer

The ``data/`` directory is auto-created and grows on every run.
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings

# ---------------------------------------------------------------------------
# Early sanity check — catch corrupted numpy *before* anything else imports
# ---------------------------------------------------------------------------
try:
    import numpy  # noqa: F401
except ImportError as exc:
    if "source directory" in str(exc):
        print(
            "\n❌  numpy import failed — the virtual environment appears corrupted.\n"
            "   This commonly happens when the Python interpreter changes (e.g.\n"
            "   entering a Nix devshell after the venv was already created).\n\n"
            "   Fix:  rm -rf .venv && uv sync && uv run python main.py\n",
            file=sys.stderr,
        )
    else:
        print(
            f"\n❌  numpy import failed: {exc}\n"
            "   Dependencies are not installed in the current Python environment.\n\n"
            "   Fix:  uv sync && uv run python main.py\n"
            "    or:  pip install -e .\n",
            file=sys.stderr,
        )
    sys.exit(1)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scalpedge.main")

# ---------------------------------------------------------------------------
# Detect optional ML dependencies
# ---------------------------------------------------------------------------
import importlib.util

_HAS_ML = (
    importlib.util.find_spec("sklearn") is not None
    and importlib.util.find_spec("torch") is not None
)

_HAS_TUI = importlib.util.find_spec("textual") is not None

# ---------------------------------------------------------------------------
# Tickers to backtest — add any ticker here to extend coverage.
# ---------------------------------------------------------------------------
TICKERS: list[str] = ["SPY", "TSLA"]

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------
HYBRID_CONFIG = dict(
    markov_order=2,
    mc_n_simulations=1000,
    mc_n_bars=12,
    mc_threshold_pct=0.0,
    markov_up_threshold=0.38,
    ml_score_threshold=0.52,
    bs_min_delta=0.40,
    bs_sigma=0.20,
    hold_bars=3,        # 15 minutes at 5m bars
    use_ml=_HAS_ML,
    use_markov=True,
    use_mc=True,
    use_bs=True,
)

BACKTEST_CONFIG = dict(
    fee_pct=0.005,       # 0.005% per side (~$5 per $100k)
    slippage_pct=0.01,   # 0.01% per side
    hold_bars=3,
    initial_capital=100_000.0,
)


def cmd_fetch(args: argparse.Namespace) -> None:
    """Fetch and persist historical OHLCV data for one or more tickers."""
    from scalpedge.data import DataManager

    # --years is a convenience alias for --start N years ago.
    start = args.start
    if getattr(args, "years", None) and start is None:
        import pandas as pd

        start = (
            (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=int(args.years)))
            .normalize()
            .strftime("%Y-%m-%d")
        )

    dm = DataManager(
        data_dir=args.output_dir or None,
        interval=args.interval,
    )

    errors: list[str] = []
    for raw_ticker in args.tickers:
        ticker = raw_ticker.upper()
        try:
            df = dm.load(ticker, start=start, end=args.end)
            first_dt = df["datetime"].min()
            last_dt = df["datetime"].max()
            print(
                f"{ticker}: {len(df):,} bars  |  "
                f"interval={args.interval}  |  "
                f"{first_dt.strftime('%Y-%m-%d')} → {last_dt.strftime('%Y-%m-%d')}"
            )
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", ticker, exc)
            errors.append(ticker)

    if errors:
        logger.warning("Failed tickers: %s", ", ".join(errors))
        sys.exit(1)


def cmd_scan(args: argparse.Namespace) -> None:
    """Print a pre-market or intraday scanner table using Polygon snapshots."""
    import os

    import pandas as pd

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print(
            "❌  POLYGON_API_KEY is not set. "
            "The scan command requires a Polygon Stocks Advanced subscription.",
            file=sys.stderr,
        )
        sys.exit(1)

    from scalpedge.data import PolygonClient

    client = PolygonClient(api_key=api_key)
    tickers: list[str] | None = [t.upper() for t in args.tickers] if args.tickers else None
    df = client.fetch_snapshot(tickers=tickers)

    if df.empty:
        print("No snapshot data returned.")
        return

    # Sort by absolute change % descending.
    if "change_pct" in df.columns:
        df = df.sort_values("change_pct", ascending=False, key=lambda s: s.abs())

    top = getattr(args, "top", None)
    if top:
        df = df.head(int(top))

    # Print table
    col_w = {"ticker": 8, "last_trade_price": 12, "change_pct": 10, "day_volume": 14, "prev_close": 12}
    header = (
        f"{'TICKER':<{col_w['ticker']}}  "
        f"{'LAST PRICE':>{col_w['last_trade_price']}}  "
        f"{'CHANGE %':>{col_w['change_pct']}}  "
        f"{'DAY VOLUME':>{col_w['day_volume']}}  "
        f"{'PREV CLOSE':>{col_w['prev_close']}}"
    )
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        ticker_str = str(row.get("ticker", ""))
        price = row.get("last_trade_price")
        chg = row.get("change_pct")
        vol = row.get("day_volume")
        prev = row.get("prev_close")

        price_str = f"{price:.2f}" if pd.notna(price) else "  N/A"
        chg_str = f"{chg:+.2f}%" if pd.notna(chg) else "   N/A"
        vol_str = f"{int(vol):,}" if pd.notna(vol) else "          N/A"
        prev_str = f"{prev:.2f}" if pd.notna(prev) else "  N/A"

        print(
            f"{ticker_str:<{col_w['ticker']}}  "
            f"{price_str:>{col_w['last_trade_price']}}  "
            f"{chg_str:>{col_w['change_pct']}}  "
            f"{vol_str:>{col_w['day_volume']}}  "
            f"{prev_str:>{col_w['prev_close']}}"
        )


def run_backtest(ticker: str, spy_df: pd.DataFrame | None = None) -> None:
    """Fetch data, compute indicators, fit ML, and run hybrid backtest.

    Parameters
    ----------
    ticker : str
        The ticker symbol to backtest.
    spy_df : pd.DataFrame or None
        Pre-loaded SPY 5-minute OHLCV DataFrame used for the market regime
        filter.  When provided and the ticker is not SPY, the regime filter
        is enabled automatically.  Pass ``None`` to disable the regime filter.
    """
    from scalpedge.data import DataManager
    from scalpedge.ta_indicators import add_all_indicators
    from scalpedge.strategies import HybridStrategy, TAStrategy
    from scalpedge.backtester import Backtester

    logger.info("=" * 60)
    logger.info("  Processing ticker: %s", ticker)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load / update data
    # ------------------------------------------------------------------
    logger.info("[%s] Loading 5-minute OHLCV data ...", ticker)
    dm = DataManager()
    df = dm.load(ticker)
    logger.info("[%s] %d bars loaded.", ticker, len(df))

    # ------------------------------------------------------------------
    # 2. Compute all TA indicators
    # ------------------------------------------------------------------
    logger.info("[%s] Computing TA indicators ...", ticker)
    df = add_all_indicators(df)

    # Drop rows with NaN indicators (warm-up period).
    df = df.dropna(subset=["ema_50", "rsi_14", "macd", "atr_14"]).reset_index(drop=True)
    logger.info("[%s] %d bars after indicator warm-up.", ticker, len(df))

    if len(df) < 100:
        logger.warning("[%s] Insufficient data for backtesting.", ticker)
        return

    # ------------------------------------------------------------------
    # 3. Run TA-only baseline
    # ------------------------------------------------------------------
    logger.info("[%s] Running TA-only baseline ...", ticker)
    ta_strategy = TAStrategy()
    ta_result = ta_strategy.backtest(df, ticker=ticker, **BACKTEST_CONFIG)
    print(ta_result.summary())

    # ------------------------------------------------------------------
    # 4. Fit ML models on first 80% of data
    # ------------------------------------------------------------------
    split = int(len(df) * 0.80)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    # ------------------------------------------------------------------
    # 4a. Configure market regime filter (SPY benchmark, non-SPY tickers)
    # ------------------------------------------------------------------
    use_regime = spy_df is not None and ticker.upper() != "SPY"
    if use_regime:
        logger.info(
            "[%s] Market regime filter ENABLED — using SPY 5-bar rolling VWAP.",
            ticker,
        )

    hybrid_cfg = dict(
        **HYBRID_CONFIG,
        use_regime_filter=use_regime,
        spy_df=spy_df if use_regime else None,
        regime_lookback=5,
    )

    logger.info("[%s] Fitting ML models on %d bars (train set) ...", ticker, len(train_df))
    strategy = HybridStrategy(**hybrid_cfg)
    strategy.fit_ml(train_df)

    # ------------------------------------------------------------------
    # 5. Run hybrid backtest on test set
    # ------------------------------------------------------------------
    logger.info("[%s] Running Hybrid backtest on %d bars (test set) ...", ticker, len(test_df))
    hybrid_result = strategy.backtest(
        test_df.reset_index(drop=True),
        ticker=ticker,
        **BACKTEST_CONFIG,
    )
    print(hybrid_result.summary())

    # ------------------------------------------------------------------
    # 6. Optional: plot equity curve
    # ------------------------------------------------------------------
    _plot_equity(hybrid_result, ticker)


def _print_signal(event) -> None:
    """Print a formatted signal notification to stdout."""
    print(str(event))


def _redirect_logs_to_file(path: str) -> None:
    """Redirect all logging output to *path* to keep the TUI terminal clean."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(file_handler)


def cmd_live(args: argparse.Namespace) -> None:
    """Seed historical buffer, fit strategy, then stream live bars from Polygon."""
    import asyncio

    from scalpedge.data import DataManager
    from scalpedge.ta_indicators import add_all_indicators
    from scalpedge.strategies import HybridStrategy
    from scalpedge.live_engine import LiveSignalEngine

    tickers: list[str] = [t.upper() for t in args.tickers] if args.tickers else TICKERS

    use_ml_live  = _HAS_ML and not getattr(args, "no_ml", False)
    buffer_size: int = getattr(args, "buffer_size", 500) or 500
    use_dashboard = _HAS_TUI and not getattr(args, "no_dashboard", False)

    logger.info(
        "Live engine starting — tickers=%s  use_ml=%s  buffer_size=%d  dashboard=%s",
        tickers, use_ml_live, buffer_size, use_dashboard,
    )

    dm = DataManager()

    # ------------------------------------------------------------------
    # 1. Load historical data + compute indicators for each ticker
    # ------------------------------------------------------------------
    ticker_dfs: dict = {}
    spy_df = None

    # Load SPY for regime filter (non-SPY tickers)
    if any(t != "SPY" for t in tickers):
        try:
            raw_spy = dm.load("SPY")
            spy_df = add_all_indicators(raw_spy)
            spy_df = spy_df.dropna(subset=["ema_50", "rsi_14", "macd", "atr_14"]).reset_index(
                drop=True
            )
            logger.info("SPY benchmark loaded: %d bars.", len(spy_df))
        except Exception as exc:
            logger.warning("Could not load SPY for regime filter (%s). Regime filter disabled.", exc)

    for ticker in tickers:
        try:
            logger.info("[%s] Loading historical data …", ticker)
            raw_df = dm.load(ticker)
            df = add_all_indicators(raw_df)
            df = df.dropna(subset=["ema_50", "rsi_14", "macd", "atr_14"]).reset_index(drop=True)
            logger.info("[%s] %d bars after indicator warm-up.", ticker, len(df))
            ticker_dfs[ticker] = df
        except Exception as exc:
            logger.error("[%s] Failed to load historical data: %s", ticker, exc)

    if not ticker_dfs:
        logger.error("No ticker data loaded. Exiting.")
        return

    # Use first ticker's data for ML fitting; fall back gracefully
    primary_ticker = next(iter(ticker_dfs))
    primary_df = ticker_dfs[primary_ticker]

    # ------------------------------------------------------------------
    # 2. Fit HybridStrategy on 80% of historical data
    # ------------------------------------------------------------------
    use_regime = spy_df is not None and any(t != "SPY" for t in tickers)
    hybrid_cfg = {
        **HYBRID_CONFIG,
        "use_ml": use_ml_live,
        "use_regime_filter": use_regime,
        "spy_df": spy_df if use_regime else None,
        "regime_lookback": 5,
    }
    strategy = HybridStrategy(**hybrid_cfg)

    split    = int(len(primary_df) * 0.80)
    train_df = primary_df.iloc[:split]
    logger.info("[%s] Fitting ML models on %d bars …", primary_ticker, len(train_df))
    strategy.fit_ml(train_df)

    # ------------------------------------------------------------------
    # 3. Create engine and seed buffers
    # ------------------------------------------------------------------
    engine = LiveSignalEngine(
        tickers=tickers,
        strategy=strategy,
        # Callbacks are wired by the dashboard; stdout fallback otherwise.
        on_signal=None if use_dashboard else _print_signal,
        buffer_size=buffer_size,
        api_key=getattr(args, "api_key", None) or None,
    )

    for ticker, df in ticker_dfs.items():
        engine.seed(ticker, df)

    # ------------------------------------------------------------------
    # 4. Launch TUI dashboard or plain asyncio loop
    # ------------------------------------------------------------------
    if use_dashboard:
        # Redirect logs to a file — the TUI owns the terminal.
        _redirect_logs_to_file("scalpedge_live.log")
        from scalpedge.dashboard import ScalpEdgeDashboard
        ScalpEdgeDashboard(tickers=tickers, engine=engine).run()
    else:
        asyncio.run(engine.run())


def _plot_equity(result, ticker: str) -> None:
    """Save an equity-curve plot to the data directory (non-blocking)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 5))
        result.equity_curve.plot(ax=ax, linewidth=1.5, color="steelblue")
        ax.set_title(f"ScalpEdge — {ticker} Hybrid Strategy — Equity Curve")
        ax.set_xlabel("Bar index")
        ax.set_ylabel("Portfolio value ($)")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        from pathlib import Path
        out_dir = Path("data")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{ticker}_equity_curve.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        logger.info("[%s] Equity curve saved → %s", ticker, path)
    except Exception as exc:
        logger.debug("Could not save equity curve: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scalpedge",
        description="ScalpEdge — intraday scalping backtester and data toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------ #
    # fetch sub-command                                                    #
    # ------------------------------------------------------------------ #
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download and persist historical OHLCV data for one or more tickers.",
    )
    fetch_parser.add_argument(
        "tickers",
        nargs="+",
        metavar="TICKER",
        help="One or more ticker symbols to fetch (e.g. SPY TSLA AAPL).",
    )
    fetch_parser.add_argument(
        "--interval",
        default="5m",
        metavar="INTERVAL",
        help=(
            "Bar interval (default: 5m). "
            "Supported: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo."
        ),
    )
    fetch_parser.add_argument(
        "--years",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Convenience shorthand: fetch the last N years of data "
            "(e.g. --years 10). Ignored when --start is also provided."
        ),
    )
    fetch_parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date for the historical range (inclusive).",
    )
    fetch_parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date for the historical range (inclusive). Defaults to today.",
    )
    fetch_parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Directory for Parquet files. Defaults to ./data.",
    )

    # ------------------------------------------------------------------ #
    # backtest sub-command (or bare invocation)                           #
    # ------------------------------------------------------------------ #
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run full hybrid backtests (default when no sub-command is given).",
    )
    backtest_parser.add_argument(
        "tickers",
        nargs="*",
        metavar="TICKER",
        help=(
            "Tickers to backtest. Defaults to SPY and TSLA when omitted."
        ),
    )

    # ------------------------------------------------------------------ #
    # scan sub-command                                                     #
    # ------------------------------------------------------------------ #
    scan_parser = subparsers.add_parser(
        "scan",
        help=(
            "Print a pre-market or intraday scanner table using Polygon snapshots "
            "(requires POLYGON_API_KEY with Stocks Advanced)."
        ),
    )
    scan_parser.add_argument(
        "tickers",
        nargs="*",
        metavar="TICKER",
        help=(
            "Tickers to scan. Fetches the whole market when omitted "
            "(e.g. SPY TSLA AAPL NVDA QQQ)."
        ),
    )
    scan_parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Limit output to the top N tickers by absolute change %% (e.g. --top 20).",
    )

    # ------------------------------------------------------------------ #
    # live sub-command                                                     #
    # ------------------------------------------------------------------ #
    live_parser = subparsers.add_parser(
        "live",
        help=(
            "Stream live bars from Polygon and emit signals in real time "
            "(requires POLYGON_API_KEY with Stocks Advanced)."
        ),
    )
    live_parser.add_argument(
        "tickers",
        nargs="*",
        metavar="TICKER",
        help=(
            "Tickers to stream.  Defaults to SPY and TSLA when omitted "
            "(e.g. SPY TSLA AAPL)."
        ),
    )
    live_parser.add_argument(
        "--no-ml",
        action="store_true",
        dest="no_ml",
        default=False,
        help="Disable the ML layer even when scikit-learn and torch are installed.",
    )
    live_parser.add_argument(
        "--buffer-size",
        type=int,
        default=500,
        dest="buffer_size",
        metavar="N",
        help="Number of bars to keep in the rolling buffer per ticker (default: 500).",
    )
    live_parser.add_argument(
        "--no-dashboard",
        action="store_true",
        dest="no_dashboard",
        default=False,
        help=(
            "Print signals to stdout instead of launching the TUI dashboard. "
            "The dashboard launches automatically when textual is installed."
        ),
    )

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
        return

    if args.command == "scan":
        cmd_scan(args)
        return

    if args.command == "live":
        cmd_live(args)
        return

    # Default / explicit backtest path.
    tickers: list[str] = TICKERS
    if args.command == "backtest" and args.tickers:
        tickers = [t.upper() for t in args.tickers]

    logger.info("ScalpEdge backtester starting ...")
    logger.info("Tickers: %s", ", ".join(tickers))
    if not _HAS_ML:
        logger.info(
            "ML layer disabled (torch/scikit-learn not installed). "
            "Install with: uv sync --extra ml  or  pip install -e '.[ml]'"
        )

    # ------------------------------------------------------------------
    # Load SPY benchmark data once for the market regime filter.
    # Used by all non-SPY tickers.  Failures are non-fatal.
    # ------------------------------------------------------------------
    spy_df: pd.DataFrame | None = None
    if any(t.upper() != "SPY" for t in tickers):
        try:
            from scalpedge.data import DataManager

            _dm = DataManager()
            spy_df = _dm.load("SPY")
            logger.info("SPY benchmark loaded: %d bars (for regime filter).", len(spy_df))
        except Exception as _exc:
            logger.warning(
                "Could not load SPY data for regime filter (%s). "
                "Regime filter will be disabled.",
                _exc,
            )

    results = {}
    for ticker in tickers:
        try:
            run_backtest(ticker, spy_df=spy_df)
            results[ticker] = "OK"
        except Exception as exc:
            logger.error("Failed to backtest %s: %s", ticker, exc, exc_info=True)
            results[ticker] = f"ERROR: {exc}"

    logger.info("\n\nAll done.")
    logger.info("Summary:")
    for ticker, status in results.items():
        logger.info("  %s: %s", ticker, status)


if __name__ == "__main__":
    main()
