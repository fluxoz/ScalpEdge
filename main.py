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
    mc_threshold_pct=0.3,
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


def run_backtest(ticker: str) -> None:
    """Fetch data, compute indicators, fit ML, and run hybrid backtest."""
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

    logger.info("[%s] Fitting ML models on %d bars (train set) ...", ticker, len(train_df))
    strategy = HybridStrategy(**HYBRID_CONFIG)
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

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
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

    results = {}
    for ticker in tickers:
        try:
            run_backtest(ticker)
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
