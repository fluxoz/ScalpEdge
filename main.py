"""ScalpEdge — main entry point.

Runs full backtests on SPY and TSLA (and any additional tickers you add)
using the hybrid strategy that combines:
  * TA indicators (EMA, RSI, MACD, Bollinger Bands, ATR, VWAP, patterns)
  * Markov chain order-2 direction probability
  * Monte Carlo random-walk probability
  * Black-Scholes 0DTE delta filter
  * RandomForest + LSTM ML score

Usage
-----
    uv run python main.py

The ``data/`` directory is auto-created and grows on every run.
"""

from __future__ import annotations

import logging
import sys
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scalpedge.main")

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
    use_ml=True,
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
    bt = Backtester(**BACKTEST_CONFIG)
    from scalpedge.backtester import Backtester
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
    logger.info("ScalpEdge backtester starting ...")
    logger.info("Tickers: %s", ", ".join(TICKERS))

    results = {}
    for ticker in TICKERS:
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
