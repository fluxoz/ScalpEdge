"""Vectorized backtesting engine.

Implements a fast, pandas/numpy-based backtester that supports:
* Any combination of entry/exit signal columns in the DataFrame.
* Configurable hold time (default 3 bars = 15 minutes).
* Per-trade fee and slippage modelling.
* Full performance metrics: win rate, expectancy, profit factor,
  Sharpe ratio, max drawdown, CAGR, total return, and more.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtesting performance metrics."""

    ticker: str
    strategy: str
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    expectancy_pct: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    equity_curve: pd.Series = field(default_factory=pd.Series)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"{'=' * 55}",
            f"  {self.ticker}  |  Strategy: {self.strategy}",
            f"{'=' * 55}",
            f"  Trades          : {self.n_trades}",
            f"  Win Rate        : {self.win_rate:.1%}",
            f"  Avg Win         : {self.avg_win_pct:.3f}%",
            f"  Avg Loss        : {self.avg_loss_pct:.3f}%",
            f"  Expectancy      : {self.expectancy_pct:.3f}%",
            f"  Profit Factor   : {self.profit_factor:.2f}",
            f"  Sharpe Ratio    : {self.sharpe_ratio:.2f}",
            f"  Max Drawdown    : {self.max_drawdown_pct:.2f}%",
            f"  Total Return    : {self.total_return_pct:.2f}%",
            f"  CAGR            : {self.cagr_pct:.2f}%",
            f"{'=' * 55}",
        ]
        return "\n".join(lines)


class Backtester:
    """Vectorized backtester for 5-minute intraday strategies.

    Parameters
    ----------
    fee_pct:
        One-way percentage fee applied per trade (e.g. 0.005 for 0.005 %).
        Applied to both entry and exit.
    slippage_pct:
        One-way percentage slippage applied per trade.
    hold_bars:
        Fixed holding period in bars (default 3 = 15 minutes at 5m).
    initial_capital:
        Starting portfolio value in dollars.
    """

    def __init__(
        self,
        fee_pct: float = 0.005,
        slippage_pct: float = 0.01,
        hold_bars: int = 3,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.hold_bars = hold_bars
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        entry_signal: pd.Series,
        ticker: str = "UNKNOWN",
        strategy_name: str = "strategy",
    ) -> BacktestResult:
        """Run a vectorized backtest.

        Parameters
        ----------
        df:
            OHLCV + indicator DataFrame.  Must contain a ``close`` column.
        entry_signal:
            Boolean/int Series aligned with *df*.  A value of 1 (True)
            triggers a long entry at the *next* bar's open price.
        ticker:
            Ticker symbol for reporting.
        strategy_name:
            Display name for the strategy.

        Returns
        -------
        :class:`BacktestResult`
        """
        close = df["close"].values.astype(float)
        open_ = df["open"].values.astype(float) if "open" in df.columns else close
        signal = np.asarray(entry_signal, dtype=int)

        n = len(close)
        trades: list[dict] = []

        # Round cost per trade (entry + exit, both directions).
        round_trip_cost = 2 * (self.fee_pct + self.slippage_pct) / 100

        i = 0
        while i < n - self.hold_bars - 1:
            if signal[i] == 1:
                entry_price = open_[i + 1] * (1 + self.slippage_pct / 100)
                exit_idx = min(i + 1 + self.hold_bars, n - 1)
                exit_price = open_[exit_idx] * (1 - self.slippage_pct / 100)

                pnl_pct = (exit_price / entry_price - 1) * 100 - round_trip_cost * 100
                trades.append(
                    {
                        "entry_idx": i + 1,
                        "exit_idx": exit_idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "win": pnl_pct > 0,
                    }
                )
                # Advance past the hold period to avoid overlapping trades.
                i = exit_idx
            else:
                i += 1

        result = self._compute_metrics(
            trades=trades,
            close=close,
            ticker=ticker,
            strategy_name=strategy_name,
        )
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        trades: list[dict],
        close: np.ndarray,
        ticker: str,
        strategy_name: str,
    ) -> BacktestResult:
        result = BacktestResult(ticker=ticker, strategy=strategy_name)

        if not trades:
            logger.warning("No trades generated for %s / %s", ticker, strategy_name)
            result.equity_curve = pd.Series(
                self.initial_capital, index=range(len(close))
            )
            return result

        trade_log = pd.DataFrame(trades)
        result.trade_log = trade_log
        result.n_trades = len(trade_log)

        wins = trade_log.loc[trade_log["win"], "pnl_pct"]
        losses = trade_log.loc[~trade_log["win"], "pnl_pct"]

        result.win_rate = len(wins) / len(trade_log)
        result.avg_win_pct = float(wins.mean()) if len(wins) > 0 else 0.0
        result.avg_loss_pct = float(losses.mean()) if len(losses) > 0 else 0.0

        result.expectancy_pct = (
            result.win_rate * result.avg_win_pct
            + (1 - result.win_rate) * result.avg_loss_pct
        )

        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve.
        equity = np.full(len(close), self.initial_capital, dtype=float)
        for trade in trades:
            idx = trade["exit_idx"]
            equity[idx:] *= (1 + trade["pnl_pct"] / 100)
        equity_series = pd.Series(equity)
        result.equity_curve = equity_series

        # Sharpe ratio (annualised, using trade-level returns).
        pnl_arr = trade_log["pnl_pct"].values / 100
        if len(pnl_arr) > 1:
            bars_per_year = 252 * 78  # 78 five-minute bars per day
            trades_per_year = result.n_trades / (len(close) / bars_per_year)
            sr_std = pnl_arr.std(ddof=1)
            if sr_std > 0:
                result.sharpe_ratio = float(
                    (pnl_arr.mean() / sr_std) * np.sqrt(trades_per_year)
                )

        # Max drawdown.
        running_max = equity_series.cummax()
        dd = (equity_series - running_max) / running_max * 100
        result.max_drawdown_pct = float(dd.min())

        # Total return.
        result.total_return_pct = float(
            (equity_series.iloc[-1] / self.initial_capital - 1) * 100
        )

        # CAGR (approximate: assume 252 * 78 bars per year).
        n_bars = len(close)
        years = n_bars / (252 * 78)
        if years > 0 and equity_series.iloc[-1] > 0:
            result.cagr_pct = float(
                ((equity_series.iloc[-1] / self.initial_capital) ** (1 / years) - 1) * 100
            )

        return result
