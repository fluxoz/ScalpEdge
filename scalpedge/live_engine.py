"""ScalpEdge — Live Signal Engine.

Wires PolygonStream → rolling bar buffer → TA indicators → HybridStrategy,
producing actionable :class:`SignalEvent` objects in real time on each new bar.

Usage
-----
    import asyncio
    from scalpedge.data import DataManager
    from scalpedge.ta_indicators import add_all_indicators
    from scalpedge.strategies import HybridStrategy
    from scalpedge.live_engine import LiveSignalEngine, SignalEvent

    def on_signal(event: SignalEvent) -> None:
        print(event)

    dm = DataManager()
    df = add_all_indicators(dm.load("SPY")).dropna(subset=["ema_50", "rsi_14", "macd", "atr_14"])
    strategy = HybridStrategy(use_ml=False)
    strategy.fit_ml(df.iloc[:int(len(df) * 0.8)])

    engine = LiveSignalEngine(["SPY"], strategy, on_signal=on_signal)
    engine.seed("SPY", df)
    asyncio.run(engine.run())
"""

from __future__ import annotations

import asyncio
import collections
import logging
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

logger = logging.getLogger("scalpedge.live_engine")

# Indicator columns captured in each SignalEvent.
_INDICATOR_KEYS = ("rsi_14", "macd", "adx_14", "ema_9", "ema_21", "vwap", "atr_14")

# Required OHLCV fields on every incoming bar dict.
_REQUIRED_BAR_FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class SignalEvent:
    """A trading signal emitted by :class:`LiveSignalEngine`.

    Attributes
    ----------
    ticker:
        Ticker symbol.
    bar_time:
        Timestamp of the bar that triggered the signal.
    price:
        Close price of the signal bar.
    signal:
        Direction integer — currently always ``1`` (long).
    indicators:
        Snapshot of key indicator values at the signal bar.
    strategy_name:
        Name attribute of the strategy that generated the signal.
    """

    ticker: str
    bar_time: pd.Timestamp
    price: float
    signal: int
    indicators: dict = field(default_factory=dict)
    strategy_name: str = ""

    def __str__(self) -> str:
        bar_time_str = self.bar_time.strftime("%Y-%m-%d %H:%M") if self.bar_time else "N/A"
        ind = self.indicators
        rsi = ind.get("rsi_14")
        macd = ind.get("macd")
        adx = ind.get("adx_14")
        vwap = ind.get("vwap")
        price = self.price

        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
        macd_str = f"{macd:+.4f}" if macd is not None else "N/A"
        adx_str = f"{adx:.1f}" if adx is not None else "N/A"
        vwap_dev_str = (
            f"{(price - vwap) / vwap * 100:+.2f}%" if vwap and vwap != 0 else "N/A"
        )

        return (
            f"🟢 LONG SIGNAL — {self.ticker}  @ ${price:.2f}  [{bar_time_str} ET]\n"
            f"   RSI: {rsi_str} | MACD: {macd_str} | ADX: {adx_str} | VWAP dev: {vwap_dev_str}"
        )


class LiveSignalEngine:
    """Wires PolygonStream → rolling bar buffer → indicators → HybridStrategy signals.

    On each new bar for a ticker:

    1. Appends the bar to a per-ticker rolling :class:`collections.deque`
       (default: last 500 bars).
    2. Converts the deque to a :class:`pandas.DataFrame`.
    3. Runs :func:`~scalpedge.ta_indicators.add_all_indicators` on the buffer.
    4. Drops NaN warm-up rows.
    5. Calls ``strategy.generate_signals(df)`` on the resulting frame.
    6. If the signal on the **last** row equals ``1``, emits a
       :class:`SignalEvent` via the *on_signal* callback.

    Parameters
    ----------
    tickers:
        Tickers to subscribe to.
    strategy:
        Pre-fitted :class:`~scalpedge.strategies.HybridStrategy` instance.
        ``fit_ml()`` must already have been called before passing it here.
    on_signal:
        Callable invoked when a new signal fires.  May be sync or async.
    buffer_size:
        Number of bars to keep in the rolling buffer per ticker (default 500).
    api_key:
        Polygon API key.  Falls back to ``POLYGON_API_KEY`` env var.
    subscriptions:
        PolygonStream subscriptions.  Default: ``["AM.*"]`` (per-minute bars).
    """

    def __init__(
        self,
        tickers: list[str],
        strategy,
        on_signal: Callable[[SignalEvent], None] | None = None,
        buffer_size: int = 500,
        api_key: str | None = None,
        subscriptions: list[str] | None = None,
    ) -> None:
        self.tickers = [t.upper() for t in tickers]
        self.strategy = strategy
        self.on_signal = on_signal
        self.buffer_size = buffer_size
        self.api_key = api_key
        self.subscriptions = subscriptions or ["AM.*"]

        # Per-ticker rolling deque: ticker -> deque[dict]
        self._buffers: dict[str, collections.deque] = {
            t: collections.deque(maxlen=buffer_size) for t in self.tickers
        }

        logger.info(
            "LiveSignalEngine initialised — tickers=%s  buffer_size=%d  "
            "strategy=%s  subscriptions=%s",
            self.tickers,
            self.buffer_size,
            getattr(strategy, "name", type(strategy).__name__),
            self.subscriptions,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, ticker: str, df: pd.DataFrame) -> None:
        """Pre-populate a ticker's buffer with historical data.

        Call this before :meth:`run` so the indicator warm-up is satisfied
        from bar 1 of the live stream.

        Parameters
        ----------
        ticker:
            Ticker symbol (case-insensitive).
        df:
            Historical OHLCV (with or without indicators).  The last
            *buffer_size* rows are loaded into the deque.
        """
        ticker = ticker.upper()
        if ticker not in self._buffers:
            self._buffers[ticker] = collections.deque(maxlen=self.buffer_size)

        rows = df.tail(self.buffer_size).to_dict("records")
        self._buffers[ticker].extend(rows)
        logger.info(
            "LiveSignalEngine.seed: %s — %d rows loaded into buffer.", ticker, len(rows)
        )

    def get_buffer(self, ticker: str) -> pd.DataFrame:
        """Return the current rolling buffer for *ticker* as a DataFrame.

        Parameters
        ----------
        ticker:
            Ticker symbol (case-insensitive).

        Returns
        -------
        pd.DataFrame
            Empty DataFrame when no data is available yet.
        """
        ticker = ticker.upper()
        buf = self._buffers.get(ticker)
        if not buf:
            return pd.DataFrame()
        return pd.DataFrame(list(buf))

    async def run(self) -> None:
        """Connect PolygonStream and process bars until cancelled."""
        from scalpedge.data import PolygonStream

        stream = PolygonStream(
            tickers=self.tickers,
            on_bar=self._on_bar,
            api_key=self.api_key,
            subscriptions=self.subscriptions,
        )
        logger.info("LiveSignalEngine.run: starting stream …")
        await stream.run()

    # ------------------------------------------------------------------
    # Internal per-bar handler
    # ------------------------------------------------------------------

    async def _on_bar(self, bar: dict) -> None:
        """Core per-bar handler — steps 1-6."""
        ticker = str(bar.get("ticker", "")).upper()
        if not ticker or ticker not in self._buffers:
            return

        # --- Step 0: validate OHLCV fields ---
        for fld in _REQUIRED_BAR_FIELDS:
            if bar.get(fld) is None:
                logger.debug(
                    "LiveSignalEngine._on_bar: skipping %s bar — missing field '%s'.",
                    ticker,
                    fld,
                )
                return

        logger.debug(
            "LiveSignalEngine._on_bar: %s  close=%.4f  time=%s",
            ticker,
            bar["close"],
            bar.get("datetime"),
        )

        # --- Step 1: append to rolling buffer ---
        self._buffers[ticker].append(bar)

        try:
            await self._process_buffer(ticker)
        except Exception as exc:
            logger.error(
                "LiveSignalEngine._on_bar: error processing %s buffer: %s",
                ticker,
                exc,
                exc_info=True,
            )

    async def _process_buffer(self, ticker: str) -> None:
        """Run indicators → signals on the current buffer for *ticker*."""
        from scalpedge.ta_indicators import add_all_indicators

        # --- Step 2: convert deque to DataFrame ---
        df = pd.DataFrame(list(self._buffers[ticker]))

        # --- Step 3: run add_all_indicators ---
        try:
            df = add_all_indicators(df)
        except Exception as exc:
            logger.warning(
                "LiveSignalEngine: add_all_indicators failed for %s (%s) — skipping bar.",
                ticker,
                exc,
            )
            return

        # --- Step 4: drop NaN warm-up rows ---
        required_cols = [c for c in ("ema_50", "rsi_14", "macd", "atr_14") if c in df.columns]
        if required_cols:
            df = df.dropna(subset=required_cols).reset_index(drop=True)

        if df.empty:
            logger.debug(
                "LiveSignalEngine: %s buffer has no clean rows after NaN drop.", ticker
            )
            return

        # --- Step 5: generate signals ---
        try:
            signals = self.strategy.generate_signals(df)
        except Exception as exc:
            logger.error(
                "LiveSignalEngine: generate_signals failed for %s: %s",
                ticker,
                exc,
                exc_info=True,
            )
            return

        last_signal = int(signals.iloc[-1]) if len(signals) > 0 else 0

        # --- Step 6: emit SignalEvent if signal == 1 ---
        if last_signal == 1:
            last_row = df.iloc[-1]
            indicators = {
                key: _safe_float(last_row.get(key)) for key in _INDICATOR_KEYS
            }
            bar_time = last_row.get("datetime")
            if not isinstance(bar_time, pd.Timestamp):
                bar_time = pd.Timestamp(bar_time) if bar_time is not None else pd.Timestamp.now(tz="UTC")

            event = SignalEvent(
                ticker=ticker,
                bar_time=bar_time,
                price=float(last_row["close"]),
                signal=last_signal,
                indicators=indicators,
                strategy_name=getattr(self.strategy, "name", type(self.strategy).__name__),
            )

            logger.info(
                "LiveSignalEngine: SIGNAL — %s  price=%.4f  bar_time=%s",
                ticker,
                event.price,
                event.bar_time,
            )

            if self.on_signal is not None:
                result = self.on_signal(event)
                if asyncio.iscoroutine(result):
                    await result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        import math
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None
