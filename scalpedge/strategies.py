"""Strategy module — hybrid signal combination.

The :class:`HybridStrategy` class combines any subset of the available
signal layers into a single entry signal:

* TA rules (EMA crossover, RSI oversold, MACD momentum, BB bounce)
* Markov chain probability filter
* Monte Carlo probability filter
* Black-Scholes delta filter (option premium reasonability gate)
* ML score filter (RF + LSTM combined probability)
* Market regime filter (SPY 5-bar rolling VWAP trend gate)

New strategies can be added by subclassing :class:`BaseStrategy` or by
passing custom ``rule_fn`` callables to :class:`HybridStrategy`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from .backtester import Backtester, BacktestResult
from .options import BlackScholes
from .probabilities import MarkovChain, MonteCarlo

if TYPE_CHECKING:
    from .ml import MLEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a boolean/int Series of entry signals aligned with *df*."""

    def backtest(
        self,
        df: pd.DataFrame,
        ticker: str = "UNKNOWN",
        **bt_kwargs,
    ) -> BacktestResult:
        """Generate signals and run the backtester. Returns :class:`BacktestResult`.

        When ``atr_sl_mult`` and ``atr_tp_mult`` are passed in *bt_kwargs*,
        the ``atr_14`` column from *df* is automatically used as the ATR
        series unless an explicit ``atr`` kwarg is provided.
        """
        signals = self.generate_signals(df)
        # Pop `atr` from bt_kwargs so it is not forwarded to Backtester.__init__
        atr = bt_kwargs.pop("atr", None)
        if atr is None and bt_kwargs.get("atr_sl_mult") is not None and bt_kwargs.get("atr_tp_mult") is not None:
            atr = df.get("atr_14")
        bt = Backtester(**bt_kwargs)
        return bt.run(df, signals, ticker=ticker, strategy_name=self.name, atr=atr)


# ---------------------------------------------------------------------------
# TA-only baseline
# ---------------------------------------------------------------------------

class TAStrategy(BaseStrategy):
    """Simple TA rule strategy (baseline reference).

    Entry rule:
    * EMA9 > EMA21 (short-term uptrend)
    * RSI_14 in 35–65 (momentum zone, avoid overbought/oversold)
    * MACD > MACD signal (upward momentum)
    * Close above lower Bollinger Band
    """

    name = "ta_only"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        c = df["close"].astype(float)
        ema9 = df.get("ema_9", c.ewm(span=9, adjust=False).mean())
        ema21 = df.get("ema_21", c.ewm(span=21, adjust=False).mean())
        rsi = df.get("rsi_14", pd.Series(50, index=df.index))
        macd = df.get("macd", pd.Series(0, index=df.index))
        macd_sig = df.get("macd_signal", pd.Series(0, index=df.index))
        bb_lower = df.get("bb_lower", c * 0.98)

        signal = (
            (ema9 > ema21)
            & (rsi >= 35)
            & (rsi <= 65)
            & (macd > macd_sig)
            & (c > bb_lower)
        ).astype(int)
        return signal


# ---------------------------------------------------------------------------
# Hybrid strategy (all layers combined)
# ---------------------------------------------------------------------------

class HybridStrategy(BaseStrategy):
    """Hybrid strategy combining TA + Markov + MonteCarlo + BS delta + ML + regime.

    Parameters
    ----------
    markov_order:
        Order for the Markov chain (default 2).
    mc_n_simulations:
        Monte Carlo simulation count.
    mc_n_bars:
        Forward horizon for Monte Carlo (in bars).
    mc_threshold_pct:
        Minimum expected move threshold for MC filter.
    markov_up_threshold:
        Minimum P(UP) from Markov chain to allow entry.
    ml_score_threshold:
        Minimum ML combined score to allow entry.
    bs_min_delta:
        Minimum Black-Scholes delta for ATM call (option quality gate).
    bs_sigma:
        Implied vol estimate for BS pricing.
    hold_bars:
        Trade holding period (bars).
    extra_rules:
        Optional list of ``Callable[[pd.DataFrame], pd.Series]`` that each
        return a boolean Series.  ALL must be True for entry.
    use_ml:
        Whether to use the ML layer (can be disabled for speed).
    use_markov:
        Whether to use the Markov layer.
    use_mc:
        Whether to use the Monte Carlo layer.
    use_bs:
        Whether to use the Black-Scholes delta filter.
    use_catalyst_filter:
        When ``True`` and *catalyst_dates* is non-empty, suppress all entry
        signals within ±*catalyst_suppress_bars* bars of any catalyst date.
    catalyst_dates:
        List of ``pd.Timestamp`` objects (date or datetime) marking known
        catalyst events (earnings, splits, macro releases, etc.).  For
        date-only catalysts every bar on that calendar day is suppressed.
    catalyst_suppress_bars:
        Half-window of bars to suppress around each catalyst (default 6,
        i.e. ±30 min at 5-minute bars).
    use_regime_filter:
        When ``True`` and *spy_df* is provided, suppress long entry signals
        during bearish market regimes (SPY close < rolling VWAP) and suppress
        short entry signals during bullish regimes.  When the ticker being
        traded IS SPY, you may still pass *spy_df* or set this to ``False``.
    spy_df:
        SPY (or other benchmark) OHLCV DataFrame used to compute the market
        regime.  Must contain ``high``, ``low``, ``close``, ``volume``, and
        ``datetime`` columns.  Required when *use_regime_filter* is ``True``.
    regime_lookback:
        Rolling-window size (in bars) for the VWAP trend calculation
        (default 5).
    """

    name = "hybrid"

    def __init__(
        self,
        markov_order: int = 2,
        mc_n_simulations: int = 1000,
        mc_n_bars: int = 12,
        mc_threshold_pct: float = 0.3,
        markov_up_threshold: float = 0.38,
        ml_score_threshold: float = 0.52,
        bs_min_delta: float = 0.40,
        bs_sigma: float = 0.20,
        hold_bars: int = 3,
        extra_rules: list[Callable[[pd.DataFrame], pd.Series]] | None = None,
        use_ml: bool = True,
        use_markov: bool = True,
        use_mc: bool = True,
        use_bs: bool = True,
        use_catalyst_filter: bool = False,
        catalyst_dates: list[pd.Timestamp] | None = None,
        catalyst_suppress_bars: int = 6,
        use_regime_filter: bool = False,
        spy_df: pd.DataFrame | None = None,
        regime_lookback: int = 5,
    ) -> None:
        self.markov_order = markov_order
        self.mc_n_simulations = mc_n_simulations
        self.mc_n_bars = mc_n_bars
        self.mc_threshold_pct = mc_threshold_pct
        self.markov_up_threshold = markov_up_threshold
        self.ml_score_threshold = ml_score_threshold
        self.bs_min_delta = bs_min_delta
        self.bs_sigma = bs_sigma
        self.hold_bars = hold_bars
        self.extra_rules = extra_rules or []
        self.use_ml = use_ml
        self.use_markov = use_markov
        self.use_mc = use_mc
        self.use_bs = use_bs
        self.use_catalyst_filter = use_catalyst_filter
        self.catalyst_dates = catalyst_dates or []
        self.catalyst_suppress_bars = catalyst_suppress_bars
        self.use_regime_filter = use_regime_filter
        self.spy_df = spy_df
        self.regime_lookback = regime_lookback

        self._markov = MarkovChain(order=markov_order)
        self._mc = MonteCarlo(n_simulations=mc_n_simulations)
        self._ml: MLEngine | None = None

    # ------------------------------------------------------------------

    def fit_ml(self, df: pd.DataFrame) -> "HybridStrategy":
        """Fit ML models (RF + LSTM) on *df*. Call before generate_signals."""
        if self.use_ml:
            from .ml import MLEngine

            self._ml = MLEngine()
            self._ml.fit(df)
        return self

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Combine all layers and return a 0/1 entry signal Series."""
        c = df["close"].astype(float)

        # ----------------------------------------------------------------
        # Layer 1 — TA rules
        # ----------------------------------------------------------------
        ema9 = df.get("ema_9", c.ewm(span=9, adjust=False).mean())
        ema21 = df.get("ema_21", c.ewm(span=21, adjust=False).mean())
        rsi = df.get("rsi_14", pd.Series(50.0, index=df.index))
        macd = df.get("macd", pd.Series(0.0, index=df.index))
        macd_sig = df.get("macd_signal", pd.Series(0.0, index=df.index))
        bb_lower = df.get("bb_lower", c * 0.98)
        adx = df.get("adx_14", pd.Series(25.0, index=df.index))
        pat_bull = df.get("pat_bull_signal", pd.Series(0, index=df.index))

        ta_signal = (
            (ema9 > ema21)
            & (rsi >= 30)
            & (rsi <= 70)
            & (macd > macd_sig)
            & (c > bb_lower)
            & (adx >= 20)
        )

        # Optional candlestick confirmation (boost but not mandatory)
        ta_signal = ta_signal | (ta_signal & (pat_bull == 1))

        combined = ta_signal.copy()

        # ----------------------------------------------------------------
        # Layer 2 — Markov chain
        # ----------------------------------------------------------------
        if self.use_markov:
            markov_signal = self._markov_filter(c)
            combined = combined & markov_signal

        # ----------------------------------------------------------------
        # Layer 3 — Monte Carlo random walk
        # ----------------------------------------------------------------
        if self.use_mc:
            mc_signal = self._mc_filter(c)
            combined = combined & mc_signal

        # ----------------------------------------------------------------
        # Layer 4 — ML score
        # ----------------------------------------------------------------
        if self.use_ml and self._ml is not None:
            ml_prob = self._ml.score(df)
            combined = combined & (ml_prob >= self.ml_score_threshold)

        # ----------------------------------------------------------------
        # Layer 5 — Black-Scholes delta filter
        # ----------------------------------------------------------------
        if self.use_bs:
            bs_signal = self._bs_filter(c)
            combined = combined & bs_signal

        # ----------------------------------------------------------------
        # Extra custom rules
        # ----------------------------------------------------------------
        for rule_fn in self.extra_rules:
            combined = combined & rule_fn(df).astype(bool)

        # ----------------------------------------------------------------
        # Layer 6 — Market regime filter (SPY rolling VWAP trend)
        # ----------------------------------------------------------------
        if self.use_regime_filter and self.spy_df is not None:
            regime_signal = self._regime_filter(df)
            combined = combined & regime_signal

        # ----------------------------------------------------------------
        # Catalyst suppression filter (final gate)
        # ----------------------------------------------------------------
        combined = self._apply_catalyst_filter(combined.astype(int), df).astype(bool)

        return combined.astype(int)

    # ------------------------------------------------------------------
    # Internal layer helpers
    # ------------------------------------------------------------------

    def _markov_filter(self, close: pd.Series) -> pd.Series:
        """Return True where Markov P(UP) >= threshold."""
        self._markov.fit(close)
        states = self._markov.get_states_series(close)
        order = self.markov_order
        result = pd.Series(False, index=close.index)
        for i in range(order, len(states)):
            context = states[i - order : i]
            proba = self._markov.predict_proba(context)
            if proba["UP"] >= self.markov_up_threshold:
                result.iloc[i] = True
        return result

    def _mc_filter(self, close: pd.Series) -> pd.Series:
        """Return True where Monte Carlo P(up) >= 0.5."""
        log_rets = np.log(close / close.shift(1)).dropna()
        if len(log_rets) < 20:
            return pd.Series(True, index=close.index)
        # Use all available history to estimate drift + vol.
        prob_up = self._mc.prob_up(
            log_rets,
            n_bars=self.mc_n_bars,
            threshold_pct=self.mc_threshold_pct,
        )
        # Scalar filter — apply uniformly (can be made rolling if desired).
        return pd.Series(prob_up >= 0.5, index=close.index)

    def _bs_filter(self, close: pd.Series) -> pd.Series:
        """Return True where ATM 0DTE call delta >= bs_min_delta.

        For ATM options the delta is determined solely by sigma and T, so
        we compute it once and broadcast over the whole series.
        """
        import math

        T = 1 / (252 * 6.5 * 12)  # ~5-minute 0DTE horizon
        sigma = self.bs_sigma
        # ATM: d1 = (r + 0.5*sigma^2) * T / (sigma * sqrt(T))
        # With T so small, d1 ≈ 0 → delta ≈ 0.5 regardless of sigma.
        try:
            d1 = (math.log(1.0) + (0.05 + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            import scipy.special as _sc

            atm_delta = 0.5 * math.erfc(-d1 / math.sqrt(2))
        except Exception:
            atm_delta = 0.5  # fallback

        passes = atm_delta >= self.bs_min_delta
        return pd.Series(passes, index=close.index)

    def _regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """Return True at bars where the market regime allows a long entry.

        Uses a rolling *regime_lookback*-bar VWAP on *spy_df* to determine
        the intraday market bias:

        * Bullish regime (SPY close > rolling VWAP) → entry allowed.
        * Bearish regime (SPY close < rolling VWAP) → entry suppressed.

        Timestamps are aligned via forward-fill so that SPY and the target
        ticker can have slightly different bar timestamps (e.g. due to NaN
        warm-up rows being dropped).  Falls back to all-True (allow all) when
        *spy_df* is ``None`` or too short to compute the rolling VWAP.

        Parameters
        ----------
        df : pd.DataFrame
            The ticker's OHLCV DataFrame.  Must contain a ``datetime`` column.

        Returns
        -------
        pd.Series
            Boolean Series (same index as *df*) — True where the regime
            allows a long entry.
        """
        from .ta_indicators import compute_market_regime

        if self.spy_df is None or self.spy_df.empty:
            return pd.Series(True, index=df.index)

        spy_regime = compute_market_regime(self.spy_df, lookback=self.regime_lookback)

        if "datetime" not in df.columns or "datetime" not in self.spy_df.columns:
            # Cannot align without datetime — fall back to allow-all.
            logger.warning(
                "regime_filter: 'datetime' column missing; regime filter skipped."
            )
            return pd.Series(True, index=df.index)

        # Build a datetime-indexed Series for SPY regime.
        spy_dt = pd.to_datetime(self.spy_df["datetime"])
        spy_regime_dt = pd.Series(spy_regime.values, index=spy_dt)

        # Remove duplicate timestamps (keep last).
        spy_regime_dt = spy_regime_dt[~spy_regime_dt.index.duplicated(keep="last")]
        ticker_dt = pd.to_datetime(df["datetime"])

        # Normalize timezone awareness so that both indices are comparable.
        # Convert to UTC if either side is tz-aware; strip tz otherwise.
        spy_tz = spy_regime_dt.index.tz
        ticker_tz = ticker_dt.dt.tz
        if spy_tz is None and ticker_tz is not None:
            spy_regime_dt.index = spy_regime_dt.index.tz_localize(ticker_tz)
        elif spy_tz is not None and ticker_tz is None:
            ticker_dt = ticker_dt.dt.tz_localize(spy_tz)
        # If both are tz-aware but different zones, convert ticker to spy's zone.
        elif spy_tz is not None and ticker_tz is not None and spy_tz != ticker_tz:
            ticker_dt = ticker_dt.dt.tz_convert(spy_tz)

        # Reindex: for each ticker bar find the most recent SPY regime value.
        ticker_idx = pd.DatetimeIndex(ticker_dt)
        combined_idx = spy_regime_dt.index.union(ticker_idx).sort_values()
        aligned = spy_regime_dt.reindex(combined_idx).ffill()
        aligned = aligned.reindex(ticker_idx)

        # Neutral (0) bars — not enough history — are treated as allowing entry.
        regime_values = aligned.fillna(0).values
        result = pd.Series(regime_values >= 0, index=df.index)
        return result

    def _apply_catalyst_filter(
        self,
        signal: pd.Series,
        df: pd.DataFrame,
    ) -> pd.Series:
        """Zero out signals within ±catalyst_suppress_bars of known catalysts.

        Returns *signal* unchanged when ``use_catalyst_filter`` is ``False``
        or when no catalyst dates are configured.

        For date-only catalysts every bar on that calendar day is suppressed.
        For datetime catalysts the suppression window is
        ±*catalyst_suppress_bars* bars around the catalyst timestamp.

        Parameters
        ----------
        signal:
            Integer (0/1) signal Series aligned with *df*.
        df:
            OHLCV + indicators DataFrame; must contain a ``datetime`` column.

        Returns
        -------
        pd.Series
            Signal with catalyst-affected bars zeroed out.
        """
        if not self.use_catalyst_filter or not self.catalyst_dates:
            return signal

        signal = signal.copy()
        if "datetime" not in df.columns:
            return signal

        dt = pd.to_datetime(df["datetime"])
        suppress_bars = self.catalyst_suppress_bars

        for cat in self.catalyst_dates:
            cat_ts = pd.Timestamp(cat)
            # Treat as a date-only catalyst if the entire time component is midnight
            # (i.e. the user passed a date string like "2024-01-25" or a date-only
            # pd.Timestamp).  For an explicit intraday time, pass a Timestamp with a
            # non-zero hour/minute/second to get the ±N-bars window instead.
            is_date_only = (
                cat_ts.hour == 0
                and cat_ts.minute == 0
                and cat_ts.second == 0
                and cat_ts.microsecond == 0
                and cat_ts.nanosecond == 0
            )
            if is_date_only:
                # Date-only catalyst — suppress entire trading day.
                mask = dt.dt.date == cat_ts.date()
                signal[mask] = 0
            else:
                # Datetime catalyst — suppress ±suppress_bars around the index.
                diffs = (dt - cat_ts).abs()
                closest_idx = diffs.idxmin()
                pos = signal.index.get_loc(closest_idx)
                lo = max(0, pos - suppress_bars)
                hi = min(len(signal) - 1, pos + suppress_bars)
                signal.iloc[lo : hi + 1] = 0

        return signal
