"""Strategy module — hybrid signal combination.

The :class:`HybridStrategy` class combines any subset of the available
signal layers into a single entry signal:

* TA rules (EMA crossover, RSI oversold, MACD momentum, BB bounce)
* Markov chain probability filter
* Monte Carlo probability filter
* Black-Scholes delta filter (option premium reasonability gate)
* ML score filter (RF + LSTM combined probability)

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
        """Generate signals and run the backtester. Returns :class:`BacktestResult`."""
        signals = self.generate_signals(df)
        bt = Backtester(**bt_kwargs)
        return bt.run(df, signals, ticker=ticker, strategy_name=self.name)


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
    """Hybrid strategy combining TA + Markov + MonteCarlo + BS delta + ML.

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
