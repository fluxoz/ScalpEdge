"""Probability & statistical modelling module.

Provides:
* :class:`MonteCarlo`  — random-walk forward probability simulation
* :class:`MarkovChain` — order-2 Markov chain on Up/Down/Flat candle states
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

State = Literal["UP", "DOWN", "FLAT"]


# ---------------------------------------------------------------------------
# Monte Carlo random walk
# ---------------------------------------------------------------------------

class MonteCarlo:
    """Estimate the probability that price moves +threshold% within *n_bars*.

    Parameters
    ----------
    n_simulations:
        Number of Monte Carlo paths to generate.
    """

    def __init__(self, n_simulations: int = 2000) -> None:
        self.n_simulations = n_simulations

    def prob_up(
        self,
        returns: pd.Series | np.ndarray,
        n_bars: int = 12,
        threshold_pct: float = 0.5,
    ) -> float:
        """Return probability that cumulative return over *n_bars* > threshold_pct %.

        Parameters
        ----------
        returns:
            Series of percentage log-returns (1 return per bar).
        n_bars:
            Forward simulation horizon in bars.
        threshold_pct:
            Price move threshold in percent.
        """
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10:
            return 0.5  # Not enough data — return neutral

        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)

        rng = np.random.default_rng(42)
        sim_returns = rng.normal(mu, sigma, size=(self.n_simulations, n_bars))
        cumulative = sim_returns.sum(axis=1)
        prob = float((cumulative > threshold_pct / 100).mean())
        return prob

    def prob_down(
        self,
        returns: pd.Series | np.ndarray,
        n_bars: int = 12,
        threshold_pct: float = 0.5,
    ) -> float:
        """Return probability that cumulative return over *n_bars* < -threshold_pct %."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10:
            return 0.5

        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)

        rng = np.random.default_rng(42)
        sim_returns = rng.normal(mu, sigma, size=(self.n_simulations, n_bars))
        cumulative = sim_returns.sum(axis=1)
        prob = float((cumulative < -threshold_pct / 100).mean())
        return prob

    def full_distribution(
        self,
        returns: pd.Series | np.ndarray,
        n_bars: int = 12,
    ) -> np.ndarray:
        """Return the array of *n_simulations* cumulative returns."""
        arr = np.asarray(returns, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 10:
            return np.zeros(self.n_simulations)

        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1)
        rng = np.random.default_rng(42)
        sim_returns = rng.normal(mu, sigma, size=(self.n_simulations, n_bars))
        return sim_returns.sum(axis=1)


# ---------------------------------------------------------------------------
# Markov Chain
# ---------------------------------------------------------------------------

class MarkovChain:
    """Order-*k* Markov chain on candle direction states (UP / DOWN / FLAT).

    Parameters
    ----------
    order:
        Number of previous states to condition on (≥ 1; default 2).
    flat_threshold:
        Returns within ±flat_threshold are classified as FLAT.
    """

    STATES: tuple[State, ...] = ("UP", "DOWN", "FLAT")

    def __init__(self, order: int = 2, flat_threshold: float = 0.0005) -> None:
        if order < 1:
            raise ValueError("order must be ≥ 1")
        self.order = order
        self.flat_threshold = flat_threshold
        self._transitions: dict[tuple[State, ...], dict[State, int]] = defaultdict(
            lambda: {"UP": 0, "DOWN": 0, "FLAT": 0}
        )
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, close: pd.Series) -> "MarkovChain":
        """Estimate transition probabilities from a price series.

        Parameters
        ----------
        close:
            Closing price series (chronological).
        """
        states = self._classify(close)
        self._transitions = defaultdict(lambda: {"UP": 0, "DOWN": 0, "FLAT": 0})

        for i in range(self.order, len(states)):
            context = tuple(states[i - self.order : i])
            next_state = states[i]
            self._transitions[context][next_state] += 1

        self._fitted = True
        logger.debug(
            "MarkovChain order=%d fitted on %d states; %d unique contexts",
            self.order,
            len(states),
            len(self._transitions),
        )
        return self

    def predict_proba(self, recent_states: list[State]) -> dict[State, float]:
        """Return transition probabilities given the last *order* states.

        Parameters
        ----------
        recent_states:
            The *order* most recent candle states (oldest first).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_proba()")
        if len(recent_states) < self.order:
            raise ValueError(f"Need {self.order} states; got {len(recent_states)}")

        context = tuple(recent_states[-self.order :])
        counts = self._transitions.get(context)

        if counts is None:
            # Unseen context: return uniform distribution.
            return {"UP": 1 / 3, "DOWN": 1 / 3, "FLAT": 1 / 3}

        total = sum(counts.values())
        if total == 0:
            return {"UP": 1 / 3, "DOWN": 1 / 3, "FLAT": 1 / 3}

        return {s: counts[s] / total for s in self.STATES}

    def get_states_series(self, close: pd.Series) -> list[State]:
        """Return the sequence of states for a price series."""
        return self._classify(close)

    # ------------------------------------------------------------------
    # Internal

    def _classify(self, close: pd.Series) -> list[State]:
        ret = close.pct_change().fillna(0)
        states: list[State] = []
        for r in ret:
            if r > self.flat_threshold:
                states.append("UP")
            elif r < -self.flat_threshold:
                states.append("DOWN")
            else:
                states.append("FLAT")
        return states
