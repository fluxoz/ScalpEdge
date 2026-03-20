"""Black-Scholes options pricing module.

Provides vanilla European call/put pricing and Greeks for **0DTE** options
on any underlying — designed for SPY/TSLA 0DTE scalp strategies.

Usage
-----
>>> bs = BlackScholes(spot=450.0, strike=451.0, r=0.05, sigma=0.20, T=1/252)
>>> bs.call_price()
1.23
>>> bs.delta("call")
0.62
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class BlackScholes:
    """Vanilla Black-Scholes pricing for European options.

    Parameters
    ----------
    spot:   Current price of the underlying.
    strike: Option strike price.
    r:      Annualised risk-free rate (e.g. 0.05 for 5 %).
    sigma:  Annualised implied volatility (e.g. 0.20 for 20 %).
    T:      Time to expiry **in years** (e.g. 1/252 for one trading day).
    q:      Continuous dividend yield (default 0.0).
    """

    spot: float
    strike: float
    r: float
    sigma: float
    T: float
    q: float = 0.0

    # cached internals — populated on first access
    _d1: float = field(default=float("nan"), init=False, repr=False)
    _d2: float = field(default=float("nan"), init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_d()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_d(self) -> None:
        S, K, r, sigma, T, q = (
            self.spot, self.strike, self.r, self.sigma, self.T, self.q
        )
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            self._d1 = float("inf")
            self._d2 = float("inf")
            return
        self._d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        self._d2 = self._d1 - sigma * math.sqrt(T)

    @staticmethod
    def _n(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    @staticmethod
    def _N(x: float) -> float:
        """Standard normal CDF (via math.erfc for numerical stability)."""
        return 0.5 * math.erfc(-x / math.sqrt(2))

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def call_price(self) -> float:
        """Black-Scholes call option premium."""
        S, K, r, T, q = self.spot, self.strike, self.r, self.T, self.q
        if self.T <= 0:
            return max(0.0, S - K)
        return (
            S * math.exp(-q * T) * self._N(self._d1)
            - K * math.exp(-r * T) * self._N(self._d2)
        )

    def put_price(self) -> float:
        """Black-Scholes put option premium."""
        S, K, r, T, q = self.spot, self.strike, self.r, self.T, self.q
        if self.T <= 0:
            return max(0.0, K - S)
        return (
            K * math.exp(-r * T) * self._N(-self._d2)
            - S * math.exp(-q * T) * self._N(-self._d1)
        )

    def price(self, kind: str) -> float:
        """Return call or put price.  *kind* is ``'call'`` or ``'put'``."""
        kind = kind.lower()
        if kind == "call":
            return self.call_price()
        if kind == "put":
            return self.put_price()
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(self, kind: str = "call") -> float:
        """Option delta (sensitivity of price to underlying move)."""
        if self.T <= 0:
            if kind.lower() == "call":
                return 1.0 if self.spot > self.strike else 0.0
            return -1.0 if self.spot < self.strike else 0.0
        exp_qt = math.exp(-self.q * self.T)
        nd1 = self._N(self._d1)
        if kind.lower() == "call":
            return exp_qt * nd1
        return exp_qt * (nd1 - 1)

    def gamma(self) -> float:
        """Option gamma (rate of change of delta)."""
        if self.T <= 0 or self.sigma <= 0:
            return 0.0
        exp_qt = math.exp(-self.q * self.T)
        return (
            exp_qt
            * self._n(self._d1)
            / (self.spot * self.sigma * math.sqrt(self.T))
        )

    def vega(self) -> float:
        """Option vega (sensitivity to vol) — returned per 1 % vol move."""
        if self.T <= 0:
            return 0.0
        exp_qt = math.exp(-self.q * self.T)
        return self.spot * exp_qt * self._n(self._d1) * math.sqrt(self.T) / 100

    def theta(self, kind: str = "call") -> float:
        """Option theta (daily time decay, per calendar day)."""
        S, K, r, sigma, T, q = (
            self.spot, self.strike, self.r, self.sigma, self.T, self.q
        )
        if T <= 0:
            return 0.0
        exp_qt = math.exp(-q * T)
        exp_rt = math.exp(-r * T)
        term1 = -S * exp_qt * self._n(self._d1) * sigma / (2 * math.sqrt(T))
        if kind.lower() == "call":
            theta_annual = (
                term1
                - r * K * exp_rt * self._N(self._d2)
                + q * S * exp_qt * self._N(self._d1)
            )
        else:
            theta_annual = (
                term1
                + r * K * exp_rt * self._N(-self._d2)
                - q * S * exp_qt * self._N(-self._d1)
            )
        return theta_annual / 365  # per calendar day

    def rho(self, kind: str = "call") -> float:
        """Option rho (sensitivity to interest rate)."""
        K, r, T = self.strike, self.r, self.T
        if T <= 0:
            return 0.0
        if kind.lower() == "call":
            return K * T * math.exp(-r * T) * self._N(self._d2) / 100
        return -K * T * math.exp(-r * T) * self._N(-self._d2) / 100

    # ------------------------------------------------------------------
    # Convenience: implied volatility via bisection
    # ------------------------------------------------------------------

    def implied_vol(
        self,
        market_price: float,
        kind: str = "call",
        tol: float = 1e-6,
        max_iter: int = 200,
    ) -> float:
        """Compute implied volatility via bisection.

        Parameters
        ----------
        market_price:
            Observed market option price.
        kind:
            ``'call'`` or ``'put'``.
        tol:
            Convergence tolerance on sigma.
        max_iter:
            Maximum bisection iterations.

        Returns the implied volatility, or ``float('nan')`` if not found.
        """
        lo, hi = 1e-6, 10.0

        def price_at(sigma: float) -> float:
            bs = BlackScholes(
                spot=self.spot, strike=self.strike, r=self.r,
                sigma=sigma, T=self.T, q=self.q,
            )
            return bs.price(kind)

        if price_at(hi) < market_price:
            return float("nan")

        for _ in range(max_iter):
            mid = (lo + hi) / 2
            if abs(hi - lo) < tol:
                return mid
            if price_at(mid) < market_price:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2

    # ------------------------------------------------------------------
    # Class-level factory helper
    # ------------------------------------------------------------------

    @classmethod
    def from_current(
        cls,
        spot: float,
        atm_offset_pct: float = 0.0,
        r: float = 0.05,
        sigma: float = 0.20,
        dte_days: float = 0.0,
        q: float = 0.0,
    ) -> "BlackScholes":
        """Convenience constructor for 0DTE strategies.

        Parameters
        ----------
        spot:
            Current price of the underlying.
        atm_offset_pct:
            Strike offset from ATM in percent (0 = ATM, +1 = 1 % OTM call).
        r:
            Annualised risk-free rate.
        sigma:
            Annualised implied volatility.
        dte_days:
            Days to expiry (default 0 = 0DTE, remaining intraday time).
        q:
            Dividend yield.
        """
        strike = spot * (1 + atm_offset_pct / 100)
        # 0DTE: assume 6.5 trading hours; convert remaining day fraction to years.
        T = max(dte_days / 252, 1 / (252 * 6.5 * 12))  # at least one 5-min bar
        return cls(spot=spot, strike=strike, r=r, sigma=sigma, T=T, q=q)
