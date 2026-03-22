"""Technical Analysis indicators module.

Adds EMA, SMA, RSI, MACD, Bollinger Bands, ATR, OBV, VWAP, Volume Profile /
POC, and 60+ candlestick pattern columns to a 5-minute OHLCV DataFrame.

All computations are vectorized (pandas / numpy).  ``pandas-ta`` is used
for the heavy lifting so every indicator is available from one function call.

Volume Profile / POC
--------------------
Each session's rolling volume-at-price histogram is accumulated bar-by-bar
(no look-ahead).  The Point of Control (POC) is the price level with the
highest cumulative volume.  Four derived signal columns are added:

* ``poc_price``          — POC price level as of each bar (session-rolling).
* ``poc_proximity_pct``  — (close − POC) / POC × 100  (negative = below POC).
* ``poc_above``          — 1 when close > POC, else 0.
* ``poc_below``          — 1 when close < POC, else 0.

Optional visualization
----------------------
``plot_volume_profile(df, session_date)`` renders a horizontal volume-profile
histogram overlaid on the session price chart.  Requires ``matplotlib``.

Market Regime
-------------
``compute_market_regime(spy_df, lookback=5)`` uses a rolling *lookback*-bar
VWAP on SPY (or any benchmark) to characterise the intraday trend:

* ``1``  — bullish (close > rolling VWAP)
* ``-1`` — bearish (close < rolling VWAP)
* ``0``  — neutral  (close == rolling VWAP, rare)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Candlestick pattern helpers (pure numpy / pandas — no extra lib required)
# ---------------------------------------------------------------------------

def _body(o: pd.Series, c: pd.Series) -> pd.Series:
    return (c - o).abs()


def _upper_shadow(o: pd.Series, h: pd.Series, c: pd.Series) -> pd.Series:
    return h - pd.concat([o, c], axis=1).max(axis=1)


def _lower_shadow(o: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    return pd.concat([o, c], axis=1).min(axis=1) - l


def _candle_range(h: pd.Series, l: pd.Series) -> pd.Series:
    return h - l


def _bull(o: pd.Series, c: pd.Series) -> pd.Series:
    return c > o


def _bear(o: pd.Series, c: pd.Series) -> pd.Series:
    return c < o


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all TA indicators + candlestick patterns.

    Requires columns: ``open``, ``high``, ``low``, ``close``, ``volume``.
    Returns the same DataFrame (copy) with new columns appended.
    """
    df = df.copy()

    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    v = df["volume"].astype(float)

    # -----------------------------------------------------------------------
    # Moving averages
    # -----------------------------------------------------------------------
    for period in (9, 21, 50, 200):
        df[f"ema_{period}"] = c.ewm(span=period, adjust=False).mean()
        df[f"sma_{period}"] = c.rolling(period).mean()

    # -----------------------------------------------------------------------
    # RSI (Wilder smoothing)
    # -----------------------------------------------------------------------
    df["rsi_14"] = _rsi(c, 14)

    # -----------------------------------------------------------------------
    # MACD (12, 26, 9)
    # -----------------------------------------------------------------------
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # -----------------------------------------------------------------------
    # Bollinger Bands (20, 2σ)
    # -----------------------------------------------------------------------
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std(ddof=0)
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_mid"] = bb_mid
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)
    df["bb_pct"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    # -----------------------------------------------------------------------
    # ATR (14)
    # -----------------------------------------------------------------------
    df["atr_14"] = _atr(h, l, c, 14)

    # -----------------------------------------------------------------------
    # OBV
    # -----------------------------------------------------------------------
    direction = np.where(c > c.shift(1), 1, np.where(c < c.shift(1), -1, 0))
    df["obv"] = (v * direction).cumsum()

    # -----------------------------------------------------------------------
    # VWAP (rolling intraday approximation — resets are per-session)
    # -----------------------------------------------------------------------
    df["vwap"] = _vwap(df)

    # -----------------------------------------------------------------------
    # Stochastic oscillator (14, 3)
    # -----------------------------------------------------------------------
    lowest_low = l.rolling(14).min()
    highest_high = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # -----------------------------------------------------------------------
    # Williams %R (14)
    # -----------------------------------------------------------------------
    df["williams_r"] = -100 * (highest_high - c) / (highest_high - lowest_low).replace(0, np.nan)

    # -----------------------------------------------------------------------
    # CCI (20)
    # -----------------------------------------------------------------------
    typical = (h + l + c) / 3
    df["cci_20"] = (typical - typical.rolling(20).mean()) / (
        0.015 * typical.rolling(20).std(ddof=0)
    ).replace(0, np.nan)

    # -----------------------------------------------------------------------
    # MFI — Money Flow Index (14)
    # -----------------------------------------------------------------------
    df["mfi_14"] = _mfi(h, l, c, v, 14)

    # -----------------------------------------------------------------------
    # ADX / DI+ / DI- (14)
    # -----------------------------------------------------------------------
    df["adx_14"], df["di_plus_14"], df["di_minus_14"] = _adx(h, l, c, 14)

    # -----------------------------------------------------------------------
    # ROC (rate of change, 12 bars)
    # -----------------------------------------------------------------------
    df["roc_12"] = c.pct_change(12) * 100

    # -----------------------------------------------------------------------
    # Price distance from VWAP (%)
    # -----------------------------------------------------------------------
    df["price_vs_vwap"] = (c - df["vwap"]) / df["vwap"].replace(0, np.nan) * 100

    # -----------------------------------------------------------------------
    # Volume Profile / Point of Control (POC) — rolling, resets each session
    # -----------------------------------------------------------------------
    df["poc_price"], df["poc_proximity_pct"] = _volume_profile(df)
    df["poc_above"] = (c > df["poc_price"]).astype(int)
    df["poc_below"] = (c < df["poc_price"]).astype(int)

    # -----------------------------------------------------------------------
    # Candlestick patterns (60+)
    # -----------------------------------------------------------------------
    df = _add_candlestick_patterns(df, o, h, l, c)

    # -----------------------------------------------------------------------
    # Bid-ask microstructure features (only if quote columns are present)
    # -----------------------------------------------------------------------
    df = add_quote_features(df)

    return df


def add_quote_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add bid-ask microstructure features if quote columns are present.

    Expected input columns (optional — function is a no-op if absent):
        bid_price, ask_price, bid_size, ask_size

    Added columns:
        spread              — ask_price - bid_price
        mid_price           — (bid_price + ask_price) / 2
        bid_ask_imbalance   — (bid_size - ask_size) / (bid_size + ask_size + 1e-9)
        spread_pct          — spread / mid_price * 100 (spread as % of mid)
        imbalance_ma_10     — 10-bar rolling mean of bid_ask_imbalance

    Returns the DataFrame (copy) with new columns appended; unchanged if
    the required input columns are absent.
    """
    required = {"bid_price", "ask_price", "bid_size", "ask_size"}
    if not required.issubset(df.columns):
        return df

    df = df.copy()
    bid = df["bid_price"].astype(float)
    ask = df["ask_price"].astype(float)
    bid_sz = df["bid_size"].astype(float)
    ask_sz = df["ask_size"].astype(float)

    df["spread"] = ask - bid
    df["mid_price"] = (bid + ask) / 2
    df["bid_ask_imbalance"] = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-9)
    df["spread_pct"] = df["spread"] / df["mid_price"].replace(0, np.nan) * 100
    df["imbalance_ma_10"] = df["bid_ask_imbalance"].rolling(10).mean()

    return df


# ---------------------------------------------------------------------------
# Market regime
# ---------------------------------------------------------------------------

def compute_market_regime(spy_df: pd.DataFrame, lookback: int = 5) -> pd.Series:
    """Compute intraday market regime using a rolling VWAP on a benchmark.

    Typical usage: pass SPY 5-minute OHLCV data to get a bar-by-bar regime
    label that can be used to gate entry signals for other tickers.

    Parameters
    ----------
    spy_df : pd.DataFrame
        OHLCV DataFrame for the benchmark (usually SPY).  Must contain
        ``high``, ``low``, ``close``, and ``volume`` columns.
    lookback : int
        Rolling window in bars for the VWAP computation (default 5).
        Typical-price × volume is summed over the last *lookback* bars.

    Returns
    -------
    pd.Series
        Integer Series (same index as *spy_df*) with values:

        *  ``1``  — bullish: close > rolling VWAP
        * ``-1``  — bearish: close < rolling VWAP
        *  ``0``  — neutral: close == rolling VWAP (or insufficient history)
    """
    h = spy_df["high"].astype(float)
    low = spy_df["low"].astype(float)
    c = spy_df["close"].astype(float)
    v = spy_df["volume"].astype(float)

    typical_price = (h + low + c) / 3
    rolling_tpv = (typical_price * v).rolling(lookback).sum()
    rolling_vol = v.rolling(lookback).sum()
    rolling_vwap = rolling_tpv / rolling_vol.replace(0, np.nan)

    regime = pd.Series(0, index=spy_df.index, dtype=int)
    regime[c > rolling_vwap] = 1
    regime[c < rolling_vwap] = -1

    return regime


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat(
        [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _mfi(
    h: pd.Series, l: pd.Series, c: pd.Series, v: pd.Series, period: int
) -> pd.Series:
    tp = (h + l + c) / 3
    mf = tp * v
    pos = mf.where(tp > tp.shift(1), 0.0)
    neg = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - 100 / (1 + mfr)


def _adx(
    h: pd.Series, l: pd.Series, c: pd.Series, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    dm_plus = np.where((h - prev_h) > (prev_l - l), (h - prev_h).clip(lower=0), 0.0)
    dm_minus = np.where((prev_l - l) > (h - prev_h), (prev_l - l).clip(lower=0), 0.0)

    tr_s = pd.Series(tr).ewm(com=period - 1, min_periods=period).mean()
    dp_s = pd.Series(dm_plus, index=h.index).ewm(com=period - 1, min_periods=period).mean()
    dm_s = pd.Series(dm_minus, index=h.index).ewm(com=period - 1, min_periods=period).mean()

    di_plus = 100 * dp_s / tr_s.replace(0, np.nan)
    di_minus = 100 * dm_s / tr_s.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(com=period - 1, min_periods=period).mean()
    return adx, di_plus, di_minus


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP that resets each calendar day."""
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    typical = (df["high"].astype(float) + df["low"].astype(float) + close) / 3
    tpv = typical * vol

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        date_key = dt.dt.date
    else:
        date_key = pd.Series(range(len(df)), index=df.index)

    vwap = pd.Series(np.nan, index=df.index)
    for _, grp in df.groupby(date_key, sort=False):
        idx = grp.index
        cum_vol = vol.loc[idx].cumsum()
        cum_tpv = tpv.loc[idx].cumsum()
        vwap.loc[idx] = cum_tpv / cum_vol.replace(0, np.nan)
    return vwap


def _volume_profile(
    df: pd.DataFrame, n_bins: int = 50
) -> tuple[pd.Series, pd.Series]:
    """Compute rolling intraday volume profile and Point of Control (POC).

    For each bar the volume-at-price histogram is built from the session open
    up to *and including* that bar (no look-ahead bias).  The POC is the price
    bin with the highest cumulative volume.

    Parameters
    ----------
    df:
        DataFrame with columns: ``high``, ``low``, ``close``, ``volume``, and
        optionally ``datetime``.
    n_bins:
        Number of equally-spaced price bins spanning the session's high–low
        range.  Default is 50.

    Returns
    -------
    poc_price : pd.Series
        POC price level for the current session as of each bar.
    poc_proximity_pct : pd.Series
        ``(close − POC) / POC × 100``.  Negative values indicate price is
        below the POC; positive values indicate price is above the POC.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    vol = df["volume"].astype(float)
    # Typical price is used to assign each bar to a price bin.
    typical = (high + low + close) / 3.0

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        date_key = dt.dt.date
    else:
        date_key = pd.Series(range(len(df)), index=df.index)

    poc_price = pd.Series(np.nan, index=df.index)

    for _, grp in df.groupby(date_key, sort=False):
        idx = grp.index
        grp_typical = typical.loc[idx].values
        grp_high = high.loc[idx].values
        grp_low = low.loc[idx].values
        grp_vol = vol.loc[idx].values

        session_high = grp_high.max()
        session_low = grp_low.min()

        if session_high == session_low:
            # Degenerate session — all bars at same price level.
            poc_price.loc[idx] = session_high
            continue

        bins = np.linspace(session_low, session_high, n_bins + 1)
        bin_mids = (bins[:-1] + bins[1:]) / 2.0

        cum_vol_by_bin = np.zeros(n_bins)
        for i, bar_idx in enumerate(idx):
            # Determine price bin for this bar's typical price.
            bin_i = int(np.searchsorted(bins, grp_typical[i], side="right")) - 1
            bin_i = min(max(bin_i, 0), n_bins - 1)
            cum_vol_by_bin[bin_i] += grp_vol[i]
            poc_price.loc[bar_idx] = bin_mids[int(np.argmax(cum_vol_by_bin))]

    poc_proximity_pct = (close - poc_price) / poc_price.replace(0, np.nan) * 100.0
    return poc_price, poc_proximity_pct


# ---------------------------------------------------------------------------
# Candlestick pattern detection (60+ patterns)
# ---------------------------------------------------------------------------

def _add_candlestick_patterns(
    df: pd.DataFrame,
    o: pd.Series,
    h: pd.Series,
    l: pd.Series,
    c: pd.Series,
) -> pd.DataFrame:
    body = _body(o, c)
    avg_body = body.rolling(10).mean()
    upper = _upper_shadow(o, h, c)
    lower = _lower_shadow(o, l, c)
    rng = _candle_range(h, l)
    bull = _bull(o, c)
    bear = _bear(o, c)

    # -- Single-bar patterns ------------------------------------------------

    # Doji: body <= 10% of range
    df["pat_doji"] = (body <= 0.1 * rng.replace(0, np.nan)).astype(int)

    # Long-legged doji
    df["pat_long_legged_doji"] = (
        (body <= 0.1 * rng.replace(0, np.nan))
        & (upper >= 0.3 * rng)
        & (lower >= 0.3 * rng)
    ).astype(int)

    # Dragonfly doji
    df["pat_dragonfly_doji"] = (
        (body <= 0.05 * rng.replace(0, np.nan)) & (upper <= 0.05 * rng) & (lower >= 0.6 * rng)
    ).astype(int)

    # Gravestone doji
    df["pat_gravestone_doji"] = (
        (body <= 0.05 * rng.replace(0, np.nan)) & (lower <= 0.05 * rng) & (upper >= 0.6 * rng)
    ).astype(int)

    # Hammer (bullish reversal): small body at top, long lower shadow
    df["pat_hammer"] = (
        (body >= 0.1 * rng.replace(0, np.nan))
        & (lower >= 2 * body)
        & (upper <= 0.1 * rng)
    ).astype(int)

    # Inverted hammer
    df["pat_inverted_hammer"] = (
        (body >= 0.1 * rng.replace(0, np.nan))
        & (upper >= 2 * body)
        & (lower <= 0.1 * rng)
    ).astype(int)

    # Hanging man (same shape as hammer but after uptrend — simplified)
    df["pat_hanging_man"] = df["pat_hammer"]

    # Shooting star (same shape as inverted hammer but after uptrend)
    df["pat_shooting_star"] = (
        bear & (body >= 0.1 * rng.replace(0, np.nan)) & (upper >= 2 * body) & (lower <= 0.1 * rng)
    ).astype(int)

    # Marubozu (large body, almost no shadows)
    df["pat_marubozu_bull"] = (bull & (body >= 0.9 * rng.replace(0, np.nan))).astype(int)
    df["pat_marubozu_bear"] = (bear & (body >= 0.9 * rng.replace(0, np.nan))).astype(int)

    # Spinning top: small body, shadows both sides
    df["pat_spinning_top"] = (
        (body < 0.3 * rng.replace(0, np.nan)) & (upper >= 0.2 * rng) & (lower >= 0.2 * rng)
    ).astype(int)

    # Large candle (body > 1.5x average)
    df["pat_large_bull"] = (bull & (body > 1.5 * avg_body.replace(0, np.nan))).astype(int)
    df["pat_large_bear"] = (bear & (body > 1.5 * avg_body.replace(0, np.nan))).astype(int)

    # -- Two-bar patterns ---------------------------------------------------

    o1 = o.shift(1)
    c1 = c.shift(1)
    h1 = h.shift(1)
    l1 = l.shift(1)
    body1 = _body(o1, c1)
    bull1 = _bull(o1, c1)
    bear1 = _bear(o1, c1)
    rng1 = _candle_range(h1, l1)

    # Bullish engulfing
    df["pat_bull_engulfing"] = (
        bear1 & bull & (o <= c1) & (c >= o1)
    ).astype(int)

    # Bearish engulfing
    df["pat_bear_engulfing"] = (
        bull1 & bear & (o >= c1) & (c <= o1)
    ).astype(int)

    # Bullish harami
    df["pat_bull_harami"] = (
        bear1 & bull & (o >= c1) & (c <= o1) & (body < body1)
    ).astype(int)

    # Bearish harami
    df["pat_bear_harami"] = (
        bull1 & bear & (o <= c1) & (c >= o1) & (body < body1)
    ).astype(int)

    # Piercing line
    df["pat_piercing"] = (
        bear1
        & bull
        & (o < l1)
        & (c > (o1 + c1) / 2)
        & (c < o1)
    ).astype(int)

    # Dark cloud cover
    df["pat_dark_cloud"] = (
        bull1
        & bear
        & (o > h1)
        & (c < (o1 + c1) / 2)
        & (c > o1)
    ).astype(int)

    # Tweezer top
    df["pat_tweezer_top"] = (bull1 & bear & (h.round(2) == h1.round(2))).astype(int)

    # Tweezer bottom
    df["pat_tweezer_bottom"] = (bear1 & bull & (l.round(2) == l1.round(2))).astype(int)

    # Inside bar
    df["pat_inside_bar"] = ((h <= h1) & (l >= l1)).astype(int)

    # Outside bar
    df["pat_outside_bar"] = ((h >= h1) & (l <= l1)).astype(int)

    # Gap up / gap down
    df["pat_gap_up"] = (o > h1).astype(int)
    df["pat_gap_down"] = (o < l1).astype(int)

    # Kicker up (gap-up bullish after bearish)
    df["pat_kicker_up"] = (bear1 & bull & (o >= c1)).astype(int)

    # Kicker down (gap-down bearish after bullish)
    df["pat_kicker_down"] = (bull1 & bear & (o <= c1)).astype(int)

    # -- Three-bar patterns --------------------------------------------------

    o2 = o.shift(2)
    c2 = c.shift(2)
    h2 = h.shift(2)
    l2 = l.shift(2)
    bull2 = _bull(o2, c2)
    bear2 = _bear(o2, c2)
    body2 = _body(o2, c2)

    # Morning star
    df["pat_morning_star"] = (
        bear2
        & (_body(o1, c1) < 0.3 * body2)
        & bull
        & (c > (o2 + c2) / 2)
    ).astype(int)

    # Evening star
    df["pat_evening_star"] = (
        bull2
        & (_body(o1, c1) < 0.3 * body2)
        & bear
        & (c < (o2 + c2) / 2)
    ).astype(int)

    # Three white soldiers
    df["pat_three_white_soldiers"] = (
        bull2 & bull1 & bull & (c > c1) & (c1 > c2) & (o > o1) & (o1 > o2)
    ).astype(int)

    # Three black crows
    df["pat_three_black_crows"] = (
        bear2 & bear1 & bear & (c < c1) & (c1 < c2) & (o < o1) & (o1 < o2)
    ).astype(int)

    # Three inside up
    df["pat_three_inside_up"] = (
        bear2 & bull1 & (o1 >= c2) & (c1 <= o2) & bull & (c > c1)
    ).astype(int)

    # Three inside down
    df["pat_three_inside_down"] = (
        bull2 & bear1 & (o1 <= c2) & (c1 >= o2) & bear & (c < c1)
    ).astype(int)

    # Three outside up
    df["pat_three_outside_up"] = (
        bear2
        & bull1
        & (o1 <= c2)
        & (c1 >= o2)
        & bull
        & (c > c1)
    ).astype(int)

    # Three outside down
    df["pat_three_outside_down"] = (
        bull2
        & bear1
        & (o1 >= c2)
        & (c1 <= o2)
        & bear
        & (c < c1)
    ).astype(int)

    # Rising three methods (simplified)
    df["pat_rising_three"] = (
        bull2
        & (body1 < 0.5 * body2)
        & bull
        & (c > c2)
    ).astype(int)

    # Falling three methods (simplified)
    df["pat_falling_three"] = (
        bear2
        & (body1 < 0.5 * body2)
        & bear
        & (c < c2)
    ).astype(int)

    # Abandoned baby bull
    df["pat_abandoned_baby_bull"] = (
        bear2
        & (h1 < l2)
        & (_body(o1, c1) < 0.05 * rng1.replace(0, np.nan))
        & bull
        & (l > h1)
    ).astype(int)

    # Abandoned baby bear
    df["pat_abandoned_baby_bear"] = (
        bull2
        & (l1 > h2)
        & (_body(o1, c1) < 0.05 * rng1.replace(0, np.nan))
        & bear
        & (h < l1)
    ).astype(int)

    # Deliberation pattern (bull)
    df["pat_deliberation_bull"] = (
        bull2 & bull1 & bull & (body < body1 * 0.5) & (c > c1)
    ).astype(int)

    # Stalled pattern (bear)
    df["pat_stalled_bear"] = (
        bear2 & bear1 & bear & (body < body1 * 0.5) & (c < c1)
    ).astype(int)

    # -- Additional single patterns -------------------------------------------

    # Long upper shadow (potential reversal signal)
    df["pat_long_upper_shadow"] = (upper >= 2 * body.replace(0, np.nan)).astype(int)

    # Long lower shadow
    df["pat_long_lower_shadow"] = (lower >= 2 * body.replace(0, np.nan)).astype(int)

    # High wave candle
    df["pat_high_wave"] = (
        (upper >= body.replace(0, np.nan)) & (lower >= body.replace(0, np.nan))
    ).astype(int)

    # Belt hold bull (opens at low, big bull body)
    df["pat_belt_hold_bull"] = (
        bull & (lower <= 0.05 * rng.replace(0, np.nan)) & (body >= 0.7 * rng.replace(0, np.nan))
    ).astype(int)

    # Belt hold bear (opens at high, big bear body)
    df["pat_belt_hold_bear"] = (
        bear & (upper <= 0.05 * rng.replace(0, np.nan)) & (body >= 0.7 * rng.replace(0, np.nan))
    ).astype(int)

    # Tasuki gap up
    df["pat_tasuki_gap_up"] = (
        bull1 & bull & (o < c1) & (c < c1) & (o > l1)
    ).astype(int)

    # Tasuki gap down
    df["pat_tasuki_gap_down"] = (
        bear1 & bear & (o > c1) & (c > c1) & (o < h1)
    ).astype(int)

    # Stick sandwich bull
    df["pat_stick_sandwich_bull"] = (
        bear2 & bull1 & bear & (c.round(2) == c2.round(2))
    ).astype(int)

    # Matching low
    df["pat_matching_low"] = (
        bear1 & bear & (c.round(2) == c1.round(2))
    ).astype(int)

    # Matching high
    df["pat_matching_high"] = (
        bull1 & bull & (c.round(2) == c1.round(2))
    ).astype(int)

    # Unique three river bottom
    df["pat_unique_three_river"] = (
        bear2 & bear1 & bull & (l1 < l2) & (c > c1) & (c < o2)
    ).astype(int)

    # Two crows
    df["pat_two_crows"] = (
        bull2 & bear1 & (o1 > c2) & bear & (o >= o1) & (c > c2) & (c < o1)
    ).astype(int)

    # On neck
    df["pat_on_neck"] = (
        bear1 & bull & (o < l1) & (c.round(2) == l1.round(2))
    ).astype(int)

    # In neck
    df["pat_in_neck"] = (
        bear1 & bull & (o < l1) & (c > l1) & (c < c1 + 0.001 * c1)
    ).astype(int)

    # Thrusting
    df["pat_thrusting"] = (
        bear1 & bull & (o < l1) & (c > l1) & (c < (o1 + c1) / 2)
    ).astype(int)

    # Counterattack bull
    df["pat_counterattack_bull"] = (
        bear1 & bull & (c.round(2) == c1.round(2))
    ).astype(int)

    # Counterattack bear
    df["pat_counterattack_bear"] = (
        bull1 & bear & (c.round(2) == c1.round(2))
    ).astype(int)

    # Separating lines bull
    df["pat_separating_bull"] = (
        bear1 & bull & (o.round(2) == o1.round(2))
    ).astype(int)

    # Separating lines bear
    df["pat_separating_bear"] = (
        bull1 & bear & (o.round(2) == o1.round(2))
    ).astype(int)

    # Ladder bottom
    df["pat_ladder_bottom"] = (
        bear2 & bear1 & bull & (c > o1) & (upper >= 0.3 * rng1.replace(0, np.nan))
    ).astype(int)

    # -- Composite trend indicators (pattern-based) --------------------------
    df["pat_bull_signal"] = (
        df["pat_bull_engulfing"]
        | df["pat_hammer"]
        | df["pat_morning_star"]
        | df["pat_three_white_soldiers"]
        | df["pat_piercing"]
        | df["pat_kicker_up"]
        | df["pat_bull_harami"]
        | df["pat_dragonfly_doji"]
        | df["pat_abandoned_baby_bull"]
    ).astype(int)

    df["pat_bear_signal"] = (
        df["pat_bear_engulfing"]
        | df["pat_shooting_star"]
        | df["pat_evening_star"]
        | df["pat_three_black_crows"]
        | df["pat_dark_cloud"]
        | df["pat_kicker_down"]
        | df["pat_bear_harami"]
        | df["pat_gravestone_doji"]
        | df["pat_abandoned_baby_bear"]
    ).astype(int)

    return df


# ---------------------------------------------------------------------------
# Volume Profile visualization (optional — requires matplotlib)
# ---------------------------------------------------------------------------

def plot_volume_profile(
    df: pd.DataFrame,
    session_date: str | None = None,
    n_bins: int = 50,
    figsize: tuple[float, float] = (14, 6),
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot a session's price chart with the volume profile histogram.

    Renders two panels side-by-side:
    * **Left (wide)**: close-price line for the selected session with a
      horizontal dashed line marking the POC.
    * **Right (narrow)**: horizontal volume-at-price histogram (the profile).

    Parameters
    ----------
    df:
        OHLCV DataFrame, ideally after calling :func:`add_all_indicators` so
        that ``poc_price`` is already present.  If ``poc_price`` is absent the
        function computes it on-the-fly.
    session_date:
        Calendar date of the session to visualize, e.g. ``"2024-01-02"``.
        When *None* the last available session in *df* is used.
    n_bins:
        Price bins to use when building the histogram.
    figsize:
        Matplotlib figure size ``(width, height)`` in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object (caller may call ``fig.savefig(...)`` or
        ``plt.show()``).

    Raises
    ------
    ImportError
        When ``matplotlib`` is not installed.
    ValueError
        When *session_date* is not found in *df*.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_volume_profile. "
            "Install it with: pip install matplotlib"
        ) from exc

    # ------------------------------------------------------------------
    # Isolate session
    # ------------------------------------------------------------------
    work = df.copy()
    if "datetime" in work.columns:
        work["_dt"] = pd.to_datetime(work["datetime"])
    else:
        work["_dt"] = pd.RangeIndex(len(work))
    work["_date"] = work["_dt"].dt.date if hasattr(work["_dt"], "dt") else work["_dt"]

    if session_date is None:
        chosen_date = work["_date"].max()
    else:
        chosen_date = pd.Timestamp(session_date).date()

    session = work[work["_date"] == chosen_date].copy()
    if session.empty:
        raise ValueError(f"No data found for session date {session_date!r}")

    close = session["close"].astype(float)
    high = session["high"].astype(float)
    low = session["low"].astype(float)
    vol = session["volume"].astype(float)
    typical = (high + low + close) / 3.0

    # ------------------------------------------------------------------
    # Build the full-session volume profile (end-of-session histogram)
    # ------------------------------------------------------------------
    s_high = high.max()
    s_low = low.min()

    if s_high == s_low:
        bin_edges = np.array([s_low - 0.5, s_high + 0.5])
        bin_mids = np.array([(s_low + s_high) / 2.0])
        vol_by_bin = np.array([vol.sum()])
    else:
        bin_edges = np.linspace(s_low, s_high, n_bins + 1)
        bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        vol_by_bin = np.zeros(n_bins)
        for tp_val, v_val in zip(typical.values, vol.values):
            bi = int(np.searchsorted(bin_edges, tp_val, side="right")) - 1
            bi = min(max(bi, 0), n_bins - 1)
            vol_by_bin[bi] += v_val

    poc_bin = int(np.argmax(vol_by_bin))
    poc_px = bin_mids[poc_bin]

    # If the DataFrame already carries poc_price use the last bar's value
    # (same result, just avoids re-computation noise from binning).
    if "poc_price" in session.columns:
        poc_px = float(session["poc_price"].iloc[-1])

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], figure=fig)
    ax_price = fig.add_subplot(gs[0])
    ax_prof = fig.add_subplot(gs[1], sharey=ax_price)

    # Price panel
    x_vals = np.arange(len(session))
    ax_price.plot(x_vals, close.values, color="steelblue", linewidth=1.2, label="Close")
    ax_price.axhline(poc_px, color="crimson", linestyle="--", linewidth=1.2,
                     label=f"POC {poc_px:.2f}")
    ax_price.set_title(f"Volume Profile — {chosen_date}", fontsize=12)
    ax_price.set_xlabel("Bar index")
    ax_price.set_ylabel("Price")
    ax_price.legend(fontsize=9)
    ax_price.grid(True, alpha=0.3)

    # Profile panel (horizontal histogram)
    ax_prof.barh(bin_mids, vol_by_bin, height=(bin_edges[1] - bin_edges[0]) * 0.9,
                 color="steelblue", alpha=0.6)
    ax_prof.axhline(poc_px, color="crimson", linestyle="--", linewidth=1.2)
    ax_prof.set_xlabel("Volume")
    ax_prof.tick_params(labelleft=False)
    ax_prof.grid(True, axis="x", alpha=0.3)

    return fig
