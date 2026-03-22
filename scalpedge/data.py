"""Data management module — fetch, store, and update 5-minute OHLCV data.

Primary data source: **Polygon.io** (Stocks Advanced plan).
  - Set the ``POLYGON_API_KEY`` environment variable to enable.
  - Stocks Advanced: unlimited API calls/minute, 5+ years of historical minute aggregates.

Fallback data source: **yfinance** (used when ``POLYGON_API_KEY`` is not set).
  - Free; limited to ~60 days of 5-min history per request (chunked automatically).

Data is persisted in Parquet format under the ``data/`` directory so the
dataset grows incrementally on every run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_INTERVAL = "5m"

# Maximum lookback (in days) supported by **yfinance** for each intraday
# interval.  Polygon's free tier caps at 2 years for all intervals.
_INTERVAL_MAX_DAYS: dict[str, int] = {
    "1m": 7,
    "2m": 60,
    "5m": 59,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
    "1d": 36500,   # daily — essentially unlimited
    "5d": 36500,
    "1wk": 36500,
    "1mo": 36500,
    "3mo": 36500,
}

# yfinance free tier returns at most ~60 days of 5-minute history in one pull.
_MAX_DAYS_PER_FETCH = 59

# Timedelta representing one bar for each supported interval, used to step
# past the last stored bar when doing incremental updates.
_INTERVAL_STEP: dict[str, pd.Timedelta] = {
    "1m": pd.Timedelta(minutes=1),
    "2m": pd.Timedelta(minutes=2),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "60m": pd.Timedelta(hours=1),
    "90m": pd.Timedelta(minutes=90),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
    "5d": pd.Timedelta(days=5),
    "1wk": pd.Timedelta(weeks=1),
    "1mo": pd.Timedelta(days=31),
    "3mo": pd.Timedelta(days=92),
}

# ---------------------------------------------------------------------------
# Polygon.io helpers
# ---------------------------------------------------------------------------

# Stocks Advanced plan: unlimited API calls.  Rate limiting is disabled.
# Set to a positive float to re-enable pacing if needed (seconds per call).
_POLYGON_RATE_LIMIT_SECONDS = 0.0

# Stocks Advanced plan provides up to 20 years of history for every ticker.
_POLYGON_MAX_HISTORY_DAYS = 7305

# Maximum bars returned per Polygon API page (their documented hard limit).
_POLYGON_MAX_LIMIT = 50_000

# Detect the yfinance parameter name for disabling multi-level column headers.
# The parameter was named 'multi_level_column' in 0.2.x and renamed to
# 'multi_level_index' in 1.0+.
_yf_major = int(yf.__version__.split(".")[0])
_MULTI_LEVEL_KWARG = "multi_level_index" if _yf_major >= 1 else "multi_level_column"

# Mapping from our interval strings to Polygon (multiplier, timespan) pairs.
_INTERVAL_TO_POLYGON: dict[str, tuple[int, str]] = {
    "1m":  (1,  "minute"),
    "2m":  (2,  "minute"),
    "5m":  (5,  "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "60m": (60, "minute"),
    "90m": (90, "minute"),
    "1h":  (1,  "hour"),
    "1d":  (1,  "day"),
    "1wk": (1,  "week"),
    "1mo": (1,  "month"),
}


class PolygonClient:
    """Thin wrapper around the Polygon.io REST API (Stocks Advanced plan).

    Parameters
    ----------
    api_key:
        Polygon API key.  If *None*, falls back to the ``POLYGON_API_KEY``
        environment variable.

    Notes
    -----
    Stocks Advanced plan features used by this client:

    * **Unlimited API calls** — rate limiting is disabled by default.
    * **5+ years of minute-bar history** — via the v2 aggregates endpoint.
    * **Tick-level trade data** — via ``/v3/trades/{ticker}``.
    * **NBBO quote data** — via ``/v3/quotes/{ticker}``.
    * **Market snapshots** — via ``/v2/snapshot/locale/us/markets/stocks/tickers``.
    * **News articles** — via ``/v2/reference/news``.
    * **Corporate events** — via ``/vX/reference/tickers/{ticker}/events``.
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not key:
            raise ValueError(
                "Polygon API key not found.  Set the POLYGON_API_KEY "
                "environment variable or pass api_key= explicitly."
            )
        self._api_key = key
        self._last_call_time: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fetch_aggs(
        self,
        ticker: str,
        interval: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch aggregate (OHLCV) bars from Polygon, following pagination.

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"SPY"``).
        interval:
            Bar interval string (e.g. ``"5m"``, ``"1h"``, ``"1d"``).
        start:
            Inclusive UTC start timestamp.
        end:
            Exclusive UTC end timestamp.

        Returns
        -------
        pd.DataFrame
            Columns: datetime (UTC, tz-aware), open, high, low, close, volume,
            ticker.  Empty DataFrame if no data is available.
        """
        if interval not in _INTERVAL_TO_POLYGON:
            raise ValueError(
                f"Interval '{interval}' is not supported by Polygon. "
                f"Choose one of: {', '.join(_INTERVAL_TO_POLYGON)}"
            )
        multiplier, timespan = _INTERVAL_TO_POLYGON[interval]

        now_utc = pd.Timestamp.now(tz="UTC")
        cutoff = now_utc - pd.Timedelta(days=_POLYGON_MAX_HISTORY_DAYS)
        if start < cutoff:
            logger.warning(
                "Requested start %s is before the Stocks Advanced plan cutoff %s "
                "(%d days); data may be incomplete.",
                start.strftime("%Y-%m-%d"),
                cutoff.strftime("%Y-%m-%d"),
                _POLYGON_MAX_HISTORY_DAYS,
            )

        from_str = start.strftime("%Y-%m-%d")
        # Polygon's `to` parameter is inclusive, so subtract one day from our
        # exclusive `end` to avoid pulling extra bars.
        to_str = (end - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        url = (
            f"{self.BASE_URL}/v2/aggs/ticker/{ticker}/range"
            f"/{multiplier}/{timespan}/{from_str}/{to_str}"
        )
        params: dict[str, object] = {
            "adjusted": "true",
            "sort": "asc",
            "limit": _POLYGON_MAX_LIMIT,
            "apiKey": self._api_key,
        }

        all_results: list[dict] = []
        page = 0
        while url:
            page += 1
            self._rate_limit()
            logger.debug("Polygon: GET %s (page %d)", url, page)
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Polygon request failed for %s: %s", ticker, exc)
                break

            body = resp.json()
            status = body.get("status", "")
            if status not in ("OK", "DELAYED"):
                logger.warning(
                    "Polygon returned status=%r for %s: %s",
                    status,
                    ticker,
                    body.get("message", ""),
                )
                break

            results = body.get("results") or []
            all_results.extend(results)

            # Follow pagination cursor if present.  On subsequent pages only
            # the apiKey param is needed — all other query params are encoded
            # in the cursor URL itself.
            url = body.get("next_url")
            params = {"apiKey": self._api_key} if url else {}

        if not all_results:
            return pd.DataFrame()

        return self._results_to_df(all_results, ticker)

    def fetch_trades(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        limit: int = 50_000,
    ) -> pd.DataFrame:
        """Fetch individual trade ticks from Polygon /v3/trades/{ticker}.

        Returns a DataFrame with columns:
            datetime (UTC, tz-aware), price, size, exchange, conditions, ticker
        Follows pagination via next_url. Requires Stocks Advanced.

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"SPY"``).
        start:
            Inclusive UTC start timestamp.
        end:
            Inclusive UTC end timestamp.
        limit:
            Maximum results per page (Polygon max: 50 000).
        """
        start_ns = int(start.value)  # pd.Timestamp.value is nanoseconds
        end_ns = int(end.value)

        url: str | None = (
            f"{self.BASE_URL}/v3/trades/{ticker}"
        )
        params: dict[str, object] = {
            "timestamp.gte": start_ns,
            "timestamp.lte": end_ns,
            "limit": limit,
            "apiKey": self._api_key,
        }

        all_results: list[dict] = []
        page = 0
        while url:
            page += 1
            self._rate_limit()
            logger.debug("Polygon trades: GET %s (page %d)", url, page)
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Polygon trades request failed for %s: %s", ticker, exc)
                break

            body = resp.json()
            status = body.get("status", "")
            if status not in ("OK", "DELAYED"):
                logger.warning(
                    "Polygon trades returned status=%r for %s: %s",
                    status,
                    ticker,
                    body.get("message", ""),
                )
                break

            results = body.get("results") or []
            all_results.extend(results)

            url = body.get("next_url")
            params = {"apiKey": self._api_key} if url else {}

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        # Use sip_timestamp (nanoseconds) if available, else participant_timestamp
        ts_col = "sip_timestamp" if "sip_timestamp" in df.columns else "participant_timestamp"
        if ts_col not in df.columns:
            logger.warning("No timestamp column found in trade data for %s", ticker)
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df[ts_col], unit="ns", utc=True)
        keep = ["datetime"]
        for src, dst in [
            ("price", "price"),
            ("size", "size"),
            ("exchange", "exchange"),
            ("conditions", "conditions"),
        ]:
            if src in df.columns:
                df[dst] = df[src]
                keep.append(dst)

        df = df[keep].copy()
        df["ticker"] = ticker
        df = df.dropna(subset=[c for c in ["price", "size"] if c in df.columns])
        return df.sort_values("datetime").reset_index(drop=True)

    def fetch_quotes(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        limit: int = 50_000,
    ) -> pd.DataFrame:
        """Fetch NBBO quote ticks from Polygon /v3/quotes/{ticker}.

        Returns a DataFrame with columns:
            datetime (UTC, tz-aware), bid_price, bid_size, ask_price, ask_size,
            spread, mid_price, bid_ask_imbalance, ticker
        Follows pagination. Requires Stocks Advanced.

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"SPY"``).
        start:
            Inclusive UTC start timestamp.
        end:
            Inclusive UTC end timestamp.
        limit:
            Maximum results per page (Polygon max: 50 000).
        """
        start_ns = int(start.value)
        end_ns = int(end.value)

        url: str | None = f"{self.BASE_URL}/v3/quotes/{ticker}"
        params: dict[str, object] = {
            "timestamp.gte": start_ns,
            "timestamp.lte": end_ns,
            "limit": limit,
            "apiKey": self._api_key,
        }

        all_results: list[dict] = []
        page = 0
        while url:
            page += 1
            self._rate_limit()
            logger.debug("Polygon quotes: GET %s (page %d)", url, page)
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("Polygon quotes request failed for %s: %s", ticker, exc)
                break

            body = resp.json()
            status = body.get("status", "")
            if status not in ("OK", "DELAYED"):
                logger.warning(
                    "Polygon quotes returned status=%r for %s: %s",
                    status,
                    ticker,
                    body.get("message", ""),
                )
                break

            results = body.get("results") or []
            all_results.extend(results)

            url = body.get("next_url")
            params = {"apiKey": self._api_key} if url else {}

        if not all_results:
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        if "sip_timestamp" not in df.columns:
            logger.warning("No sip_timestamp column found in quote data for %s", ticker)
            return pd.DataFrame()

        df["datetime"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)

        for col in ("bid_price", "bid_size", "ask_price", "ask_size"):
            if col not in df.columns:
                df[col] = float("nan")

        df = df[["datetime", "bid_price", "bid_size", "ask_price", "ask_size"]].copy()
        df = df.dropna(subset=["bid_price", "ask_price"])

        bid = df["bid_price"].astype(float)
        ask = df["ask_price"].astype(float)
        bid_sz = df["bid_size"].astype(float)
        ask_sz = df["ask_size"].astype(float)

        df["spread"] = ask - bid
        df["mid_price"] = (bid + ask) / 2
        df["bid_ask_imbalance"] = (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-9)
        df["ticker"] = ticker

        return df.sort_values("datetime").reset_index(drop=True)

    def fetch_snapshot(
        self,
        tickers: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch market snapshot(s) from Polygon.

        Endpoint: ``/v2/snapshot/locale/us/markets/stocks/tickers``

        If *tickers* is ``None`` or empty, fetches the whole market.

        Returns a DataFrame with columns:
            ticker, day_open, day_high, day_low, day_close, day_volume,
            prev_close, change_pct, last_trade_price, last_trade_size,
            min_open, min_high, min_low, min_close, min_volume

        Parameters
        ----------
        tickers:
            Optional list of ticker symbols to fetch (e.g. ``["AAPL", "TSLA"]``).
            Pass ``None`` or an empty list to fetch the whole market.
        """
        url = f"{self.BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers"
        params: dict[str, object] = {"apiKey": self._api_key}
        if tickers:
            params["tickers"] = ",".join(t.upper() for t in tickers)

        self._rate_limit()
        logger.debug("Polygon snapshot: GET %s", url)
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Polygon snapshot request failed: %s", exc)
            return pd.DataFrame()

        body = resp.json()
        status = body.get("status", "")
        if status not in ("OK", "DELAYED"):
            logger.warning(
                "Polygon snapshot returned status=%r: %s",
                status,
                body.get("message", ""),
            )
            return pd.DataFrame()

        raw = body.get("tickers") or []
        if not raw:
            return pd.DataFrame()

        rows = []
        for item in raw:
            day = item.get("day") or {}
            prev_day = item.get("prevDay") or {}
            last_trade = item.get("lastTrade") or {}
            minute = item.get("min") or {}
            rows.append(
                {
                    "ticker": item.get("ticker", ""),
                    "day_open": day.get("o"),
                    "day_high": day.get("h"),
                    "day_low": day.get("l"),
                    "day_close": day.get("c"),
                    "day_volume": day.get("v"),
                    "prev_close": prev_day.get("c"),
                    "change_pct": item.get("todaysChangePerc"),
                    "last_trade_price": last_trade.get("p"),
                    "last_trade_size": last_trade.get("s"),
                    "min_open": minute.get("o"),
                    "min_high": minute.get("h"),
                    "min_low": minute.get("l"),
                    "min_close": minute.get("c"),
                    "min_volume": minute.get("v"),
                }
            )

        return pd.DataFrame(rows)

    def fetch_news(
        self,
        ticker: str,
        limit: int = 50,
        published_utc_gte: str | None = None,
    ) -> pd.DataFrame:
        """Fetch news articles for *ticker* from Polygon /v2/reference/news.

        Returns a DataFrame with columns:
            published_utc, title, description, article_url, tickers, keywords

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"SPY"``).
        limit:
            Maximum number of articles to return (default: 50).
        published_utc_gte:
            Optional ISO-8601 date string to filter articles published on or
            after this date (e.g. ``"2024-01-01"``).
        """
        url = f"{self.BASE_URL}/v2/reference/news"
        params: dict[str, object] = {
            "ticker": ticker.upper(),
            "limit": limit,
            "order": "desc",
            "apiKey": self._api_key,
        }
        if published_utc_gte is not None:
            params["published_utc.gte"] = published_utc_gte

        self._rate_limit()
        logger.debug("Polygon news: GET %s", url)
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Polygon news request failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        body = resp.json()
        results = body.get("results") or []
        if not results:
            return pd.DataFrame(
                columns=["published_utc", "title", "description", "article_url", "tickers", "keywords"]
            )

        rows = []
        for item in results:
            rows.append(
                {
                    "published_utc": item.get("published_utc"),
                    "title": item.get("title"),
                    "description": item.get("description"),
                    "article_url": item.get("article_url"),
                    "tickers": item.get("tickers"),
                    "keywords": item.get("keywords"),
                }
            )
        return pd.DataFrame(rows)

    def fetch_events(
        self,
        ticker: str,
    ) -> pd.DataFrame:
        """Fetch corporate events (earnings, splits, dividends) for *ticker*.

        Endpoint: ``/vX/reference/tickers/{ticker}/events``

        Returns a DataFrame with columns:
            event_type, date, name, description
        where ``date`` is a ``pd.Timestamp`` (UTC midnight for date-only strings).

        Parameters
        ----------
        ticker:
            Ticker symbol (e.g. ``"AAPL"``).
        """
        url = f"{self.BASE_URL}/vX/reference/tickers/{ticker.upper()}/events"
        params: dict[str, object] = {"apiKey": self._api_key}

        self._rate_limit()
        logger.debug("Polygon events: GET %s", url)
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Polygon events request failed for %s: %s", ticker, exc)
            return pd.DataFrame(columns=["event_type", "date", "name", "description"])

        body = resp.json()
        results = body.get("results") or {}
        events = results.get("events") if isinstance(results, dict) else None
        if not events:
            return pd.DataFrame(columns=["event_type", "date", "name", "description"])

        rows = []
        for event in events:
            date_raw = event.get("date")
            try:
                date_ts = pd.Timestamp(date_raw, tz="UTC") if date_raw else pd.NaT
            except Exception:
                date_ts = pd.NaT
            rows.append(
                {
                    "event_type": event.get("type", ""),
                    "date": date_ts,
                    "name": event.get("name", ""),
                    "description": event.get("description", ""),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Pace API calls if ``_POLYGON_RATE_LIMIT_SECONDS`` > 0; otherwise a no-op."""
        if _POLYGON_RATE_LIMIT_SECONDS <= 0:
            return
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < _POLYGON_RATE_LIMIT_SECONDS:
            time.sleep(_POLYGON_RATE_LIMIT_SECONDS - elapsed)
        self._last_call_time = time.monotonic()

    @staticmethod
    def _results_to_df(results: list[dict], ticker: str) -> pd.DataFrame:
        """Convert Polygon aggregate result dicts to a normalised DataFrame."""
        df = pd.DataFrame(results)
        # Polygon returns timestamps as Unix milliseconds in the 't' column.
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in Polygon response for {ticker}: {missing}")
        df = df[required].copy()
        df["ticker"] = ticker
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        return df.sort_values("datetime").reset_index(drop=True)


class PolygonStream:
    """Real-time WebSocket stream from Polygon.io (Stocks Advanced).

    Subscribes to aggregate bars (AM.*), per-second bars (A.*), and/or
    trade ticks (T.*) for a list of tickers and calls a user-supplied
    callback on each message.

    Parameters
    ----------
    tickers:
        List of ticker symbols to subscribe to (e.g. ["SPY", "TSLA"]).
    on_bar:
        Async callable(bar: dict) invoked for every aggregate bar message.
    on_trade:
        Optional async callable(trade: dict) invoked for every trade tick.
    api_key:
        Polygon API key. Falls back to POLYGON_API_KEY env var.
    feed:
        WebSocket feed URL. Defaults to ``wss://socket.polygon.io/stocks``.
    subscriptions:
        List of channel prefixes to subscribe. Defaults to ``["AM.*"]``
        (per-minute bars). Other options: ``"A.*"`` (per-second),
        ``"T.*"`` (trades), ``"Q.*"`` (quotes).

    Usage
    -----
        async def handle_bar(bar):
            print(bar)

        stream = PolygonStream(["SPY", "TSLA"], on_bar=handle_bar)
        asyncio.run(stream.run())

    Notes
    -----
    Requires the ``websockets`` package (install with
    ``pip install scalpedge[streaming]``).  The import is deferred to
    :meth:`run` so the rest of the module works without it.

    Reconnects automatically on disconnect with exponential back-off
    (max 60 s).
    """

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(
        self,
        tickers: list[str],
        on_bar: Callable | None = None,
        on_trade: Callable | None = None,
        api_key: str | None = None,
        feed: str | None = None,
        subscriptions: list[str] | None = None,
    ) -> None:
        self._tickers = [t.upper() for t in tickers]
        self._on_bar = on_bar
        self._on_trade = on_trade
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Polygon API key not found.  Set the POLYGON_API_KEY "
                "environment variable or pass api_key= explicitly."
            )
        self._feed = feed or self.WS_URL
        self._subscriptions = subscriptions or ["AM.*"]

    async def run(self) -> None:
        """Connect, authenticate, subscribe, and stream until cancelled."""
        try:
            import websockets  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'websockets' package is required for PolygonStream. "
                "Install it with: pip install 'scalpedge[streaming]'"
            ) from exc

        attempt = 0
        while True:
            try:
                await self._connect_and_stream(websockets)
                attempt = 0  # reset on clean disconnect
            except Exception as exc:
                delay = min(2**attempt, 60)
                logger.warning(
                    "PolygonStream disconnected (%s). Reconnecting in %ds …", exc, delay
                )
                attempt += 1
                import asyncio
                await asyncio.sleep(delay)

    async def _connect_and_stream(self, websockets) -> None:  # type: ignore[no-untyped-def]
        """Single connection attempt: auth → subscribe → receive loop."""
        import asyncio

        async with websockets.connect(self._feed) as ws:
            # Wait for connected status
            raw = await ws.recv()
            msgs = json.loads(raw)
            if not any(m.get("status") == "connected" for m in msgs):
                raise RuntimeError(f"Unexpected connect message: {msgs}")

            # Authenticate
            await ws.send(json.dumps({"action": "auth", "params": self._api_key}))
            raw = await ws.recv()
            msgs = json.loads(raw)
            if not any(m.get("status") == "auth_success" for m in msgs):
                raise RuntimeError(f"Authentication failed: {msgs}")

            logger.info("PolygonStream: authenticated, subscribing …")

            # Subscribe
            subscribe_msg = self._build_subscribe_message()
            await ws.send(subscribe_msg)

            attempt_reset_logged = False
            async for raw_msg in ws:
                if not attempt_reset_logged:
                    logger.info("PolygonStream: receiving messages …")
                    attempt_reset_logged = True
                messages = json.loads(raw_msg)
                for msg in messages:
                    ev = msg.get("ev", "")
                    if ev in ("AM", "A"):
                        if self._on_bar is not None:
                            bar = self._normalise_bar(msg)
                            coro = self._on_bar(bar)
                            if asyncio.iscoroutine(coro):
                                await coro
                    elif ev == "T":
                        if self._on_trade is not None:
                            trade = self._normalise_trade(msg)
                            coro = self._on_trade(trade)
                            if asyncio.iscoroutine(coro):
                                await coro

    def _build_subscribe_message(self) -> str:
        """Build the JSON subscribe payload for the chosen channels and tickers."""
        channels: list[str] = []
        for sub in self._subscriptions:
            prefix = sub.rstrip(".*").rstrip(".")
            if sub.endswith("*"):
                for t in self._tickers:
                    channels.append(f"{prefix}.{t}")
            else:
                channels.append(sub)
        params = ",".join(channels)
        return json.dumps({"action": "subscribe", "params": params})

    @staticmethod
    def _normalise_bar(msg: dict) -> dict:
        """Normalise an AM/A bar message to a standard dict."""
        ts_ms = msg.get("s") or msg.get("e", 0)
        return {
            "ticker": msg.get("sym", ""),
            "open": msg.get("o"),
            "high": msg.get("h"),
            "low": msg.get("l"),
            "close": msg.get("c"),
            "volume": msg.get("v"),
            "datetime": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
            "ev": msg.get("ev"),
        }

    @staticmethod
    def _normalise_trade(msg: dict) -> dict:
        """Normalise a T (trade) message to a standard dict."""
        ts_ms = msg.get("t", 0)
        return {
            "ticker": msg.get("sym", ""),
            "price": msg.get("p"),
            "size": msg.get("s"),
            "exchange": msg.get("x"),
            "conditions": msg.get("c"),
            "datetime": pd.Timestamp(ts_ms, unit="ms", tz="UTC"),
        }


class DataManager:
    """Fetch and persist OHLCV bars for one or many tickers.

    Data is stored as ``data/<TICKER>_<INTERVAL>.parquet``.  On each call
    to :meth:`load` the file is checked for the most recent timestamp and
    only *new* bars are downloaded — no duplicates, no full re-downloads.

    Data source selection
    ---------------------
    * **Polygon.io** (primary) — used when the ``POLYGON_API_KEY`` environment
      variable is set.  Stocks Advanced plan: unlimited calls, 5+ years of
      minute aggregates.
    * **yfinance** (fallback) — used when ``POLYGON_API_KEY`` is absent.
      Free; limited to ~60 days of 5-min history per request (chunked
      automatically for longer ranges).

    Parameters
    ----------
    data_dir:
        Root directory for parquet files (defaults to ``./data``).
    interval:
        Bar interval string (e.g. ``"1m"``, ``"5m"``, ``"1d"``).
        Defaults to ``"5m"``.
    api_key:
        Polygon API key.  Overrides ``POLYGON_API_KEY`` env var when set.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        interval: str = _INTERVAL,
        api_key: str | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if interval not in _INTERVAL_MAX_DAYS:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Choose one of: {', '.join(_INTERVAL_MAX_DAYS)}"
            )
        self.interval = interval

        # Resolve Polygon client — use yfinance fallback if no key available.
        _key = api_key or os.environ.get("POLYGON_API_KEY", "")
        if _key:
            self._polygon: PolygonClient | None = PolygonClient(api_key=_key)
            logger.debug("DataManager: using Polygon.io as data source.")
        else:
            self._polygon = None
            logger.debug("DataManager: POLYGON_API_KEY not set — using yfinance fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        ticker: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Return a fully-updated OHLCV DataFrame for *ticker*.

        Steps:
        1. Load existing Parquet file (if any).
        2. Determine the latest stored timestamp.
        3. Fetch only bars newer than that timestamp from Polygon or yfinance.
        4. Append new bars, de-duplicate, sort, and save.
        5. Return the complete DataFrame.

        Parameters
        ----------
        ticker:
            Ticker symbol (case-insensitive).
        start:
            Optional ISO-8601 start date (``"YYYY-MM-DD"``).  When
            provided the full requested range is (re-)fetched regardless
            of what is already stored.
        end:
            Optional ISO-8601 end date (``"YYYY-MM-DD"``).  Defaults to
            today when *start* is provided.
        """
        ticker = ticker.upper()
        safe_interval = self.interval.replace("/", "-")
        path = self.data_dir / f"{ticker}_{safe_interval}.parquet"

        # Legacy path: files created before multi-interval support used
        # <TICKER>.parquet.  Migrate transparently on first access.
        legacy_path = self.data_dir / f"{ticker}.parquet"
        if not path.exists() and legacy_path.exists() and self.interval == _INTERVAL:
            path = legacy_path

        existing = self._load_parquet(path)

        new_bars = self._fetch_new_bars(ticker, existing, start=start, end=end)

        if new_bars is not None and not new_bars.empty:
            combined = (
                pd.concat([existing, new_bars])
                .drop_duplicates(subset=["datetime"])
                .sort_values("datetime")
                .reset_index(drop=True)
            )
            self._save_parquet(combined, path)
            logger.info(
                "%s: appended %d new bars (total %d)", ticker, len(new_bars), len(combined)
            )
        else:
            combined = existing
            logger.info("%s: data is up-to-date (%d bars total)", ticker, len(combined))

        if combined.empty:
            raise ValueError(
                f"No data available for {ticker}. "
                "Check the ticker symbol and your internet connection."
            )

        # When a date range is requested, filter the returned DataFrame to
        # match even if the parquet file holds more history.
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            combined = combined[combined["datetime"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            combined = combined[combined["datetime"] < end_ts]

        return combined.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_parquet(path: Path) -> pd.DataFrame:
        if path.exists():
            df = pd.read_parquet(path)
            # Ensure the datetime column is timezone-aware (UTC).
            if "datetime" in df.columns and df["datetime"].dt.tz is None:
                df["datetime"] = df["datetime"].dt.tz_localize("UTC")
            return df
        return pd.DataFrame()

    @staticmethod
    def _save_parquet(df: pd.DataFrame, path: Path) -> None:
        df.to_parquet(path, index=False)

    def _fetch_new_bars(
        self,
        ticker: str,
        existing: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame | None:
        """Download bars that are not yet in *existing*.

        Routes to Polygon.io when ``POLYGON_API_KEY`` is configured, otherwise
        falls back to yfinance.  For yfinance intraday intervals whose
        per-request window is shorter than the requested range, the range is
        split into consecutive chunks and concatenated.

        Parameters
        ----------
        ticker:
            Ticker symbol.
        existing:
            Already-stored bars (may be empty).
        start:
            Optional explicit start date (``"YYYY-MM-DD"``).
        end:
            Optional explicit end date (``"YYYY-MM-DD"``).
        """
        now_utc = pd.Timestamp.now(tz="UTC")

        if start is not None:
            # Explicit range requested — always fetch the full window.
            fetch_start = pd.Timestamp(start, tz="UTC")
        elif existing.empty:
            if self._polygon is not None:
                # For Polygon, default to the full Stocks Advanced plan window.
                fetch_start = now_utc - pd.Timedelta(days=_POLYGON_MAX_HISTORY_DAYS)
            else:
                max_days = _INTERVAL_MAX_DAYS.get(self.interval, _MAX_DAYS_PER_FETCH)
                fetch_start = now_utc - pd.Timedelta(days=max_days)
        else:
            last_ts = existing["datetime"].max()
            # Step forward by one bar so we don't re-download the last known bar.
            step = _INTERVAL_STEP.get(self.interval, pd.Timedelta(minutes=1))
            fetch_start = last_ts + step
            if fetch_start >= now_utc:
                return None  # Already up-to-date.

        if end is not None:
            fetch_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        else:
            fetch_end = now_utc + pd.Timedelta(days=1)

        # ------------------------------------------------------------------
        # Route to the appropriate data source.
        # ------------------------------------------------------------------
        if self._polygon is not None:
            df = self._fetch_polygon(ticker, fetch_start, fetch_end)
        else:
            max_days = _INTERVAL_MAX_DAYS.get(self.interval, _MAX_DAYS_PER_FETCH)
            total_days = (fetch_end - fetch_start).days
            if total_days <= max_days:
                df = self._fetch_single_window(ticker, fetch_start, fetch_end)
            else:
                df = self._fetch_range_chunked(ticker, fetch_start, fetch_end, max_days)

        if df is None or df.empty:
            return None

        # When doing an incremental update (no explicit start), keep only
        # bars that are strictly newer than the last stored bar.
        if start is None and not existing.empty:
            last_ts = existing["datetime"].max()
            df = df[df["datetime"] > last_ts]
        return df if not df.empty else None

    def _fetch_polygon(
        self,
        ticker: str,
        fetch_start: pd.Timestamp,
        fetch_end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Fetch bars from Polygon.io (handles pagination internally)."""
        assert self._polygon is not None  # guarded by caller
        try:
            df = self._polygon.fetch_aggs(
                ticker=ticker,
                interval=self.interval,
                start=fetch_start,
                end=fetch_end,
            )
        except Exception as exc:
            logger.warning("Polygon fetch failed for %s: %s", ticker, exc)
            return None
        return df if not df.empty else None

    def _fetch_single_window(
        self,
        ticker: str,
        fetch_start: pd.Timestamp,
        fetch_end: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Download a single yfinance window and return a normalised DataFrame."""
        try:
            raw = yf.download(
                ticker,
                start=fetch_start.strftime("%Y-%m-%d"),
                end=fetch_end.strftime("%Y-%m-%d"),
                interval=self.interval,
                progress=False,
                auto_adjust=True,
                **{_MULTI_LEVEL_KWARG: False},
            )
        except Exception as exc:
            logger.warning("yfinance download failed for %s: %s", ticker, exc)
            return None

        if raw is None or raw.empty:
            return None
        return self._normalise(raw, ticker)

    def _fetch_range_chunked(
        self,
        ticker: str,
        fetch_start: pd.Timestamp,
        fetch_end: pd.Timestamp,
        max_days: int,
    ) -> pd.DataFrame | None:
        """Fetch a large date range by issuing multiple consecutive API calls.

        The range ``[fetch_start, fetch_end)`` is split into windows of at
        most *max_days* days.  Each chunk is fetched individually and the
        results are concatenated.  Progress is logged so long-running
        downloads remain observable.

        Parameters
        ----------
        ticker:
            Ticker symbol.
        fetch_start:
            Inclusive UTC start of the overall range.
        fetch_end:
            Exclusive UTC end of the overall range.
        max_days:
            Maximum number of days per individual API call.
        """
        chunks: list[pd.DataFrame] = []
        chunk_start = fetch_start
        chunk_delta = pd.Timedelta(days=max_days)

        total_days = (fetch_end - fetch_start).days
        n_chunks = math.ceil(total_days / max_days)
        logger.info(
            "%s: fetching %d days of %s data in %d chunk(s) of ≤%d days ...",
            ticker,
            total_days,
            self.interval,
            n_chunks,
            max_days,
        )

        chunk_num = 0
        while chunk_start < fetch_end:
            chunk_end = min(chunk_start + chunk_delta, fetch_end)
            chunk_num += 1
            logger.info(
                "%s: chunk %d/%d  %s → %s",
                ticker,
                chunk_num,
                n_chunks,
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
            )
            chunk_df = self._fetch_single_window(ticker, chunk_start, chunk_end)
            if chunk_df is not None and not chunk_df.empty:
                chunks.append(chunk_df)
            chunk_start = chunk_end

        if not chunks:
            return None

        combined = (
            pd.concat(chunks)
            .drop_duplicates(subset=["datetime"])
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        logger.info(
            "%s: chunked fetch complete — %d bars across %d chunk(s)",
            ticker,
            len(combined),
            n_chunks,
        )
        return combined

    @staticmethod
    def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Convert yfinance output to a clean flat DataFrame."""
        df = raw.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        df.index.name = "datetime"
        df = df.reset_index()

        # Normalise column names (handle MultiIndex if present).
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]

        rename_map = {}
        for col in df.columns:
            low = col.lower()
            if "open" in low:
                rename_map[col] = "open"
            elif "high" in low:
                rename_map[col] = "high"
            elif "low" in low:
                rename_map[col] = "low"
            elif "close" in low:
                rename_map[col] = "close"
            elif "volume" in low:
                rename_map[col] = "volume"
            elif "datetime" in low or low == "date":
                rename_map[col] = "datetime"
        df = df.rename(columns=rename_map)

        required = {"datetime", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns after normalise for {ticker}: {missing}")

        df = df[list(required)].copy()
        df["ticker"] = ticker
        # Ensure tz-aware datetime.
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC")
        # Drop rows with NaN OHLCV.
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        return df.reset_index(drop=True)
