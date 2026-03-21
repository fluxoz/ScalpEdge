"""Data management module — fetch, store, and update 5-minute OHLCV data.

Primary data source: **Polygon.io** (free tier).
  - Set the ``POLYGON_API_KEY`` environment variable to enable.
  - Free tier: 5 API calls/minute, 2 years of historical minute aggregates.

Fallback data source: **yfinance** (used when ``POLYGON_API_KEY`` is not set).
  - Free; limited to ~60 days of 5-min history per request (chunked automatically).

Data is persisted in Parquet format under the ``data/`` directory so the
dataset grows incrementally on every run.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

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

# Polygon free tier: 5 calls/minute.  We pace at 13 s/call (~4.6 calls/min)
# rather than the theoretical minimum of 12 s/call (exactly 5 calls/min) to
# provide a small safety margin against clock jitter and network latency.
_POLYGON_RATE_LIMIT_SECONDS = 13.0

# Polygon free tier: 2 years of minute-aggregate history.
_POLYGON_MAX_HISTORY_DAYS = 730

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
    """Thin wrapper around the Polygon.io v2 aggregates REST endpoint.

    Parameters
    ----------
    api_key:
        Polygon API key.  If *None*, falls back to the ``POLYGON_API_KEY``
        environment variable.

    Notes
    -----
    Free-tier limits observed by this client:

    * **5 calls / minute** — enforced via a configurable sleep between
      successive requests.
    * **2 years of history** — a warning is logged (but not an error) if the
      requested range extends beyond 2 years ago.
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
                "Polygon free tier only provides %d days of history. "
                "Requested start %s is before the cutoff %s; data may be incomplete.",
                _POLYGON_MAX_HISTORY_DAYS,
                start.strftime("%Y-%m-%d"),
                cutoff.strftime("%Y-%m-%d"),
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

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Sleep if necessary to stay within the 5-calls/minute limit."""
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


class DataManager:
    """Fetch and persist OHLCV bars for one or many tickers.

    Data is stored as ``data/<TICKER>_<INTERVAL>.parquet``.  On each call
    to :meth:`load` the file is checked for the most recent timestamp and
    only *new* bars are downloaded — no duplicates, no full re-downloads.

    Data source selection
    ---------------------
    * **Polygon.io** (primary) — used when the ``POLYGON_API_KEY`` environment
      variable is set.  Free tier: 5 calls/min, 2 years of minute aggregates.
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
                # For Polygon, default to the full 2-year free-tier window.
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
