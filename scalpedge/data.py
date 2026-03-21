"""Data management module — fetch, store, and update 5-minute OHLCV data.

Supports any ticker via yfinance. Data is persisted in Parquet format under
the ``data/`` directory so the dataset grows incrementally on every run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Detect the yfinance parameter name for disabling multi-level column headers.
# The parameter was named 'multi_level_column' in 0.2.x and renamed to
# 'multi_level_index' in 1.0+.
_yf_major = int(yf.__version__.split(".")[0])
_MULTI_LEVEL_KWARG = "multi_level_index" if _yf_major >= 1 else "multi_level_column"

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_INTERVAL = "5m"
# yfinance free tier returns at most ~60 days of 5-minute history in one pull.
_MAX_DAYS_PER_FETCH = 59

# Maximum lookback (in days) supported by yfinance for each intraday interval.
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


class DataManager:
    """Fetch and persist OHLCV bars for one or many tickers.

    Data is stored as ``data/<TICKER>_<INTERVAL>.parquet``.  On each call
    to :meth:`load` the file is checked for the most recent timestamp and
    only *new* bars are downloaded from yfinance — no duplicates, no
    full re-downloads.

    Parameters
    ----------
    data_dir:
        Root directory for parquet files (defaults to ``./data``).
    interval:
        yfinance bar interval (e.g. ``"1m"``, ``"5m"``, ``"1d"``).
        Defaults to ``"5m"``.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        interval: str = _INTERVAL,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if interval not in _INTERVAL_MAX_DAYS:
            raise ValueError(
                f"Unsupported interval '{interval}'. "
                f"Choose one of: {', '.join(_INTERVAL_MAX_DAYS)}"
            )
        self.interval = interval

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
        3. Fetch only bars newer than that timestamp from yfinance.
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
        now_utc = pd.Timestamp.utcnow()
        max_days = _INTERVAL_MAX_DAYS.get(self.interval, _MAX_DAYS_PER_FETCH)

        if start is not None:
            # Explicit range requested — always fetch the full window.
            fetch_start = pd.Timestamp(start, tz="UTC")
        elif existing.empty:
            # Fresh incremental download: grab the maximum window yfinance allows.
            fetch_start = now_utc - pd.Timedelta(days=max_days)
        else:
            last_ts = existing["datetime"].max()
            # Step forward by one bar so we don't re-download the last known bar.
            step = _INTERVAL_STEP.get(self.interval, pd.Timedelta(minutes=1))
            fetch_start = last_ts + step
            if fetch_start >= now_utc:
                return None  # Already up-to-date.

        if end is not None:
            fetch_end = (pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
        else:
            fetch_end = now_utc + pd.Timedelta(days=1)

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

        df = self._normalise(raw, ticker)
        # When doing an incremental update (no explicit start), keep only
        # bars that are strictly newer than the last stored bar.
        if start is None and not existing.empty:
            last_ts = existing["datetime"].max()
            df = df[df["datetime"] > last_ts]
        return df if not df.empty else None

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
