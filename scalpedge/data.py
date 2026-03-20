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

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_INTERVAL = "5m"
# yfinance free tier returns at most ~60 days of 5-minute history in one pull.
_MAX_DAYS_PER_FETCH = 59


class DataManager:
    """Fetch and persist 5-minute OHLCV bars for one or many tickers.

    Data is stored as ``data/<TICKER>.parquet``.  On each call to
    :meth:`load` the file is checked for the most recent timestamp and
    only *new* bars are downloaded from yfinance — no duplicates, no
    full re-downloads.

    Parameters
    ----------
    data_dir:
        Root directory for parquet files (defaults to ``./data``).
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, ticker: str) -> pd.DataFrame:
        """Return a fully-updated OHLCV DataFrame for *ticker*.

        Steps:
        1. Load existing Parquet file (if any).
        2. Determine the latest stored timestamp.
        3. Fetch only bars newer than that timestamp from yfinance.
        4. Append new bars, de-duplicate, sort, and save.
        5. Return the complete DataFrame.
        """
        ticker = ticker.upper()
        path = self.data_dir / f"{ticker}.parquet"

        existing = self._load_parquet(path)

        new_bars = self._fetch_new_bars(ticker, existing)

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
        return combined

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
        self, ticker: str, existing: pd.DataFrame
    ) -> pd.DataFrame | None:
        """Download bars that are not yet in *existing*."""
        import datetime

        now_utc = pd.Timestamp.utcnow()

        if existing.empty:
            # Fresh download: grab the maximum window yfinance allows.
            start = now_utc - pd.Timedelta(days=_MAX_DAYS_PER_FETCH)
        else:
            last_ts = existing["datetime"].max()
            # Add one 5-minute bar so we don't re-download the last known bar.
            start = last_ts + pd.Timedelta(minutes=5)
            if start >= now_utc:
                return None  # Already up-to-date.

        try:
            raw = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=(now_utc + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=_INTERVAL,
                progress=False,
                auto_adjust=True,
                multi_level_column=False,
            )
        except Exception as exc:
            logger.warning("yfinance download failed for %s: %s", ticker, exc)
            return None

        if raw is None or raw.empty:
            return None

        df = self._normalise(raw, ticker)
        # Keep only bars that are strictly newer than the last stored bar.
        if not existing.empty:
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
