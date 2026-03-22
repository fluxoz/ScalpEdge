"""Tests for ScalpEdge modules using synthetic data (no network required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _has_pkg(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


_skip_no_sklearn = pytest.mark.skipif(
    not _has_pkg("sklearn"), reason="scikit-learn not installed"
)
_skip_no_torch = pytest.mark.skipif(
    not _has_pkg("torch"), reason="torch not installed"
)
_skip_no_ml = pytest.mark.skipif(
    not (_has_pkg("sklearn") and _has_pkg("torch")),
    reason="scikit-learn and/or torch not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """500-bar synthetic 5-minute OHLCV DataFrame."""
    np.random.seed(42)
    n = 500
    close = 450.0 * np.cumprod(1 + np.random.normal(0.0001, 0.002, n))
    df = pd.DataFrame(
        {
            "datetime": pd.date_range(
                "2024-01-02 09:30", periods=n, freq="5min", tz="UTC"
            ),
            "open": close * (1 + np.random.normal(0, 0.001, n)),
            "high": close * (1 + np.abs(np.random.normal(0, 0.002, n))),
            "low": close * (1 - np.abs(np.random.normal(0, 0.002, n))),
            "close": close,
            "volume": np.random.randint(100_000, 500_000, n).astype(float),
            "ticker": "TEST",
        }
    )
    # Ensure high >= close >= low
    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)
    return df


@pytest.fixture
def indicator_df(synthetic_df):
    """synthetic_df with all TA indicators added."""
    from scalpedge.ta_indicators import add_all_indicators

    return add_all_indicators(synthetic_df)


@pytest.fixture
def clean_df(indicator_df):
    """indicator_df after dropping NaN warm-up rows."""
    return (
        indicator_df.dropna(subset=["ema_50", "rsi_14", "macd", "atr_14"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Data module
# ---------------------------------------------------------------------------

class TestDataManager:
    def test_normalise(self, synthetic_df):
        from scalpedge.data import DataManager

        # _normalise expects a raw yfinance-style df with datetime as index
        raw = synthetic_df.drop(columns=["ticker"]).set_index("datetime")
        result = DataManager._normalise(raw, "TEST")
        assert "datetime" in result.columns
        assert set(result.columns) >= {"open", "high", "low", "close", "volume"}
        assert len(result) == len(synthetic_df)

    def test_parquet_roundtrip(self, synthetic_df, tmp_path):
        from scalpedge.data import DataManager

        dm = DataManager(data_dir=tmp_path)
        path = tmp_path / "TEST.parquet"
        dm._save_parquet(synthetic_df, path)
        loaded = dm._load_parquet(path)
        assert len(loaded) == len(synthetic_df)
        assert set(loaded.columns) == set(synthetic_df.columns)

    def test_custom_interval_accepted(self, tmp_path):
        """DataManager should accept any supported interval string."""
        from scalpedge.data import DataManager

        for interval in ("1m", "5m", "15m", "1h", "1d"):
            dm = DataManager(data_dir=tmp_path, interval=interval)
            assert dm.interval == interval

    def test_invalid_interval_raises(self, tmp_path):
        """DataManager should raise ValueError for unsupported intervals."""
        from scalpedge.data import DataManager

        with pytest.raises(ValueError, match="Unsupported interval"):
            DataManager(data_dir=tmp_path, interval="99x")

    def test_load_filters_by_start_end(self, synthetic_df, tmp_path):
        """load() should honour start/end date filters against stored data."""
        from scalpedge.data import DataManager

        dm = DataManager(data_dir=tmp_path, interval="5m")
        path = tmp_path / "TEST_5m.parquet"
        dm._save_parquet(synthetic_df, path)

        # Use a narrow date window in the middle of the synthetic data.
        start_ts = synthetic_df["datetime"].quantile(0.25)
        end_ts = synthetic_df["datetime"].quantile(0.75)
        start_str = start_ts.strftime("%Y-%m-%d")
        end_str = end_ts.strftime("%Y-%m-%d")

        # Mock _fetch_new_bars to return None (simulate up-to-date cache).
        original = dm._fetch_new_bars
        dm._fetch_new_bars = lambda *a, **kw: None
        result = dm.load("TEST", start=start_str, end=end_str)
        dm._fetch_new_bars = original

        assert len(result) < len(synthetic_df)
        assert result["datetime"].min() >= pd.Timestamp(start_str, tz="UTC")
        assert result["datetime"].max() < pd.Timestamp(end_str, tz="UTC") + pd.Timedelta(days=1)

    def test_parquet_filename_includes_interval(self, synthetic_df, tmp_path):
        """Parquet files should be named <TICKER>_<INTERVAL>.parquet."""
        from scalpedge.data import DataManager

        dm = DataManager(data_dir=tmp_path, interval="1d")
        dm._fetch_new_bars = lambda *a, **kw: None  # skip network call
        dm._save_parquet(synthetic_df, tmp_path / "TEST_1d.parquet")

        assert (tmp_path / "TEST_1d.parquet").exists()

    def test_fetch_range_chunked_combines_results(self, synthetic_df, tmp_path):
        """_fetch_range_chunked should combine results from multiple windows."""
        from scalpedge.data import DataManager

        dm = DataManager(data_dir=tmp_path, interval="5m")

        # Build three non-overlapping chunks of 100 bars each.
        chunk_size = 100
        chunk1 = synthetic_df.iloc[:chunk_size].copy()
        chunk2 = synthetic_df.iloc[chunk_size : chunk_size * 2].copy()
        chunk3 = synthetic_df.iloc[chunk_size * 2 : chunk_size * 3].copy()
        chunks_iter = iter([chunk1, chunk2, chunk3])

        def fake_fetch_single(ticker, start, end):
            try:
                return next(chunks_iter)
            except StopIteration:
                return None

        dm._fetch_single_window = fake_fetch_single

        # Use a range that spans 3× max_days to force 3 chunks.
        fetch_start = pd.Timestamp("2020-01-01", tz="UTC")
        fetch_end = fetch_start + pd.Timedelta(days=180)  # 3 × 59-day chunks

        result = dm._fetch_range_chunked(
            "TEST", fetch_start, fetch_end, max_days=59
        )

        assert result is not None
        assert len(result) == chunk_size * 3
        # Results must be sorted by datetime and duplicate-free.
        assert result["datetime"].is_monotonic_increasing

    def test_fetch_range_chunked_deduplicates(self, synthetic_df, tmp_path):
        """_fetch_range_chunked must deduplicate overlapping bars."""
        from scalpedge.data import DataManager

        dm = DataManager(data_dir=tmp_path, interval="5m")

        overlap_df = synthetic_df.iloc[:200].copy()
        # Return the same data twice to simulate overlap.
        calls = [overlap_df, overlap_df]
        call_iter = iter(calls)

        def fake_fetch_single(ticker, start, end):
            try:
                return next(call_iter)
            except StopIteration:
                return None

        dm._fetch_single_window = fake_fetch_single

        fetch_start = pd.Timestamp("2020-01-01", tz="UTC")
        fetch_end = fetch_start + pd.Timedelta(days=120)

        result = dm._fetch_range_chunked("TEST", fetch_start, fetch_end, max_days=59)

        assert result is not None
        # After deduplication the row count should equal the unique bar count.
        assert len(result) == result["datetime"].nunique()

    def test_fetch_new_bars_uses_chunking_for_large_range(self, synthetic_df, tmp_path):
        """_fetch_new_bars should delegate to _fetch_range_chunked for ranges
        exceeding max_days for the given interval."""
        from scalpedge.data import DataManager

        dm = DataManager(data_dir=tmp_path, interval="5m")

        chunked_called: list[bool] = []

        def fake_chunked(ticker, start, end, max_days):
            chunked_called.append(True)
            return synthetic_df.copy()

        dm._fetch_range_chunked = fake_chunked

        # Request > 59 days — should trigger chunked path.
        result = dm._fetch_new_bars(
            "TEST",
            pd.DataFrame(),
            start="2015-01-01",
            end="2025-01-01",
        )

        assert chunked_called, "_fetch_range_chunked was not called for a large range"
        assert result is not None


# ---------------------------------------------------------------------------
# PolygonClient
# ---------------------------------------------------------------------------

class TestPolygonClient:
    """Unit tests for PolygonClient — all HTTP calls are mocked."""

    def _make_agg_response(self, bars: list[dict], next_url: str | None = None) -> dict:
        """Build a minimal Polygon aggregates API response."""
        body: dict = {"status": "OK", "results": bars}
        if next_url:
            body["next_url"] = next_url
        return body

    def _bar(self, ts_ms: int, o=100.0, h=101.0, low=99.0, c=100.5, v=10000) -> dict:
        return {"t": ts_ms, "o": o, "h": h, "l": low, "c": c, "v": v}

    def test_results_to_df_basic(self):
        """_results_to_df should produce a properly-typed DataFrame."""
        from scalpedge.data import PolygonClient

        bars = [self._bar(1_700_000_000_000 + i * 300_000) for i in range(5)]
        df = PolygonClient._results_to_df(bars, "SPY")

        assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume", "ticker"]
        assert len(df) == 5
        assert df["datetime"].dt.tz is not None  # UTC-aware
        assert df["ticker"].iloc[0] == "SPY"
        assert df["datetime"].is_monotonic_increasing

    def test_fetch_aggs_single_page(self, tmp_path):
        """fetch_aggs should return bars from a single-page response."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = PolygonClient.__new__(PolygonClient)
        client._api_key = "test"
        client._last_call_time = 0.0

        bars = [self._bar(1_700_000_000_000 + i * 300_000) for i in range(3)]
        mock_resp = MagicMock()
        mock_resp.json.return_value = self._make_agg_response(bars)
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            with patch("scalpedge.data.time.sleep"):  # skip rate-limit sleep
                df = client.fetch_aggs(
                    "SPY",
                    "5m",
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                )

        assert len(df) == 3
        assert set(df.columns) >= {"datetime", "open", "high", "low", "close", "volume"}

    def test_fetch_aggs_pagination(self):
        """fetch_aggs should follow next_url to collect multiple pages."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = PolygonClient.__new__(PolygonClient)
        client._api_key = "test"
        client._last_call_time = 0.0

        page1_bars = [self._bar(1_700_000_000_000 + i * 300_000) for i in range(3)]
        page2_bars = [self._bar(1_700_000_000_000 + (i + 3) * 300_000) for i in range(2)]

        next_url = "https://api.polygon.io/v2/aggs/ticker/SPY/range/5/minute/...?cursor=abc"
        page1_resp = MagicMock()
        page1_resp.json.return_value = self._make_agg_response(page1_bars, next_url=next_url)
        page1_resp.raise_for_status.return_value = None

        page2_resp = MagicMock()
        page2_resp.json.return_value = self._make_agg_response(page2_bars)
        page2_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", side_effect=[page1_resp, page2_resp]):
            with patch("scalpedge.data.time.sleep"):
                df = client.fetch_aggs(
                    "SPY",
                    "5m",
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                )

        assert len(df) == 5  # 3 + 2 bars across two pages

    def test_fetch_aggs_empty_response(self):
        """fetch_aggs should return an empty DataFrame when no results."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = PolygonClient.__new__(PolygonClient)
        client._api_key = "test"
        client._last_call_time = 0.0

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "OK", "results": []}
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            with patch("scalpedge.data.time.sleep"):
                df = client.fetch_aggs(
                    "SPY",
                    "5m",
                    pd.Timestamp("2024-01-01", tz="UTC"),
                    pd.Timestamp("2024-01-02", tz="UTC"),
                )

        assert df.empty

    def test_fetch_aggs_unsupported_interval(self):
        """fetch_aggs should raise ValueError for unmapped intervals."""
        from scalpedge.data import PolygonClient

        client = PolygonClient.__new__(PolygonClient)
        client._api_key = "test"
        client._last_call_time = 0.0

        with pytest.raises(ValueError, match="not supported by Polygon"):
            client.fetch_aggs(
                "SPY",
                "99x",
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-02", tz="UTC"),
            )

    def test_no_api_key_raises(self, monkeypatch):
        """PolygonClient should raise ValueError when no API key is available."""
        from scalpedge.data import PolygonClient

        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        with pytest.raises(ValueError, match="POLYGON_API_KEY"):
            PolygonClient()

    def test_data_manager_uses_polygon_when_key_set(self, tmp_path, monkeypatch, synthetic_df):
        """DataManager should route to Polygon when POLYGON_API_KEY is set."""
        from scalpedge.data import DataManager

        monkeypatch.setenv("POLYGON_API_KEY", "fake_key")
        dm = DataManager(data_dir=tmp_path)
        assert dm._polygon is not None

        # Verify _fetch_new_bars delegates to _fetch_polygon.
        polygon_called: list[bool] = []

        def fake_polygon(ticker, start, end):
            polygon_called.append(True)
            return synthetic_df.copy()

        dm._fetch_polygon = fake_polygon
        dm._fetch_new_bars("TEST", pd.DataFrame())
        assert polygon_called, "_fetch_polygon was not called when Polygon key is set"

    def test_data_manager_falls_back_to_yfinance(self, tmp_path, monkeypatch):
        """DataManager should use yfinance when POLYGON_API_KEY is absent."""
        from scalpedge.data import DataManager

        monkeypatch.delenv("POLYGON_API_KEY", raising=False)
        dm = DataManager(data_dir=tmp_path)
        assert dm._polygon is None


# ---------------------------------------------------------------------------
# CLI fetch sub-command
# ---------------------------------------------------------------------------

class TestCLIFetch:
    """Tests for the 'fetch' sub-command in main.py."""

    def _make_dm(self, synthetic_df, tmp_path, monkeypatch):
        """Patch DataManager.load to return synthetic data without network."""
        from unittest.mock import MagicMock

        mock_dm = MagicMock()
        mock_dm.load.return_value = synthetic_df

        import main as main_module
        monkeypatch.setattr(
            main_module,
            "cmd_fetch",
            lambda args: print(
                f"{args.tickers[0].upper()}: {len(synthetic_df):,} bars"
            ),
        )
        return mock_dm

    def test_fetch_help_exits_cleanly(self):
        """scalpedge fetch --help should exit with code 0."""
        import argparse

        with pytest.raises(SystemExit) as exc_info:
            parser = argparse.ArgumentParser()
            sub = parser.add_subparsers(dest="command")
            fp = sub.add_parser("fetch")
            fp.add_argument("tickers", nargs="+")
            fp.add_argument("--interval", default="5m")
            fp.add_argument("--start", default=None)
            fp.add_argument("--end", default=None)
            fp.add_argument("--output-dir", default=None)
            fp.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_cmd_fetch_success(self, synthetic_df, tmp_path, monkeypatch, capsys):
        """cmd_fetch prints a summary line per ticker on success."""
        import argparse
        import main as main_module
        from unittest.mock import MagicMock, patch

        mock_dm_instance = MagicMock()
        mock_dm_instance.load.return_value = synthetic_df

        with patch("scalpedge.data.DataManager", return_value=mock_dm_instance):
            args = argparse.Namespace(
                tickers=["TEST"],
                interval="5m",
                start=None,
                end=None,
                output_dir=None,
            )
            main_module.cmd_fetch(args)

        captured = capsys.readouterr()
        assert "TEST" in captured.out
        assert "bars" in captured.out

    def test_cmd_fetch_multiple_tickers(self, synthetic_df, tmp_path, capsys):
        """cmd_fetch handles multiple tickers in one call."""
        import argparse
        import main as main_module
        from unittest.mock import MagicMock, patch

        mock_dm_instance = MagicMock()
        mock_dm_instance.load.return_value = synthetic_df

        with patch("scalpedge.data.DataManager", return_value=mock_dm_instance):
            args = argparse.Namespace(
                tickers=["SPY", "TSLA", "AAPL"],
                interval="1d",
                start="2023-01-01",
                end="2023-12-31",
                output_dir=str(tmp_path),
            )
            main_module.cmd_fetch(args)

        captured = capsys.readouterr()
        assert "SPY" in captured.out
        assert "TSLA" in captured.out
        assert "AAPL" in captured.out

    def test_cmd_fetch_error_exits_nonzero(self, monkeypatch, capsys):
        """cmd_fetch exits with code 1 when a ticker fails to fetch."""
        import argparse
        import main as main_module
        from unittest.mock import MagicMock, patch

        mock_dm_instance = MagicMock()
        mock_dm_instance.load.side_effect = ValueError("No data available for BAD")

        with patch("scalpedge.data.DataManager", return_value=mock_dm_instance):
            args = argparse.Namespace(
                tickers=["BAD"],
                interval="5m",
                start=None,
                end=None,
                output_dir=None,
            )
            with pytest.raises(SystemExit) as exc_info:
                main_module.cmd_fetch(args)

        assert exc_info.value.code == 1

    def test_cmd_fetch_years_sets_start(self, synthetic_df, tmp_path, capsys):
        """--years N should compute a start date ~N years ago and pass it to load()."""
        import argparse
        import datetime
        import main as main_module
        from unittest.mock import MagicMock, patch

        mock_dm_instance = MagicMock()
        captured_start: list[str] = []

        def recording_load(ticker, start=None, end=None):
            captured_start.append(start)
            return synthetic_df

        mock_dm_instance.load.side_effect = recording_load

        with patch("scalpedge.data.DataManager", return_value=mock_dm_instance):
            args = argparse.Namespace(
                tickers=["SPY"],
                interval="5m",
                start=None,
                end=None,
                output_dir=None,
                years=10,
            )
            main_module.cmd_fetch(args)

        assert captured_start, "load() was never called"
        computed_date = datetime.date.fromisoformat(captured_start[0])
        expected_year = datetime.date.today().year - 10
        # The computed year should be exactly 10 years ago (within ±1 year for safety).
        assert abs(computed_date.year - expected_year) <= 1

    def test_cmd_fetch_start_overrides_years(self, synthetic_df, tmp_path, capsys):
        """When --start is provided, --years should be ignored."""
        import argparse
        import main as main_module
        from unittest.mock import MagicMock, patch

        mock_dm_instance = MagicMock()
        captured_start: list[str] = []

        def recording_load(ticker, start=None, end=None):
            captured_start.append(start)
            return synthetic_df

        mock_dm_instance.load.side_effect = recording_load

        with patch("scalpedge.data.DataManager", return_value=mock_dm_instance):
            args = argparse.Namespace(
                tickers=["SPY"],
                interval="5m",
                start="2020-01-01",
                end=None,
                output_dir=None,
                years=10,  # should be ignored because start is set
            )
            main_module.cmd_fetch(args)

        assert captured_start[0] == "2020-01-01"


# ---------------------------------------------------------------------------
# TA indicators
# ---------------------------------------------------------------------------

class TestTAIndicators:
    def test_column_count(self, indicator_df):
        """Expect at least 60 new columns (indicators + patterns)."""
        base_cols = {"datetime", "open", "high", "low", "close", "volume", "ticker"}
        new_cols = set(indicator_df.columns) - base_cols
        assert len(new_cols) >= 60, f"Only {len(new_cols)} new indicator columns"

    def test_ema_values(self, indicator_df):
        assert "ema_9" in indicator_df.columns
        assert "ema_21" in indicator_df.columns
        assert "ema_50" in indicator_df.columns
        assert "ema_200" in indicator_df.columns
        # EMA-9 should be closer to current price than EMA-200 (shorter window)
        tail = indicator_df.tail(50)
        assert (tail["ema_9"] - tail["close"]).abs().mean() <= (
            tail["ema_200"] - tail["close"]
        ).abs().mean()

    def test_rsi_range(self, indicator_df):
        rsi = indicator_df["rsi_14"].dropna()
        assert rsi.between(0, 100).all(), "RSI values must be in [0, 100]"

    def test_bb_ordering(self, indicator_df):
        bb = indicator_df[["bb_lower", "bb_mid", "bb_upper"]].dropna()
        assert (bb["bb_lower"] <= bb["bb_mid"]).all()
        assert (bb["bb_mid"] <= bb["bb_upper"]).all()

    def test_atr_positive(self, indicator_df):
        atr = indicator_df["atr_14"].dropna()
        assert (atr >= 0).all()

    def test_candlestick_patterns(self, indicator_df):
        pat_cols = [c for c in indicator_df.columns if c.startswith("pat_")]
        assert len(pat_cols) >= 40, f"Only {len(pat_cols)} pattern columns found"
        # Pattern columns should be 0 or 1
        for col in pat_cols:
            vals = indicator_df[col].dropna()
            assert vals.isin([0, 1]).all(), f"{col} has non-binary values"

    def test_vwap_present(self, indicator_df):
        assert "vwap" in indicator_df.columns
        assert indicator_df["vwap"].notna().any()

    # ------------------------------------------------------------------
    # Volume Profile / POC
    # ------------------------------------------------------------------

    def test_poc_columns_present(self, indicator_df):
        """add_all_indicators must produce the four POC signal columns."""
        for col in ("poc_price", "poc_proximity_pct", "poc_above", "poc_below"):
            assert col in indicator_df.columns, f"Missing column: {col}"

    def test_poc_price_positive(self, indicator_df):
        """POC price must be a positive number wherever it is not NaN."""
        poc = indicator_df["poc_price"].dropna()
        assert (poc > 0).all(), "poc_price must be positive"

    def test_poc_proximity_sign(self, indicator_df):
        """poc_proximity_pct must be positive when close > poc and negative when close < poc."""
        sub = indicator_df[["close", "poc_price", "poc_proximity_pct"]].dropna()
        above = sub[sub["close"] > sub["poc_price"]]
        below = sub[sub["close"] < sub["poc_price"]]
        assert (above["poc_proximity_pct"] > 0).all(), (
            "poc_proximity_pct should be positive when close > poc_price"
        )
        assert (below["poc_proximity_pct"] < 0).all(), (
            "poc_proximity_pct should be negative when close < poc_price"
        )

    def test_poc_above_below_binary(self, indicator_df):
        """poc_above and poc_below must only contain 0 or 1."""
        for col in ("poc_above", "poc_below"):
            vals = indicator_df[col].dropna()
            assert vals.isin([0, 1]).all(), f"{col} contains non-binary values"

    def test_poc_above_below_mutually_exclusive(self, indicator_df):
        """poc_above and poc_below should never both be 1 at the same time."""
        both_set = (indicator_df["poc_above"] == 1) & (indicator_df["poc_below"] == 1)
        assert not both_set.any(), "poc_above and poc_below are mutually exclusive"

    def test_poc_resets_per_session(self, synthetic_df):
        """POC should be computed independently per session (calendar day)."""
        from scalpedge.ta_indicators import _volume_profile

        # Build a two-session frame so we can check independence.
        day1 = synthetic_df[synthetic_df["datetime"].dt.date == synthetic_df["datetime"].dt.date.iloc[0]]
        day2 = synthetic_df[synthetic_df["datetime"].dt.date == synthetic_df["datetime"].dt.date.iloc[-1]]
        if len(day1) < 2 or len(day2) < 2:
            return  # not enough data in synthetic fixture for this check

        poc1, _ = _volume_profile(day1)
        poc2, _ = _volume_profile(day2)

        # Both series must be non-NaN.
        assert poc1.notna().all()
        assert poc2.notna().all()

    def test_volume_profile_no_lookahead(self, synthetic_df):
        """POC computed on a partial session prefix must not change when more bars are added."""
        from scalpedge.ta_indicators import _volume_profile

        # Grab the first session.
        first_date = synthetic_df["datetime"].dt.date.iloc[0]
        session = synthetic_df[synthetic_df["datetime"].dt.date == first_date].copy()
        if len(session) < 4:
            return  # degenerate fixture

        half = len(session) // 2
        poc_partial, _ = _volume_profile(session.iloc[:half])
        poc_last_partial = poc_partial.iloc[-1]

        poc_full, _ = _volume_profile(session)
        poc_at_half_in_full = poc_full.iloc[half - 1]

        # The POC value at bar `half-1` must be the same whether or not
        # future bars are present (rolling, no look-ahead).
        assert poc_last_partial == poc_at_half_in_full, (
            "POC look-ahead bias detected: value changes when future bars are appended"
        )

    def test_plot_volume_profile_returns_figure(self, indicator_df):
        """plot_volume_profile should return a matplotlib Figure without errors."""
        import matplotlib
        matplotlib.use("Agg")
        from scalpedge.ta_indicators import plot_volume_profile

        fig = plot_volume_profile(indicator_df)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_volume_profile_specific_date(self, indicator_df):
        """plot_volume_profile accepts a session_date string."""
        import matplotlib
        matplotlib.use("Agg")
        from scalpedge.ta_indicators import plot_volume_profile

        session_date = str(indicator_df["datetime"].dt.date.iloc[0])
        fig = plot_volume_profile(indicator_df, session_date=session_date)
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_volume_profile_invalid_date_raises(self, indicator_df):
        """plot_volume_profile raises ValueError for an unknown session date."""
        import matplotlib
        matplotlib.use("Agg")
        from scalpedge.ta_indicators import plot_volume_profile

        with pytest.raises(ValueError, match="No data found for session date"):
            plot_volume_profile(indicator_df, session_date="1900-01-01")


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

class TestBlackScholes:
    def test_call_put_parity(self):
        """C - P = S*exp(-qT) - K*exp(-rT)  (put-call parity)."""
        from scalpedge.options import BlackScholes

        bs = BlackScholes(spot=450.0, strike=452.0, r=0.05, sigma=0.25, T=30 / 252)
        call = bs.call_price()
        put = bs.put_price()
        expected = 450.0 * np.exp(-bs.q * bs.T) - 452.0 * np.exp(-bs.r * bs.T)
        assert abs((call - put) - expected) < 1e-6

    def test_delta_range(self):
        from scalpedge.options import BlackScholes

        for spot in [400, 450, 500]:
            bs = BlackScholes(spot=spot, strike=450.0, r=0.05, sigma=0.20, T=1 / 252)
            assert 0 <= bs.delta("call") <= 1
            assert -1 <= bs.delta("put") <= 0

    def test_atm_call_delta_near_half(self):
        from scalpedge.options import BlackScholes

        bs = BlackScholes(spot=450.0, strike=450.0, r=0.0, sigma=0.20, T=1 / 252)
        assert abs(bs.delta("call") - 0.5) < 0.15

    def test_intrinsic_at_expiry(self):
        from scalpedge.options import BlackScholes

        bs = BlackScholes(spot=460.0, strike=450.0, r=0.05, sigma=0.20, T=0.0)
        assert bs.call_price() == 10.0
        assert bs.put_price() == 0.0

    def test_gamma_positive(self):
        from scalpedge.options import BlackScholes

        bs = BlackScholes(spot=450.0, strike=450.0, r=0.05, sigma=0.20, T=1 / 252)
        assert bs.gamma() > 0

    def test_vega_positive(self):
        from scalpedge.options import BlackScholes

        bs = BlackScholes(spot=450.0, strike=450.0, r=0.05, sigma=0.20, T=30 / 252)
        assert bs.vega() > 0

    def test_from_current_factory(self):
        from scalpedge.options import BlackScholes

        bs = BlackScholes.from_current(spot=450.0, dte_days=0.0)
        assert bs.spot == 450.0
        assert bs.T > 0


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_prob_up_range(self, synthetic_df):
        from scalpedge.probabilities import MonteCarlo

        mc = MonteCarlo(n_simulations=200)
        rets = np.log(synthetic_df["close"] / synthetic_df["close"].shift(1)).dropna()
        prob = mc.prob_up(rets, n_bars=12, threshold_pct=0.3)
        assert 0.0 <= prob <= 1.0

    def test_prob_down_range(self, synthetic_df):
        from scalpedge.probabilities import MonteCarlo

        mc = MonteCarlo(n_simulations=200)
        rets = np.log(synthetic_df["close"] / synthetic_df["close"].shift(1)).dropna()
        prob = mc.prob_down(rets, n_bars=12, threshold_pct=0.3)
        assert 0.0 <= prob <= 1.0

    def test_insufficient_data_returns_neutral(self):
        from scalpedge.probabilities import MonteCarlo

        mc = MonteCarlo(n_simulations=100)
        assert mc.prob_up(np.array([0.001, 0.002])) == 0.5

    def test_distribution_shape(self, synthetic_df):
        from scalpedge.probabilities import MonteCarlo

        mc = MonteCarlo(n_simulations=500)
        rets = np.log(synthetic_df["close"] / synthetic_df["close"].shift(1)).dropna()
        dist = mc.full_distribution(rets, n_bars=12)
        assert dist.shape == (500,)


class TestMarkovChain:
    def test_fit_and_predict(self, synthetic_df):
        from scalpedge.probabilities import MarkovChain

        mc = MarkovChain(order=2)
        mc.fit(synthetic_df["close"])
        states = mc.get_states_series(synthetic_df["close"])
        proba = mc.predict_proba(states[-2:])
        assert set(proba.keys()) == {"UP", "DOWN", "FLAT"}
        assert abs(sum(proba.values()) - 1.0) < 1e-9

    def test_unseen_context_uniform(self, synthetic_df):
        from scalpedge.probabilities import MarkovChain

        mc = MarkovChain(order=2)
        mc.fit(synthetic_df["close"])
        proba = mc.predict_proba(["UP", "UP"])  # may be unseen
        assert abs(sum(proba.values()) - 1.0) < 1e-9

    def test_order_1(self, synthetic_df):
        from scalpedge.probabilities import MarkovChain

        mc = MarkovChain(order=1)
        mc.fit(synthetic_df["close"])
        states = mc.get_states_series(synthetic_df["close"])
        proba = mc.predict_proba(states[-1:])
        assert all(0 <= v <= 1 for v in proba.values())

    def test_invalid_order(self):
        from scalpedge.probabilities import MarkovChain

        with pytest.raises(ValueError):
            MarkovChain(order=0)


# ---------------------------------------------------------------------------
# ML
# ---------------------------------------------------------------------------

class TestMLModels:
    @_skip_no_sklearn
    def test_rf_fit_predict(self, clean_df):
        from scalpedge.ml import RandomForestModel

        rf = RandomForestModel(n_estimators=20)
        rf.fit(clean_df)
        proba = rf.predict_proba(clean_df)
        assert proba.shape == (len(clean_df),)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_rf_without_fit_returns_half(self, clean_df):
        from scalpedge.ml import RandomForestModel

        rf = RandomForestModel()
        proba = rf.predict_proba(clean_df)
        assert np.all(proba == 0.5)

    @_skip_no_ml
    def test_lstm_fit_predict(self, clean_df):
        from scalpedge.ml import LSTMModel

        lstm = LSTMModel(seq_len=10, epochs=2, hidden_size=16)
        lstm.fit(clean_df)
        proba = lstm.predict_proba(clean_df)
        assert proba.shape == (len(clean_df),)
        assert np.all((proba >= 0) & (proba <= 1))

    @_skip_no_ml
    def test_ml_engine_score(self, clean_df):
        from scalpedge.ml import MLEngine

        engine = MLEngine(
            rf_kwargs={"n_estimators": 20},
            lstm_kwargs={"seq_len": 10, "epochs": 2, "hidden_size": 16},
        )
        engine.fit(clean_df)
        score = engine.score(clean_df)
        assert len(score) == len(clean_df)
        assert np.all((score >= 0) & (score <= 1))

    # ------------------------------------------------------------------
    # Rolling / online retraining
    # ------------------------------------------------------------------

    @_skip_no_sklearn
    def test_rf_partial_fit_grows_trees(self, clean_df):
        from scalpedge.ml import RandomForestModel

        rf = RandomForestModel(n_estimators=20)
        rf.fit(clean_df)
        initial_trees = rf._model.n_estimators
        rf.partial_fit(clean_df, n_new_trees=10)
        assert rf._model.n_estimators == initial_trees + 10
        proba = rf.predict_proba(clean_df)
        assert proba.shape == (len(clean_df),)
        assert np.all((proba >= 0) & (proba <= 1))

    @_skip_no_sklearn
    def test_rf_partial_fit_without_prior_fit(self, clean_df):
        from scalpedge.ml import RandomForestModel

        rf = RandomForestModel(n_estimators=20)
        rf.partial_fit(clean_df)
        assert rf._fitted
        assert rf._last_fit_dt is not None

    @_skip_no_sklearn
    def test_rf_staleness(self):
        from datetime import datetime, timedelta, timezone

        from scalpedge.ml import RandomForestModel

        rf = RandomForestModel(max_staleness=timedelta(hours=1))
        assert rf.is_stale()  # Never fitted

        rf._last_fit_dt = datetime.now(timezone.utc)
        assert not rf.is_stale()

        stale_time = datetime.now(timezone.utc) + timedelta(hours=2)
        assert rf.is_stale(now=stale_time)

    @_skip_no_sklearn
    def test_rf_last_fit_dt_set(self, clean_df):
        from scalpedge.ml import RandomForestModel

        rf = RandomForestModel(n_estimators=20)
        assert rf.last_fit_dt is None
        rf.fit(clean_df)
        assert rf.last_fit_dt is not None

    @_skip_no_ml
    def test_lstm_partial_fit(self, clean_df):
        from scalpedge.ml import LSTMModel

        lstm = LSTMModel(seq_len=10, epochs=2, hidden_size=16)
        lstm.fit(clean_df)
        first_fit_dt = lstm.last_fit_dt
        lstm.partial_fit(clean_df, epochs=1)
        assert lstm.last_fit_dt >= first_fit_dt
        proba = lstm.predict_proba(clean_df)
        assert proba.shape == (len(clean_df),)
        assert np.all((proba >= 0) & (proba <= 1))

    @_skip_no_ml
    def test_lstm_partial_fit_without_prior_fit(self, clean_df):
        from scalpedge.ml import LSTMModel

        lstm = LSTMModel(seq_len=10, epochs=2, hidden_size=16)
        lstm.partial_fit(clean_df)
        assert lstm._fitted
        assert lstm.last_fit_dt is not None

    @_skip_no_ml
    def test_lstm_staleness(self):
        from datetime import datetime, timedelta, timezone

        from scalpedge.ml import LSTMModel

        lstm = LSTMModel(max_staleness=timedelta(hours=1))
        assert lstm.is_stale()

        lstm._last_fit_dt = datetime.now(timezone.utc)
        assert not lstm.is_stale()

        stale_time = datetime.now(timezone.utc) + timedelta(hours=2)
        assert lstm.is_stale(now=stale_time)

    @_skip_no_ml
    def test_engine_partial_fit(self, clean_df):
        from scalpedge.ml import MLEngine

        engine = MLEngine(
            rf_kwargs={"n_estimators": 20},
            lstm_kwargs={"seq_len": 10, "epochs": 2, "hidden_size": 16},
        )
        engine.fit(clean_df)
        assert not engine.is_stale()

        engine.partial_fit(clean_df, rf_n_new_trees=5, lstm_epochs=1)
        score = engine.score(clean_df)
        assert len(score) == len(clean_df)
        assert np.all((score >= 0) & (score <= 1))

    @_skip_no_ml
    def test_engine_staleness_warning(self, clean_df, caplog):
        import logging
        from datetime import timedelta

        from scalpedge.ml import MLEngine

        engine = MLEngine(
            rf_kwargs={"n_estimators": 20, "max_staleness": timedelta(seconds=0)},
            lstm_kwargs={
                "seq_len": 10,
                "epochs": 2,
                "hidden_size": 16,
                "max_staleness": timedelta(seconds=0),
            },
        )
        engine.fit(clean_df)
        with caplog.at_level(logging.WARNING, logger="scalpedge.ml"):
            engine.score(clean_df)
        assert "stale" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class TestBacktester:
    def test_no_trades(self, clean_df):
        from scalpedge.backtester import Backtester

        bt = Backtester()
        signal = pd.Series(0, index=clean_df.index)
        result = bt.run(clean_df, signal, ticker="TEST")
        assert result.n_trades == 0
        assert result.total_return_pct == 0.0

    def test_all_signals(self, clean_df):
        from scalpedge.backtester import Backtester

        bt = Backtester(hold_bars=3)
        signal = pd.Series(1, index=clean_df.index)
        result = bt.run(clean_df, signal, ticker="TEST")
        assert result.n_trades > 0
        assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_length(self, clean_df):
        from scalpedge.backtester import Backtester

        bt = Backtester()
        signal = pd.Series(1, index=clean_df.index)
        result = bt.run(clean_df, signal)
        assert len(result.equity_curve) == len(clean_df)

    def test_metrics_consistency(self, clean_df):
        from scalpedge.backtester import Backtester

        bt = Backtester()
        signal = pd.Series([1 if i % 10 == 0 else 0 for i in range(len(clean_df))], index=clean_df.index)
        result = bt.run(clean_df, signal)
        if result.n_trades > 0:
            assert result.profit_factor >= 0
            assert result.max_drawdown_pct <= 0

    def test_atr_exits_basic(self, clean_df):
        """ATR-based exits should fire and populate exit_type in trade log."""
        from scalpedge.backtester import Backtester

        bt = Backtester(hold_bars=6, atr_sl_mult=1.0, atr_tp_mult=1.5)
        signal = pd.Series([1 if i % 8 == 0 else 0 for i in range(len(clean_df))], index=clean_df.index)
        # Use a small constant ATR so stops/targets are well-defined
        atr = pd.Series(clean_df["close"].mean() * 0.005, index=clean_df.index)
        result = bt.run(clean_df, signal, ticker="TEST", atr=atr)
        if result.n_trades > 0:
            assert "exit_type" in result.trade_log.columns
            assert result.trade_log["exit_type"].isin(["tp", "sl", "time"]).all()
            # Exit rates should sum to ~1
            total = result.atr_tp_rate + result.atr_sl_rate + result.atr_time_exit_rate
            assert abs(total - 1.0) < 1e-9

    def test_atr_exits_sl_triggered(self):
        """A trade where price drops sharply should hit the stop loss."""
        from scalpedge.backtester import Backtester

        # Build a synthetic price series: entry at 100, then drops significantly
        prices = [100.0] * 3 + [90.0] * 10
        n = len(prices)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
        })
        # Override low on bar 2 (i+1=2 relative to entry) to go well below SL
        df.loc[2, "low"] = 95.0  # below 100 - 1.5 * 1.0 = 98.5

        atr = pd.Series(1.0, index=df.index)
        bt = Backtester(hold_bars=8, atr_sl_mult=1.5, atr_tp_mult=3.0, fee_pct=0.0, slippage_pct=0.0)
        signal = pd.Series([1] + [0] * (n - 1), index=df.index)
        result = bt.run(df, signal, atr=atr)
        assert result.n_trades == 1
        assert result.trade_log.iloc[0]["exit_type"] == "sl"
        assert result.atr_sl_rate == 1.0

    def test_atr_exits_tp_triggered(self):
        """A trade where price rises sharply should hit the take profit."""
        from scalpedge.backtester import Backtester

        prices = [100.0] * 10
        n = len(prices)
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
        })
        # Override high on bar 2 to go well above TP (100 + 2.0 * 1.0 = 102)
        df.loc[2, "high"] = 103.0

        atr = pd.Series(1.0, index=df.index)
        bt = Backtester(hold_bars=8, atr_sl_mult=1.5, atr_tp_mult=2.0, fee_pct=0.0, slippage_pct=0.0)
        signal = pd.Series([1] + [0] * (n - 1), index=df.index)
        result = bt.run(df, signal, atr=atr)
        assert result.n_trades == 1
        assert result.trade_log.iloc[0]["exit_type"] == "tp"
        assert result.atr_tp_rate == 1.0

    def test_atr_exits_fallback_without_atr(self, clean_df):
        """When atr_sl_mult/atr_tp_mult are set but atr=None, falls back to hold_bars."""
        from scalpedge.backtester import Backtester

        bt = Backtester(hold_bars=3, atr_sl_mult=1.5, atr_tp_mult=2.0)
        signal = pd.Series([1 if i % 5 == 0 else 0 for i in range(len(clean_df))], index=clean_df.index)
        result = bt.run(clean_df, signal, ticker="TEST")
        if result.n_trades > 0:
            # All exits should be time exits (fallback to hold_bars)
            assert (result.trade_log["exit_type"] == "time").all()

    def test_atr_exits_summary_shows_breakdown(self, clean_df):
        """BacktestResult.summary() should include ATR breakdown when relevant."""
        from scalpedge.backtester import Backtester, BacktestResult

        bt = Backtester(hold_bars=6, atr_sl_mult=1.0, atr_tp_mult=1.5)
        signal = pd.Series([1 if i % 8 == 0 else 0 for i in range(len(clean_df))], index=clean_df.index)
        atr = pd.Series(clean_df["close"].mean() * 0.005, index=clean_df.index)
        result = bt.run(clean_df, signal, atr=atr)
        summary = result.summary()
        if result.atr_tp_rate + result.atr_sl_rate > 0:
            assert "ATR Exit Breakdown" in summary
            assert "TP Exits" in summary
            assert "SL Exits" in summary

    def test_strategy_atr_exits_via_backtest(self, clean_df):
        """TAStrategy.backtest() should auto-use atr_14 from df for ATR exits."""
        from scalpedge.strategies import TAStrategy
        from scalpedge.ta_indicators import add_all_indicators

        df_with_indicators = add_all_indicators(clean_df.copy())
        result = TAStrategy().backtest(
            df_with_indicators,
            ticker="TEST",
            hold_bars=6,
            atr_sl_mult=1.5,
            atr_tp_mult=2.0,
        )
        assert result.ticker == "TEST"
        if result.n_trades > 0:
            assert "exit_type" in result.trade_log.columns
            total = result.atr_tp_rate + result.atr_sl_rate + result.atr_time_exit_rate
            assert abs(total - 1.0) < 1e-9

class TestStrategies:
    def test_ta_strategy_signals(self, clean_df):
        from scalpedge.strategies import TAStrategy

        strategy = TAStrategy()
        signals = strategy.generate_signals(clean_df)
        assert len(signals) == len(clean_df)
        assert signals.isin([0, 1]).all()

    def test_ta_strategy_backtest(self, clean_df):
        from scalpedge.strategies import TAStrategy

        result = TAStrategy().backtest(clean_df, ticker="TEST",
                                       fee_pct=0.005, slippage_pct=0.01, hold_bars=3)
        assert result.ticker == "TEST"
        assert result.n_trades >= 0

    def test_hybrid_strategy_signals(self, clean_df):
        from scalpedge.strategies import HybridStrategy

        strategy = HybridStrategy(
            use_ml=False, use_markov=True, use_mc=True, use_bs=True,
            mc_n_simulations=100,
        )
        signals = strategy.generate_signals(clean_df)
        assert len(signals) == len(clean_df)
        assert signals.isin([0, 1]).all()

    @_skip_no_ml
    def test_hybrid_with_ml(self, clean_df):
        from scalpedge.strategies import HybridStrategy

        split = int(len(clean_df) * 0.8)
        train = clean_df.iloc[:split]
        test = clean_df.iloc[split:].reset_index(drop=True)

        strategy = HybridStrategy(
            use_ml=True, use_markov=True, use_mc=True, use_bs=True,
            mc_n_simulations=100,
        )
        strategy.fit_ml(train)
        result = strategy.backtest(test, ticker="TEST",
                                    fee_pct=0.005, slippage_pct=0.01, hold_bars=3)
        assert result.ticker == "TEST"
        assert result.strategy == "hybrid"


# ---------------------------------------------------------------------------
# PolygonClient — Stocks Advanced features
# ---------------------------------------------------------------------------

class TestPolygonClientAdvanced:
    """Tests for the new Stocks Advanced methods on PolygonClient."""

    def _make_client(self):
        from scalpedge.data import PolygonClient

        client = PolygonClient.__new__(PolygonClient)
        client._api_key = "test"
        client._last_call_time = 0.0
        return client

    def test_rate_limit_disabled(self):
        """_rate_limit() should not sleep when _POLYGON_RATE_LIMIT_SECONDS == 0."""
        import scalpedge.data as data_mod
        from unittest.mock import patch

        original = data_mod._POLYGON_RATE_LIMIT_SECONDS
        try:
            data_mod._POLYGON_RATE_LIMIT_SECONDS = 0.0
            client = self._make_client()
            with patch("scalpedge.data.time.sleep") as mock_sleep:
                client._rate_limit()
                mock_sleep.assert_not_called()
        finally:
            data_mod._POLYGON_RATE_LIMIT_SECONDS = original

    def test_fetch_trades_basic(self):
        """fetch_trades should return a DataFrame with expected columns and dtypes."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        ts_ns = pd.Timestamp("2024-01-02 10:00", tz="UTC").value
        results = [
            {"sip_timestamp": ts_ns, "price": 450.0, "size": 100, "exchange": 4,
             "conditions": [14, 41]},
            {"sip_timestamp": ts_ns + 1_000_000, "price": 450.5, "size": 200,
             "exchange": 4, "conditions": [14]},
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "OK", "results": results}
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            df = client.fetch_trades(
                "SPY",
                pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
            )

        assert not df.empty
        assert "datetime" in df.columns
        assert "price" in df.columns
        assert "size" in df.columns
        assert df["datetime"].dt.tz is not None
        assert len(df) == 2

    def test_fetch_quotes_derived_columns(self):
        """fetch_quotes should compute spread, mid_price, bid_ask_imbalance."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        ts_ns = pd.Timestamp("2024-01-02 10:00", tz="UTC").value
        results = [
            {
                "sip_timestamp": ts_ns,
                "bid_price": 449.0,
                "bid_size": 300,
                "ask_price": 451.0,
                "ask_size": 100,
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "OK", "results": results}
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            df = client.fetch_quotes(
                "SPY",
                pd.Timestamp("2024-01-02", tz="UTC"),
                pd.Timestamp("2024-01-03", tz="UTC"),
            )

        assert not df.empty
        assert "spread" in df.columns
        assert "mid_price" in df.columns
        assert "bid_ask_imbalance" in df.columns
        assert abs(df["spread"].iloc[0] - 2.0) < 1e-9
        assert abs(df["mid_price"].iloc[0] - 450.0) < 1e-9
        # bid_size=300, ask_size=100 → positive imbalance
        assert df["bid_ask_imbalance"].iloc[0] > 0

    def test_fetch_snapshot_flattening(self):
        """fetch_snapshot should flatten nested Polygon response into expected columns."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        raw_tickers = [
            {
                "ticker": "SPY",
                "todaysChangePerc": 0.52,
                "day": {"o": 449.0, "h": 452.0, "l": 448.0, "c": 451.0, "v": 5_000_000},
                "prevDay": {"c": 448.67},
                "lastTrade": {"p": 451.12, "s": 100},
                "min": {"o": 450.9, "h": 451.2, "l": 450.7, "c": 451.0, "v": 12_000},
            }
        ]
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "OK", "tickers": raw_tickers}
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            df = client.fetch_snapshot(tickers=["SPY"])

        expected_cols = {
            "ticker", "day_open", "day_high", "day_low", "day_close", "day_volume",
            "prev_close", "change_pct", "last_trade_price", "last_trade_size",
            "min_open", "min_high", "min_low", "min_close", "min_volume",
        }
        assert expected_cols.issubset(set(df.columns))
        assert df["ticker"].iloc[0] == "SPY"
        assert df["change_pct"].iloc[0] == 0.52

    def test_fetch_news_empty(self):
        """fetch_news should return an empty DataFrame when there are no results."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "OK", "results": []}
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            df = client.fetch_news("SPY")

        assert df.empty
        assert "title" in df.columns

    def test_fetch_events_basic(self):
        """fetch_events should parse date strings into pd.Timestamp objects."""
        from scalpedge.data import PolygonClient
        from unittest.mock import MagicMock, patch

        client = self._make_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "status": "OK",
            "results": {
                "events": [
                    {"type": "earnings", "date": "2024-01-25", "name": "Q4 2023 Earnings"},
                    {"type": "split", "date": "2024-03-15", "name": "4:1 Stock Split"},
                ]
            },
        }
        mock_resp.raise_for_status.return_value = None

        with patch("scalpedge.data.requests.get", return_value=mock_resp):
            df = client.fetch_events("AAPL")

        assert not df.empty
        assert "event_type" in df.columns
        assert "date" in df.columns
        assert isinstance(df["date"].iloc[0], pd.Timestamp)
        assert df["event_type"].iloc[0] == "earnings"


# ---------------------------------------------------------------------------
# Quote features (ta_indicators)
# ---------------------------------------------------------------------------

class TestQuoteFeatures:
    def test_add_quote_features_present(self):
        """add_quote_features should compute derived columns when input cols present."""
        from scalpedge.ta_indicators import add_quote_features

        df = pd.DataFrame(
            {
                "bid_price": [100.0, 100.5, 101.0],
                "ask_price": [100.2, 100.7, 101.3],
                "bid_size": [200.0, 300.0, 150.0],
                "ask_size": [100.0, 100.0, 200.0],
            }
        )
        result = add_quote_features(df)

        assert "spread" in result.columns
        assert "mid_price" in result.columns
        assert "bid_ask_imbalance" in result.columns
        assert "spread_pct" in result.columns
        assert "imbalance_ma_10" in result.columns

        # Spot-check first row: spread = 0.2, mid = 100.1
        assert abs(result["spread"].iloc[0] - 0.2) < 1e-9
        assert abs(result["mid_price"].iloc[0] - 100.1) < 1e-6
        # bid_size=200 > ask_size=100 → positive imbalance
        assert result["bid_ask_imbalance"].iloc[0] > 0

    def test_add_quote_features_absent(self):
        """add_quote_features should return the DataFrame unchanged if bid/ask absent."""
        from scalpedge.ta_indicators import add_quote_features

        df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [100, 200, 300]})
        result = add_quote_features(df)

        assert list(result.columns) == list(df.columns)
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Catalyst suppression filter (strategies)
# ---------------------------------------------------------------------------

class TestCatalystFilter:
    def _make_df(self, n: int = 50) -> pd.DataFrame:
        """Small synthetic DataFrame with a datetime column."""
        return pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-15 09:30", periods=n, freq="5min", tz="UTC"),
                "close": 450.0 + np.arange(n) * 0.01,
                "open": 449.0 + np.arange(n) * 0.01,
                "high": 451.0 + np.arange(n) * 0.01,
                "low": 448.0 + np.arange(n) * 0.01,
                "volume": 100_000.0 * np.ones(n),
                "ticker": "TEST",
            }
        )

    def test_catalyst_suppression(self):
        """Signals near a catalyst date should be zeroed out."""
        from scalpedge.strategies import HybridStrategy

        df = self._make_df(50)
        # All signals are 1.
        signal = pd.Series(1, index=df.index)

        # Catalyst at the 25th bar's datetime.
        catalyst_dt = df["datetime"].iloc[25]
        strategy = HybridStrategy(
            use_ml=False,
            use_markov=False,
            use_mc=False,
            use_bs=False,
            use_catalyst_filter=True,
            catalyst_dates=[catalyst_dt],
            catalyst_suppress_bars=3,
        )
        filtered = strategy._apply_catalyst_filter(signal, df)

        # Bars 22–28 (±3 around bar 25) should be 0.
        assert filtered.iloc[22] == 0
        assert filtered.iloc[25] == 0
        assert filtered.iloc[28] == 0
        # Bars well outside the window should still be 1.
        assert filtered.iloc[0] == 1
        assert filtered.iloc[49] == 1

    def test_catalyst_filter_disabled(self):
        """Signals should be unchanged when use_catalyst_filter=False."""
        from scalpedge.strategies import HybridStrategy

        df = self._make_df(30)
        signal = pd.Series(1, index=df.index)
        catalyst_dt = df["datetime"].iloc[15]
        strategy = HybridStrategy(
            use_ml=False,
            use_markov=False,
            use_mc=False,
            use_bs=False,
            use_catalyst_filter=False,
            catalyst_dates=[catalyst_dt],
            catalyst_suppress_bars=3,
        )
        filtered = strategy._apply_catalyst_filter(signal, df)
        # No suppression — all 1.
        assert (filtered == 1).all()

    def test_catalyst_date_only_suppresses_full_day(self):
        """A date-only catalyst should suppress all bars on that calendar day."""
        from scalpedge.strategies import HybridStrategy

        df = self._make_df(50)
        signal = pd.Series(1, index=df.index)
        # Date-only catalyst: UTC midnight on 2024-01-15.
        catalyst_date = pd.Timestamp("2024-01-15", tz="UTC")  # hour=0, min=0, sec=0
        strategy = HybridStrategy(
            use_ml=False,
            use_markov=False,
            use_mc=False,
            use_bs=False,
            use_catalyst_filter=True,
            catalyst_dates=[catalyst_date],
            catalyst_suppress_bars=3,
        )
        filtered = strategy._apply_catalyst_filter(signal, df)
        # All bars are on 2024-01-15, so all should be 0.
        assert (filtered == 0).all()
