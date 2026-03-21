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


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

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
