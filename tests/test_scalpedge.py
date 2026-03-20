"""Tests for ScalpEdge modules using synthetic data (no network required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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

    def test_lstm_fit_predict(self, clean_df):
        from scalpedge.ml import LSTMModel

        lstm = LSTMModel(seq_len=10, epochs=2, hidden_size=16)
        lstm.fit(clean_df)
        proba = lstm.predict_proba(clean_df)
        assert proba.shape == (len(clean_df),)
        assert np.all((proba >= 0) & (proba <= 1))

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
