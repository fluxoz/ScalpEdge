"""Machine Learning & Deep Learning module.

Provides:
* :class:`MLEngine` — orchestrates both RF and LSTM models
* :class:`RandomForestModel` — sklearn RF classifier for next-bar direction
* :class:`LSTMModel`         — PyTorch LSTM sequence classifier

Both models return a ``prob_up`` float ∈ [0, 1] that can be used as a
filter in hybrid strategies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "atr_14", "cci_20", "mfi_14", "adx_14",
    "stoch_k", "stoch_d", "williams_r", "roc_12",
    "price_vs_vwap",
    "ema_9", "ema_21", "ema_50",
    "pat_bull_signal", "pat_bear_signal",
    # Bid-ask microstructure features (present when quote data is joined)
    "spread_pct", "bid_ask_imbalance", "imbalance_ma_10",
]


def _make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and forward-fill ML features."""
    available = [c for c in _FEATURE_COLS if c in df.columns]
    X = df[available].copy().ffill().fillna(0.0)
    return X.astype(float)


def _make_labels(close: pd.Series, n_bars_ahead: int = 1) -> pd.Series:
    """Binary label: 1 if close rises over next *n_bars_ahead* bars, else 0."""
    future = close.shift(-n_bars_ahead)
    return (future > close).astype(int)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class RandomForestModel:
    """Sklearn RandomForestClassifier for next-candle direction.

    Parameters
    ----------
    n_estimators:   Number of trees.
    n_bars_ahead:   Prediction horizon in bars.
    random_state:   Random seed for reproducibility.
    max_staleness:  Maximum age of model before it is considered stale.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        n_bars_ahead: int = 1,
        random_state: int = 42,
        max_staleness: timedelta = timedelta(hours=24),
    ) -> None:
        self.n_estimators = n_estimators
        self.n_bars_ahead = n_bars_ahead
        self.random_state = random_state
        self.max_staleness = max_staleness
        self._model = None
        self._fitted = False
        self._last_fit_dt: datetime | None = None

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "RandomForestModel":
        """Fit on TA-enriched DataFrame."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.utils.class_weight import compute_class_weight
        except ImportError:
            logger.error("scikit-learn not installed — RandomForest will return 0.5.")
            return self

        X = _make_features(df)
        y = _make_labels(df["close"], self.n_bars_ahead)

        # Remove rows where label is NaN (last n_bars_ahead rows).
        mask = y.notna()
        X, y = X[mask], y[mask]

        if len(X) < 50:
            logger.warning("Too few samples (%d) to fit RandomForest.", len(X))
            return self

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        cw = dict(zip(classes, weights))

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight=cw,
            warm_start=True,
        )
        self._model.fit(X, y)
        self._feature_cols = list(X.columns)
        self._fitted = True
        self._last_fit_dt = datetime.now(timezone.utc)
        return self

    def partial_fit(
        self,
        df: pd.DataFrame,
        n_new_trees: int = 50,
    ) -> "RandomForestModel":
        """Incrementally add trees trained on new data via ``warm_start``.

        If the model has not been fitted yet, falls back to a full :meth:`fit`.
        """
        if not self._fitted or self._model is None:
            logger.info("No existing RF model — falling back to full fit.")
            return self.fit(df)

        try:
            from sklearn.utils.class_weight import compute_class_weight
        except ImportError:
            logger.error("scikit-learn not installed — cannot partial-fit.")
            return self

        X = _make_features(df)
        y = _make_labels(df["close"], self.n_bars_ahead)
        mask = y.notna()
        X, y = X[mask], y[mask]

        if len(X) < 50:
            logger.warning("Too few samples (%d) to partial-fit RandomForest.", len(X))
            return self

        # Align features with original training columns.
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._feature_cols]

        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        self._model.class_weight = dict(zip(classes, weights))

        # Grow additional trees using warm_start.
        self._model.n_estimators += n_new_trees
        self._model.fit(X, y)
        self._last_fit_dt = datetime.now(timezone.utc)
        logger.info(
            "RF partial-fit complete — total trees: %d", self._model.n_estimators
        )
        return self

    # ------------------------------------------------------------------

    @property
    def last_fit_dt(self) -> datetime | None:
        """UTC timestamp of the last fit/partial-fit, or *None*."""
        return self._last_fit_dt

    def is_stale(self, now: datetime | None = None) -> bool:
        """Return *True* if model has never been fitted or exceeds *max_staleness*."""
        if self._last_fit_dt is None:
            return True
        ref = now if now is not None else datetime.now(timezone.utc)
        return (ref - self._last_fit_dt) > self.max_staleness

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return array of P(up) for each row in *df*."""
        if not self._fitted or self._model is None:
            logger.warning("RandomForest not fitted — returning 0.5 array.")
            return np.full(len(df), 0.5)

        X = _make_features(df)
        # Align to training features.
        for col in self._feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._feature_cols]
        proba = self._model.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def feature_importances(self) -> pd.Series:
        """Return feature importances as a Series (requires fitted model)."""
        if not self._fitted or self._model is None:
            return pd.Series(dtype=float)
        return pd.Series(self._model.feature_importances_, index=self._feature_cols).sort_values(
            ascending=False
        )


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------

class LSTMModel:
    """PyTorch LSTM for sequence-based price direction probability.

    Parameters
    ----------
    seq_len:        Look-back window length in bars.
    hidden_size:    LSTM hidden units.
    num_layers:     Number of stacked LSTM layers.
    n_bars_ahead:   Prediction horizon in bars.
    epochs:         Training epochs.
    lr:             Learning rate.
    batch_size:     Mini-batch size.
    device:         ``'cpu'`` or ``'cuda'``.
    max_staleness:  Maximum age of model before it is considered stale.
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_bars_ahead: int = 1,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: str = "cpu",
        max_staleness: timedelta = timedelta(hours=24),
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_bars_ahead = n_bars_ahead
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device_name = device
        self.max_staleness = max_staleness
        self._net = None
        self._fitted = False
        self._scaler = None
        self._feature_cols: list[str] = []
        self._last_fit_dt: datetime | None = None

    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "LSTMModel":
        """Train the LSTM on an indicator-enriched DataFrame."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch not installed — LSTM will return 0.5.")
            return self

        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed — LSTM will return 0.5.")
            return self

        X_df = _make_features(df)
        y = _make_labels(df["close"], self.n_bars_ahead)
        mask = y.notna()
        X_df, y = X_df[mask], y[mask]

        if len(X_df) < self.seq_len + 10:
            logger.warning("Too few samples for LSTM training.")
            return self

        self._feature_cols = list(X_df.columns)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df.values.astype(float))
        self._scaler = scaler

        # Build sequences.
        seqs, labels = [], []
        for i in range(self.seq_len, len(X_scaled)):
            seqs.append(X_scaled[i - self.seq_len : i])
            labels.append(int(y.iloc[i]))

        X_tensor = torch.FloatTensor(np.array(seqs))
        y_tensor = torch.FloatTensor(labels)

        device = torch.device(self.device_name)
        n_features = X_tensor.shape[2]

        net = _LSTMNet(n_features, self.hidden_size, self.num_layers).to(device)
        optim = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                preds = net(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optim.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.debug("LSTM epoch %d/%d — loss %.4f", epoch + 1, self.epochs, total_loss)

        net.eval()
        self._net = net
        self._device = device
        self._fitted = True
        self._last_fit_dt = datetime.now(timezone.utc)
        return self

    def partial_fit(
        self,
        df: pd.DataFrame,
        epochs: int | None = None,
    ) -> "LSTMModel":
        """Continue training the existing LSTM on new data.

        If the model has not been fitted yet, falls back to a full :meth:`fit`.

        Parameters
        ----------
        df:
            New indicator-enriched DataFrame to train on.
        epochs:
            Number of additional epochs.  Defaults to ``self.epochs``.
        """
        if not self._fitted or self._net is None:
            logger.info("No existing LSTM model — falling back to full fit.")
            return self.fit(df)

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch not installed — LSTM will return 0.5.")
            return self

        epochs = epochs if epochs is not None else self.epochs

        X_df = _make_features(df)
        y = _make_labels(df["close"], self.n_bars_ahead)
        mask = y.notna()
        X_df, y = X_df[mask], y[mask]

        if len(X_df) < self.seq_len + 10:
            logger.warning("Too few samples for LSTM partial-fit.")
            return self

        # Align features.
        for col in self._feature_cols:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[self._feature_cols]

        X_scaled = self._scaler.transform(X_df.values.astype(float))

        seqs, labels = [], []
        for i in range(self.seq_len, len(X_scaled)):
            seqs.append(X_scaled[i - self.seq_len : i])
            labels.append(int(y.iloc[i]))

        X_tensor = torch.FloatTensor(np.array(seqs))
        y_tensor = torch.FloatTensor(labels)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        criterion = nn.BCELoss()
        optim = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self._net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optim.zero_grad()
                preds = self._net(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optim.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.debug(
                    "LSTM partial-fit epoch %d/%d — loss %.4f", epoch + 1, epochs, total_loss
                )

        self._net.eval()
        self._last_fit_dt = datetime.now(timezone.utc)
        logger.info("LSTM partial-fit complete (%d epochs).", epochs)
        return self

    # ------------------------------------------------------------------

    @property
    def last_fit_dt(self) -> datetime | None:
        """UTC timestamp of the last fit/partial-fit, or *None*."""
        return self._last_fit_dt

    def is_stale(self, now: datetime | None = None) -> bool:
        """Return *True* if model has never been fitted or exceeds *max_staleness*."""
        if self._last_fit_dt is None:
            return True
        ref = now if now is not None else datetime.now(timezone.utc)
        return (ref - self._last_fit_dt) > self.max_staleness

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(up) for each row. Uses a rolling window of seq_len rows."""
        if not self._fitted or self._net is None:
            return np.full(len(df), 0.5)

        try:
            import torch
        except ImportError:
            return np.full(len(df), 0.5)

        X_df = _make_features(df)
        for col in self._feature_cols:
            if col not in X_df.columns:
                X_df[col] = 0.0
        X_df = X_df[self._feature_cols]
        X_scaled = self._scaler.transform(X_df.values.astype(float))

        probs = np.full(len(df), 0.5)
        self._net.eval()
        with torch.no_grad():
            for i in range(self.seq_len, len(X_scaled)):
                seq = torch.FloatTensor(X_scaled[i - self.seq_len : i]).unsqueeze(0)
                seq = seq.to(self._device)
                prob = self._net(seq).squeeze().item()
                probs[i] = prob
        return probs


class _LSTMNet:
    """Internal PyTorch LSTM network."""

    def __new__(cls, n_features: int, hidden_size: int, num_layers: int) -> "_LSTMNet":
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.2 if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.fc(out[:, -1, :])
                return self.sigmoid(out)

        return _Net()


# ---------------------------------------------------------------------------
# Combined ML Engine
# ---------------------------------------------------------------------------

class MLEngine:
    """High-level interface that fits and queries both RF and LSTM models.

    Parameters
    ----------
    rf_kwargs:   kwargs forwarded to :class:`RandomForestModel`.
    lstm_kwargs: kwargs forwarded to :class:`LSTMModel`.
    """

    def __init__(
        self,
        rf_kwargs: dict | None = None,
        lstm_kwargs: dict | None = None,
    ) -> None:
        self.rf = RandomForestModel(**(rf_kwargs or {}))
        self.lstm = LSTMModel(**(lstm_kwargs or {}))
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "MLEngine":
        """Fit both models on the indicator-enriched DataFrame."""
        logger.info("Fitting RandomForest ...")
        self.rf.fit(df)
        logger.info("Fitting LSTM ...")
        self.lstm.fit(df)
        self._fitted = True
        return self

    def partial_fit(
        self,
        df: pd.DataFrame,
        rf_n_new_trees: int = 50,
        lstm_epochs: int | None = None,
    ) -> "MLEngine":
        """Incrementally retrain both models on new data.

        Parameters
        ----------
        df:
            New indicator-enriched DataFrame.
        rf_n_new_trees:
            Number of additional trees to grow in the RF model.
        lstm_epochs:
            Number of additional epochs for LSTM.  Defaults to the model's
            configured epoch count.
        """
        logger.info("Partial-fitting RandomForest ...")
        self.rf.partial_fit(df, n_new_trees=rf_n_new_trees)
        logger.info("Partial-fitting LSTM ...")
        self.lstm.partial_fit(df, epochs=lstm_epochs)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Staleness monitoring
    # ------------------------------------------------------------------

    def is_stale(self, now: datetime | None = None) -> bool:
        """Return *True* if **either** sub-model is stale."""
        return self.rf.is_stale(now=now) or self.lstm.is_stale(now=now)

    def score(self, df: pd.DataFrame, weights: tuple[float, float] = (0.5, 0.5)) -> pd.Series:
        """Return weighted-average P(up) signal for every row in *df*.

        Parameters
        ----------
        df:
            Indicator-enriched DataFrame.
        weights:
            (rf_weight, lstm_weight) — must sum to 1.
        """
        if self.is_stale():
            logger.warning(
                "ML model is stale (RF last fit: %s, LSTM last fit: %s). "
                "Consider calling partial_fit() or fit() with recent data.",
                self.rf.last_fit_dt,
                self.lstm.last_fit_dt,
            )

        rf_fitted = self.rf._fitted
        lstm_fitted = self.lstm._fitted

        if not rf_fitted and not lstm_fitted:
            # Neither model could be fitted (e.g. ML libraries not installed).
            # Return a pass-through score so the ML layer does not block signals.
            logger.warning(
                "MLEngine: neither RF nor LSTM is fitted — returning pass-through scores."
            )
            return pd.Series(1.0, index=df.index, name="ml_score")

        w_rf, w_lstm = weights
        rf_prob = self.rf.predict_proba(df)
        lstm_prob = self.lstm.predict_proba(df)

        if rf_fitted and not lstm_fitted:
            return pd.Series(rf_prob, index=df.index, name="ml_score")
        if lstm_fitted and not rf_fitted:
            return pd.Series(lstm_prob, index=df.index, name="ml_score")

        combined = w_rf * rf_prob + w_lstm * lstm_prob
        return pd.Series(combined, index=df.index, name="ml_score")
