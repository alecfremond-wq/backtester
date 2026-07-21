from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from backtester.strategy.base import Strategy

FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "volatility_20",
    "price_to_ma20",
    "price_to_ma50",
    "rsi_14",
    "volume_ratio",
]

MIN_TRAIN_SAMPLES = 50


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    features = pd.DataFrame(index=df.index)
    features["ret_1"] = close.pct_change(1)
    features["ret_5"] = close.pct_change(5)
    features["ret_10"] = close.pct_change(10)
    features["ret_20"] = close.pct_change(20)
    features["volatility_20"] = close.pct_change().rolling(20).std()
    features["price_to_ma20"] = close / close.rolling(20).mean() - 1
    features["price_to_ma50"] = close / close.rolling(50).mean() - 1
    features["rsi_14"] = _rsi(close, 14)
    features["volume_ratio"] = (
        df["volume"] / df["volume"].rolling(20).mean() if "volume" in df.columns else np.nan
    )
    return features


class MLClassifier(Strategy):
    """Logistic regression predicting the sign of the forward return.

    Retrained on an expanding window every `retrain_every` bars. At each
    retrain checkpoint, training only uses rows whose label (the return
    `horizon` bars later) is already resolved as of that checkpoint --
    the model never sees information from after the bar it is trained up
    to, mirroring the walk-forward discipline used elsewhere in this repo.
    """

    def __init__(
        self,
        horizon: int = 10,
        retrain_every: int = 63,
        min_train_bars: int = 500,
        prob_threshold: float = 0.55,
        regularization: float = 0.5,
        long_only: bool = False,
    ):
        if not 0.5 < prob_threshold < 1.0:
            raise ValueError("prob_threshold must be strictly between 0.5 and 1.0")
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.min_train_bars = min_train_bars
        self.prob_threshold = prob_threshold
        self.regularization = regularization
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        signal = pd.Series(0, index=df.index, dtype="int64")
        if n <= self.min_train_bars:
            return signal

        features = compute_features(df)
        close = df["close"]
        forward_return = close.shift(-self.horizon) / close - 1
        label = (forward_return > 0).astype("float64")
        label[forward_return.isna()] = np.nan

        checkpoint = self.min_train_bars
        while checkpoint < n:
            train_end = checkpoint - self.horizon
            train_idx = features.index[:train_end]
            X_train = features.loc[train_idx, FEATURE_COLUMNS]
            y_train = label.loc[train_idx]
            valid_train = X_train.notna().all(axis=1) & y_train.notna()
            X_train, y_train = X_train[valid_train], y_train[valid_train]

            predict_end = min(checkpoint + self.retrain_every, n)
            predict_idx = features.index[checkpoint:predict_end]
            X_pred = features.loc[predict_idx, FEATURE_COLUMNS]
            valid_pred_idx = X_pred.index[X_pred.notna().all(axis=1)]

            if len(y_train) >= MIN_TRAIN_SAMPLES and y_train.nunique() == 2 and len(valid_pred_idx) > 0:
                model = make_pipeline(
                    StandardScaler(), LogisticRegression(C=self.regularization, max_iter=1000)
                )
                model.fit(X_train, y_train.astype(int))
                proba_up = model.predict_proba(X_pred.loc[valid_pred_idx])[:, 1]
                proba = pd.Series(proba_up, index=valid_pred_idx)

                block = pd.Series(0, index=predict_idx, dtype="int64")
                block.loc[proba[proba > self.prob_threshold].index] = 1
                if not self.long_only:
                    block.loc[proba[proba < (1 - self.prob_threshold)].index] = -1
                signal.loc[predict_idx] = block

            checkpoint += self.retrain_every

        return signal
