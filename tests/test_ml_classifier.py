import numpy as np
import pandas as pd
import pytest

from backtester.strategy.examples.ml_classifier import MLClassifier


def test_prob_threshold_must_be_between_half_and_one():
    with pytest.raises(ValueError):
        MLClassifier(prob_threshold=0.4)
    with pytest.raises(ValueError):
        MLClassifier(prob_threshold=1.0)


def test_flat_before_min_train_bars():
    df = make_df(n=100)
    strat = MLClassifier(min_train_bars=500)
    signal = strat.generate_signals(df)
    assert (signal == 0).all()
    assert len(signal) == len(df)


def make_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    # momentum regime switches every 40 bars: informative signal for the classifier to learn
    regime = np.repeat(np.resize([1, -1], n // 40 + 1), 40)[:n]
    daily_returns = regime * 0.004 + rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + daily_returns)
    volume = rng.integers(1_000_000, 2_000_000, n).astype("float64")
    return pd.DataFrame(
        {
            "date": pd.date_range("2015-01-01", periods=n, freq="D"),
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": volume,
        }
    )


def test_generate_signals_runs_and_returns_valid_values():
    df = make_df(n=400)
    strat = MLClassifier(horizon=5, retrain_every=40, min_train_bars=150)
    signal = strat.generate_signals(df)

    assert len(signal) == len(df)
    assert set(signal.unique()).issubset({-1, 0, 1})
    assert signal.dtype == np.int64
    # the model should actually take a position at some point past warmup
    assert (signal.iloc[150:] != 0).any()


def test_long_only_never_shorts():
    df = make_df(n=400)
    strat = MLClassifier(horizon=5, retrain_every=40, min_train_bars=150, long_only=True)
    signal = strat.generate_signals(df)
    assert (signal >= 0).all()


def test_deterministic_across_runs():
    df = make_df(n=400)
    strat = MLClassifier(horizon=5, retrain_every=40, min_train_bars=150)
    signal_a = strat.generate_signals(df)
    signal_b = strat.generate_signals(df)
    pd.testing.assert_series_equal(signal_a, signal_b)
