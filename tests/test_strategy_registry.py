import pytest

from backtester.strategy.examples.breakout import RangeBreakout
from backtester.strategy.examples.ma_crossover import MovingAverageCrossover
from backtester.strategy.examples.mean_reversion import MeanReversionScalper
from backtester.strategy.examples.ml_classifier import MLClassifier
from backtester.strategy.examples.trend_filtered_pullback import TrendFilteredPullback
from backtester.strategy.registry import STRATEGIES, build_strategy


def test_registry_contains_baseline_strategies():
    assert set(STRATEGIES) == {
        "ma_crossover", "breakout", "ml_classifier", "mean_reversion", "trend_pullback"
    }


def test_build_strategy_ma_crossover():
    strat = build_strategy("ma_crossover", {"fast": 5, "slow": 20})
    assert isinstance(strat, MovingAverageCrossover)


def test_build_strategy_breakout():
    strat = build_strategy("breakout", {"window": 15})
    assert isinstance(strat, RangeBreakout)


def test_build_strategy_ml_classifier():
    strat = build_strategy("ml_classifier", {"horizon": 5, "retrain_every": 40})
    assert isinstance(strat, MLClassifier)


def test_build_strategy_mean_reversion():
    strat = build_strategy("mean_reversion", {"lookback": 5, "entry_z": 1.2})
    assert isinstance(strat, MeanReversionScalper)


def test_build_strategy_trend_pullback():
    strat = build_strategy("trend_pullback", {"trend_ma": 100, "lookback": 5})
    assert isinstance(strat, TrendFilteredPullback)


def test_build_strategy_unknown_raises():
    with pytest.raises(ValueError):
        build_strategy("nope", {})
