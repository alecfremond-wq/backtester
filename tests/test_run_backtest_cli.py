import pytest

from run_backtest import build_strategy, parse_strategy_params
from backtester.strategy.examples.breakout import RangeBreakout
from backtester.strategy.examples.ma_crossover import MovingAverageCrossover


def test_parse_strategy_params_casts_types():
    params = parse_strategy_params(["fast=10", "slow=50", "long_only=true"])
    assert params == {"fast": 10, "slow": 50, "long_only": "true"}


def test_build_strategy_ma_crossover():
    strat = build_strategy("ma_crossover", {"fast": 5, "slow": 20})
    assert isinstance(strat, MovingAverageCrossover)
    assert strat.fast == 5 and strat.slow == 20


def test_build_strategy_breakout():
    strat = build_strategy("breakout", {"window": 15})
    assert isinstance(strat, RangeBreakout)
    assert strat.window == 15


def test_build_strategy_unknown_raises():
    with pytest.raises(ValueError):
        build_strategy("nope", {})
