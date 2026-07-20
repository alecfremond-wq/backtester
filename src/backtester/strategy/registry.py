from __future__ import annotations

from backtester.strategy.base import Strategy
from backtester.strategy.examples.breakout import RangeBreakout
from backtester.strategy.examples.ma_crossover import MovingAverageCrossover

STRATEGIES: dict[str, type[Strategy]] = {
    "ma_crossover": MovingAverageCrossover,
    "breakout": RangeBreakout,
}


def build_strategy(name: str, params: dict) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}', choices: {list(STRATEGIES)}")
    return STRATEGIES[name](**params)
