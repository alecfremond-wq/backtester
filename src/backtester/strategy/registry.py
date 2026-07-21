from __future__ import annotations

from backtester.strategy.base import Strategy
from backtester.strategy.examples.breakout import RangeBreakout
from backtester.strategy.examples.ma_crossover import MovingAverageCrossover
from backtester.strategy.examples.mean_reversion import MeanReversionScalper
from backtester.strategy.examples.ml_classifier import MLClassifier
from backtester.strategy.examples.trend_filtered_pullback import TrendFilteredPullback

STRATEGIES: dict[str, type[Strategy]] = {
    "ma_crossover": MovingAverageCrossover,
    "breakout": RangeBreakout,
    "ml_classifier": MLClassifier,
    "mean_reversion": MeanReversionScalper,
    "trend_pullback": TrendFilteredPullback,
}


def build_strategy(name: str, params: dict) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy '{name}', choices: {list(STRATEGIES)}")
    return STRATEGIES[name](**params)
