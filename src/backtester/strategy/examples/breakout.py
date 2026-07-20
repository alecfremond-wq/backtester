from __future__ import annotations

import pandas as pd

from backtester.strategy.base import Strategy


class RangeBreakout(Strategy):
    def __init__(self, window: int = 20, long_only: bool = False):
        self.window = window
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rolling_high = df["high"].rolling(self.window).max().shift(1)
        rolling_low = df["low"].rolling(self.window).min().shift(1)

        raw = pd.Series(index=df.index, dtype="float64")
        raw[df["close"] > rolling_high] = 1
        if not self.long_only:
            raw[df["close"] < rolling_low] = -1

        return raw.ffill().fillna(0).astype("int64")
