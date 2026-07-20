from __future__ import annotations

import pandas as pd

from backtester.strategy.base import Strategy


class MovingAverageCrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50, long_only: bool = False):
        if fast >= slow:
            raise ValueError("fast window must be smaller than slow window")
        self.fast = fast
        self.slow = slow
        self.long_only = long_only

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()

        signal = pd.Series(0, index=df.index, dtype="int64")
        signal[fast_ma > slow_ma] = 1
        if not self.long_only:
            signal[fast_ma < slow_ma] = -1

        warmup = fast_ma.isna() | slow_ma.isna()
        signal[warmup] = 0
        return signal
