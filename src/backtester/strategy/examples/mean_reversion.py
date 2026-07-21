from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.strategy.base import Strategy


class MeanReversionScalper(Strategy):
    """Short-term mean reversion: buy dips / sell spikes, exit fast.

    Entry: z-score of price vs its own rolling mean, normalized by rolling
    volatility. |z| > entry_z triggers a position betting on a snap-back.

    Exit (checked every bar while a position is open, first trigger wins):
    profit target reached, stop loss hit, or max_holding_bars elapsed.
    Unlike ma_crossover/breakout, exits are decoupled from the entry signal
    -- a position can close well before the entry condition would reverse.
    """

    def __init__(
        self,
        lookback: int = 10,
        entry_z: float = 1.0,
        profit_target: float = 0.02,
        stop_loss: float = 0.02,
        max_holding_bars: int = 5,
        long_only: bool = False,
    ):
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if profit_target <= 0 or stop_loss <= 0:
            raise ValueError("profit_target and stop_loss must be positive")
        if max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")
        self.lookback = lookback
        self.entry_z = entry_z
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_bars = max_holding_bars
        self.long_only = long_only

    def _zscore(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        return (close - ma) / std.replace(0, np.nan)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].to_numpy()
        z = self._zscore(df).to_numpy()
        n = len(close)
        signal = np.zeros(n, dtype="int64")

        position = 0
        entry_price = 0.0
        bars_held = 0

        for t in range(n):
            if position == 0:
                if not np.isnan(z[t]):
                    if z[t] < -self.entry_z:
                        position, entry_price, bars_held = 1, close[t], 0
                    elif not self.long_only and z[t] > self.entry_z:
                        position, entry_price, bars_held = -1, close[t], 0
            else:
                bars_held += 1
                unrealized = position * (close[t] - entry_price) / entry_price
                if (
                    unrealized >= self.profit_target
                    or unrealized <= -self.stop_loss
                    or bars_held >= self.max_holding_bars
                ):
                    position, entry_price, bars_held = 0, 0.0, 0

            signal[t] = position

        return pd.Series(signal, index=df.index, dtype="int64")

    def indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        close = df["close"]
        ma = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        return {
            f"MA ({self.lookback})": ma,
            f"Bande +{self.entry_z}σ": ma + self.entry_z * std,
            f"Bande -{self.entry_z}σ": ma - self.entry_z * std,
        }
