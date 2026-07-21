from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.strategy.base import Strategy


class TrendFilteredPullback(Strategy):
    """Short-term pullback entries, filtered by a slow trend regime.

    Regime: price vs a slow moving average sets the allowed direction --
    only long above it (uptrend), only short below it (downtrend). Entry:
    a z-score dip/spike vs a short rolling mean, taken only *with* the
    regime (buy dips in an uptrend, sell spikes in a downtrend -- never
    against it). Exit: first of profit target, stop loss, or max holding
    bars, decoupled from the entry signal, same as MeanReversionScalper.
    """

    def __init__(
        self,
        trend_ma: int = 150,
        lookback: int = 10,
        entry_z: float = 1.0,
        profit_target: float = 0.02,
        stop_loss: float = 0.02,
        max_holding_bars: int = 5,
        long_only: bool = False,
    ):
        if trend_ma < 2:
            raise ValueError("trend_ma must be >= 2")
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        if profit_target <= 0 or stop_loss <= 0:
            raise ValueError("profit_target and stop_loss must be positive")
        if max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")
        self.trend_ma = trend_ma
        self.lookback = lookback
        self.entry_z = entry_z
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.max_holding_bars = max_holding_bars
        self.long_only = long_only

    def _regime(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        trend = close.rolling(self.trend_ma).mean()
        return np.sign(close - trend)

    def _zscore(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        ma = close.rolling(self.lookback).mean()
        std = close.rolling(self.lookback).std()
        return (close - ma) / std.replace(0, np.nan)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"].to_numpy()
        regime = self._regime(df).to_numpy()
        z = self._zscore(df).to_numpy()
        n = len(close)
        signal = np.zeros(n, dtype="int64")

        position = 0
        entry_price = 0.0
        bars_held = 0

        for t in range(n):
            if position == 0:
                if not np.isnan(z[t]) and not np.isnan(regime[t]):
                    if regime[t] > 0 and z[t] < -self.entry_z:
                        position, entry_price, bars_held = 1, close[t], 0
                    elif not self.long_only and regime[t] < 0 and z[t] > self.entry_z:
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
        return {
            f"Tendance ({self.trend_ma})": close.rolling(self.trend_ma).mean(),
            f"MA courte ({self.lookback})": close.rolling(self.lookback).mean(),
        }
