from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return target position per bar, indexed like df.

        Values are position direction/size decided at the close of each bar
        (e.g. -1/0/1 for short/flat/long, or a continuous size). Must only use
        data up to and including that bar — the backtest engine shifts this
        series by one bar before applying it, so signals here represent
        decisions, not positions actually held during that bar.
        """
        raise NotImplementedError
