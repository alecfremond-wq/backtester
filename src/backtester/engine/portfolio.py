from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Trade:
    side: int  # 1 = long, -1 = short
    size: float
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None = None
    exit_price: float | None = None
    costs: float = 0.0
    pnl: float | None = None

    def close(self, date: pd.Timestamp, price: float, exit_cost: float) -> None:
        self.exit_date = date
        self.exit_price = price
        self.costs += exit_cost
        gross = self.side * self.size * (price - self.entry_price)
        self.pnl = gross - self.costs
