from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.engine.costs import CostModel
from backtester.engine.portfolio import Trade
from backtester.strategy.base import Strategy


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade]


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    pct_per_trade: float = 1.0,
    costs: CostModel | None = None,
) -> BacktestResult:
    costs = costs or CostModel()

    signal = strategy.generate_signals(df)
    # decision made at close of bar t is only actionable starting bar t+1
    position = signal.shift(1).fillna(0).astype(int)

    equity = initial_capital
    dates: list[pd.Timestamp] = []
    equities: list[float] = []
    trades: list[Trade] = []
    open_trade: Trade | None = None
    prev_close: float | None = None

    for i in range(len(df)):
        date = df["date"].iloc[i]
        o, c = df["open"].iloc[i], df["close"].iloc[i]
        target = int(position.iloc[i])
        current_side = open_trade.side if open_trade is not None else 0

        if target != current_side:
            if open_trade is not None:
                exit_price = costs.execution_price(o, is_buy=open_trade.side == -1)
                exit_cost = costs.fee(open_trade.size * exit_price)
                equity += open_trade.side * open_trade.size * (exit_price - prev_close) - exit_cost
                open_trade.close(date, exit_price, exit_cost)
                trades.append(open_trade)
                open_trade = None

            if target != 0:
                entry_price = costs.execution_price(o, is_buy=target == 1)
                size = (pct_per_trade * equity) / entry_price
                entry_cost = costs.fee(size * entry_price)
                equity -= entry_cost
                open_trade = Trade(side=target, size=size, entry_date=date, entry_price=entry_price)
                equity += target * size * (c - entry_price)
        elif open_trade is not None:
            equity += open_trade.side * open_trade.size * (c - prev_close)

        dates.append(date)
        equities.append(equity)
        prev_close = c

    if open_trade is not None:
        open_trade.close(df["date"].iloc[-1], df["close"].iloc[-1], 0.0)
        trades.append(open_trade)

    equity_curve = pd.Series(equities, index=pd.to_datetime(dates), name="equity")
    return BacktestResult(equity_curve=equity_curve, trades=trades)
