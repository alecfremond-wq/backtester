from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.cross_sectional.signals import RankFn
from backtester.engine.costs import CostModel
from backtester.engine.portfolio import Trade


@dataclass
class CrossSectionalResult:
    equity_curve: pd.Series
    trades: list[Trade]


def run_cross_sectional_backtest(
    dfs: dict[str, pd.DataFrame],
    rank_fn: RankFn,
    top_n: int = 3,
    bottom_n: int = 3,
    rebalance_every: int = 21,
    initial_capital: float = 100_000.0,
    long_only: bool = False,
    costs: CostModel | None = None,
) -> CrossSectionalResult:
    """Rank tickers by rank_fn, go long the top_n / short the bottom_n, equal-weighted.

    Rebalances every `rebalance_every` bars. The ranking at the close of bar
    t-1 decides the portfolio executed at bar t's open (with slippage) --
    same one-bar decision-to-execution lag as the single-ticker engine,
    just applied to a basket instead of one instrument.
    """
    costs = costs or CostModel()
    tickers = list(dfs)

    common_dates = None
    for df in dfs.values():
        idx = pd.DatetimeIndex(df["date"])
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    n = len(common_dates)

    open_px, close_px, score = {}, {}, {}
    for ticker, df in dfs.items():
        indexed = df.set_index("date")
        open_px[ticker] = indexed["open"].reindex(common_dates)
        close_px[ticker] = indexed["close"].reindex(common_dates)
        score[ticker] = rank_fn(df).set_axis(df["date"]).reindex(common_dates)

    equity = initial_capital
    dates_out: list[pd.Timestamp] = []
    equities_out: list[float] = []
    trades: list[Trade] = []
    positions: dict[str, Trade] = {}

    for i in range(n):
        date = common_dates[i]

        if i > 0:
            prev_date = common_dates[i - 1]
            for ticker, trade in positions.items():
                prev_close, close = close_px[ticker].loc[prev_date], close_px[ticker].loc[date]
                if pd.notna(prev_close) and pd.notna(close):
                    equity += trade.side * trade.size * (close - prev_close)

        is_decision_bar = i > 0 and (i - 1) % rebalance_every == 0
        if is_decision_bar:
            for ticker, trade in list(positions.items()):
                open_price = open_px[ticker].loc[date]
                if pd.isna(open_price):
                    continue
                exit_price = costs.execution_price(open_price, is_buy=trade.side == -1)
                fee = costs.fee(trade.size * exit_price)
                prev_close = close_px[ticker].loc[common_dates[i - 1]]
                equity += trade.side * trade.size * (exit_price - prev_close) - fee
                trade.close(date, exit_price, fee)
                trades.append(trade)
            positions = {}

            ranking_date = common_dates[i - 1]
            ranked = sorted(
                ((t, score[t].loc[ranking_date]) for t in tickers if pd.notna(score[t].loc[ranking_date])),
                key=lambda kv: kv[1],
                reverse=True,
            )
            longs = [t for t, _ in ranked[:top_n]]
            shorts = [] if long_only else [t for t, _ in ranked[-bottom_n:] if t not in longs]

            selected = [(t, 1) for t in longs] + [(t, -1) for t in shorts]
            if selected:
                weight = 1.0 / len(selected)
                for ticker, side in selected:
                    open_price = open_px[ticker].loc[date]
                    if pd.isna(open_price):
                        continue
                    entry_price = costs.execution_price(open_price, is_buy=side == 1)
                    size = (weight * equity) / entry_price
                    fee = costs.fee(size * entry_price)
                    equity -= fee
                    trade = Trade(
                        side=side, size=size, entry_date=date, entry_price=entry_price,
                        costs=fee, ticker=ticker,
                    )
                    close = close_px[ticker].loc[date]
                    if pd.notna(close):
                        equity += side * size * (close - entry_price)
                    positions[ticker] = trade

        dates_out.append(date)
        equities_out.append(equity)

    last_date = common_dates[-1]
    for ticker, trade in positions.items():
        last_close = close_px[ticker].loc[last_date]
        trade.close(last_date, last_close, 0.0)
        trades.append(trade)

    equity_curve = pd.Series(equities_out, index=pd.to_datetime(dates_out), name="equity")
    return CrossSectionalResult(equity_curve=equity_curve, trades=trades)
