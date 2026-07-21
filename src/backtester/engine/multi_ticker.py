from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.engine.backtest import run_backtest
from backtester.engine.costs import CostModel
from backtester.engine.portfolio import Trade
from backtester.strategy.base import Strategy


@dataclass
class MultiTickerResult:
    equity_curve: pd.Series
    trades: list[Trade]
    per_ticker_equity: dict[str, pd.Series]


def run_multi_ticker_backtest(
    dfs: dict[str, pd.DataFrame],
    strategy: Strategy,
    initial_capital: float = 100_000.0,
    pct_per_trade: float = 1.0,
    costs: CostModel | None = None,
) -> MultiTickerResult:
    """Run the same strategy independently on each ticker, capital split equally.

    Each ticker gets its own slice of capital (initial_capital / n) and its
    own independent single-ticker backtest -- there's no cross-ticker
    signal here, just diversification of the same edge across names so no
    single ticker's idiosyncratic outcome dominates the result.

    Tickers with data starting later than the combined date range have
    their equity held flat at their allocated capital until they start
    trading (that slice sits idle, not invested, rather than back-filled
    from future values).
    """
    if not dfs:
        raise ValueError("dfs must contain at least one ticker")
    capital_per_ticker = initial_capital / len(dfs)

    all_trades: list[Trade] = []
    equity_curves: dict[str, pd.Series] = {}
    for ticker, df in dfs.items():
        result = run_backtest(
            df, strategy, initial_capital=capital_per_ticker, pct_per_trade=pct_per_trade, costs=costs
        )
        for trade in result.trades:
            trade.ticker = ticker
        all_trades.extend(result.trades)
        equity_curves[ticker] = result.equity_curve

    full_index = None
    for equity in equity_curves.values():
        full_index = equity.index if full_index is None else full_index.union(equity.index)
    full_index = full_index.sort_values()

    aligned = [
        equity.reindex(full_index).ffill().fillna(capital_per_ticker) for equity in equity_curves.values()
    ]
    combined = sum(aligned)
    combined.name = "equity"

    return MultiTickerResult(equity_curve=combined, trades=all_trades, per_ticker_equity=equity_curves)
