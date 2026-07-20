import pandas as pd
import pytest

from backtester.engine.backtest import run_backtest
from backtester.engine.costs import CostModel
from backtester.strategy.base import Strategy


class FixedSignal(Strategy):
    def __init__(self, signal):
        self._signal = signal

    def generate_signals(self, df):
        return pd.Series(self._signal, index=df.index)


def make_df(opens, closes):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(opens), freq="D"),
            "open": opens,
            "close": closes,
        }
    )


def test_no_lookahead():
    # if the signal were applied on the same bar (bug), bar0 equity would already
    # jump from the open->close move; with the correct 1-bar shift it must not.
    df = make_df(opens=[100, 100, 100], closes=[150, 150, 150])
    result = run_backtest(df, FixedSignal([1, 1, 1]), initial_capital=1000)
    assert result.equity_curve.iloc[0] == 1000


def test_long_trade_pnl_matches_manual_calc():
    df = make_df(opens=[100, 110, 120], closes=[100, 110, 120])
    result = run_backtest(df, FixedSignal([1, 1, 1]), initial_capital=1000, pct_per_trade=1.0)

    trade = result.trades[0]
    expected_size = 1000 / 110  # entered at bar1's open with full capital
    assert trade.side == 1
    assert trade.entry_price == 110
    assert trade.size == pytest.approx(expected_size)
    assert trade.pnl == pytest.approx(expected_size * (120 - 110))
    assert result.equity_curve.iloc[-1] == pytest.approx(1000 + trade.pnl)


def test_reversal_closes_and_reopens_same_bar():
    df = make_df(opens=[100, 100, 100, 100], closes=[100, 100, 100, 100])
    result = run_backtest(df, FixedSignal([1, -1, -1, -1]), initial_capital=1000)
    # position: [0, 1, -1, -1] -> long opened bar1, reversed to short at bar2's open
    assert len(result.trades) == 2
    assert result.trades[0].side == 1
    assert result.trades[1].side == -1


def test_costs_reduce_pnl():
    df = make_df(opens=[100, 110, 120], closes=[100, 110, 120])
    zero_cost = run_backtest(df, FixedSignal([1, 1, 1]), initial_capital=1000)
    with_cost = run_backtest(
        df, FixedSignal([1, 1, 1]), initial_capital=1000, costs=CostModel(slippage_bps=10, fee_bps=5)
    )
    assert with_cost.equity_curve.iloc[-1] < zero_cost.equity_curve.iloc[-1]
