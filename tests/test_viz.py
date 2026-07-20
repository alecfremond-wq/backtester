import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.figure import Figure

from backtester.engine.portfolio import Trade
from backtester.viz.plots import plot_drawdown, plot_equity_curve, plot_trade_distribution


def make_equity():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.Series([1000, 1050, 980, 1010, 1100], index=dates)


def make_trades():
    trades = []
    for pnl in (50.0, -20.0, 30.0):
        t = Trade(side=1, size=1.0, entry_date=pd.Timestamp("2024-01-01"), entry_price=100.0)
        t.pnl = pnl
        trades.append(t)
    return trades


def test_plot_equity_curve_returns_figure():
    fig = plot_equity_curve(make_equity())
    assert isinstance(fig, Figure)


def test_plot_drawdown_returns_figure():
    fig = plot_drawdown(make_equity())
    assert isinstance(fig, Figure)


def test_plot_trade_distribution_returns_figure():
    fig = plot_trade_distribution(make_trades())
    assert isinstance(fig, Figure)
