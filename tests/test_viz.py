import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.figure import Figure

from backtester.engine.portfolio import Trade
from backtester.viz.plots import plot_drawdown, plot_equity_curve, plot_price, plot_trade_distribution


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


def make_price_df(with_volume=True):
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    data = {
        "date": dates,
        "open": [100, 101, 102, 101, 103, 104],
        "high": [101, 102, 103, 102, 104, 105],
        "low": [99, 100, 101, 100, 102, 103],
        "close": [100.5, 101.5, 102.5, 101.5, 103.5, 104.5],
    }
    if with_volume:
        data["volume"] = [1000, 1200, 900, 1100, 1300, 1000]
    return pd.DataFrame(data)


def make_price_trades():
    long_trade = Trade(side=1, size=1.0, entry_date=pd.Timestamp("2024-01-02"), entry_price=101.5)
    long_trade.exit_date = pd.Timestamp("2024-01-05")
    long_trade.exit_price = 103.5
    long_trade.pnl = 2.0
    short_trade = Trade(side=-1, size=1.0, entry_date=pd.Timestamp("2024-01-03"), entry_price=102.5)
    return [long_trade, short_trade]


def test_plot_price_returns_figure_with_volume():
    fig = plot_price(make_price_df(with_volume=True))
    assert isinstance(fig, Figure)


def test_plot_price_without_volume_column():
    fig = plot_price(make_price_df(with_volume=False))
    assert isinstance(fig, Figure)


def test_plot_price_with_indicators_and_trades():
    df = make_price_df()
    indicators = {"MA rapide (2)": df["close"].rolling(2).mean()}
    fig = plot_price(df, indicators=indicators, trades=make_price_trades())
    assert isinstance(fig, Figure)
