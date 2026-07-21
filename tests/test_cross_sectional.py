import pandas as pd
import pytest

from backtester.cross_sectional.engine import run_cross_sectional_backtest
from backtester.cross_sectional.signals import momentum_score


def make_ticker_df(start_price, daily_pct, n=40):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    closes = [start_price * (1 + daily_pct) ** i for i in range(n)]
    return pd.DataFrame({"date": dates, "open": closes, "close": closes})


def test_momentum_score_matches_pct_change():
    df = make_ticker_df(100, 0.01, n=20)
    score = momentum_score(window=5)(df)
    expected = df["close"].pct_change(5)
    pd.testing.assert_series_equal(score, expected, check_names=False)


def test_selects_top_and_bottom_momentum_tickers():
    dfs = {
        "UP": make_ticker_df(100, 0.02),
        "MID": make_ticker_df(100, 0.0),
        "DOWN": make_ticker_df(100, -0.02),
    }
    result = run_cross_sectional_backtest(
        dfs, momentum_score(window=5), top_n=1, bottom_n=1, rebalance_every=10, initial_capital=10_000,
    )
    tickers_traded = {t.ticker for t in result.trades}
    assert tickers_traded == {"UP", "DOWN"}
    assert all(t.side == 1 for t in result.trades if t.ticker == "UP")
    assert all(t.side == -1 for t in result.trades if t.ticker == "DOWN")


def test_long_only_never_shorts():
    dfs = {
        "UP": make_ticker_df(100, 0.02),
        "MID": make_ticker_df(100, 0.0),
        "DOWN": make_ticker_df(100, -0.02),
    }
    result = run_cross_sectional_backtest(
        dfs, momentum_score(window=5), top_n=1, bottom_n=1, rebalance_every=10,
        initial_capital=10_000, long_only=True,
    )
    assert all(t.side == 1 for t in result.trades)
    assert all(t.ticker != "DOWN" for t in result.trades)


def test_equity_curve_length_matches_common_dates():
    dfs = {
        "A": make_ticker_df(100, 0.01, n=30),
        "B": make_ticker_df(100, -0.01, n=30),
    }
    result = run_cross_sectional_backtest(
        dfs, momentum_score(window=5), top_n=1, bottom_n=1, rebalance_every=10, initial_capital=10_000,
    )
    assert len(result.equity_curve) == 30


def test_no_positions_before_first_rebalance():
    dfs = {
        "A": make_ticker_df(100, 0.01, n=30),
        "B": make_ticker_df(100, -0.01, n=30),
    }
    result = run_cross_sectional_backtest(
        dfs, momentum_score(window=5), top_n=1, bottom_n=1, rebalance_every=10, initial_capital=10_000,
    )
    # equity flat until the momentum window is warmed up and the first rebalance fires
    assert (result.equity_curve.iloc[:6] == 10_000).all()
