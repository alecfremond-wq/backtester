import numpy as np
import pandas as pd
import pytest

from backtester.engine.portfolio import Trade
from backtester.metrics.performance import annualized_sharpe, compute_performance, trade_sharpe


def test_annualized_sharpe_zero_std_is_zero():
    assert annualized_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0


def test_annualized_sharpe_positive_for_positive_drift():
    returns = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    assert annualized_sharpe(returns) > 0


def test_trade_sharpe_unannualized():
    returns = np.array([0.05, -0.02, 0.03])
    result = trade_sharpe(returns)
    expected = returns.mean() / returns.std(ddof=1)
    assert result == pytest.approx(expected)


def test_drawdown_known_equity_curve():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0, 130.0])
    trades = []
    report = compute_performance(equity, trades)
    assert report.max_drawdown_pct == pytest.approx(-0.25)
    assert report.max_drawdown_duration_days == 2


def make_trade(pnl: float, entry_price: float = 100.0, size: float = 1.0) -> Trade:
    t = Trade(side=1, size=size, entry_date=pd.Timestamp("2024-01-01"), entry_price=entry_price)
    t.exit_date = pd.Timestamp("2024-01-05")
    t.exit_price = entry_price + pnl / size
    t.pnl = pnl
    return t


def test_win_rate_profit_factor_reward_risk():
    equity = pd.Series([1000.0, 1050.0, 1150.0, 1100.0])
    trades = [make_trade(100.0), make_trade(100.0), make_trade(-50.0)]

    report = compute_performance(equity, trades)

    assert report.num_trades == 3
    assert report.win_rate == pytest.approx(2 / 3)
    assert report.avg_win == pytest.approx(100.0)
    assert report.avg_loss == pytest.approx(-50.0)
    assert report.reward_risk_ratio == pytest.approx(2.0)
    assert report.profit_factor == pytest.approx(200.0 / 50.0)


def test_no_trades_gives_neutral_stats():
    equity = pd.Series([1000.0, 1000.0])
    report = compute_performance(equity, [])
    assert report.num_trades == 0
    assert report.win_rate == 0.0
    assert report.sharpe_per_trade == 0.0
