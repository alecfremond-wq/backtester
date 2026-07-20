import pandas as pd
import pytest

from backtester.engine.portfolio import Trade
from backtester.propfirm.rules import PropFirmRules, validate


def make_equity(values):
    dates = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=dates)


def test_passes_when_within_bounds():
    equity = make_equity([1000, 990, 985, 995, 1010])
    rules = PropFirmRules(initial_capital=1000, daily_loss_pct=0.05, max_total_loss_pct=0.10)

    result = validate(equity, [], rules)

    assert result.passed
    assert result.breaches == []
    assert result.first_breach_date is None


def test_detects_daily_loss_breach():
    equity = make_equity([1000, 940, 930])  # day2 drop of 6% > 5% limit
    rules = PropFirmRules(initial_capital=1000, daily_loss_pct=0.05, max_total_loss_pct=0.50)

    result = validate(equity, [], rules)

    assert not result.passed
    assert any(b.rule == "daily_loss" for b in result.breaches)
    assert result.first_breach_date == equity.index[1]


def test_detects_max_total_loss_breach():
    equity = make_equity([1000, 970, 940, 900, 880])  # ends 12% below initial capital
    rules = PropFirmRules(initial_capital=1000, daily_loss_pct=0.50, max_total_loss_pct=0.10)

    result = validate(equity, [], rules)

    assert not result.passed
    assert any(b.rule == "max_total_loss" for b in result.breaches)


def make_trade(pnl: float, exit_offset_days: int) -> Trade:
    t = Trade(
        side=1,
        size=1.0,
        entry_date=pd.Timestamp("2024-01-01"),
        entry_price=100.0,
    )
    t.exit_date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=exit_offset_days)
    t.exit_price = 100.0 + pnl
    t.pnl = pnl
    return t


def test_detects_consistency_breach():
    equity = make_equity([1000, 1000, 1000])
    trades = [make_trade(500, 1), make_trade(50, 2), make_trade(50, 3)]  # one trade = 500/600 = 83% of profit
    rules = PropFirmRules(
        initial_capital=1000, daily_loss_pct=0.99, max_total_loss_pct=0.99, max_single_trade_profit_pct=0.30
    )

    result = validate(equity, trades, rules)

    assert not result.passed
    assert any(b.rule == "consistency" for b in result.breaches)


def test_consistency_rule_ignored_when_not_set():
    equity = make_equity([1000, 1000, 1000])
    trades = [make_trade(500, 1), make_trade(50, 2)]
    rules = PropFirmRules(initial_capital=1000, daily_loss_pct=0.99, max_total_loss_pct=0.99)

    result = validate(equity, trades, rules)

    assert result.passed
