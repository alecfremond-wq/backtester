import pandas as pd
import pytest

from backtester.engine.multi_ticker import run_multi_ticker_backtest
from backtester.strategy.base import Strategy


class AlwaysLong(Strategy):
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=df.index, dtype="int64")


def make_df(closes, start="2020-01-01"):
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(closes), freq="D"),
            "open": closes,
            "close": closes,
        }
    )


def test_capital_split_equally_across_tickers():
    dfs = {
        "A": make_df([100.0] * 10),
        "B": make_df([100.0] * 10),
    }
    result = run_multi_ticker_backtest(dfs, AlwaysLong(), initial_capital=10_000)
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert set(result.per_ticker_equity) == {"A", "B"}
    for equity in result.per_ticker_equity.values():
        assert equity.iloc[0] == pytest.approx(5_000)


def test_trades_are_tagged_with_ticker():
    dfs = {
        "UP": make_df([100.0, 105.0, 110.0, 115.0]),
        "DOWN": make_df([100.0, 95.0, 90.0, 85.0]),
    }
    result = run_multi_ticker_backtest(dfs, AlwaysLong(), initial_capital=10_000)
    tickers_seen = {t.ticker for t in result.trades}
    assert tickers_seen == {"UP", "DOWN"}


def test_no_tickers_raises():
    with pytest.raises(ValueError):
        run_multi_ticker_backtest({}, AlwaysLong())


def test_late_starting_ticker_holds_flat_until_its_data_begins():
    dfs = {
        "EARLY": make_df([100.0] * 10, start="2020-01-01"),
        "LATE": make_df([100.0] * 5, start="2020-01-06"),
    }
    result = run_multi_ticker_backtest(dfs, AlwaysLong(), initial_capital=10_000)
    # combined index spans the full union of dates, and starts at the full initial capital
    # even though LATE has no data yet on day 1 (its slice sits idle, not NaN/missing)
    assert len(result.equity_curve) == 10
    assert result.equity_curve.iloc[0] == pytest.approx(10_000)
    assert not result.equity_curve.isna().any()
