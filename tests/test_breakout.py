import pandas as pd

from backtester.strategy.examples.breakout import RangeBreakout


def make_df(closes):
    return pd.DataFrame({"high": closes, "low": closes, "close": closes}, dtype="float64")


def test_flat_during_warmup_and_range():
    df = make_df([100, 100, 100, 100])
    strat = RangeBreakout(window=3)
    signal = strat.generate_signals(df)
    assert (signal == 0).all()


def test_holds_long_until_opposite_breakout():
    closes = [100, 100, 100, 100, 105, 105, 105, 105, 90, 90, 90, 90]
    df = make_df(closes)
    strat = RangeBreakout(window=3)
    signal = strat.generate_signals(df)

    assert (signal.iloc[:4] == 0).all()
    assert (signal.iloc[4:8] == 1).all()
    assert (signal.iloc[8:12] == -1).all()


def test_long_only_never_shorts():
    closes = [100, 100, 100, 100, 105, 105, 105, 105, 90, 90, 90, 90]
    df = make_df(closes)
    strat = RangeBreakout(window=3, long_only=True)
    signal = strat.generate_signals(df)
    assert (signal >= 0).all()
