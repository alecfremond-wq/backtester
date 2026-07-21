import pandas as pd
import pytest

from backtester.strategy.examples.mean_reversion import MeanReversionScalper


def make_df(closes):
    return pd.DataFrame({"close": closes}, dtype="float64")


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        MeanReversionScalper(lookback=1)
    with pytest.raises(ValueError):
        MeanReversionScalper(profit_target=0)
    with pytest.raises(ValueError):
        MeanReversionScalper(stop_loss=-0.01)
    with pytest.raises(ValueError):
        MeanReversionScalper(max_holding_bars=0)


def test_flat_during_warmup():
    df = make_df([100] * 5)
    strat = MeanReversionScalper(lookback=5)
    signal = strat.generate_signals(df)
    assert (signal.iloc[:4] == 0).all()


def test_sharp_dip_triggers_long_entry():
    # flat around 100 for the lookback window, then a sharp one-bar drop
    closes = [100, 100, 100, 100, 100, 90]
    df = make_df(closes)
    strat = MeanReversionScalper(lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5, max_holding_bars=10)
    signal = strat.generate_signals(df)
    assert signal.iloc[5] == 1


def test_exits_on_profit_target():
    # enter long on the dip at t=5 (price 90), then bounce past +2% target at t=6 (price 92)
    closes = [100, 100, 100, 100, 100, 90, 93]
    df = make_df(closes)
    strat = MeanReversionScalper(lookback=5, entry_z=1.0, profit_target=0.02, stop_loss=0.5, max_holding_bars=10)
    signal = strat.generate_signals(df)
    assert signal.iloc[5] == 1
    assert signal.iloc[6] == 0


def test_exits_on_stop_loss():
    closes = [100, 100, 100, 100, 100, 90, 87]
    df = make_df(closes)
    strat = MeanReversionScalper(lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.02, max_holding_bars=10)
    signal = strat.generate_signals(df)
    assert signal.iloc[5] == 1
    assert signal.iloc[6] == 0


def test_forced_exit_after_max_holding_bars():
    # price stays flat at 90 after the dip -- neither target nor stop trigger
    closes = [100, 100, 100, 100, 100, 90, 90, 90, 90]
    df = make_df(closes)
    strat = MeanReversionScalper(lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5, max_holding_bars=3)
    signal = strat.generate_signals(df)
    assert signal.iloc[5] == 1
    assert (signal.iloc[6:8] == 1).all()
    assert signal.iloc[8] == 0


def test_long_only_never_shorts():
    closes = [100, 100, 100, 100, 100, 112, 112, 112, 112, 112]
    df = make_df(closes)
    strat = MeanReversionScalper(lookback=5, entry_z=1.0, long_only=True)
    signal = strat.generate_signals(df)
    assert (signal >= 0).all()


def test_indicators_returns_bands():
    df = make_df([100, 101, 102, 103, 104, 105])
    strat = MeanReversionScalper(lookback=5, entry_z=1.0)
    indicators = strat.indicators(df)
    assert set(indicators) == {"MA (5)", "Bande +1.0σ", "Bande -1.0σ"}
