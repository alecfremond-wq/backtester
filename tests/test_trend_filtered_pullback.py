import pandas as pd
import pytest

from backtester.strategy.examples.trend_filtered_pullback import TrendFilteredPullback


def make_df(closes):
    return pd.DataFrame({"close": closes}, dtype="float64")


def uptrend_with_dip():
    base = [50 + 2 * i for i in range(25)]  # steady uptrend, 25 bars
    base.append(base[-1] - 15)  # one-bar dip at index 25
    return base


def uptrend_with_spike():
    base = [50 + 2 * i for i in range(25)]
    base.append(base[-1] + 15)  # one-bar spike at index 25
    return base


def downtrend_with_spike():
    base = [98 - 2 * i for i in range(25)]  # steady downtrend, 25 bars
    base.append(base[-1] + 15)  # one-bar spike (bounce) at index 25
    return base


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        TrendFilteredPullback(trend_ma=1)
    with pytest.raises(ValueError):
        TrendFilteredPullback(lookback=1)
    with pytest.raises(ValueError):
        TrendFilteredPullback(profit_target=0)
    with pytest.raises(ValueError):
        TrendFilteredPullback(stop_loss=-0.01)
    with pytest.raises(ValueError):
        TrendFilteredPullback(max_holding_bars=0)


def test_flat_during_warmup():
    df = make_df([100] * 10)
    strat = TrendFilteredPullback(trend_ma=20, lookback=5)
    signal = strat.generate_signals(df)
    assert (signal == 0).all()


def test_dip_in_uptrend_triggers_long():
    df = make_df(uptrend_with_dip())
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5, max_holding_bars=10
    )
    signal = strat.generate_signals(df)
    assert signal.iloc[25] == 1


def test_spike_in_uptrend_does_not_enter():
    # a spike (not a dip) inside an uptrend matches neither the long nor the short condition
    df = make_df(uptrend_with_spike())
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5, max_holding_bars=10
    )
    signal = strat.generate_signals(df)
    assert signal.iloc[25] == 0


def test_spike_in_downtrend_triggers_short():
    df = make_df(downtrend_with_spike())
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5, max_holding_bars=10
    )
    signal = strat.generate_signals(df)
    assert signal.iloc[25] == -1


def test_long_only_suppresses_short_in_downtrend():
    df = make_df(downtrend_with_spike())
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5,
        max_holding_bars=10, long_only=True,
    )
    signal = strat.generate_signals(df)
    assert (signal >= 0).all()


def test_exits_on_profit_target():
    closes = uptrend_with_dip() + [95]  # bounce past +2% target the bar after entry
    df = make_df(closes)
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.02, stop_loss=0.5, max_holding_bars=10
    )
    signal = strat.generate_signals(df)
    assert signal.iloc[25] == 1
    assert signal.iloc[26] == 0


def test_exits_on_stop_loss():
    closes = uptrend_with_dip() + [70]  # falls further, past -2% stop
    df = make_df(closes)
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.02, max_holding_bars=10
    )
    signal = strat.generate_signals(df)
    assert signal.iloc[25] == 1
    assert signal.iloc[26] == 0


def test_forced_exit_after_max_holding_bars():
    dip_value = uptrend_with_dip()[-1]
    closes = uptrend_with_dip() + [dip_value] * 4  # flat after entry, neither target nor stop hit
    df = make_df(closes)
    strat = TrendFilteredPullback(
        trend_ma=20, lookback=5, entry_z=1.0, profit_target=0.5, stop_loss=0.5, max_holding_bars=3
    )
    signal = strat.generate_signals(df)
    assert signal.iloc[25] == 1
    assert (signal.iloc[26:28] == 1).all()
    assert signal.iloc[28] == 0


def test_indicators_returns_trend_and_short_ma():
    df = make_df(uptrend_with_dip())
    strat = TrendFilteredPullback(trend_ma=20, lookback=5)
    indicators = strat.indicators(df)
    assert set(indicators) == {"Tendance (20)", "MA courte (5)"}
