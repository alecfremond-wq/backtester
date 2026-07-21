import pandas as pd

from backtester.strategy.examples.ma_crossover import MovingAverageCrossover


def test_flat_during_warmup():
    df = pd.DataFrame({"close": range(1, 11)}, dtype="float64")
    strat = MovingAverageCrossover(fast=3, slow=5)
    signal = strat.generate_signals(df)
    assert (signal.iloc[:4] == 0).all()


def test_long_when_fast_above_slow():
    # strictly increasing prices -> fast MA ends up above slow MA
    df = pd.DataFrame({"close": range(1, 21)}, dtype="float64")
    strat = MovingAverageCrossover(fast=3, slow=5)
    signal = strat.generate_signals(df)
    assert signal.iloc[-1] == 1


def test_short_when_fast_below_slow():
    # strictly decreasing prices -> fast MA ends up below slow MA
    df = pd.DataFrame({"close": range(20, 0, -1)}, dtype="float64")
    strat = MovingAverageCrossover(fast=3, slow=5)
    signal = strat.generate_signals(df)
    assert signal.iloc[-1] == -1


def test_long_only_never_shorts():
    df = pd.DataFrame({"close": range(20, 0, -1)}, dtype="float64")
    strat = MovingAverageCrossover(fast=3, slow=5, long_only=True)
    signal = strat.generate_signals(df)
    assert (signal >= 0).all()


def test_fast_must_be_smaller_than_slow():
    try:
        MovingAverageCrossover(fast=10, slow=5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_indicators_returns_fast_and_slow_ma():
    df = pd.DataFrame({"close": range(1, 11)}, dtype="float64")
    strat = MovingAverageCrossover(fast=3, slow=5)
    indicators = strat.indicators(df)

    assert set(indicators) == {"MA rapide (3)", "MA lente (5)"}
    pd.testing.assert_series_equal(
        indicators["MA rapide (3)"], df["close"].rolling(3).mean(), check_names=False
    )
