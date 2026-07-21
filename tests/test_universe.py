from backtester.data.universe import LARGE_CAP_UNIVERSE, UNIVERSES


def test_no_duplicate_tickers():
    assert len(LARGE_CAP_UNIVERSE) == len(set(LARGE_CAP_UNIVERSE))


def test_reasonable_breadth():
    assert len(LARGE_CAP_UNIVERSE) >= 75


def test_tickers_are_uppercase_strings():
    assert all(isinstance(t, str) and t == t.upper() for t in LARGE_CAP_UNIVERSE)


def test_registered_in_universes_dict():
    assert UNIVERSES["large_cap"] is LARGE_CAP_UNIVERSE
