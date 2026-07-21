from optimize import PARAM_GRIDS


def test_ma_crossover_grid_respects_fast_lt_slow():
    assert all(p["fast"] < p["slow"] for p in PARAM_GRIDS["ma_crossover"])
    assert len(PARAM_GRIDS["ma_crossover"]) > 0


def test_breakout_grid_has_positive_windows():
    assert all(p["window"] > 0 for p in PARAM_GRIDS["breakout"])
    assert len(PARAM_GRIDS["breakout"]) > 0
