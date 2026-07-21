import pandas as pd

from walkforward import make_fold_boundaries, slice_df


def test_make_fold_boundaries_count_and_order():
    boundaries = make_fold_boundaries("2010-01-01", "2024-01-01", 5)
    assert len(boundaries) == 6
    assert all(boundaries[i] < boundaries[i + 1] for i in range(len(boundaries) - 1))
    assert boundaries[0] == pd.Timestamp("2010-01-01")
    assert boundaries[-1] == pd.Timestamp("2024-01-01")


def test_slice_df_respects_half_open_interval():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10, freq="D"), "close": range(10)})
    start, end = pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")

    sliced = slice_df(df, start, end)

    assert list(sliced["date"]) == list(pd.date_range("2020-01-03", "2020-01-05"))
