import pandas as pd

from backtester.data import store


def test_load_filters_by_date(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "open": [1, 2, 3, 4, 5],
            "close": [1, 2, 3, 4, 5],
        }
    )
    df.to_parquet(tmp_path / "TEST.parquet", index=False)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)

    result = store.load("TEST", start="2023-01-02", end="2023-01-04")

    assert list(result["date"].dt.strftime("%Y-%m-%d")) == ["2023-01-02", "2023-01-03", "2023-01-04"]


def test_load_missing_ticker_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    try:
        store.load("NOPE")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
