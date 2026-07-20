import pandas as pd

from backtester.data import ingest, store


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


def seed_cache(tmp_path, ticker, start, end):
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates, "open": range(len(dates)), "close": range(len(dates))})
    df.to_parquet(tmp_path / f"{ticker}.parquet", index=False)


def test_ensure_cached_downloads_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(
        ingest, "ingest", lambda ticker, start, end=None, interval="1d": calls.append((ticker, start, end))
    )

    ingest.ensure_cached("AAPL", start="2020-01-01", end="2020-01-10")

    assert calls == [("AAPL", "2020-01-01", "2020-01-10")]


def test_ensure_cached_skips_when_fully_covered(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    seed_cache(tmp_path, "AAPL", "2020-01-01", "2020-06-01")
    calls = []
    monkeypatch.setattr(ingest, "ingest", lambda *a, **k: calls.append((a, k)))

    ingest.ensure_cached("AAPL", start="2020-02-01", end="2020-03-01")

    assert calls == []


def test_ensure_cached_refetches_when_range_not_covered(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store, "RAW_DIR", tmp_path)
    seed_cache(tmp_path, "AAPL", "2023-01-01", "2023-06-01")
    calls = []
    monkeypatch.setattr(
        ingest, "ingest", lambda ticker, start, end=None, interval="1d": calls.append((ticker, start, end))
    )

    ingest.ensure_cached("AAPL", start="2018-01-01", end="2023-06-01")

    assert len(calls) == 1
    ticker, start, end = calls[0]
    assert ticker == "AAPL"
    assert start == "2018-01-01"
    assert end is None
