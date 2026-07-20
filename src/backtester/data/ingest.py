from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from backtester.data import store

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def download(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"no data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
    df.index.name = "date"
    return df.reset_index()


def save_raw(df: pd.DataFrame, ticker: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    return path


def ingest(ticker: str, start: str, end: str | None = None, interval: str = "1d") -> Path:
    df = download(ticker, start, end, interval)
    return save_raw(df, ticker)


def ensure_cached(ticker: str, start: str, end: str | None = None) -> None:
    """Make sure the local parquet cache for `ticker` covers [start, end].

    Re-downloads (merging with whatever range is already cached) if the file
    is missing or doesn't cover the requested range, so callers can't silently
    get an empty/partial slice back from store.load().
    """
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        ingest(ticker, start=start, end=end)
        return

    cached = store.load(ticker)
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    if cached.empty:
        ingest(ticker, start=start, end=end)
        return

    cached_start, cached_end = cached["date"].min(), cached["date"].max()
    if requested_start < cached_start or requested_end > cached_end:
        refresh_start = min(requested_start, cached_start)
        ingest(ticker, start=refresh_start.strftime("%Y-%m-%d"))
