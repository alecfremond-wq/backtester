from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

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
