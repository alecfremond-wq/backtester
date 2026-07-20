from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def load(ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no raw data for {ticker}, run ingest() first")

    conditions = []
    params: list[str] = [str(path)]
    if start:
        conditions.append("date >= ?")
        params.append(start)
    if end:
        conditions.append("date <= ?")
        params.append(end)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM read_parquet(?) {where} ORDER BY date"
    return duckdb.execute(query, params).df()
