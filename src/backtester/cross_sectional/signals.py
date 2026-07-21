from __future__ import annotations

from typing import Callable

import pandas as pd

RankFn = Callable[[pd.DataFrame], pd.Series]


def momentum_score(window: int = 126) -> RankFn:
    """Trailing `window`-bar return, in %. Higher = stronger momentum.

    Classic cross-sectional momentum lookback (Jegadeesh & Titman use
    3-12 months). Computed per ticker using only that ticker's own past
    prices -- ranking across tickers happens in the engine, not here.
    """

    def _score(df: pd.DataFrame) -> pd.Series:
        return df["close"].pct_change(window)

    return _score
