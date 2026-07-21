from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from backtester.engine.portfolio import Trade


def plot_price(
    df: pd.DataFrame,
    indicators: dict[str, pd.Series] | None = None,
    trades: list[Trade] | None = None,
) -> Figure:
    indicators = indicators or {}
    trades = trades or []
    has_volume = "volume" in df.columns
    dates = df["date"]

    if has_volume:
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_price = plt.subplots(figsize=(10, 4.5))
        ax_vol = None

    ax_price.plot(dates, df["close"], color="#1f77b4", linewidth=1.3, label="Clôture", zorder=3)
    ax_price.fill_between(dates, df["close"], df["close"].min(), color="#1f77b4", alpha=0.06)

    for label, series in indicators.items():
        ax_price.plot(dates, series, linewidth=1.0, linestyle="--", label=label, zorder=2)

    long_entries = [(t.entry_date, t.entry_price) for t in trades if t.side == 1]
    short_entries = [(t.entry_date, t.entry_price) for t in trades if t.side == -1]
    exits = [(t.exit_date, t.exit_price) for t in trades if t.exit_date is not None]

    if long_entries:
        xs, ys = zip(*long_entries)
        ax_price.scatter(xs, ys, marker="^", color="green", s=45, zorder=5, label="Entrée long")
    if short_entries:
        xs, ys = zip(*short_entries)
        ax_price.scatter(xs, ys, marker="v", color="crimson", s=45, zorder=5, label="Entrée short")
    if exits:
        xs, ys = zip(*exits)
        ax_price.scatter(xs, ys, marker="x", color="black", s=30, zorder=5, label="Sortie")

    ax_price.set_title("Évolution du prix")
    ax_price.set_ylabel("Prix")
    ax_price.legend(loc="upper left", fontsize=7.5, ncol=2)

    if ax_vol is not None:
        ax_vol.bar(dates, df["volume"], color="#94A3B8", width=1.0)
        ax_vol.set_ylabel("Volume")
        ax_vol.set_xlabel("Date")
    else:
        ax_price.set_xlabel("Date")

    fig.tight_layout()
    return fig


def plot_equity_curve(equity_curve: pd.Series) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve.index, equity_curve.values)
    ax.set_title("Equity curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    fig.tight_layout()
    return fig


def plot_drawdown(equity_curve: pd.Series) -> Figure:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color="crimson", alpha=0.4)
    ax.plot(drawdown.index, drawdown.values * 100, color="crimson")
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    fig.tight_layout()
    return fig


def plot_trade_distribution(trades: list[Trade], bins: int = 30) -> Figure:
    pnls = [t.pnl for t in trades if t.pnl is not None]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pnls, bins=bins, color="steelblue", edgecolor="black")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Trade PnL distribution")
    ax.set_xlabel("PnL")
    ax.set_ylabel("Number of trades")
    fig.tight_layout()
    return fig
