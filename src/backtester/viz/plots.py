from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from backtester.engine.portfolio import Trade


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
