from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.engine.portfolio import Trade

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceReport:
    num_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    reward_risk_ratio: float
    profit_factor: float
    sharpe_daily: float
    sharpe_per_trade: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    total_return_pct: float


def annualized_sharpe(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR, risk_free_annual: float = 0.0
) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    excess = returns - risk_free_annual / periods_per_year
    return float(excess.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def trade_sharpe(trade_returns: np.ndarray) -> float:
    if len(trade_returns) < 2 or trade_returns.std(ddof=1) == 0:
        return 0.0
    return float(trade_returns.mean() / trade_returns.std(ddof=1))


def _trade_pct_returns(trades: list[Trade]) -> np.ndarray:
    returns = [t.pnl / (t.size * t.entry_price) for t in trades if t.pnl is not None]
    return np.array(returns)


def _drawdown(equity: pd.Series) -> tuple[float, int]:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    max_duration = 0
    current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0

    return float(drawdown.min()), max_duration


def compute_performance(
    equity_curve: pd.Series, trades: list[Trade], risk_free_annual: float = 0.0
) -> PerformanceReport:
    daily_returns = equity_curve.pct_change().dropna().to_numpy()
    trade_returns = _trade_pct_returns(trades)

    closed = [t for t in trades if t.pnl is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]

    win_rate = len(wins) / len(closed) if closed else 0.0
    avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    reward_risk_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float("inf")

    max_dd_pct, max_dd_duration = _drawdown(equity_curve)
    total_return_pct = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)

    return PerformanceReport(
        num_trades=len(closed),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        reward_risk_ratio=reward_risk_ratio,
        profit_factor=profit_factor,
        sharpe_daily=annualized_sharpe(daily_returns, risk_free_annual=risk_free_annual),
        sharpe_per_trade=trade_sharpe(trade_returns),
        max_drawdown_pct=max_dd_pct,
        max_drawdown_duration_days=max_dd_duration,
        total_return_pct=total_return_pct,
    )
