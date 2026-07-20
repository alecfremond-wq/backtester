from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.engine.portfolio import Trade


@dataclass
class PropFirmRules:
    initial_capital: float
    daily_loss_pct: float = 0.05
    max_total_loss_pct: float = 0.10
    max_single_trade_profit_pct: float | None = None  # e.g. 0.30 -> no trade > 30% of total profit


@dataclass
class RuleBreach:
    rule: str
    date: pd.Timestamp
    detail: str


@dataclass
class ValidationResult:
    passed: bool
    breaches: list[RuleBreach]
    first_breach_date: pd.Timestamp | None


def validate(equity_curve: pd.Series, trades: list[Trade], rules: PropFirmRules) -> ValidationResult:
    breaches: list[RuleBreach] = []
    prev_equity = rules.initial_capital

    for date, equity in equity_curve.items():
        daily_pnl_pct = (equity - prev_equity) / prev_equity
        if daily_pnl_pct < -rules.daily_loss_pct:
            breaches.append(
                RuleBreach(
                    rule="daily_loss",
                    date=date,
                    detail=f"daily loss {daily_pnl_pct:.2%} exceeds limit -{rules.daily_loss_pct:.2%}",
                )
            )

        total_loss_pct = (equity - rules.initial_capital) / rules.initial_capital
        if total_loss_pct < -rules.max_total_loss_pct:
            breaches.append(
                RuleBreach(
                    rule="max_total_loss",
                    date=date,
                    detail=f"total loss {total_loss_pct:.2%} exceeds limit -{rules.max_total_loss_pct:.2%}",
                )
            )

        prev_equity = equity

    if rules.max_single_trade_profit_pct is not None:
        closed = [t for t in trades if t.pnl is not None]
        total_profit = sum(t.pnl for t in closed if t.pnl > 0)
        if total_profit > 0:
            for t in closed:
                if t.pnl > 0 and t.pnl / total_profit > rules.max_single_trade_profit_pct:
                    share = t.pnl / total_profit
                    breaches.append(
                        RuleBreach(
                            rule="consistency",
                            date=t.exit_date,
                            detail=(
                                f"trade pnl {t.pnl:.2f} is {share:.2%} of total profit, "
                                f"exceeds limit {rules.max_single_trade_profit_pct:.2%}"
                            ),
                        )
                    )

    breaches.sort(key=lambda b: b.date)
    first_breach_date = breaches[0].date if breaches else None
    return ValidationResult(passed=len(breaches) == 0, breaches=breaches, first_breach_date=first_breach_date)
