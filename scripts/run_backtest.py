#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester.data import ingest, store  # noqa: E402
from backtester.engine.backtest import run_backtest  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_performance  # noqa: E402
from backtester.propfirm.rules import PropFirmRules, validate  # noqa: E402
from backtester.strategy.registry import STRATEGIES, build_strategy  # noqa: E402
from backtester.viz.plots import plot_drawdown, plot_equity_curve, plot_trade_distribution  # noqa: E402


def parse_strategy_params(pairs: list[str]) -> dict:
    params: dict = {}
    for pair in pairs:
        key, value = pair.split("=", 1)
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        params[key] = value
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-ticker backtest end-to-end.")
    parser.add_argument("ticker")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGIES))
    parser.add_argument("--param", action="append", default=[], help="strategy param as key=value, repeatable")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--pct-per-trade", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--daily-loss-pct", type=float, default=0.05)
    parser.add_argument("--max-total-loss-pct", type=float, default=0.10)
    parser.add_argument("--max-single-trade-profit-pct", type=float, default=None)
    parser.add_argument("--refresh-data", action="store_true", help="re-download data even if cached locally")
    parser.add_argument("--out-dir", default="reports/last_run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path = store.RAW_DIR / f"{args.ticker}.parquet"
    if args.refresh_data or not raw_path.exists():
        ingest.ingest(args.ticker, start=args.start, end=args.end)
    df = store.load(args.ticker, start=args.start, end=args.end)

    strategy = build_strategy(args.strategy, parse_strategy_params(args.param))
    costs = CostModel(slippage_bps=args.slippage_bps, fee_bps=args.fee_bps)
    result = run_backtest(
        df,
        strategy,
        initial_capital=args.initial_capital,
        pct_per_trade=args.pct_per_trade,
        costs=costs,
    )
    report = compute_performance(result.equity_curve, result.trades)

    rules = PropFirmRules(
        initial_capital=args.initial_capital,
        daily_loss_pct=args.daily_loss_pct,
        max_total_loss_pct=args.max_total_loss_pct,
        max_single_trade_profit_pct=args.max_single_trade_profit_pct,
    )
    validation = validate(result.equity_curve, result.trades, rules)

    print(f"=== {args.ticker} | {args.strategy} | {args.start} -> {args.end or 'latest'} ===")
    for field, value in vars(report).items():
        print(f"{field}: {value}")

    print(f"\nprop firm rules passed: {validation.passed}")
    if not validation.passed:
        print(f"first breach: {validation.first_breach_date}")
        for b in validation.breaches[:10]:
            print(f"  - {b.rule} {b.date.date()}: {b.detail}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_equity_curve(result.equity_curve).savefig(out_dir / "equity.png")
    plot_drawdown(result.equity_curve).savefig(out_dir / "drawdown.png")
    plot_trade_distribution(result.trades).savefig(out_dir / "trades.png")
    print(f"\nplots saved to {out_dir}")


if __name__ == "__main__":
    main()
