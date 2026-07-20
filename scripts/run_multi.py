#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtester.data import ingest, store  # noqa: E402
from backtester.engine.backtest import run_backtest  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_trade_stats  # noqa: E402
from backtester.strategy.registry import STRATEGIES, build_strategy  # noqa: E402
from run_backtest import parse_strategy_params  # noqa: E402

MIN_TRADES_FOR_SIGNIFICANCE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate trade-level stats for one strategy across multiple tickers."
    )
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy", default="ma_crossover", choices=list(STRATEGIES))
    parser.add_argument("--param", action="append", default=[], help="strategy param as key=value, repeatable")
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--pct-per-trade", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy_params = parse_strategy_params(args.param)
    costs = CostModel(slippage_bps=args.slippage_bps, fee_bps=args.fee_bps)

    all_trades = []
    print(f"{'ticker':<8}{'trades':>8}{'win_rate':>10}{'profit_factor':>15}")
    for ticker in args.tickers:
        if args.refresh_data:
            ingest.ingest(ticker, start=args.start, end=args.end)
        else:
            ingest.ensure_cached(ticker, start=args.start, end=args.end)
        df = store.load(ticker, start=args.start, end=args.end)

        strategy = build_strategy(args.strategy, strategy_params)
        result = run_backtest(
            df,
            strategy,
            initial_capital=args.initial_capital,
            pct_per_trade=args.pct_per_trade,
            costs=costs,
        )
        stats = compute_trade_stats(result.trades)
        all_trades.extend(result.trades)
        print(f"{ticker:<8}{stats.num_trades:>8}{stats.win_rate:>10.2%}{stats.profit_factor:>15.2f}")

    combined = compute_trade_stats(all_trades)
    print(f"\n=== combined across {len(args.tickers)} tickers ({args.strategy}) ===")
    for field, value in vars(combined).items():
        print(f"{field}: {value}")

    if combined.num_trades < MIN_TRADES_FOR_SIGNIFICANCE:
        print(f"\nwarning: {combined.num_trades} trades is below the ~{MIN_TRADES_FOR_SIGNIFICANCE} threshold "
              "usually wanted for statistical significance")

    print(
        "\nnote: each ticker is backtested independently with its own starting capital — "
        "this measures the strategy's edge across a broader trade sample, not a single combined account. "
        "Use run_backtest.py on one ticker for equity-curve-based prop firm validation."
    )


if __name__ == "__main__":
    main()
