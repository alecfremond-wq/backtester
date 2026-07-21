#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester.cross_sectional.engine import run_cross_sectional_backtest  # noqa: E402
from backtester.cross_sectional.signals import momentum_score  # noqa: E402
from backtester.data import ingest, store  # noqa: E402
from backtester.data.universe import UNIVERSES  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_performance  # noqa: E402
from backtester.propfirm.rules import PropFirmRules, validate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-sectional momentum long/short across a ticker basket.")
    parser.add_argument("tickers", nargs="*", help="explicit tickers, or omit and pass --universe")
    parser.add_argument("--universe", choices=list(UNIVERSES), default=None,
                         help="use a predefined ticker basket instead of listing tickers")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--momentum-window", type=int, default=126)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--bottom-n", type=int, default=3)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--daily-loss-pct", type=float, default=0.05)
    parser.add_argument("--max-total-loss-pct", type=float, default=0.10)
    parser.add_argument("--refresh-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = args.tickers if args.tickers else UNIVERSES.get(args.universe, [])
    if not tickers:
        raise SystemExit("provide tickers or --universe")

    dfs = {}
    skipped = []
    for ticker in tickers:
        try:
            if args.refresh_data:
                ingest.ingest(ticker, start=args.start, end=args.end)
            else:
                ingest.ensure_cached(ticker, start=args.start, end=args.end)
            dfs[ticker] = store.load(ticker, start=args.start, end=args.end)
        except Exception as exc:
            skipped.append(ticker)
            print(f"skipping {ticker}: {exc}")

    if skipped:
        print(f"skipped {len(skipped)}/{len(tickers)} tickers: {', '.join(skipped)}\n")

    costs = CostModel(slippage_bps=args.slippage_bps, fee_bps=args.fee_bps)
    result = run_cross_sectional_backtest(
        dfs,
        momentum_score(args.momentum_window),
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        rebalance_every=args.rebalance_every,
        initial_capital=args.initial_capital,
        long_only=args.long_only,
        costs=costs,
    )
    report = compute_performance(result.equity_curve, result.trades)

    print(
        f"=== cross-sectional momentum | {len(dfs)} tickers | "
        f"window={args.momentum_window} top={args.top_n} bottom={args.bottom_n} "
        f"rebalance={args.rebalance_every} | {args.start} -> {args.end or 'latest'} ==="
    )
    for field, value in vars(report).items():
        print(f"{field}: {value}")

    rules = PropFirmRules(
        initial_capital=args.initial_capital,
        daily_loss_pct=args.daily_loss_pct,
        max_total_loss_pct=args.max_total_loss_pct,
    )
    validation = validate(result.equity_curve, result.trades, rules)
    print(f"\nprop firm rules passed: {validation.passed}")
    if not validation.passed:
        print(f"first breach: {validation.first_breach_date}")
        for b in validation.breaches[:10]:
            print(f"  - {b.rule} {b.date.date()}: {b.detail}")


if __name__ == "__main__":
    main()
