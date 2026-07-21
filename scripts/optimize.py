#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester.data import ingest, store  # noqa: E402
from backtester.engine.backtest import run_backtest  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_performance, compute_trade_stats  # noqa: E402
from backtester.propfirm.rules import PropFirmRules, validate  # noqa: E402
from backtester.strategy.registry import build_strategy  # noqa: E402

MIN_TRADES_FOR_SIGNIFICANCE = 100

PARAM_GRIDS = {
    "ma_crossover": [
        {"fast": f, "slow": s}
        for f in (5, 10, 15, 20, 30, 50)
        for s in (20, 50, 75, 100, 150, 200)
        if f < s
    ],
    "breakout": [{"window": w} for w in (10, 15, 20, 30, 40, 60, 80, 100, 120)],
    "ml_classifier": [
        {"horizon": h, "retrain_every": r, "prob_threshold": p}
        for h in (5, 10, 20)
        for r in (63, 126)
        for p in (0.55, 0.60)
    ],
    "mean_reversion": [
        {"lookback": lb, "entry_z": z, "profit_target": pt, "stop_loss": pt, "max_holding_bars": 5}
        for lb in (5, 10, 20)
        for z in (1.0, 1.5, 2.0)
        for pt in (0.01, 0.02, 0.03)
    ],
    "trend_pullback": [
        {
            "trend_ma": tm, "lookback": lb, "entry_z": z,
            "profit_target": pt, "stop_loss": pt, "max_holding_bars": 5,
        }
        for tm in (100, 150, 200)
        for lb in (5, 10, 20)
        for z in (1.0, 1.5)
        for pt in (0.02, 0.03)
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-search strategy parameters across a ticker basket to look for real edge."
    )
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", default=None)
    parser.add_argument("--strategy", required=True, choices=list(PARAM_GRIDS))
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--pct-per-trade", type=float, default=1.0,
                         help="sizing used during the sweep itself — irrelevant to profit_factor/"
                              "win_rate/sharpe_per_trade, which are computed from pct returns")
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--verify-pct-per-trade", type=float, default=0.15,
                         help="realistic sizing used to verify the winning combo on one account")
    parser.add_argument("--daily-loss-pct", type=float, default=0.05)
    parser.add_argument("--max-total-loss-pct", type=float, default=0.10)
    return parser.parse_args()


def load_tickers(tickers: list[str], start: str, end: str | None) -> dict:
    dfs = {}
    for ticker in tickers:
        ingest.ensure_cached(ticker, start=start, end=end)
        dfs[ticker] = store.load(ticker, start=start, end=end)
    return dfs


def sweep(strategy_name: str, dfs: dict, initial_capital: float, pct_per_trade: float, costs: CostModel):
    results = []
    grid = PARAM_GRIDS[strategy_name]
    for i, params in enumerate(grid, 1):
        all_trades = []
        for df in dfs.values():
            strategy = build_strategy(strategy_name, params)
            result = run_backtest(
                df, strategy, initial_capital=initial_capital, pct_per_trade=pct_per_trade, costs=costs
            )
            all_trades.extend(result.trades)
        stats = compute_trade_stats(all_trades)
        results.append((params, stats))
        print(f"[{i}/{len(grid)}] {params} -> trades={stats.num_trades} pf={stats.profit_factor:.2f}")
    return results


def main() -> None:
    args = parse_args()
    costs = CostModel(slippage_bps=args.slippage_bps, fee_bps=args.fee_bps)
    dfs = load_tickers(args.tickers, args.start, args.end)

    results = sweep(args.strategy, dfs, args.initial_capital, args.pct_per_trade, costs)

    significant = [(p, s) for p, s in results if s.num_trades >= MIN_TRADES_FOR_SIGNIFICANCE]
    ranked = sorted(significant, key=lambda ps: ps[1].profit_factor, reverse=True)

    print(f"\n=== top {args.top} combos ({args.strategy}), min {MIN_TRADES_FOR_SIGNIFICANCE} trades combinés ===")
    print(f"{'params':<28}{'trades':>8}{'win_rate':>10}{'profit_factor':>15}{'sharpe_trade':>14}")
    for params, stats in ranked[: args.top]:
        print(
            f"{str(params):<28}{stats.num_trades:>8}{stats.win_rate:>10.2%}"
            f"{stats.profit_factor:>15.2f}{stats.sharpe_per_trade:>14.3f}"
        )

    if not ranked:
        print(
            f"\nAucune combinaison n'atteint {MIN_TRADES_FOR_SIGNIFICANCE} trades combinés sur ce panier — "
            "élargir la période ou le nombre de tickers avant de conclure quoi que ce soit."
        )
        return

    best_params, best_stats = ranked[0]
    verify_ticker = args.tickers[0]
    print(
        f"\n=== vérification de la meilleure combinaison {best_params} sur {verify_ticker}, "
        f"sizing réaliste {args.verify_pct_per_trade:.0%} ==="
    )
    strategy = build_strategy(args.strategy, best_params)
    result = run_backtest(
        dfs[verify_ticker], strategy, initial_capital=args.initial_capital,
        pct_per_trade=args.verify_pct_per_trade, costs=costs,
    )
    report = compute_performance(result.equity_curve, result.trades)
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


if __name__ == "__main__":
    main()
