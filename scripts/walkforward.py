#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from backtester.data import ingest, store  # noqa: E402
from backtester.engine.backtest import run_backtest  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_trade_stats  # noqa: E402
from backtester.strategy.registry import build_strategy  # noqa: E402
from optimize import PARAM_GRIDS  # noqa: E402

MIN_TRADES_PER_FOLD = 25


def make_fold_boundaries(start: str, end: str, n_segments: int) -> list[pd.Timestamp]:
    return list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end), periods=n_segments + 1))


def slice_df(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] < end)
    return df.loc[mask].reset_index(drop=True)


def trades_for(strategy_name: str, params: dict, dfs: dict, initial_capital: float,
                pct_per_trade: float, costs: CostModel) -> list:
    all_trades = []
    for df in dfs.values():
        if len(df) < 5:
            continue
        strategy = build_strategy(strategy_name, params)
        result = run_backtest(
            df, strategy, initial_capital=initial_capital, pct_per_trade=pct_per_trade, costs=costs
        )
        all_trades.extend(result.trades)
    return all_trades


def best_params_on(strategy_name: str, dfs: dict, initial_capital: float,
                    pct_per_trade: float, costs: CostModel):
    best = None
    for params in PARAM_GRIDS[strategy_name]:
        trades = trades_for(strategy_name, params, dfs, initial_capital, pct_per_trade, costs)
        stats = compute_trade_stats(trades)
        if stats.num_trades < MIN_TRADES_PER_FOLD:
            continue
        if best is None or stats.profit_factor > best[1].profit_factor:
            best = (params, stats)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation: optimize on an expanding train window, "
                     "test on the next unseen fold, repeat."
    )
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--strategy", required=True, choices=list(PARAM_GRIDS))
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--initial-capital", type=float, default=100_000.0)
    parser.add_argument("--pct-per-trade", type=float, default=1.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fee-bps", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    costs = CostModel(slippage_bps=args.slippage_bps, fee_bps=args.fee_bps)

    dfs = {}
    for ticker in args.tickers:
        ingest.ensure_cached(ticker, start=args.start, end=args.end)
        dfs[ticker] = store.load(ticker, start=args.start, end=args.end)

    boundaries = make_fold_boundaries(args.start, args.end, args.folds + 1)
    print("segments:", [b.date().isoformat() for b in boundaries])
    print(f"(seuil de {MIN_TRADES_PER_FOLD} trades minimum par fold pour retenir une combinaison "
          f"— plus bas que le seuil de {100} utilisé sur la période complète, car chaque fenêtre "
          f"d'entraînement est plus courte)\n")

    all_oos_trades = []
    for k in range(1, args.folds + 1):
        train_start, train_end = boundaries[0], boundaries[k]
        test_start, test_end = boundaries[k], boundaries[k + 1]

        train_slices = {t: slice_df(df, train_start, train_end) for t, df in dfs.items()}
        test_slices = {t: slice_df(df, test_start, test_end) for t, df in dfs.items()}

        best = best_params_on(args.strategy, train_slices, args.initial_capital, args.pct_per_trade, costs)
        if best is None:
            print(f"Fold {k}: train {train_start.date()}..{train_end.date()} "
                  "-> pas assez de trades pour optimiser, fold ignoré\n")
            continue
        best_params, train_stats = best

        oos_trades = trades_for(args.strategy, best_params, test_slices,
                                 args.initial_capital, args.pct_per_trade, costs)
        oos_stats = compute_trade_stats(oos_trades)
        all_oos_trades.extend(oos_trades)

        print(f"Fold {k}: train {train_start.date()}..{train_end.date()} -> best {best_params} "
              f"(train pf={train_stats.profit_factor:.2f}, {train_stats.num_trades} trades)")
        print(f"         test  {test_start.date()}..{test_end.date()}  -> "
              f"oos pf={oos_stats.profit_factor:.2f}, oos trades={oos_stats.num_trades}, "
              f"oos win_rate={oos_stats.win_rate:.2%}\n")

    if not all_oos_trades:
        print("Aucun fold exploitable — élargir la période ou réduire le nombre de folds.")
        return

    overall = compute_trade_stats(all_oos_trades)
    print("=== performance OUT-OF-SAMPLE agrégée sur tous les folds ===")
    for field, value in vars(overall).items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
