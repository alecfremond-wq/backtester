# backtester

Framework de backtesting réutilisable pour stratégies actions/indices, avec validation des règles prop firm.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Structure

- `src/backtester/data` — ingestion et stockage OHLCV (DuckDB/parquet)
- `src/backtester/strategy` — interface de stratégie + exemples (MA crossover, breakout)
- `src/backtester/engine` — moteur de simulation (portfolio, coûts, exécution sans look-ahead)
- `src/backtester/metrics` — métriques de performance (Sharpe, drawdown, profit factor, ...)
- `src/backtester/propfirm` — règles de validation prop firm (daily loss, max loss, consistency)
- `src/backtester/viz` — visualisation (equity curve, distribution des trades, drawdown)
