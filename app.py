from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backtester.data import ingest, store  # noqa: E402
from backtester.engine.backtest import run_backtest  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_performance  # noqa: E402
from backtester.propfirm.rules import PropFirmRules, validate  # noqa: E402
from backtester.strategy.registry import STRATEGIES, build_strategy  # noqa: E402
from backtester.viz.plots import plot_drawdown, plot_equity_curve, plot_price, plot_trade_distribution  # noqa: E402

st.set_page_config(page_title="Backtester", layout="wide")

STRATEGY_PARAM_SPECS = {
    # défauts calibrés via scripts/walkforward.py — profit factor out-of-sample
    # > 1 sur les 4 folds testés (2010-2024, 10 tickers), contrairement aux
    # anciens défauts (fast=20/slow=50, window=20) qui montraient un edge faible
    "ma_crossover": [
        ("fast", 5, 100, 30, 1),
        ("slow", 10, 300, 150, 1),
    ],
    "breakout": [
        ("window", 5, 200, 100, 1),
    ],
    "ml_classifier": [
        ("horizon", 3, 40, 10, 1),
        ("retrain_every", 21, 252, 63, 1),
        ("prob_threshold", 0.50, 0.90, 0.55, 0.01),
    ],
    "mean_reversion": [
        ("lookback", 3, 60, 10, 1),
        ("entry_z", 0.5, 3.0, 1.0, 0.1),
        ("profit_target", 0.005, 0.10, 0.02, 0.005),
        ("stop_loss", 0.005, 0.10, 0.02, 0.005),
        ("max_holding_bars", 1, 30, 5, 1),
    ],
}


@st.cache_data(show_spinner="Téléchargement des données...")
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    ingest.ensure_cached(ticker, start=start, end=end)
    return store.load(ticker, start=start, end=end)


def strategy_param_widgets(strategy_name: str) -> dict:
    params = {}
    for name, lo, hi, default, step in STRATEGY_PARAM_SPECS[strategy_name]:
        params[name] = st.sidebar.slider(name, lo, hi, default, step, key=f"{strategy_name}_{name}")
    params["long_only"] = st.sidebar.checkbox("Long only", value=False, key=f"{strategy_name}_long_only")
    return params


st.title("Backtester — exploration de stratégies")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    col1, col2 = st.columns(2)
    date_bounds = (pd.Timestamp("1990-01-01"), pd.Timestamp.today())
    start = col1.date_input("Début", value=pd.Timestamp("2010-01-01"), min_value=date_bounds[0],
                             max_value=date_bounds[1]).isoformat()
    end = col2.date_input("Fin", value=pd.Timestamp("2024-01-01"), min_value=date_bounds[0],
                           max_value=date_bounds[1]).isoformat()

    st.subheader("Stratégie")
    strategy_name = st.selectbox("Stratégie", list(STRATEGIES))
    strategy_params = strategy_param_widgets(strategy_name)
    st.caption(
        "Défauts calibrés via scripts/walkforward.py : profit factor "
        "out-of-sample > 1 sur 4 folds (2010–2024, panier de 10 tickers). "
        "C'est un résultat de portefeuille diversifié — un seul ticker sur "
        "une autre période peut très bien ne pas le reproduire."
    )

    st.subheader("Capital & coûts")
    initial_capital = st.number_input("Capital initial", value=100_000.0, step=10_000.0)
    pct_per_trade = st.slider("% du capital par trade", 0.01, 1.0, 1.0, 0.01)
    slippage_bps = st.slider("Slippage (bps)", 0.0, 20.0, 2.0, 0.5)
    fee_bps = st.slider("Frais (bps)", 0.0, 20.0, 1.0, 0.5)

    st.subheader("Règles prop firm")
    daily_loss_pct = st.slider("Perte journalière max (%)", 1.0, 20.0, 5.0, 0.5) / 100
    max_total_loss_pct = st.slider("Perte totale max (%)", 1.0, 30.0, 10.0, 0.5) / 100
    enable_consistency = st.checkbox("Règle de consistency", value=False)
    max_single_trade_profit_pct = (
        st.slider("Profit max d'un trade (% du profit total)", 5.0, 90.0, 30.0, 5.0) / 100
        if enable_consistency
        else None
    )

    run_clicked = st.button("Lancer le backtest", type="primary")

if not run_clicked:
    st.info("Configure les paramètres dans la barre latérale puis clique sur 'Lancer le backtest'.")
    st.stop()

if not ticker:
    st.error("Ticker manquant.")
    st.stop()

try:
    df = load_data(ticker, start, end)
except Exception as exc:
    st.error(f"Impossible de charger les données pour {ticker} : {exc}")
    st.stop()

if df.empty:
    st.error(f"Aucune donnée pour {ticker} entre {start} et {end}.")
    st.stop()

strategy = build_strategy(strategy_name, strategy_params)
costs = CostModel(slippage_bps=slippage_bps, fee_bps=fee_bps)
result = run_backtest(df, strategy, initial_capital=initial_capital, pct_per_trade=pct_per_trade, costs=costs)
report = compute_performance(result.equity_curve, result.trades)

rules = PropFirmRules(
    initial_capital=initial_capital,
    daily_loss_pct=daily_loss_pct,
    max_total_loss_pct=max_total_loss_pct,
    max_single_trade_profit_pct=max_single_trade_profit_pct,
)
validation = validate(result.equity_curve, result.trades, rules)

st.subheader(f"{ticker} — {strategy_name} — {start} → {end}")

cols = st.columns(4)
cols[0].metric("Trades", report.num_trades)
cols[1].metric("Win rate", f"{report.win_rate:.1%}")
cols[2].metric("Profit factor", f"{report.profit_factor:.2f}")
cols[3].metric("Rendement total", f"{report.total_return_pct:.1%}")

cols = st.columns(4)
cols[0].metric("Sharpe (journalier, annualisé)", f"{report.sharpe_daily:.2f}")
cols[1].metric("Sharpe (par trade)", f"{report.sharpe_per_trade:.2f}")
cols[2].metric("Max drawdown", f"{report.max_drawdown_pct:.1%}")
cols[3].metric("Durée max drawdown", f"{report.max_drawdown_duration_days} jours")

if validation.passed:
    st.success("Règles prop firm respectées sur toute la période testée.")
else:
    st.error(f"Règles prop firm violées — première violation le {validation.first_breach_date.date()}.")
    breach_df = pd.DataFrame(
        [{"date": b.date.date(), "règle": b.rule, "détail": b.detail} for b in validation.breaches]
    )
    st.dataframe(breach_df, width="stretch")

st.subheader("Évolution du prix")
st.pyplot(plot_price(df, indicators=strategy.indicators(df), trades=result.trades))

st.subheader("Equity curve")
st.pyplot(plot_equity_curve(result.equity_curve))

st.subheader("Drawdown")
st.pyplot(plot_drawdown(result.equity_curve))

st.subheader("Distribution des trades")
st.pyplot(plot_trade_distribution(result.trades))

st.subheader("Journal des trades")
trades_df = pd.DataFrame(
    [
        {
            "entrée": t.entry_date.date(),
            "sortie": t.exit_date.date() if t.exit_date is not None else None,
            "sens": "long" if t.side == 1 else "short",
            "taille": round(t.size, 4),
            "prix entrée": round(t.entry_price, 2),
            "prix sortie": round(t.exit_price, 2) if t.exit_price is not None else None,
            "pnl": round(t.pnl, 2) if t.pnl is not None else None,
        }
        for t in result.trades
    ]
)
st.dataframe(trades_df, width="stretch")
