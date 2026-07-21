from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from backtester.data import ingest, store  # noqa: E402
from backtester.engine.backtest import run_backtest  # noqa: E402
from backtester.engine.costs import CostModel  # noqa: E402
from backtester.metrics.performance import compute_performance  # noqa: E402
from backtester.propfirm.rules import PropFirmRules, validate  # noqa: E402
from backtester.strategy.registry import STRATEGIES, build_strategy  # noqa: E402

st.set_page_config(page_title="Backtester", page_icon=":material/monitoring:", layout="wide")

GREEN, RED, GRAY = "#34D399", "#F87171", "#94A3B8"

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
    "trend_pullback": [
        ("trend_ma", 20, 300, 150, 5),
        ("lookback", 3, 60, 10, 1),
        ("entry_z", 0.5, 3.0, 1.5, 0.1),
        ("profit_target", 0.005, 0.10, 0.02, 0.005),
        ("stop_loss", 0.005, 0.10, 0.02, 0.005),
        ("max_holding_bars", 1, 30, 5, 1),
    ],
}

# résumé honnête de scripts/walkforward.py (2010-2024, 10 tickers, 4 folds) pour les
# défauts de chaque stratégie -- résultat de portefeuille diversifié, pas une garantie
# pour un seul ticker sur une autre période
WALKFORWARD_NOTES = {
    "ma_crossover": (
        "Défauts validés : profit factor out-of-sample > 1 sur les 4 folds "
        "(1.95 / 5.15 / 5.81 / 3.03), agrégé ≈ 3.3. L'edge le plus solide des quatre."
    ),
    "breakout": (
        "Défauts validés : profit factor out-of-sample > 1 sur les 4 folds "
        "(1.26 / 6.90 / 3.35 / 4.23), agrégé ≈ 2.8."
    ),
    "ml_classifier": (
        "Edge réel mais modeste : profit factor out-of-sample 1.56 / 1.66 / 1.68 / 1.11, "
        "agrégé ≈ 1.5. Le fold le plus récent est le plus faible — signe possible de "
        "dérive du modèle dans le temps."
    ),
    "mean_reversion": (
        "Aucun edge détecté : profit factor ≈ 1.0 sur toute la grille testée "
        "(0.90–1.04) — statistiquement indiscernable du bruit après coûts."
    ),
    "trend_pullback": (
        "Edge réel mais modeste : profit factor out-of-sample 1.09 / 1.96 / 1.19 / 0.97, "
        "agrégé ≈ 1.2. Le fold le plus récent (2021-2024) est légèrement perdant hors "
        "échantillon — à surveiller. Avantage : drawdown nettement plus faible que les "
        "stratégies de tendance pure."
    ),
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


# --------------------------------------------------------------- charts --
def price_chart(df: pd.DataFrame, indicators: dict, trades: list) -> alt.LayerChart:
    series_names = ["Clôture", *indicators.keys()]
    palette = ["#60A5FA", "#FBBF24", "#A78BFA", "#38BDF8", "#FB923C", "#34D399"]
    color_scale = alt.Scale(domain=series_names, range=palette[: len(series_names)])
    dash_scale = alt.Scale(domain=series_names, range=[[1, 0], *([[5, 3]] * (len(series_names) - 1))])

    lines_df = pd.concat(
        [pd.DataFrame({"date": df["date"], "valeur": df["close"], "série": "Clôture"})]
        + [
            pd.DataFrame({"date": df["date"], "valeur": pd.Series(values).to_numpy(), "série": label})
            for label, values in indicators.items()
        ],
        ignore_index=True,
    )

    zoom = alt.selection_interval(bind="scales", encodings=["x"])

    price_lines = (
        alt.Chart(lines_df)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=18, opacity=0), strokeWidth=1.8)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("valeur:Q", title="Prix ($)", scale=alt.Scale(zero=False)),
            color=alt.Color("série:N", scale=color_scale, title="Courbe", legend=alt.Legend(orient="top")),
            strokeDash=alt.StrokeDash("série:N", scale=dash_scale, legend=None),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("série:N", title="Courbe"),
                alt.Tooltip("valeur:Q", title="Valeur", format="$,.2f"),
            ],
        )
    )
    layers = [price_lines]

    if trades:
        markers = pd.DataFrame(
            [
                {"date": t.entry_date, "prix": t.entry_price, "type": "Entrée long",
                 "détail": f"Long @ ${t.entry_price:,.2f}"}
                for t in trades if t.side == 1
            ]
            + [
                {"date": t.entry_date, "prix": t.entry_price, "type": "Entrée short",
                 "détail": f"Short @ ${t.entry_price:,.2f}"}
                for t in trades if t.side == -1
            ]
            + [
                {
                    "date": t.exit_date, "prix": t.exit_price, "type": "Sortie",
                    "détail": f"Sortie @ ${t.exit_price:,.2f}"
                    + (f" · PnL ${t.pnl:,.2f}" if t.pnl is not None else ""),
                }
                for t in trades if t.exit_date is not None
            ]
        )
        layers.append(
            alt.Chart(markers)
            .mark_point(size=90, filled=True, opacity=0.9, stroke="#0F172A", strokeWidth=0.6)
            .encode(
                x="date:T",
                y="prix:Q",
                color=alt.Color(
                    "type:N",
                    title="Trade",
                    scale=alt.Scale(
                        domain=["Entrée long", "Entrée short", "Sortie"], range=[GREEN, RED, GRAY]
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                shape=alt.Shape(
                    "type:N",
                    scale=alt.Scale(
                        domain=["Entrée long", "Entrée short", "Sortie"],
                        range=["triangle-up", "triangle-down", "circle"],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                    alt.Tooltip("détail:N", title=""),
                ],
            )
        )

    price = (
        alt.layer(*layers)
        .resolve_scale(color="independent", shape="independent")
        .properties(height=380)
        .add_params(zoom)
    )

    if "volume" not in df.columns:
        return price

    volume = (
        alt.Chart(df)
        .mark_bar(color="#334155")
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("volume:Q", title="Volume"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("volume:Q", title="Volume", format=",.0f"),
            ],
        )
        .properties(height=100)
        .add_params(zoom)
    )

    return alt.vconcat(price, volume).resolve_scale(x="shared")


def equity_chart(equity_curve: pd.Series) -> alt.Chart:
    df = equity_curve.rename("equity").reset_index()
    df.columns = ["date", "equity"]
    return (
        alt.Chart(df)
        .mark_area(opacity=0.15, color="#60A5FA", line={"color": "#60A5FA", "strokeWidth": 1.8})
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("equity:Q", title="Equity", scale=alt.Scale(zero=False)),
        )
        .properties(height=320)
    )


def drawdown_chart(equity_curve: pd.Series) -> alt.Chart:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max - 1.0) * 100
    df = drawdown.rename("drawdown").reset_index()
    df.columns = ["date", "drawdown"]
    return (
        alt.Chart(df)
        .mark_area(opacity=0.3, color=RED, line={"color": RED, "strokeWidth": 1.4})
        .encode(x=alt.X("date:T", title="Date"), y=alt.Y("drawdown:Q", title="Drawdown (%)"))
        .properties(height=220)
    )


def trade_distribution_chart(trades: list) -> alt.Chart:
    pnls = [t.pnl for t in trades if t.pnl is not None]
    df = pd.DataFrame({"pnl": pnls})
    return (
        alt.Chart(df)
        .mark_bar(color="#60A5FA")
        .encode(
            x=alt.X("pnl:Q", bin=alt.Bin(maxbins=30), title="PnL"),
            y=alt.Y("count():Q", title="Nombre de trades"),
        )
        .properties(height=280)
    )


# ---------------------------------------------------------------- page --
st.title(":material/monitoring: Backtester")
st.caption("Exploration de stratégies de trading — actions & indices")

with st.sidebar:
    st.header("Configuration")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    col1, col2 = st.columns(2)
    date_bounds = (pd.Timestamp("1990-01-01"), pd.Timestamp.today())
    start = col1.date_input("Début", value=pd.Timestamp("2010-01-01"), min_value=date_bounds[0],
                             max_value=date_bounds[1]).isoformat()
    end = col2.date_input("Fin", value=pd.Timestamp("2024-01-01"), min_value=date_bounds[0],
                           max_value=date_bounds[1]).isoformat()

    st.subheader(":material/query_stats: Stratégie")
    strategy_name = st.selectbox("Stratégie", list(STRATEGIES), label_visibility="collapsed")
    strategy_params = strategy_param_widgets(strategy_name)
    st.caption(WALKFORWARD_NOTES.get(strategy_name, ""))

    with st.expander("Capital & coûts", icon=":material/payments:"):
        initial_capital = st.number_input("Capital initial", value=100_000.0, step=10_000.0)
        pct_per_trade = st.slider("% du capital par trade", 0.01, 1.0, 1.0, 0.01)
        slippage_bps = st.slider("Slippage (bps)", 0.0, 20.0, 2.0, 0.5)
        fee_bps = st.slider("Frais (bps)", 0.0, 20.0, 1.0, 0.5)

    with st.expander("Règles prop firm", icon=":material/shield:"):
        daily_loss_pct = st.slider("Perte journalière max (%)", 1.0, 20.0, 5.0, 0.5) / 100
        max_total_loss_pct = st.slider("Perte totale max (%)", 1.0, 30.0, 10.0, 0.5) / 100
        enable_consistency = st.checkbox("Règle de consistency", value=False)
        max_single_trade_profit_pct = (
            st.slider("Profit max d'un trade (% du profit total)", 5.0, 90.0, 30.0, 5.0) / 100
            if enable_consistency
            else None
        )

    run_clicked = st.button("Lancer le backtest", type="primary", icon=":material/play_arrow:", width="stretch")

if not run_clicked:
    st.info("Configure les paramètres dans la barre latérale puis clique sur « Lancer le backtest ».",
             icon=":material/tune:")
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

st.subheader(f"{ticker} · {strategy_name} · {start} → {end}")

with st.container(horizontal=True):
    st.metric("Trades", report.num_trades, border=True)
    st.metric("Win rate", f"{report.win_rate:.1%}", border=True)
    st.metric("Profit factor", f"{report.profit_factor:.2f}", border=True)
    st.metric("Rendement total", f"{report.total_return_pct:.1%}", border=True)

with st.container(horizontal=True):
    st.metric("Sharpe (journalier)", f"{report.sharpe_daily:.2f}", border=True)
    st.metric("Sharpe (par trade)", f"{report.sharpe_per_trade:.2f}", border=True)
    st.metric("Max drawdown", f"{report.max_drawdown_pct:.1%}", border=True)
    st.metric("Durée max drawdown", f"{report.max_drawdown_duration_days} j.", border=True)

if validation.passed:
    st.success("Règles prop firm respectées sur toute la période testée.", icon=":material/check_circle:")
else:
    st.error(f"Règles prop firm violées — première violation le {validation.first_breach_date.date()}.",
              icon=":material/error:")
    breach_df = pd.DataFrame(
        [{"date": b.date.date(), "règle": b.rule, "détail": b.detail} for b in validation.breaches]
    )
    with st.expander(f"Voir les {len(validation.breaches)} violations"):
        st.dataframe(breach_df, width="stretch", hide_index=True)

tab_price, tab_perf, tab_trades = st.tabs([
    ":material/candlestick_chart: Prix & signaux",
    ":material/trending_up: Performance",
    ":material/receipt_long: Trades",
])

with tab_price:
    with st.container(border=True):
        st.altair_chart(price_chart(df, strategy.indicators(df), result.trades), width="stretch")
        st.caption("Survole une courbe pour la valeur exacte · molette ou glisser pour zoomer/naviguer")

with tab_perf:
    with st.container(border=True):
        st.markdown("**Equity curve**")
        st.altair_chart(equity_chart(result.equity_curve), width="stretch")
    with st.container(border=True):
        st.markdown("**Drawdown**")
        st.altair_chart(drawdown_chart(result.equity_curve), width="stretch")

with tab_trades:
    with st.container(border=True):
        st.markdown("**Distribution des trades**")
        st.altair_chart(trade_distribution_chart(result.trades), width="stretch")
    with st.container(border=True):
        st.markdown("**Journal des trades**")
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
        st.dataframe(
            trades_df,
            width="stretch",
            hide_index=True,
            column_config={
                "prix entrée": st.column_config.NumberColumn(format="$%.2f"),
                "prix sortie": st.column_config.NumberColumn(format="$%.2f"),
                "pnl": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
