"""Streamlit dashboard: the first credible demo surface.

Shows a site's probabilistic forecast against actuals with a P10 to P90
uncertainty band, the backtest scorecard against the persistence baseline, and
the battery and genset advisory. Everything runs through the engine facade, so
the dashboard shows exactly what the API would return.

Front end polish stays behind forecast quality on purpose. This is a working
demo of a real forecast, not a mockup.

Run with:
    uv sync --extra dashboard
    uv run streamlit run vaticore/dashboard/app.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from vaticore import engine
from vaticore.datasets import make_synthetic_fleet
from vaticore.forecasting.base import quantile_column
from vaticore.schemas import GENERATION_KW, LOAD_KW, OPERATOR_ID, SITE_ID, TIMESTAMP

QUANTILES = (0.1, 0.5, 0.9)
HISTORY_TAIL = 24 * 7  # show one week of context behind the forecast


def _load_fleet(days: int) -> pd.DataFrame:
    # Demo data. Replace with a repository read against real operator feeds.
    return make_synthetic_fleet(days=days, seed=1)


def _forecast_figure(history: pd.DataFrame, target: str, forecast: pd.DataFrame) -> Any:
    import plotly.graph_objects as go

    context = history.tail(HISTORY_TAIL)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=context[TIMESTAMP],
            y=context[target],
            name="actual",
            line={"color": "#1f77b4"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast[quantile_column(0.9)],
            name="P90",
            line={"width": 0},
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast[quantile_column(0.1)],
            name="P10 to P90",
            fill="tonexty",
            fillcolor="rgba(255,127,14,0.2)",
            line={"width": 0},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast[quantile_column(0.5)],
            name="forecast (P50)",
            line={"color": "#ff7f0e"},
        )
    )
    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h"},
        yaxis_title="kW",
        height=420,
    )
    return fig


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Vaticore", layout="wide")
    st.title("Vaticore")
    st.caption("Probabilistic load and solar forecasting for distributed energy operators.")

    fleet = _load_fleet(days=90)
    sites = fleet[[OPERATOR_ID, SITE_ID]].drop_duplicates()

    with st.sidebar:
        st.header("Controls")
        labels = [f"{r[OPERATOR_ID]} / {r[SITE_ID]}" for _, r in sites.iterrows()]
        choice = st.selectbox("Site", labels)
        operator_id, site_id = (part.strip() for part in choice.split("/"))
        target_label = st.radio("Target", ["Load", "Solar generation"])
        target = LOAD_KW if target_label == "Load" else GENERATION_KW
        model = st.selectbox(
            "Model", engine.available_models(), index=len(engine.available_models()) - 1
        )
        horizon = st.slider("Forecast horizon (hours)", 6, 48, 24, step=6)
        battery = st.number_input("Usable battery (kWh)", min_value=0.0, value=600.0, step=50.0)

    history = engine.select_site(fleet, operator_id, site_id)

    forecast = engine.forecast_site(
        history, target, horizon=horizon, model=model, quantiles=QUANTILES
    )

    left, right = st.columns([3, 1])
    with left:
        st.subheader(f"{target_label} forecast: {site_id}")
        st.plotly_chart(_forecast_figure(history, target, forecast), width="stretch")
    with right:
        st.subheader("Advisory")
        advisory = engine.advisory_for_site(
            history, horizon=horizon, usable_battery_kwh=battery, model=model, quantiles=QUANTILES
        )
        st.metric("Genset needed", "Yes" if advisory.genset_recommended else "No")
        st.metric("Recommended reserve", f"{advisory.recommended_reserve_kwh:.0f} kWh")
        st.metric("Expected net load", f"{advisory.expected_net_load_kwh:.0f} kWh")
        st.caption(advisory.note)

    st.subheader("Backtest vs persistence baseline")
    st.caption(
        "Rolling origin evaluation. Lower pinball loss is better. The candidate model "
        "must beat persistence to earn its place."
    )
    with st.spinner("Running backtest..."):
        initial = min(len(history) - horizon - 1, 24 * 45)
        result = engine.run_backtest(
            history,
            target,
            horizon=horizon,
            initial=initial,
            step=horizon * 3,
            model=model,
            quantiles=QUANTILES,
        )
    st.dataframe(result.summary().round(3), width="stretch")


if __name__ == "__main__":
    main()
