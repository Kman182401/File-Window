"""Dash application for trading performance analytics."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html

from compute_metrics import (
    compute_equity,
    resample_daily_returns,
    rolling_sharpe,
    rolling_sortino,
    daily_return_distribution,
    pnl_heatmap_hour_weekday,
    pnl_by_symbol_month,
    symbol_contribution,
)
from data_access import load_trades
import theming as th


app = Dash(__name__)
app.title = "Performance & Risk (MVP)"

DEFAULT_HEIGHT = 440


def fig_equity_and_drawdown(eq: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=eq["ts_exit"], y=eq["equity"], name="Equity", mode="lines"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=eq["ts_exit"], y=eq["dd"], name="Drawdown", mode="lines"),
        secondary_y=True,
    )
    fig.update_layout(template=th.PLOTLY_TEMPLATE, height=DEFAULT_HEIGHT, **th.LAYOUT_KW)
    fig.update_layout(legend_orientation="h", legend_y=1.1)
    fig.update_yaxes(title_text="Equity", secondary_y=False)
    fig.update_yaxes(title_text="Drawdown", secondary_y=True)
    return fig


def fig_rolling_ratios(daily_ret: pd.DataFrame) -> go.Figure:
    window = max(5, min(60, len(daily_ret)))
    sharpe_window = rolling_sharpe(daily_ret, window)
    sortino_window = rolling_sortino(daily_ret, window)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sharpe_window["ts"],
            y=sharpe_window[f"sharpe_{window}"],
            name=f"Sharpe ({window}d)",
            mode="lines",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=sortino_window["ts"],
            y=sortino_window[f"sortino_{window}"],
            name=f"Sortino ({window}d)",
            mode="lines",
        ),
    )
    fig.update_layout(template=th.PLOTLY_TEMPLATE, height=DEFAULT_HEIGHT, **th.LAYOUT_KW)
    fig.update_yaxes(title="Ratio")
    return fig


def fig_daily_distribution(daily_ret: pd.DataFrame) -> go.Figure:
    series = daily_return_distribution(daily_ret)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=series.values, nbinsx=60, name="Daily Returns"))
    fig.update_layout(
        template=th.PLOTLY_TEMPLATE,
        height=DEFAULT_HEIGHT,
        **th.LAYOUT_KW,
        xaxis_title="Daily Return",
        yaxis_title="Count",
    )
    return fig


def fig_heatmap_hour_weekday(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            coloraxis="coloraxis",
        )
    )
    fig.update_layout(template=th.PLOTLY_TEMPLATE, height=DEFAULT_HEIGHT, **th.LAYOUT_KW)
    fig.update_layout(
        xaxis_title="Hour (0–23)",
        yaxis_title="Weekday (0=Mon)",
        coloraxis_colorscale="RdBu",
        coloraxis_reversescale=True,
    )
    return fig


def fig_symbol_contrib(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["symbol"], y=df["pnl"], name="PnL by Symbol"))
    fig.update_layout(
        template=th.PLOTLY_TEMPLATE,
        height=DEFAULT_HEIGHT,
        **th.LAYOUT_KW,
        xaxis_title="Symbol",
        yaxis_title="PnL",
    )
    return fig


def fig_symbol_month_heatmap(pivot: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, coloraxis="coloraxis"),
    )
    fig.update_layout(
        template=th.PLOTLY_TEMPLATE,
        height=DEFAULT_HEIGHT,
        **th.LAYOUT_KW,
        xaxis_title="Month",
        yaxis_title="Symbol",
        coloraxis_colorscale="Viridis",
    )
    return fig


def layout_page(trades: pd.DataFrame) -> html.Div:
    if trades.empty:
        return html.Div([
            html.H2("Trading Performance & Risk — MVP"),
            html.P("No trades available. Please populate data/trades.parquet or data/trades.csv."),
        ])

    equity = compute_equity(trades)
    daily = resample_daily_returns(equity)
    heatmap_hw = pnl_heatmap_hour_weekday(trades)
    symbol_contrib_df = symbol_contribution(trades)
    symbol_month = pnl_by_symbol_month(trades)

    kpis = [
        ("Total PnL", f"{equity['equity'].iloc[-1]:,.2f}"),
        ("Max DD", f"{equity['dd'].min():,.2f}"),
        ("Max DD %", f"{equity['dd_pct'].min() * 100:,.2f}%"),
        ("Days", f"{len(daily):d}"),
        ("Symbols", f"{trades['symbol'].nunique():d}"),
    ]
    tiles = [
        html.Div(
            [html.Div(title, className="kpi-title"), html.Div(value, className="kpi-value")],
            className="kpi",
        )
        for title, value in kpis
    ]

    return html.Div(
        [
            html.H2("Trading Performance & Risk — MVP"),
            html.Div(tiles, className="kpi-row"),
            dcc.Graph(
                figure=fig_equity_and_drawdown(equity),
                style={"height": f"{DEFAULT_HEIGHT}px"},
                config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            ),
            dcc.Graph(
                figure=fig_rolling_ratios(daily),
                style={"height": f"{DEFAULT_HEIGHT}px"},
                config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            ),
            dcc.Graph(
                figure=fig_daily_distribution(daily),
                style={"height": f"{DEFAULT_HEIGHT}px"},
                config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            ),
            dcc.Graph(
                figure=fig_heatmap_hour_weekday(heatmap_hw),
                style={"height": f"{DEFAULT_HEIGHT}px"},
                config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            ),
            dcc.Graph(
                figure=fig_symbol_contrib(symbol_contrib_df),
                style={"height": f"{DEFAULT_HEIGHT}px"},
                config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            ),
            dcc.Graph(
                figure=fig_symbol_month_heatmap(symbol_month),
                style={"height": f"{DEFAULT_HEIGHT}px"},
                config={"displaylogo": False, "responsive": True, "scrollZoom": True},
            ),
        ],
        className="container",
    )


app.layout = lambda: layout_page(load_trades())


if __name__ == "__main__":  # pragma: no cover
    app.run_server(
        debug=True,
        dev_tools_hot_reload=False,
        host="127.0.0.1",
        port=8050,
    )
