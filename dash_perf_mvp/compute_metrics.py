"""Metric calculations for trading performance analytics."""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
ANNUALIZATION_FACTOR = 252


def compute_equity(trades: pd.DataFrame) -> pd.DataFrame:
    """Return cumulative equity and drawdown series."""
    eq = trades[["ts_exit", "pnl"]].copy().sort_values("ts_exit")
    eq["equity"] = eq["pnl"].cumsum()
    eq["equity_cummax"] = eq["equity"].cummax()
    eq["dd"] = eq["equity"] - eq["equity_cummax"]
    eq["dd_pct"] = np.where(
        eq["equity_cummax"].abs() > EPS,
        eq["dd"] / eq["equity_cummax"].abs(),
        0.0,
    )
    return eq.reset_index(drop=True)


def resample_daily_returns(equity_df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Compute daily returns from equity curve."""
    series = equity_df.set_index("ts_exit")["equity"].copy()
    if tz:
        series = series.tz_convert(tz)
    daily = series.resample("1D").last().ffill()
    returns = daily.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pd.DataFrame({"ts": returns.index, "ret": returns.values})


def _safe_roll(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window)


def rolling_sharpe(daily_ret: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Return rolling Sharpe ratio."""
    roll = _safe_roll(daily_ret["ret"], window)
    mu = roll.mean()
    sigma = roll.std().replace(0, np.nan)
    sharpe = np.sqrt(ANNUALIZATION_FACTOR) * (mu / sigma)
    out = daily_ret.copy()
    out[f"sharpe_{window}"] = sharpe.bfill().fillna(0.0)
    return out


def rolling_sortino(daily_ret: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Return rolling Sortino ratio."""
    returns = daily_ret["ret"]
    mu = _safe_roll(returns, window).mean()
    downside = returns.where(returns < 0.0, 0.0) ** 2
    ddv = _safe_roll(downside, window).mean().apply(np.sqrt)
    denom = ddv.replace(0, np.nan)
    sortino = np.sqrt(ANNUALIZATION_FACTOR) * (mu / denom)
    out = daily_ret.copy()
    out[f"sortino_{window}"] = sortino.bfill().fillna(0.0)
    return out


def daily_return_distribution(daily_ret: pd.DataFrame) -> pd.Series:
    """Return daily return series for distribution plotting."""
    return daily_ret.set_index("ts")["ret"]


def pnl_heatmap_hour_weekday(trades: pd.DataFrame) -> pd.DataFrame:
    """Pivot PnL by exit hour and weekday."""
    df = trades.copy()
    df["hour"] = df["ts_exit"].dt.hour
    df["weekday"] = df["ts_exit"].dt.weekday
    pivot = df.pivot_table(
        index="weekday",
        columns="hour",
        values="pnl",
        aggfunc="sum",
        fill_value=0.0,
    )
    return pivot.sort_index()


def pnl_by_symbol_month(trades: pd.DataFrame) -> pd.DataFrame:
    """Pivot PnL by symbol and calendar month."""
    df = trades.copy()
    df["month"] = df["ts_exit"].dt.to_period("M").astype(str)
    return df.pivot_table(
        index="symbol",
        columns="month",
        values="pnl",
        aggfunc="sum",
        fill_value=0.0,
    )


def symbol_contribution(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total PnL contribution per symbol."""
    grouped = trades.groupby("symbol", dropna=False)["pnl"].sum().sort_values(ascending=False)
    return grouped.reset_index(name="pnl")
