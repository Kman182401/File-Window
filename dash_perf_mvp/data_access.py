"""Data access helpers for performance dashboard."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


DEFAULT_PARQUET = Path("data/trades.parquet")
DEFAULT_CSV = Path("data/trades.csv")


def _resolve_path(env_var: str, default: Path) -> Path:
    """Return the path indicated by environment or default."""
    override = os.getenv(env_var)
    return Path(override) if override else default


def load_trades() -> pd.DataFrame:
    """Load trade-level results from Parquet or CSV.

    Returns
    -------
    pandas.DataFrame
        Sorted trades with normalized column names and timestamps.
    """
    parquet_path = _resolve_path("TRADES_PARQUET", DEFAULT_PARQUET)
    csv_path = _resolve_path("TRADES_CSV", DEFAULT_CSV)

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            "Provide data/trades.parquet or data/trades.csv or set TRADES_PARQUET/CSV"
        )

    rename = {
        "timestamp_exit": "ts_exit",
        "time_exit": "ts_exit",
        "exit_ts": "ts_exit",
        "pl": "pnl",
        "profit": "pnl",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["ts_exit"] = pd.to_datetime(df["ts_exit"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_exit", "pnl"])

    if "symbol" not in df:
        df["symbol"] = "UNKNOWN"

    df = df.sort_values("ts_exit").reset_index(drop=True)
    return df
