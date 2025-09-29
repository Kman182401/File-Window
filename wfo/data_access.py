"""Unified data access helpers for WFO runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow.dataset as ds
    _HAS_PYARROW = True
except Exception:  # pragma: no cover - pyarrow optional in some environments
    _HAS_PYARROW = False

try:  # pragma: no cover - optional dependency for live data
    from market_data_ibkr_adapter import IBKRIngestor
    _HAS_IBKR = True
except Exception:  # pragma: no cover
    IBKRIngestor = None
    _HAS_IBKR = False



# --- Safe alias mapping (works even if IBKRIngestor lacks SYMBOL_ALIASES) ---
try:
    from market_data_ibkr_adapter import IBKRIngestor  # optional dependency
except Exception:
    IBKRIngestor = None  # type: ignore

SYMBOL_ALIASES = {
    "ES1!": "ES", "NQ1!": "NQ",
    "XAUUSD": "GC",
    "EURUSD": "6E", "GBPUSD": "6B", "AUDUSD": "6A",
    "ES": "ES", "NQ": "NQ", "GC": "GC", "6E": "6E", "6B": "6B", "6A": "6A",
}
if IBKRIngestor is not None:
    SYMBOL_ALIASES.update(getattr(IBKRIngestor, "SYMBOL_ALIASES", {}))
# ---------------------------------------------------------------------------
DEFAULT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _as_utc(ts_like: Any) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp for arbitrary datetime-like inputs."""
    return pd.to_datetime(ts_like, utc=True)


@dataclass
class DataAccessConfig:
    parquet_path: Optional[str] = None
    columns: Optional[Dict[str, str]] = None
    synthetic_days: int = 30
    event_bars: Optional[Dict[str, Any]] = None
    fracdiff: Optional[Dict[str, Any]] = None


class MarketDataAccess:
    """Loader that prefers local parquet data and falls back gracefully."""

    def __init__(self, config: Optional[DataAccessConfig] = None):
        self.config = config or DataAccessConfig()
        parquet_path = self.config.parquet_path
        if parquet_path is None:
            # Prefer freshly promoted dataset, fall back to _v2 alias
            candidates = [
                Path.home() / "data/ibkr_partitioned/minute_bars",
                Path.home() / "data/ibkr_partitioned_v2/minute_bars",
            ]
            parquet_path = next((p for p in candidates if p.exists()), candidates[0])
        self.parquet_path = Path(parquet_path)
        self._dataset = None
        if self.parquet_path.exists() and _HAS_PYARROW:
            self._dataset = ds.dataset(str(self.parquet_path), format="parquet", partitioning="hive")

    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tz: str = "America/New_York",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load bars for a symbol between start and end (inclusive)."""
        if self._dataset is not None:
            df = self._from_dataset(symbol, start, end, tz, limit)
        else:
            df = self._synthetic(symbol, start, end, tz, limit)
        df = self._maybe_event_bars(df)
        df = self._maybe_fracdiff(df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _from_dataset(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tz: str,
        limit: Optional[int],
    ) -> pd.DataFrame:
        columns = DEFAULT_COLUMNS
        if self.config.columns:
            columns = list(self.config.columns.values())
        if _HAS_IBKR and IBKRIngestor is not None:
            symbol_root = SYMBOL_ALIASES.get(symbol.upper(), symbol.upper())
        else:
            symbol_root = symbol.upper()
        start_ts = _as_utc(start)
        end_ts = _as_utc(end)
        filt = (
            (ds.field("symbol_root") == symbol_root)
            & (ds.field("timestamp") >= start_ts)
            & (ds.field("timestamp") < end_ts)
        )
        scanner = self._dataset.scanner(filter=filt, columns=columns)
        table = scanner.to_table()
        df = table.to_pandas()
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
        df = df.sort_values("timestamp")
        if limit is not None:
            df = df.head(limit)
        return df

    def _synthetic(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        tz: str,
        limit: Optional[int],
    ) -> pd.DataFrame:
        """Generate deterministic synthetic data when local files are missing."""
        minutes = int((end - start).total_seconds() // 60) + 1
        if limit is not None:
            minutes = min(minutes, limit)
        if minutes <= 0:
            minutes = self.config.synthetic_days * 390
        index = pd.date_range(start=start, periods=minutes, freq="T", tz=tz)
        base = np.linspace(0, 1, len(index))
        price = 100 + 5 * np.sin(2 * np.pi * base) + 0.5 * np.random.default_rng(42).standard_normal(len(index))
        df = pd.DataFrame(
            {
                "timestamp": index,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price + np.random.default_rng(123).normal(0, 0.2, len(index)),
                "volume": 1000 + 100 * np.sin(5 * base),
            }
        )
        return df.head(limit) if limit is not None else df

    # ------------------------------------------------------------------
    # Post-processing hooks
    # ------------------------------------------------------------------
    def _maybe_event_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.event_bars or {}
        mode = str(cfg.get("mode", "")).lower()
        if mode not in {"", "time", "tick", "volume", "dollar"}:
            mode = ""
        if mode in {"", "time"} or df.empty:
            return df
        required = {"close", "volume"}
        if not required.issubset(df.columns):
            return df

        threshold = cfg.get("threshold")
        if threshold is None:
            threshold = 1000 if mode == "tick" else 1_000_000
        bars = _build_event_bars(df, mode=mode, threshold=threshold)
        if bars is None or bars.empty:
            return df
        return bars

    def _maybe_fracdiff(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.fracdiff or {}
        if not cfg.get("enabled", False) or df.empty:
            return df
        columns = cfg.get("columns") or ["close"]
        d = float(cfg.get("d", 0.5))
        thresh = float(cfg.get("thresh", 1e-4))
        for col in columns:
            if col not in df.columns:
                continue
            name = f"{col}_fracdiff_{d:.2f}".replace(".", "p")
            df[name] = fracdiff_series(df[col], d=d, thresh=thresh)
        return df


__all__ = ["MarketDataAccess", "DataAccessConfig"]


def _build_event_bars(df: pd.DataFrame, *, mode: str, threshold: float) -> pd.DataFrame:
    mode = mode.lower()
    threshold = float(threshold)
    if threshold <= 0:
        return df

    values = []
    start_idx = 0
    acc = 0.0
    dollar_col = df["close"].to_numpy(dtype=float) * df["volume"].to_numpy(dtype=float)

    for idx in range(len(df)):
        if mode == "tick":
            acc += 1
        elif mode == "volume":
            acc += float(df["volume"].iat[idx])
        elif mode == "dollar":
            acc += float(dollar_col[idx])
        else:
            break

        if acc >= threshold:
            window = df.iloc[start_idx : idx + 1]
            values.append(_aggregate_window(window))
            start_idx = idx + 1
            acc = 0.0

    if start_idx < len(df):
        window = df.iloc[start_idx:]
        if not window.empty:
            values.append(_aggregate_window(window))

    return pd.DataFrame(values)


def _aggregate_window(window: pd.DataFrame) -> Dict[str, Any]:
    open_price = float(window["open"].iloc[0])
    close_price = float(window["close"].iloc[-1])
    high_price = float(window["high"].max())
    low_price = float(window["low"].min())
    volume = float(window["volume"].sum())
    timestamp = window["timestamp"].iloc[-1]
    out = {
        "timestamp": timestamp,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
    }
    if "symbol" in window.columns:
        out["symbol"] = window["symbol"].iloc[-1]
    if "symbol_root" in window.columns:
        out["symbol_root"] = window["symbol_root"].iloc[-1]
    out["dollar_volume"] = float((window["close"] * window["volume"]).sum())
    return out


def fracdiff_series(series: pd.Series, *, d: float, thresh: float = 1e-4) -> pd.Series:
    """Fractional differentiation with d typically in [0, 1].

    When ``d = 0`` the original series is returned (after dropping NaNs); as ``d``
    approaches 1 the transformation converges to a first difference.
    """
    values = series.astype(float)
    weights = [1.0]
    k = 1
    while True:
        weight = -weights[-1] * (d - k + 1) / k
        if abs(weight) < thresh:
            break
        weights.append(weight)
        k += 1
    weights = np.array(weights[::-1])

    output = np.full(len(values), np.nan, dtype=float)
    span = len(weights)
    for i in range(span - 1, len(values)):
        window = values.iloc[i - span + 1 : i + 1]
        if window.isnull().any():
            continue
        output[i] = np.dot(weights, window.to_numpy())
    return pd.Series(output, index=series.index)
