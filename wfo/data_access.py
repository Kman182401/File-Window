"""Unified data access helpers for WFO runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

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


DEFAULT_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class DataAccessConfig:
    parquet_path: Optional[str] = None
    columns: Optional[Dict[str, str]] = None
    synthetic_days: int = 30


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
            return self._from_dataset(symbol, start, end, tz, limit)
        return self._synthetic(symbol, start, end, tz, limit)

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
            symbol_root = IBKRIngestor.SYMBOL_ALIASES.get(symbol.upper(), symbol.upper())
        else:
            symbol_root = symbol.upper()
        filt = (
            (ds.field("symbol_root") == symbol_root)
            & (ds.field("timestamp") >= pd.Timestamp(start, tz="UTC"))
            & (ds.field("timestamp") <= pd.Timestamp(end, tz="UTC"))
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


__all__ = ["MarketDataAccess", "DataAccessConfig"]
