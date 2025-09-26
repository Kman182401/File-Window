#!/usr/bin/env python3
"""Bulk backfill IBKR historical bars using the shared ingestor.

Walks a fixed lookback window in safe chunks while reusing the
existing adapter so pacing, contract mapping, and persistence logic
stay centralized. The script is idempotent; it resumes from the
latest persisted timestamp if data already exists.
"""

import os
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional, Tuple

import pandas as pd

from market_data_ibkr_adapter import IBKRIngestor
from market_data_config import IBKR_SYMBOLS

DATA_DIR = os.path.expanduser(os.getenv("DATA_DIR", "~/.local/share/m5_trader/data"))
WINDOW_DAYS = int(os.getenv("BACKFILL_WINDOW_DAYS", "10"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "730"))
SLEEP_BETWEEN_REQ = float(os.getenv("BACKFILL_SLEEP_SECS", "0.6"))
MAX_BARS_PER_REQUEST = int(os.getenv("BACKFILL_MAX_BARS", "5000"))

_REQUEST_HISTORY: Deque[float] = deque()
_SYMBOL_HISTORY: Dict[str, Deque[float]] = {}
_LAST_IDENTICAL: Dict[Tuple[str, str, str], float] = {}


def _enforce_pacing(symbol: str, duration: str, end_dt: datetime) -> None:
    """Block until IB's historical pacing thresholds are satisfied."""

    key = (symbol, duration, end_dt.strftime("%Y%m%d %H:%M:%S"))

    while True:
        now = time.monotonic()

        # Global 60 requests per 10-minute window.
        while _REQUEST_HISTORY and now - _REQUEST_HISTORY[0] > 600:
            _REQUEST_HISTORY.popleft()
        if len(_REQUEST_HISTORY) >= 60:
            sleep_for = 600 - (now - _REQUEST_HISTORY[0]) + 0.05
            time.sleep(max(0.05, sleep_for))
            continue

        bucket = _SYMBOL_HISTORY.setdefault(symbol, deque())
        while bucket and now - bucket[0] > 2:
            bucket.popleft()
        if len(bucket) >= 5:  # sixth call would violate 6-in-2s limit
            sleep_for = 2 - (now - bucket[0]) + 0.05
            time.sleep(max(0.05, sleep_for))
            continue

        last_identical = _LAST_IDENTICAL.get(key)
        if last_identical is not None and now - last_identical < 15:
            sleep_for = 15 - (now - last_identical) + 0.05
            time.sleep(max(0.05, sleep_for))
            continue

        break


def _record_request(symbol: str, duration: str, end_dt: datetime) -> None:
    now = time.monotonic()
    _REQUEST_HISTORY.append(now)
    _SYMBOL_HISTORY.setdefault(symbol, deque()).append(now)
    key = (symbol, duration, end_dt.strftime("%Y%m%d %H:%M:%S"))
    _LAST_IDENTICAL[key] = now


def _latest_timestamp(symbol: str) -> Optional[pd.Timestamp]:
    base = os.path.join(DATA_DIR, f"symbol={symbol}")
    if not os.path.isdir(base):
        return None
    for date_dir in sorted((d for d in os.listdir(base) if d.startswith("date=")), reverse=True):
        parquet_path = os.path.join(base, date_dir, "bars.parquet")
        if not os.path.exists(parquet_path):
            continue
        try:
            df = pd.read_parquet(parquet_path, columns=["timestamp"])
        except Exception:
            continue
        if df.empty:
            continue
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        return ts.max()
    return None


def _backfill_symbol(ib: IBKRIngestor, symbol: str) -> None:
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=LOOKBACK_DAYS)

    latest = _latest_timestamp(symbol)
    if latest is not None:
        start_utc = max(start_utc, (latest + pd.Timedelta(minutes=1)).to_pydatetime())

    provider_limit = now_utc - timedelta(days=730)
    if start_utc < provider_limit:
        print(
            f"[{symbol}] start {start_utc.isoformat()} predates IB's ~2y futures retention; "
            "older data may be unavailable."
        )

    cur = start_utc
    print(f"[{symbol}] backfill start={start_utc.isoformat()} end={now_utc.isoformat()}")

    while cur < now_utc:
        nxt = min(cur + timedelta(days=WINDOW_DAYS), now_utc)

        minutes = max(1, int((nxt - cur).total_seconds() / 60))
        if minutes > MAX_BARS_PER_REQUEST:
            nxt = cur + timedelta(minutes=MAX_BARS_PER_REQUEST)
        if nxt - cur > timedelta(days=1):
            nxt = cur + timedelta(days=1)
        if nxt > now_utc:
            nxt = now_utc

        duration_minutes = max(1, int((nxt - cur).total_seconds() / 60))
        duration_days = max(1, (nxt - cur).days or (duration_minutes + 1439) // 1440)
        duration = f"{duration_days} D"
        try:
            _enforce_pacing(symbol, duration, nxt)
            ib.fetch_data(
                symbol,
                duration=duration,
                barSize="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                endDateTime=nxt.strftime("%Y%m%d %H:%M:%S"),
                asof=nxt,
            )
            _record_request(symbol, duration, nxt)
            print(f"  [OK] {symbol} {cur} -> {nxt}")
        except Exception as exc:
            print(f"  [WARN] {symbol} {cur} -> {nxt}: {exc}")
        cur = nxt
        time.sleep(SLEEP_BETWEEN_REQ)


def main() -> None:
    ib = IBKRIngestor(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=int(os.getenv("IBKR_PORT", "4002")),
        clientId=int(os.getenv("IBKR_CLIENT_ID", "9003")),
    )

    for symbol in IBKR_SYMBOLS:
        _backfill_symbol(ib, symbol)


if __name__ == "__main__":
    main()
