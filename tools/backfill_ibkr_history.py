#!/usr/bin/env python3
"""Bulk backfill IBKR historical bars using the shared ingestor.

Walks a fixed lookback window in safe chunks while reusing the
existing adapter so pacing, contract mapping, and persistence logic
stay centralized. The script is idempotent; it resumes from the
latest persisted timestamp if data already exists.
"""

import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ib_insync import Future

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

ALIAS_TO_ROOT = {
    "ES1!": "ES",
    "NQ1!": "NQ",
    "XAUUSD": "GC",
    "EURUSD": "6E",
    "GBPUSD": "6B",
    "AUDUSD": "6A",
}

ROOT_TO_EXCH = {
    "ES": "CME",
    "NQ": "CME",
    "6E": "CME",
    "6B": "CME",
    "6A": "CME",
    "GC": "COMEX",
}

ROOT_TO_SYMBOL_OVERRIDE = {
    "6E": "EUR",
    "6B": "GBP",
    "6A": "AUD",
}


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


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "error 166" in text:
        return "retention"
    if "error 162" in text or "error 165" in text or "no data returned" in text:
        return "nodata"
    return "other"


def _resolve_front_month_contract(ib_ingestor: IBKRIngestor, symbol: str, asof: datetime):
    """Resolve the front-month futures contract for a symbol as of *asof*."""

    root = ALIAS_TO_ROOT.get(symbol, symbol)
    exch = ROOT_TO_EXCH.get(root)
    if exch is None:
        return None

    symbol_override = ROOT_TO_SYMBOL_OVERRIDE.get(root, root)
    future_contract = Future(
        symbol=symbol_override,
        exchange=exch,
        tradingClass=root,
    )
    details = ib_ingestor.ib.reqContractDetails(future_contract)
    if not details:
        return None

    yyyymmdd = asof.strftime("%Y%m%d")

    def _expiry_key(cd):
        month = cd.contract.lastTradeDateOrContractMonth or "99999999"
        return month

    candidates = [cd for cd in details if (cd.contract.lastTradeDateOrContractMonth or "00000000") >= yyyymmdd]
    picked = sorted(candidates, key=_expiry_key)[0] if candidates else sorted(details, key=_expiry_key)[-1]
    return picked.contract


def _probe_head_timestamp(ib_ingestor: IBKRIngestor, contract) -> Optional[pd.Timestamp]:
    """Return earliest available timestamp for the contract via reqHeadTimeStamp."""

    try:
        head = ib_ingestor.ib.reqHeadTimeStamp(contract, whatToShow="TRADES", useRTH=False, formatDate=1)
    except Exception:
        return None

    ts = pd.to_datetime(str(head), utc=True, errors="coerce")
    return ts if pd.notna(ts) else None


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


def _earliest_timestamp(symbol: str) -> Optional[pd.Timestamp]:
    base = os.path.join(DATA_DIR, f"symbol={symbol}")
    if not os.path.isdir(base):
        return None
    for date_dir in sorted((d for d in os.listdir(base) if d.startswith("date="))):
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
        return ts.min()
    return None


def _backfill_symbol(ib: IBKRIngestor, symbol: str) -> None:
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(days=LOOKBACK_DAYS)

    latest = _latest_timestamp(symbol)
    if latest is not None:
        start_utc = max(start_utc, (latest + pd.Timedelta(minutes=1)).to_pydatetime())

    earliest = _earliest_timestamp(symbol)
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

        if os.getenv("USE_HEAD_TS", "1") == "1":
            contract = _resolve_front_month_contract(ib, symbol, nxt)
            if contract is not None:
                earliest = _probe_head_timestamp(ib, contract)
                if earliest is not None:
                    earliest_dt = earliest.to_pydatetime()
                    if earliest_dt > cur:
                        cur = earliest_dt
                        if cur >= nxt:
                            continue

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
            end_utc_str = nxt.strftime("%Y%m%d-%H:%M:%S")
            ib.fetch_data(
                symbol,
                duration=duration,
                barSize="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                endDateTime=end_utc_str,
                asof=nxt,
            )
            _record_request(symbol, duration, nxt)
            print(f"  [OK] {symbol} {cur} -> {nxt}")
        except Exception as exc:
            kind = _classify_error(exc)
            if kind == "retention":
                print(
                    f"  [STOP] {symbol} {cur} -> {nxt}: IBKR retention limit reached ({exc}); stopping forward fill."
                )
                break
            print(f"  [WARN] {symbol} {cur} -> {nxt}: {exc}")
        cur = nxt
        time.sleep(SLEEP_BETWEEN_REQ)

    target_start = now_utc - timedelta(days=LOOKBACK_DAYS)
    if earliest is None:
        earliest = _earliest_timestamp(symbol)

    if earliest is not None and earliest - target_start > timedelta(minutes=1):
        print(f"[{symbol}] backward fill target={target_start.isoformat()} current_earliest={earliest.isoformat()}")
        end_point = earliest
        while end_point - target_start > timedelta(minutes=1):
            nxt_end = end_point
            start = max(target_start, nxt_end - timedelta(days=1))
            minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            if minutes > MAX_BARS_PER_REQUEST:
                start = nxt_end - timedelta(minutes=MAX_BARS_PER_REQUEST)
            duration_minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            duration_days = max(1, (nxt_end - start).days or (duration_minutes + 1439) // 1440)
            duration = f"{duration_days} D"

            if os.getenv("USE_HEAD_TS", "1") == "1":
                contract = _resolve_front_month_contract(ib, symbol, nxt_end)
                if contract is not None:
                    earliest_available = _probe_head_timestamp(ib, contract)
                    if earliest_available is not None and earliest_available.to_pydatetime() > start:
                        start = earliest_available.to_pydatetime()
                        if nxt_end - start <= timedelta(minutes=1):
                            end_point = start
                            continue

            try:
                _enforce_pacing(symbol, duration, nxt_end)
                end_utc_str = nxt_end.strftime("%Y%m%d-%H:%M:%S")
                ib.fetch_data(
                    symbol,
                    duration=duration,
                    barSize="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                    endDateTime=end_utc_str,
                    asof=nxt_end,
                )
                _record_request(symbol, duration, nxt_end)
                print(f"  [OK] {symbol} backward {start} -> {nxt_end}")
            except Exception as exc:
                kind = _classify_error(exc)
                if kind == "retention":
                    print(
                        f"  [STOP] {symbol} backward {start} -> {nxt_end}: IBKR retention limit reached ({exc}); stopping backward fill."
                    )
                    break
                print(f"  [WARN] {symbol} backward {start} -> {nxt_end}: {exc}")
            end_point = start
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
