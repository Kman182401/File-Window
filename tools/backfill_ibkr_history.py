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
from typing import Deque, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ib_insync import Future

from market_data_ibkr_adapter import IBKRIngestor
from market_data_config import IBKR_SYMBOLS
from utils.persist_market_data import persist_bars

DATA_DIR = os.path.expanduser(os.getenv("DATA_DIR", "~/.local/share/m5_trader/data"))
WINDOW_DAYS = int(os.getenv("BACKFILL_WINDOW_DAYS", "10"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "730"))
BACKFILL_MODE = os.getenv("BACKFILL_MODE", os.getenv("BACKFILL_ORDER", "forward_first")).lower()
# Max number of whole days to include per request slice (bars-first guard still applies)
MAX_DAYS_PER_REQ = int(os.getenv("BACKFILL_MAX_DAYS_PER_REQ", "3"))
# Optional boundary padding in days applied at the backward target (floored to UTC midnight first)
BOUNDARY_PAD_DAYS = int(os.getenv("BACKFILL_BOUNDARY_PAD_DAYS", "0"))
SLEEP_BETWEEN_REQ = float(os.getenv("BACKFILL_SLEEP_SECS", "0.6"))
MAX_BARS_PER_REQUEST = int(os.getenv("BACKFILL_MAX_BARS", "5000"))

_REQUEST_HISTORY: Deque[float] = deque()
_SYMBOL_HISTORY: Dict[str, Deque[float]] = {}
_LAST_IDENTICAL: Dict[Tuple[str, str, str], float] = {}
_CONTRACT_CACHE: Dict[Tuple[str, datetime], Optional[Future]] = {}
_CONTRACT_CHAINS: Dict[str, List[Future]] = {}

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

ROLLOVER_BUFFER_DAYS = int(os.getenv("IBKR_ROLLOVER_BUFFER_DAYS", "2"))


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


def _req_cd_with_retry(ib_ingestor: IBKRIngestor, contract: Future, tries: int = 4):
    for attempt in range(tries):
        try:
            # Ensure connection is alive before making contract calls
            try:
                ib_ingestor.ensure_connected()
            except Exception:
                time.sleep(1.5 * (attempt + 1))
                continue
            return ib_ingestor.ib.reqContractDetails(contract)
        except Exception as exc:
            if attempt == tries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "error 166" in text:
        return "retention"
    if "error 200" in text or "no security definition" in text:
        return "secdef"
    if "error 162" in text or "error 165" in text or "no data returned" in text:
        return "nodata"
    return "other"


def _expiry_int(contract) -> int:
    exp = getattr(contract, "lastTradeDateOrContractMonth", "") or ""
    if len(exp) == 6:
        exp = exp + "01"
    try:
        return int(exp)
    except ValueError:
        return 99991231


def _load_contract_chain(ib_ingestor: IBKRIngestor, root: str, exch: str) -> List[Future]:
    chain = _CONTRACT_CHAINS.get(root)
    if chain is not None:
        return chain

    symbol_override = ROOT_TO_SYMBOL_OVERRIDE.get(root, root)
    base_contract = Future(
        symbol=symbol_override,
        exchange=exch,
        tradingClass=root,
        includeExpired=True,
    )
    details = _req_cd_with_retry(ib_ingestor, base_contract)
    if not details:
        _CONTRACT_CHAINS[root] = []
        return []

    chain = []
    for cd in details:
        contract = cd.contract
        try:
            contract.includeExpired = True
        except Exception:
            pass
        chain.append(contract)
    chain.sort(key=_expiry_int)
    _CONTRACT_CHAINS[root] = chain
    return chain


def _resolve_front_month_contract(ib_ingestor: IBKRIngestor, symbol: str, asof: datetime):
    """Resolve the front-month futures contract for a symbol as of *asof*."""

    cache_key = (symbol, asof.replace(minute=0, second=0, microsecond=0))
    if cache_key in _CONTRACT_CACHE:
        return _CONTRACT_CACHE[cache_key]

    root = ALIAS_TO_ROOT.get(symbol, symbol)
    exch = ROOT_TO_EXCH.get(root)
    if exch is None:
        return None

    chain = _load_contract_chain(ib_ingestor, root, exch)
    if not chain:
        _CONTRACT_CACHE[cache_key] = None
        return None

    asof_key = int(asof.strftime("%Y%m%d"))
    selected = None
    for contract in chain:
        if _expiry_int(contract) >= asof_key:
            selected = contract
            break
    if selected is None:
        selected = chain[-1]

    _CONTRACT_CACHE[cache_key] = selected
    return selected


def _parse_last_trade(contract) -> Optional[datetime]:
    lt = getattr(contract, "lastTradeDateOrContractMonth", None)
    if not lt:
        return None
    lt = lt.strip()
    try:
        if len(lt) == 8:
            return datetime.strptime(lt, "%Y%m%d").replace(tzinfo=timezone.utc)
        if len(lt) == 6:
            return datetime.strptime(lt + "01", "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return None


def _maybe_roll_contract(ib_ingestor: IBKRIngestor, symbol: str, contract, asof: datetime):
    last_trade = _parse_last_trade(contract)
    if not last_trade:
        return contract
    if asof.replace(tzinfo=timezone.utc) >= last_trade - timedelta(days=ROLLOVER_BUFFER_DAYS):
        root = ALIAS_TO_ROOT.get(symbol, symbol)
        exch = ROOT_TO_EXCH.get(root)
        chain = _load_contract_chain(ib_ingestor, root, exch)
        if chain:
            current_id = getattr(contract, "conId", None)
            for idx, c in enumerate(chain):
                if getattr(c, "conId", None) == current_id:
                    if idx > 0:
                        return chain[idx - 1]
                    break
        alt_asof = asof - timedelta(days=30)
        alt_contract = _resolve_front_month_contract(ib_ingestor, symbol, alt_asof)
        if alt_contract and alt_contract.conId != getattr(contract, "conId", None):
            return alt_contract
    return contract


def _previous_contract(ib_ingestor: IBKRIngestor, symbol: str, contract) -> Optional[Future]:
    if contract is None:
        return None
    root = ALIAS_TO_ROOT.get(symbol, symbol)
    exch = ROOT_TO_EXCH.get(root)
    chain = _load_contract_chain(ib_ingestor, root, exch)
    if not chain:
        return None
    current_id = getattr(contract, "conId", None)
    for idx, c in enumerate(chain):
        if getattr(c, "conId", None) == current_id:
            if idx > 0:
                prev = chain[idx - 1]
                try:
                    prev.includeExpired = True
                except Exception:
                    pass
                return prev
            break
    return None


def _probe_head_timestamp(ib_ingestor: IBKRIngestor, contract) -> Optional[pd.Timestamp]:
    """Return earliest available timestamp for the contract via reqHeadTimeStamp."""

    try:
        try:
            ib_ingestor.ensure_connected()
        except Exception:
            return None
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
    # Compute a midnight-aligned boundary target with optional pad for backward phase
    base = now_utc - timedelta(days=LOOKBACK_DAYS)
    boundary_midnight = base.replace(hour=0, minute=0, second=0, microsecond=0)
    boundary_with_pad = boundary_midnight - timedelta(days=BOUNDARY_PAD_DAYS)
    start_utc = base  # forward phase still resumes from latest below; backward uses boundary_with_pad

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

    def forward_fill(from_dt: datetime):
        cur = from_dt
        print(f"[{symbol}] forward fill start={cur.isoformat()} end={now_utc.isoformat()}")
        while cur < now_utc:
            nxt = min(cur + timedelta(days=WINDOW_DAYS), now_utc)

            contract = None
            if os.getenv("USE_HEAD_TS", "1") == "1":
                contract = _resolve_front_month_contract(ib, symbol, nxt)
                if contract is not None:
                    contract = _maybe_roll_contract(ib, symbol, contract, nxt)
                    _ear = _probe_head_timestamp(ib, contract)
                    if _ear is not None:
                        earliest_dt = _ear.to_pydatetime()
                        if earliest_dt > cur:
                            cur = earliest_dt
                            if cur >= nxt:
                                continue

            # Bars-first guard, then days cap
            minutes = max(1, int((nxt - cur).total_seconds() / 60))
            if minutes > MAX_BARS_PER_REQUEST:
                nxt = cur + timedelta(minutes=MAX_BARS_PER_REQUEST)
            elif nxt - cur > timedelta(days=MAX_DAYS_PER_REQ):
                nxt = cur + timedelta(days=MAX_DAYS_PER_REQ)
            if nxt > now_utc:
                nxt = now_utc

            duration_minutes = max(1, int((nxt - cur).total_seconds() / 60))
            duration_days = max(1, (nxt - cur).days or (duration_minutes + 1439) // 1440)
            duration = f"{duration_days} D"
            try:
                _enforce_pacing(symbol, duration, nxt)
                end_utc_str = nxt.strftime("%Y%m%d-%H:%M:%S")
                df = ib.fetch_data(
                    symbol,
                    duration=duration,
                    barSize="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                    endDateTime=end_utc_str,
                    asof=nxt,
                    contract_override=contract,
                )
                _record_request(symbol, duration, nxt)
                if df is not None and not df.empty:
                    out_path = persist_bars(symbol, df)
                    print(f"  [OK] {symbol} {cur} -> {nxt} wrote={out_path}")
                else:
                    print(f"  [OK] {symbol} {cur} -> {nxt} (no rows)")
            except Exception as exc:
                kind = _classify_error(exc)
                if kind == "retention":
                    print(
                        f"  [STOP] {symbol} {cur} -> {nxt}: IBKR retention limit reached ({exc}); stopping forward fill."
                    )
                    break
                if kind == "secdef":
                    try:
                        alt_asof = nxt - timedelta(days=1)
                        df = ib.fetch_data(
                            symbol,
                            duration=duration,
                            barSize="1 min",
                            whatToShow="TRADES",
                            useRTH=False,
                            formatDate=1,
                            endDateTime=end_utc_str,
                            asof=alt_asof,
                        )
                        _record_request(symbol, duration, nxt)
                        if df is not None and not df.empty:
                            out_path = persist_bars(symbol, df)
                            print(
                                f"  [OK] {symbol} {cur} -> {nxt} (retry after error 200; asof={alt_asof:%Y-%m-%d}) wrote={out_path}"
                            )
                            cur = nxt
                            time.sleep(SLEEP_BETWEEN_REQ)
                            continue
                    except Exception as exc2:
                        print(f"  [WARN] {symbol} secdef-retry failed: {exc2}")
                print(f"  [WARN] {symbol} {cur} -> {nxt}: {exc}")
            cur = nxt
            time.sleep(SLEEP_BETWEEN_REQ)

    def backward_fill(to_dt: datetime):
        target_start = to_dt
        earliest_local = _earliest_timestamp(symbol)
        if earliest_local is None:
            return
        if earliest_local - target_start <= timedelta(minutes=1):
            return
        print(f"[{symbol}] backward fill target={target_start.isoformat()} current_earliest={earliest_local.isoformat()}")
        end_point = earliest_local
        prev_end_point = None
        contract_override = None
        while end_point - target_start > timedelta(minutes=1):
            nxt_end = end_point
            # Start with a generous window, then clamp by bars then by days
            start = max(target_start, nxt_end - timedelta(days=MAX_DAYS_PER_REQ))
            minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            if minutes > MAX_BARS_PER_REQUEST:
                start = nxt_end - timedelta(minutes=MAX_BARS_PER_REQUEST)
                minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            elif nxt_end - start > timedelta(days=MAX_DAYS_PER_REQ):
                start = nxt_end - timedelta(days=MAX_DAYS_PER_REQ)
                minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            duration_minutes = minutes
            duration_days = max(1, (nxt_end - start).days or (duration_minutes + 1439) // 1440)
            duration = f"{duration_days} D"

            contract = contract_override
            if contract is None:
                contract = _resolve_front_month_contract(ib, symbol, nxt_end)
                if contract is not None:
                    contract = _maybe_roll_contract(ib, symbol, contract, nxt_end)

            earliest_available = None
            if os.getenv("USE_HEAD_TS", "1") == "1" and contract is not None:
                earliest_available = _probe_head_timestamp(ib, contract)
                if earliest_available is not None:
                    ea_dt = earliest_available.to_pydatetime()
                    if ea_dt > start:
                        if nxt_end - ea_dt <= timedelta(minutes=1):
                            previous = _previous_contract(ib, symbol, contract)
                            if previous is not None:
                                contract_override = previous
                                prev_end_point = nxt_end
                                continue
                            candidate_end = ea_dt - timedelta(days=1, minutes=1)
                            if candidate_end <= target_start:
                                print(f"  [STOP] {symbol} reached HMDS floor at {ea_dt}; ending backward fill.")
                                break
                            if prev_end_point is not None and candidate_end >= nxt_end - timedelta(minutes=1):
                                candidate_end = nxt_end - timedelta(days=1)
                            end_point = candidate_end
                            prev_end_point = nxt_end
                            continue
                        start = max(start, ea_dt)

            if prev_end_point is not None and start >= prev_end_point - timedelta(minutes=1):
                print(f"  [STOP] {symbol} backward fill detected no progress (start {start}); ending backward fill.")
                break

            if contract is None:
                print(f"  [WARN] {symbol} backward {start} -> {nxt_end}: unable to resolve contract")
                break

            try:
                _enforce_pacing(symbol, duration, nxt_end)
                end_utc_str = nxt_end.strftime("%Y%m%d-%H:%M:%S")
                df = ib.fetch_data(
                    symbol,
                    duration=duration,
                    barSize="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                    endDateTime=end_utc_str,
                    asof=nxt_end,
                    contract_override=contract,
                )
                _record_request(symbol, duration, nxt_end)
                if df is not None and not df.empty:
                    out_path = persist_bars(symbol, df)
                    print(f"  [OK] {symbol} backward {start} -> {nxt_end} wrote={out_path}")
                else:
                    print(f"  [OK] {symbol} backward {start} -> {nxt_end} (no rows)")
            except Exception as exc:
                kind = _classify_error(exc)
                if kind == "retention":
                    print(f"  [STOP] {symbol} backward {start} -> {nxt_end}: IBKR retention limit reached ({exc}); stopping backward fill.")
                    break
                if kind == "secdef":
                    try:
                        alt_asof = nxt_end - timedelta(days=1)
                        df = ib.fetch_data(
                            symbol,
                            duration=duration,
                            barSize="1 min",
                            whatToShow="TRADES",
                            useRTH=False,
                            formatDate=1,
                            endDateTime=end_utc_str,
                            asof=alt_asof,
                        )
                        _record_request(symbol, duration, nxt_end)
                        if df is not None and not df.empty:
                            out_path = persist_bars(symbol, df)
                            print(f"  [OK] {symbol} backward {start} -> {nxt_end} (retry after error 200; asof={alt_asof:%Y-%m-%d}) wrote={out_path}")
                            prev_end_point = end_point
                            end_point = start
                            contract_override = contract
                            time.sleep(SLEEP_BETWEEN_REQ)
                            continue
                    except Exception as exc2:
                        print(f"  [WARN] {symbol} backward secdef-retry failed: {exc2}")
                if kind in {"nodata", "secdef"}:
                    previous = _previous_contract(ib, symbol, contract)
                    if previous is not None:
                        contract_override = previous
                        print(f"  [INFO] {symbol} switching to prior contract conId={previous.conId} after {kind}")
                        prev_end_point = nxt_end
                        continue
                print(f"  [WARN] {symbol} backward {start} -> {nxt_end}: {exc}")
            prev_end_point = end_point
            end_point = start
            contract_override = contract
            time.sleep(SLEEP_BETWEEN_REQ)

    # Choose fill order
    if BACKFILL_MODE in ("backward_first", "backward_then_forward", "backward"):
        # Backward fill to historical target, then forward to now
        if earliest is None:
            earliest = _earliest_timestamp(symbol)
        if earliest is not None:
            backward_fill(boundary_with_pad)
        forward_fill(max(_latest_timestamp(symbol) or start_utc, start_utc))
    else:
        # Forward to now, then backward fill leftovers
        forward_fill(start_utc)
        backward_fill(start_utc)

    target_start = now_utc - timedelta(days=LOOKBACK_DAYS)
    if earliest is None:
        earliest = _earliest_timestamp(symbol)

    if earliest is not None and earliest - target_start > timedelta(minutes=1):
        print(f"[{symbol}] backward fill target={target_start.isoformat()} current_earliest={earliest.isoformat()}")
        end_point = earliest
        prev_end_point = None
        contract_override = None
        while end_point - target_start > timedelta(minutes=1):
            nxt_end = end_point
            start = max(target_start, nxt_end - timedelta(days=1))
            minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            if minutes > MAX_BARS_PER_REQUEST:
                start = nxt_end - timedelta(minutes=MAX_BARS_PER_REQUEST)
            duration_minutes = max(1, int((nxt_end - start).total_seconds() / 60))
            duration_days = max(1, (nxt_end - start).days or (duration_minutes + 1439) // 1440)
            duration = f"{duration_days} D"

            contract = contract_override
            if contract is None:
                contract = _resolve_front_month_contract(ib, symbol, nxt_end)
                if contract is not None:
                    contract = _maybe_roll_contract(ib, symbol, contract, nxt_end)

            earliest_available = None
            if os.getenv("USE_HEAD_TS", "1") == "1" and contract is not None:
                earliest_available = _probe_head_timestamp(ib, contract)
                if earliest_available is not None:
                    ea_dt = earliest_available.to_pydatetime()
                    if ea_dt > start:
                        if nxt_end - ea_dt <= timedelta(minutes=1):
                            previous = _previous_contract(ib, symbol, contract)
                            if previous is not None:
                                contract_override = previous
                                prev_end_point = nxt_end
                                continue
                            candidate_end = ea_dt - timedelta(days=1, minutes=1)
                            if candidate_end <= target_start:
                                print(
                                    f"  [STOP] {symbol} reached HMDS floor at {ea_dt}; ending backward fill."
                                )
                                break
                            if prev_end_point is not None and candidate_end >= nxt_end - timedelta(minutes=1):
                                candidate_end = nxt_end - timedelta(days=1)
                            end_point = candidate_end
                            prev_end_point = nxt_end
                            continue
                        start = max(start, ea_dt)

            if prev_end_point is not None and start >= prev_end_point - timedelta(minutes=1):
                print(
                    f"  [STOP] {symbol} backward fill detected no progress (start {start}); ending backward fill."
                )
                break

            if contract is None:
                print(f"  [WARN] {symbol} backward {start} -> {nxt_end}: unable to resolve contract")
                break

            try:
                _enforce_pacing(symbol, duration, nxt_end)
                end_utc_str = nxt_end.strftime("%Y%m%d-%H:%M:%S")
                df = ib.fetch_data(
                    symbol,
                    duration=duration,
                    barSize="1 min",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1,
                    endDateTime=end_utc_str,
                    asof=nxt_end,
                    contract_override=contract,
                )
                _record_request(symbol, duration, nxt_end)
                if df is not None and not df.empty:
                    out_path = persist_bars(symbol, df)
                    print(f"  [OK] {symbol} backward {start} -> {nxt_end} wrote={out_path}")
                else:
                    print(f"  [OK] {symbol} backward {start} -> {nxt_end} (no rows)")
            except Exception as exc:
                kind = _classify_error(exc)
                if kind == "retention":
                    print(
                        f"  [STOP] {symbol} backward {start} -> {nxt_end}: IBKR retention limit reached ({exc}); stopping backward fill."
                    )
                    break
                if kind == "secdef":
                    # Re-resolve with a slightly earlier as-of to avoid ambiguous/underspecified contracts
                    try:
                        alt_asof = nxt_end - timedelta(days=1)
                        df = ib.fetch_data(
                            symbol,
                            duration=duration,
                            barSize="1 min",
                            whatToShow="TRADES",
                            useRTH=False,
                            formatDate=1,
                            endDateTime=end_utc_str,
                            asof=alt_asof,
                        )
                        _record_request(symbol, duration, nxt_end)
                        if df is not None and not df.empty:
                            out_path = persist_bars(symbol, df)
                            print(
                                f"  [OK] {symbol} backward {start} -> {nxt_end} (retry after error 200; asof={alt_asof:%Y-%m-%d}) wrote={out_path}"
                            )
                            prev_end_point = end_point
                            end_point = start
                            contract_override = contract
                            time.sleep(SLEEP_BETWEEN_REQ)
                            continue
                    except Exception as exc2:
                        print(f"  [WARN] {symbol} backward secdef-retry failed: {exc2}")
                if kind in {"nodata", "secdef"}:
                    previous = _previous_contract(ib, symbol, contract)
                    if previous is not None:
                        contract_override = previous
                        print(
                            f"  [INFO] {symbol} switching to prior contract conId={previous.conId} after {kind}"
                        )
                        prev_end_point = nxt_end
                        continue
                print(f"  [WARN] {symbol} backward {start} -> {nxt_end}: {exc}")
            prev_end_point = end_point
            end_point = start
            contract_override = contract
            time.sleep(SLEEP_BETWEEN_REQ)


def main() -> None:
    ib = IBKRIngestor(
        host=os.getenv("IBKR_HOST", "127.0.0.1"),
        port=int(os.getenv("IBKR_PORT", "4002")),
        clientId=int(os.getenv("IBKR_CLIENT_ID", "9003")),
    )

    try:
        ib_request_timeout = int(os.getenv("IBKR_REQUEST_TIMEOUT", "120"))
    except ValueError:
        ib_request_timeout = 120
    ib.ib.RequestTimeout = ib_request_timeout

    for symbol in IBKR_SYMBOLS:
        _backfill_symbol(ib, symbol)


if __name__ == "__main__":
    main()
