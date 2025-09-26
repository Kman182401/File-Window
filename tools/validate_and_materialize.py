#!/usr/bin/env python3
"""Validate cached IBKR bars and materialize continuous feeds."""

import os
import sys
from datetime import datetime, time, timedelta, timezone
from typing import Iterable, List, Tuple

import pandas as pd

DATA_DIR = os.path.expanduser(os.getenv("DATA_DIR", "~/.local/share/m5_trader/data"))
OUT_DIR = os.path.join(DATA_DIR, "ibkr_continuous")


def _parquet_paths(symbol: str) -> Iterable[str]:
    base = os.path.join(DATA_DIR, f"symbol={symbol}")
    if not os.path.isdir(base):
        return []
    for date_dir in sorted(os.listdir(base)):
        parquet_path = os.path.join(base, date_dir, "bars.parquet")
        if os.path.exists(parquet_path):
            yield parquet_path


def _load_symbol(symbol: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in _parquet_paths(symbol):
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            print(f"  [WARN] failed to read {path}: {exc}")
    if not frames:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    cols = [c for c in ["timestamp", "open", "high", "low", "close", "volume", "wap", "barCount"] if c in df.columns]
    return df[cols]


def _is_cme_maintenance(ts: pd.Timestamp) -> bool:
    et_time = ts.tz_convert("America/New_York").time()
    return time(17, 0) <= et_time < time(18, 0)


def _detect_gaps(
    df: pd.DataFrame,
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp, float]], int, int]:
    if df.empty:
        return [], 0, 0
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
    inside = 0
    outside = 0
    timestamps = df["timestamp"].array
    for i in range(1, len(timestamps)):
        prev = timestamps[i - 1]
        cur = timestamps[i]
        delta_minutes = (cur - prev).total_seconds() / 60.0
        if delta_minutes <= 1.01:
            continue
        in_prev = _is_cme_maintenance(prev)
        in_cur = _is_cme_maintenance(cur)
        if in_prev and in_cur:
            inside += 1
            continue
        outside += 1
        gaps.append((prev, cur, delta_minutes))
    return gaps, inside, outside


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: validate_and_materialize.py SYMBOL [SYMBOL ...]")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(days=730)

    for symbol in sys.argv[1:]:
        df = _load_symbol(symbol)
        gaps, maintenance_skips, outside_gaps = _detect_gaps(df)
        if df.empty:
            print(f"[{symbol}] no data found under {DATA_DIR}")
            continue

        first = df["timestamp"].min()
        last = df["timestamp"].max()
        print(
            f"[{symbol}] rows={len(df)} first={first} last={last} gaps_outside={outside_gaps}"
            f" gaps_inside_maintenance={maintenance_skips}"
        )
        for prev, cur, mins in gaps[:20]:
            print(f"  GAP {mins:.1f} min: {prev} -> {cur}")

        if first is not pd.NaT and first > cutoff:
            delta_days = (cutoff - first).days
            if delta_days < 0:
                print(
                    "  [WARN] Earliest cached data is newer than the ~2 year IBKR retention window;"
                    " older gaps may reflect provider limits."
                )

        out_path = os.path.join(OUT_DIR, f"{symbol}_continuous_1min.csv")
        df.to_csv(out_path, index=False)
        print(f"  [WRITE] {out_path}")


if __name__ == "__main__":
    main()
