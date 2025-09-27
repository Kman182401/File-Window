#!/usr/bin/env python3
"""Enrich legacy IB minute bars with contract metadata and rewrite as partitioned Parquet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Reuse the project’s existing contract helpers for perfect consistency.
# ---------------------------------------------------------------------------
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.backfill_ibkr_history import (  # noqa: E402
    ALIAS_TO_ROOT,
    ROOT_TO_EXCH,
    _load_contract_chain,
    _parse_last_trade,
    _probe_head_timestamp,
    IBKRIngestor,
)
from market_data_config import IBKR_SYMBOLS  # noqa: E402

UTC = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass
class ContractWindow:
    symbol_root: str
    exchange: str
    conId: int
    expiry: str
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp


def build_contract_windows(ingestor: IBKRIngestor, symbol_alias: str) -> pd.DataFrame:
    """Return contract metadata windows for a symbol alias as a DataFrame."""
    root = ALIAS_TO_ROOT.get(symbol_alias, symbol_alias)
    exch = ROOT_TO_EXCH[root]
    chain = _load_contract_chain(ingestor, root, exch)

    windows: List[ContractWindow] = []
    for contract in chain:
        con_id = getattr(contract, "conId", None)
        if con_id is None:
            continue

        expiry_dt = _parse_last_trade(contract, ingestor)
        if expiry_dt is None:
            continue

        lt_raw = (getattr(contract, "lastTradeDateOrContractMonth", "") or "").strip()
        if len(lt_raw) == 6:
            lt_raw = f"{lt_raw}01"
        if len(lt_raw) != 8:
            lt_raw = expiry_dt.strftime("%Y%m%d")

        head_ts = _probe_head_timestamp(ingestor, contract)
        start_ts = head_ts.to_pydatetime() if head_ts is not None else None
        start_ts = pd.Timestamp(start_ts, tz="UTC") if start_ts else UTC
        end_ts = pd.Timestamp(expiry_dt, tz="UTC")

        windows.append(
            ContractWindow(
                symbol_root=root,
                exchange=exch,
                conId=con_id,
                expiry=lt_raw,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        )

    if not windows:
        return pd.DataFrame(columns=["symbol_root", "exchange", "conId", "expiry", "start_ts", "end_ts"])

    windows_df = pd.DataFrame([w.__dict__ for w in windows]).sort_values("start_ts").reset_index(drop=True)
    return windows_df


def attach_metadata(df: pd.DataFrame, windows_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised assignment of contract metadata based on timestamp intervals."""
    if windows_df.empty:
        meta = pd.DataFrame(
            {
                "symbol_root": pd.Series([pd.NA] * len(df), index=df.index, dtype="string"),
                "exchange": pd.Series([pd.NA] * len(df), index=df.index, dtype="string"),
                "conId": pd.Series([pd.NA] * len(df), index=df.index, dtype="float64"),
                "expiry": pd.Series([pd.NA] * len(df), index=df.index, dtype="string"),
            }
        )
        return meta

    # merge_asof requires sorted inputs
    ts_sorted = df[["timestamp"]].copy().sort_values("timestamp")
    merged = pd.merge_asof(
        ts_sorted,
        windows_df,
        left_on="timestamp",
        right_on="start_ts",
        direction="backward",
    )

    # Re-align to original index order
    merged = merged.set_index(ts_sorted.index).reindex(df.index)

    # Mask out timestamps falling after the contract’s last trade
    valid = merged["end_ts"].notna() & (df["timestamp"] <= merged["end_ts"])
    merged.loc[~valid, ["symbol_root", "exchange", "conId", "expiry"]] = pd.NA

    meta = merged[["symbol_root", "exchange", "conId", "expiry"]].copy()
    meta["conId"] = meta["conId"].astype("Int64")
    return meta


def read_legacy_file(path: Path) -> pd.DataFrame:
    """Load a legacy parquet/csv file and normalise columns."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if "average" in df.columns and "wap" not in df.columns:
        df = df.rename(columns={"average": "wap"})
    if "wap" not in df.columns:
        df["wap"] = pd.NA

    expected = ["timestamp", "open", "high", "low", "close", "volume", "barCount", "wap"]
    for col in expected:
        if col not in df.columns:
            df[col] = pd.NA
        elif col != "timestamp":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[expected]
    return df


def migrate_symbol(legacy_root: Path, output_root: Path, ingestor: IBKRIngestor, symbol: str) -> None:
    symbol_dir = legacy_root / f"symbol={symbol}"
    if not symbol_dir.exists():
        print(f"[SKIP] {symbol}: no legacy directory")
        return

    windows_df = build_contract_windows(ingestor, symbol)
    files = sorted(symbol_dir.glob("**/bars.parquet")) + sorted(symbol_dir.glob("**/bars.csv"))
    if not files:
        print(f"[WARN] {symbol}: no legacy files found")
        return

    enriched_frames: List[pd.DataFrame] = []
    for path in tqdm(files, desc=f"{symbol} files"):
        df = read_legacy_file(path)
        if df.empty:
            continue
        meta = attach_metadata(df, windows_df)
        enriched_frames.append(pd.concat([df, meta], axis=1))

    if not enriched_frames:
        print(f"[WARN] {symbol}: nothing to rewrite")
        return

    merged = (
        pd.concat(enriched_frames, ignore_index=True)
        .drop_duplicates(subset=["timestamp", "conId"], keep="last")
        .sort_values("timestamp")
    )

    merged["year"] = merged["timestamp"].dt.year.astype("Int64")
    merged["month"] = merged["timestamp"].dt.month.astype("Int64")

    table = pa.Table.from_pandas(merged, preserve_index=False)
    root = output_root / "minute_bars"
    root.mkdir(parents=True, exist_ok=True)

    # Remove pre-existing partitions we are about to rewrite so reruns stay clean
    for key in merged[["symbol_root", "year", "month", "expiry"]].drop_duplicates().itertuples(index=False):
        part_path = root
        part_path /= f"symbol_root={key.symbol_root}"
        part_path /= f"year={key.year}"
        part_path /= f"month={key.month}"
        part_path /= f"expiry={key.expiry}"
        if part_path.exists():
            shutil.rmtree(part_path)

    pq.write_to_dataset(
        table,
        root_path=str(root),
        partition_cols=["symbol_root", "year", "month", "expiry"],
        existing_data_behavior="overwrite_or_ignore",
    )
    print(f"[DONE] {symbol}: wrote {len(merged)} rows under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add contract metadata and rewrite bars.")
    parser.add_argument("--legacy-root", default="~/.local/share/m5_trader/data", type=Path)
    parser.add_argument("--output-root", default="~/data/ibkr_partitioned", type=Path)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--client-id", default=9010, type=int)
    args = parser.parse_args()

    legacy_root = args.legacy_root.expanduser()
    output_root = args.output_root.expanduser()

    ib = IBKRIngestor(host=args.host, port=args.port, clientId=args.client_id)
    try:
        for symbol in IBKR_SYMBOLS:
            migrate_symbol(legacy_root, output_root, ib, symbol)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
