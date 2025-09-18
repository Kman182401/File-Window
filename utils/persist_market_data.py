import os, io, gzip, datetime as dt
import pandas as pd
try:
    import boto3  # optional if S3_ENABLE=0
except Exception:  # pragma: no cover
    boto3 = None

S3_ENABLE = int(os.getenv("S3_ENABLE", "0"))
S3_BUCKET = os.getenv("S3_BUCKET", "omega-singularity-ml")
S3_PREFIX = os.getenv("S3_PREFIX", "market_data")
LOCAL_DIR = os.path.expanduser(os.getenv("LOCAL_FALLBACK_DIR", "~/data/market_data"))
from utils.io_config import partition_path
from utils.io_parquet import write_parquet_any

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly normalize a DataFrame to a UTC DatetimeIndex named "timestamp".
    Accepts either:
      - a DatetimeIndex (tz-naive or tz-aware), or
      - a "timestamp" column (string/TS), or
      - a "date" column (string/TS)
    """
    # Case 1: already has a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        out = df.copy()
        out.index = idx
        out.index.name = out.index.name or "timestamp"
        return out

    # Case 2: has a timestamp/date column
    ts_col = None
    for c in ("timestamp","date","Datetime","Date","time"):
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("No DatetimeIndex and no timestamp-like column found")

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out.dropna(subset=[ts_col])
    out = out.set_index(ts_col)
    out.index.name = "timestamp"
    return out

def _csv_gz_bytes(df: pd.DataFrame) -> bytes:
    bio = io.StringIO(); df.to_csv(bio, index=True)
    return gzip.compress(bio.getvalue().encode("utf-8"))

def _s3_key(symbol: str, when_utc: dt.datetime) -> str:
    d = when_utc.date().isoformat()
    ts = when_utc.strftime("%Y-%m-%dT%H-%M-%S.%fZ")
    return f"{S3_PREFIX}/symbol={symbol}/date={d}/{symbol}_{ts}.csv.gz"

def persist_bars(symbol: str, df: pd.DataFrame) -> str:
    """
    Persist bars to local Parquet under env-first partition layout:
      $DATA_DIR/symbol=<symbol>/date=<YYYY-MM-DD>/bars.parquet

    Optional: if OUTPUT_FORMAT=csv, write gzip CSV instead for backwards compatibility.
    S3 uploads are disabled by default (S3_ENABLE=0).
    """
    if df is None or df.empty:
        return ""

    # Normalize to UTC index named 'timestamp'
    df = _to_utc_index(df)

    # Column ordering: OHLCV first if present
    cols = [c for c in ["open", "high", "low", "close", "volume", "wap", "barCount"] if c in df.columns]
    if cols and list(df.columns) != cols:
        df = df[cols + [c for c in df.columns if c not in cols]]

    # Choose partition date from the last bar's date (UTC)
    try:
        date_str = df.index[-1].date().isoformat()
    except Exception:
        date_str = dt.datetime.now(dt.timezone.utc).date().isoformat()

    # Build path and write according to OUTPUT_FORMAT
    out_path = partition_path(symbol, date_str, "bars", ext="parquet")
    fmt = os.getenv("OUTPUT_FORMAT", "parquet").lower()
    if fmt == "csv":
        csvp = str(out_path).replace(".parquet", ".csv.gz")
        os.makedirs(os.path.dirname(csvp), exist_ok=True)
        # reset index so 'timestamp' becomes a column
        df.reset_index().to_csv(csvp, index=False, compression="gzip")
        return csvp

    # Default: Parquet via pyarrow with on-write de-duplication by timestamp
    try:
        from pathlib import Path
        df_new = df.reset_index()
        p = Path(out_path)
        if p.exists():
            try:
                df_old = pd.read_parquet(p)
                combined = pd.concat([df_old, df_new], ignore_index=True)
                combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
                combined.to_parquet(p, index=False)
            except Exception:
                # Fallback to simple write
                write_parquet_any(df_new, out_path)
        else:
            write_parquet_any(df_new, out_path)
    except Exception:
        # Last-resort fallback: simple write
        write_parquet_any(df.reset_index(), out_path)

    # Optional S3 upload (disabled by default)
    if S3_ENABLE and boto3 is not None:
        try:
            rel = f"symbol={symbol}/date={date_str}/bars.parquet"
            key = f"{S3_PREFIX}/{rel}"
            with open(out_path, "rb") as fh:
                boto3.client("s3").put_object(
                    Bucket=S3_BUCKET,
                    Key=key,
                    Body=fh.read(),
                    ContentType="application/octet-stream",
                )
            return f"s3://{S3_BUCKET}/{key}"
        except Exception as e:  # pragma: no cover
            print(f"[persist] s3 put failed: {e}")

    return str(out_path)
