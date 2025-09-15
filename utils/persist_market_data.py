import os, io, gzip, datetime as dt
import pandas as pd
import boto3

S3_ENABLE = int(os.getenv("S3_ENABLE","1"))
S3_BUCKET = os.getenv("S3_BUCKET","omega-singularity-ml")
S3_PREFIX = os.getenv("S3_PREFIX","market_data")
LOCAL_DIR = os.path.expanduser(os.getenv("LOCAL_FALLBACK_DIR","~/data/market_data"))

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
    if df is None or df.empty: return ""
    df = _to_utc_index(df)
    cols = [c for c in ["open","high","low","close","volume","wap","barCount"] if c in df.columns]
    if cols and list(df.columns) != cols:
        df = df[cols + [c for c in df.columns if c not in cols]]
    now_utc = dt.datetime.now(dt.timezone.utc)
    key = _s3_key(symbol, now_utc)
    try:
        if int(os.getenv("S3_ENABLE","1")):
            boto3.client("s3").put_object(
                Bucket=os.getenv("S3_BUCKET","omega-singularity-ml"),
                Key=key,
                Body=_csv_gz_bytes(df),
                ContentType="text/csv",
                ContentEncoding="gzip",
            )
            return f"s3://{os.getenv('S3_BUCKET','omega-singularity-ml')}/{key}"
    except Exception as e:
        print(f"[persist] s3 put failed: {e}")
    # local fallback
    path = os.path.join(LOCAL_DIR, key[len(S3_PREFIX)+1:])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(_csv_gz_bytes(df))
    return path
