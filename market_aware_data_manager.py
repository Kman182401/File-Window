#!/usr/bin/env python3
"""
Market-Aware Data Manager

Routes data requests between live IBKR and historical (S3/local) based on
market hours, with optional blending around open/close transitions.

Environment flags (precedence high→low):
  - FORCE_LIVE_DATA=1         Always use live IBKR
  - FORCE_HISTORICAL_DATA=1   Always use historical fallback
  - MARKET_AWARE_MODE=1       Auto-select by market hours (default)

Other knobs:
  - MARKET_WINDOW_MINS       Target bar window in minutes (default 360)
  - FALLBACK_MINUTES         Extra minutes to pull from fallback for blending (default 120)
  - USE_RTH                  Not used (futures use extended hours); defaults False

Returns pandas DataFrames with a UTC DatetimeIndex named 'timestamp'.
"""

from __future__ import annotations

import io
import os
import gzip
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List

import pandas as pd
import pytz

from market_hours_detector import MarketHoursDetector
import logging


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
        out.index = idx
        out.index.name = out.index.name or 'timestamp'
        return out.sort_index()
    # otherwise try common columns
    for c in ("timestamp","date","Datetime","Date","time"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], utc=True, errors='coerce')
            out = out.dropna(subset=[c]).set_index(c)
            out.index.name = 'timestamp'
            return out.sort_index()
    return pd.DataFrame()


def _dedupe_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    parts = [d for d in dfs if d is not None and not d.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=0, ignore_index=False)
    out = _to_utc_index(out)
    out = out[~out.index.duplicated(keep='last')].sort_index()
    return out


class MarketAwareDataManager:
    def __init__(self, ingestor, detector: Optional[MarketHoursDetector] = None):
        self.ingestor = ingestor  # Expected to expose fetch_data(symbol, duration, barSize, ...)
        self.detector = detector or MarketHoursDetector()

        # Env flags and defaults
        self.market_aware = os.getenv('MARKET_AWARE_MODE', '1') not in ('0', 'false', 'False')
        self.force_live = os.getenv('FORCE_LIVE_DATA', '0') in ('1', 'true', 'True')
        self.force_hist = os.getenv('FORCE_HISTORICAL_DATA', '0') in ('1', 'true', 'True')
        self.window_mins = int(os.getenv('MARKET_WINDOW_MINS', '360'))
        self.fallback_mins = int(os.getenv('FALLBACK_MINUTES', '120'))

    # Pass-through to underlying IB adapter when needed (e.g., health/keepalive utilities)
    @property
    def ib(self):
        return getattr(self.ingestor, 'ib', None)

    # Public API mirrors adapter
    def fetch_data(self, symbol: str) -> pd.DataFrame:
        # Decide route
        route, reason = self._decide_route(symbol)

        if route == 'live':
            live_df = self._fetch_live(symbol, self.window_mins)
            # Optional blend at open: add small fallback tail if available
            if self.fallback_mins > 0:
                hist_tail = self._load_historical(symbol, self.fallback_mins)
                blended = _dedupe_concat([hist_tail, live_df])
                out = self._trim_window(blended, self.window_mins)
            else:
                out = self._trim_window(live_df, self.window_mins)
            try:
                logging.info(
                    f"[data-route] symbol={symbol} source=live reason={reason} "
                    f"bars={len(out)} req_window_mins={self.window_mins} tail_blend_mins={self.fallback_mins}"
                )
            except Exception:
                pass
            return out

        # historical route
        hist_df = self._load_historical(symbol, self.window_mins + self.fallback_mins)
        out = self._trim_window(hist_df, self.window_mins)
        try:
            logging.info(
                f"[data-route] symbol={symbol} source=historical reason={reason} "
                f"bars={len(out)} req_window_mins={self.window_mins}"
            )
        except Exception:
            pass
        return out

    # Internals
    def _decide_route(self, symbol: str) -> Tuple[str, str]:
        # Overrides first
        if self.force_live and not self.force_hist:
            return 'live', 'FORCE_LIVE_DATA'
        if self.force_hist and not self.force_live:
            return 'historical', 'FORCE_HISTORICAL_DATA'

        if not self.market_aware:
            return 'live', 'MARKET_AWARE_MODE=0'

        # Symbol group mapping → futures
        market_name = 'US_FUTURES'
        state, next_at = self.detector.get_market_state(market_name)
        if state == 'open':
            return 'live', 'market open'
        elif state == 'maintenance':
            return 'historical', 'maintenance window'
        else:
            return 'historical', 'market closed'

    def _fetch_live(self, symbol: str, window_mins: int) -> pd.DataFrame:
        # Convert minutes to IB duration string (coarse)
        # Use days for larger windows to reduce rate limits
        # Use day-based duration to comply with IBKR API (avoid minute string ambiguity)
        days = max(1, (window_mins + 1439) // 1440)
        duration = f"{days} D"
        try:
            df = self.ingestor.fetch_data(
                symbol,
                duration=duration,
                barSize="1 min",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                endDateTime="",
                asof=None,
            )
            return _to_utc_index(df)
        except Exception:
            # Fail-closed to historical path
            return pd.DataFrame()

    def _trim_window(self, df: pd.DataFrame, window_mins: int) -> pd.DataFrame:
        df = _to_utc_index(df)
        if df.empty:
            return df
        end = df.index.max()
        start = end - pd.Timedelta(minutes=window_mins)
        return df.loc[df.index >= start]

    def _load_historical(self, symbol: str, window_mins: int) -> pd.DataFrame:
        # Try local training cache first
        local = self._load_local_cache(symbol)
        local = self._trim_window(local, window_mins)
        if not local.empty:
            return local

        # Optionally load from S3 recent objects
        if os.getenv('S3_ENABLE', '1') in ('1', 'true', 'True'):
            s3_df = self._load_s3_recent(symbol, max_objects=20)
            return self._trim_window(s3_df, window_mins)
        return pd.DataFrame()

    def _load_local_cache(self, symbol: str) -> pd.DataFrame:
        try:
            base = os.path.expanduser('~/training_data')
            sym = symbol.replace('!', '')
            path = os.path.join(base, f"{sym}_training_data.parquet")
            if os.path.exists(path):
                df = pd.read_parquet(path)
                return _to_utc_index(df)
        except Exception:
            pass
        # Also try local persisted gz (latest few)
        try:
            local_dir = os.path.expanduser(os.getenv('LOCAL_FALLBACK_DIR', '~/data/market_data'))
            sym_dir = os.path.join(local_dir, f"symbol={symbol}")
            if not os.path.isdir(sym_dir):
                return pd.DataFrame()
            # Walk dates descending, pick latest N files
            dates = sorted((d for d in os.listdir(sym_dir)), reverse=True)
            frames: List[pd.DataFrame] = []
            count = 0
            for d in dates:
                date_path = os.path.join(sym_dir, d)
                if not os.path.isdir(date_path):
                    continue
                files = sorted(os.listdir(date_path), reverse=True)
                for fn in files:
                    if not fn.endswith('.csv.gz'):
                        continue
                    full = os.path.join(date_path, fn)
                    with gzip.open(full, 'rb') as f:
                        df = pd.read_csv(io.BytesIO(f.read()))
                        frames.append(df)
                        count += 1
                    if count >= 10:
                        break
                if count >= 10:
                    break
            return _dedupe_concat(frames)
        except Exception:
            return pd.DataFrame()

    def _load_s3_recent(self, symbol: str, max_objects: int = 20) -> pd.DataFrame:
        try:
            import boto3
            s3 = boto3.client('s3')
            bucket = os.getenv('S3_BUCKET', 'omega-singularity-ml')
            prefix = os.getenv('S3_PREFIX', 'market_data')
            pfx = f"{prefix}/symbol={symbol}/"
            token = None
            objs = []
            while True:
                kwargs = dict(Bucket=bucket, Prefix=pfx, MaxKeys=1000)
                if token:
                    kwargs['ContinuationToken'] = token
                resp = s3.list_objects_v2(**kwargs)
                for o in resp.get('Contents', []):
                    if o['Key'].endswith('.csv.gz'):
                        objs.append(o)
                if not resp.get('IsTruncated'):
                    break
                token = resp.get('NextContinuationToken')
            # Sort by LastModified desc and take top N
            objs.sort(key=lambda o: o['LastModified'], reverse=True)
            frames: List[pd.DataFrame] = []
            for o in objs[:max_objects]:
                body = s3.get_object(Bucket=bucket, Key=o['Key'])['Body'].read()
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as gz:
                    df = pd.read_csv(gz)
                    frames.append(df)
            return _dedupe_concat(frames)
        except Exception:
            return pd.DataFrame()
