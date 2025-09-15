import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from ib_news_fetch_dynamic import fetch_ib_news_for_tickers

TARGET_COLS = ["title","summary","published_at","tickers","entities",
               "sentiment_score","source","url","keywords",
               "provider","article_id","headline"]

def _local_normalize_ib(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TARGET_COLS)
    out = df.copy()
    # map headline -> title; keep provider/article_id/headline/published_at
    out["title"] = out.get("title")
    if "headline" in out.columns:
        out["title"] = out["title"].where(out["title"].notna(), out["headline"])
    out["summary"] = out.get("summary")
    # ensure 'tickers' exists; if not provided, fill from 'ticker'
    if "tickers" not in out.columns:
        out["tickers"] = None
    if "ticker" in out.columns:
        out["tickers"] = out["tickers"].where(out["tickers"].notna(), out["ticker"])
    out["entities"] = out.get("entities")
    out["sentiment_score"] = out.get("sentiment_score")
    out["source"] = "IBKR"
    out["url"] = out.get("url")
    out["keywords"] = out.get("keywords")
    # coerce timestamp
    out["published_at"] = pd.to_datetime(out.get("published_at"), utc=True, errors="coerce")
    for c in TARGET_COLS:
        if c not in out.columns:
            out[c] = None
    return out[TARGET_COLS]

def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=TARGET_COLS)
    out = df.copy()
    for c in TARGET_COLS:
        if c not in out.columns:
            out[c] = None
    return out[TARGET_COLS]

def _bad_or_empty(df: pd.DataFrame) -> bool:
    """
    Treat as bad if empty OR published_at all NaT OR both title/headline all missing.
    """
    if df is None or df.empty:
        return True
    pa = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
    title = df.get("title")
    headline = df.get("headline")
    all_no_timestamp = pa.isna().all()
    all_no_text = (title is None or title.isna().all()) and (headline is None or headline.isna().all())
    return bool(all_no_timestamp or all_no_text)

# Try project normalizers; may not be compatible with our columns
try:
    from news_ingestion_ibkr import normalize_ibkr_news as proj_norm_ib
except Exception:
    proj_norm_ib = None

try:
    from news_ingestion_marketaux import fetch_marketaux_news, normalize_marketaux_news as proj_norm_ma
except Exception:
    fetch_marketaux_news = proj_norm_ma = None

def get_combined_news(tickers, lookback_hours=48, max_per_ticker=50):
    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PORT", "4002"))

    # 1) IB raw (BZ via conId/proxies)
    ib_raw = fetch_ib_news_for_tickers(tickers, lookback_hours=lookback_hours,
                                       max_per_ticker=max_per_ticker, host=host, port=port)

    # Normalize with project function if it returns a "good" frame; else fallback
    if proj_norm_ib:
        try:
            tmp = proj_norm_ib(ib_raw.copy())
            ib_df = _ensure_cols(tmp)
            if _bad_or_empty(ib_df) and not ib_raw.empty:
                ib_df = _local_normalize_ib(ib_raw)
        except Exception:
            ib_df = _local_normalize_ib(ib_raw)
    else:
        ib_df = _local_normalize_ib(ib_raw)

    # 2) MarketAux (if available). Some impls don’t accept lookback_hours.
    ma_df = pd.DataFrame(columns=TARGET_COLS)
    if fetch_marketaux_news and proj_norm_ma and os.getenv("MARKETAUX_API_KEY"):
        try:
            ma_raw = fetch_marketaux_news(limit=100)  # no lookback_hours arg
            tmp = proj_norm_ma(ma_raw.copy())
            tmp = _ensure_cols(tmp)
            if not tmp.empty:
                # filter to lookback_hours locally
                now = datetime.now(timezone.utc)
                cutoff = now - timedelta(hours=lookback_hours)
                tmp["published_at"] = pd.to_datetime(tmp["published_at"], utc=True, errors="coerce")
                ma_df = tmp[tmp["published_at"] >= cutoff]
        except Exception:
            ma_df = pd.DataFrame(columns=TARGET_COLS)

    # 3) Merge, dedup, sort
    frames = [df for df in (ib_df, ma_df) if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame(columns=TARGET_COLS)

    both = pd.concat(frames, ignore_index=True)
    both["published_at"] = pd.to_datetime(both["published_at"], utc=True, errors="coerce")
    if "article_id" in both.columns:
        both = both.drop_duplicates(subset=["article_id"], keep="last")
    else:
        both = both.drop_duplicates(subset=["provider","published_at","title","headline"], keep="last")
    both = both.sort_values("published_at").reset_index(drop=True)
    return both
