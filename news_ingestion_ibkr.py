from ib_insync import IB
from datetime import datetime, timedelta
import pandas as pd
import os

# PHASE4A-Port-Fix: Use environment variables for IBKR connection with 4004 default
_IB_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
_IB_PORT = int(os.getenv('IBKR_PORT', '4004'))  # Changed from hardcoded 4002 to env var with 4004 default
_IB_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '101'))

# Provider codes: common free sources; adjust if your account has others enabled
# Examples: 'BRFG' (Briefing.com), 'BRFUPDN'; leave empty to include all available
_PROVIDER_CODES = None  # None = all

def _connect():
    ib = IB()
    ib.connect(_IB_HOST, _IB_PORT, clientId=_IB_CLIENT_ID)
    return ib

def fetch_ibkr_news(lookback_hours=24, max_results=200):
    """
    Returns a DataFrame with columns:
    ['title','summary','published_at','tickers','entities','sentiment_score','source','url','keywords']
    """
    ib = _connect()
    try:
        # Discover providers if not specified
        providers = ib.reqNewsProviders()
        provider_codes = ",".join([p.code for p in providers]) if _PROVIDER_CODES is None else _PROVIDER_CODES

        end = datetime.utcnow()
        start = end - timedelta(hours=lookback_hours)

        # Using conId=0 pulls general news not tied to a specific contract.
        items = ib.reqHistoricalNews(
            conId=0,
            providerCodes=provider_codes,
            startDateTime=start,
            endDateTime=end,
            totalResults=max_results,
            )

        recs = []
        for it in items:
            # Optionally fetch full article body:
            # article = ib.reqNewsArticle(it.providerCode, it.articleId)
            recs.append({
                "title": it.headline,
                "summary": None,  # body fetch optional; keep light for now
                "published_at": pd.to_datetime(it.time, utc=True),
                "tickers": [],
                "entities": None,
                "sentiment_score": None,
                "source": it.providerCode,
                "url": None,
                "keywords": None,
            })
        df = pd.DataFrame(recs)
        return (df if "published_at" not in df.columns else df.sort_values("published_at")).reset_index(drop=True)
    finally:
        ib.disconnect()

def normalize_ibkr_news(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure identical schema to MarketAux normalize for easy union."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["title","summary","published_at","tickers","entities","sentiment_score","source","url","keywords"])
    keep = ["title","summary","published_at","tickers","entities","sentiment_score","source","url","keywords"]
    for c in keep:
        if c not in df.columns:
            df[c] = None
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    return df[keep]
