from datetime import datetime, timedelta, timezone
from typing import Iterable
from ib_insync import IB, Stock
import pandas as pd

# ETF proxies for macro/futures/FX tickers
PROXY = {
    'ES1!':'SPY', 'NQ1!':'QQQ',
    'GC':'GLD', 'XAUUSD':'GLD',
    'EURUSD':'FXE', 'GBPUSD':'FXB', 'AUDUSD':'FXA',
    '6E':'FXE', '6B':'FXB', '6A':'FXA'
}

def _rows(items, label):
    return [{
        "published_at": pd.to_datetime(getattr(it, "time", None), utc=True, errors="coerce"),
        "provider": getattr(it, "providerCode", None),
        "article_id": getattr(it, "articleId", None),
        "headline": getattr(it, "headline", None),
        "extra": getattr(it, "extraData", None),
        "ticker": label,
    } for it in items or []]

def fetch_ib_news_for_tickers(
    tickers: Iterable[str],
    lookback_hours: int = 48,
    max_per_ticker: int = 50,
    host: str = "127.0.0.1",
    port: int = 4002,
    clientId: int = 2401,
) -> pd.DataFrame:
    ib = IB(); ib.connect(host, port, clientId=clientId, timeout=10)

    # (Optional) ignore only error code 200 messages from ib_insync
    def _on_error(reqId, code, msg, advancedOrderReject=''):
        if code == 200:
            return
    ib.errorEvent += _on_error

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=lookback_hours)
    frames = []

    provs = {p.code for p in ib.reqNewsProviders()}
    if 'BZ' not in provs:
        ib.errorEvent -= _on_error
        ib.disconnect()
        return pd.DataFrame(columns=["published_at","provider","article_id","headline","extra","ticker"])

    for t in tickers:
        got = False

        # 1) Try proxy first for macro/futures/FX symbols
        if t in PROXY:
            try:
                pc = Stock(PROXY[t], 'SMART', 'USD'); [pc] = ib.qualifyContracts(pc)
                items = ib.reqHistoricalNews(pc.conId, 'BZ', start, end, max_per_ticker)
                df = pd.DataFrame(_rows(items, t))
                if not df.empty:
                    frames.append(df); got = True
            except Exception:
                pass

        # 2) If still nothing, try the ticker itself (works for stocks like AAPL/MSFT)
        if not got:
            try:
                c = Stock(t, 'SMART', 'USD'); [c] = ib.qualifyContracts(c)
                items = ib.reqHistoricalNews(c.conId, 'BZ', start, end, max_per_ticker)
                df = pd.DataFrame(_rows(items, t))
                if not df.empty:
                    frames.append(df); got = True
            except Exception:
                pass

        # 3) Final fallback: broad feed (rarely returns anything but cheap to check)
        if not got:
            try:
                items = ib.reqHistoricalNews(0, 'BZ', start, end, max_per_ticker)
                df = pd.DataFrame(_rows(items, t))
                if not df.empty:
                    frames.append(df)
            except Exception:
                pass

    ib.errorEvent -= _on_error
    ib.disconnect()

    if not frames:
        return pd.DataFrame(columns=["published_at","provider","article_id","headline","extra","ticker"])

    out = pd.concat(frames, ignore_index=True)
    if "published_at" in out.columns:
        out = out.drop_duplicates("article_id").sort_values("published_at").reset_index(drop=True)
    return out
