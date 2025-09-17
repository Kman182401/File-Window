set -euo pipefail
. "$HOME/.profile" 2>/dev/null || true
. "$HOME/M5-Trader/.venv/bin/activate"
python - <<'PY'
import os, datetime as dt
from ib_insync import IB, Future, util
from utils.persist_market_data import persist_bars
ib = IB()
ib.connect(os.getenv("IBKR_HOST","127.0.0.1"), int(os.getenv("IBKR_PORT","4002")), clientId=9001, timeout=10)
for sym in ("ES", "NQ"):
    cds = ib.reqContractDetails(Future(sym,'',exchange='CME'))
    if not cds:
        continue
    today = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d')
    contracts = sorted([cd.contract for cd in cds if (cd.contract.lastTradeDateOrContractMonth or '0000') >= today],
                       key=lambda c: c.lastTradeDateOrContractMonth or '9999')
    c = contracts[0]
    bars = ib.reqHistoricalData(c, endDateTime='', durationStr='900 S', barSizeSetting='1 min',
                                whatToShow='TRADES', useRTH=False, formatDate=1)
    df = util.df(bars)
    if df is not None and len(df):
        persist_bars(f"{sym}1!", df)
ib.disconnect()
PY
