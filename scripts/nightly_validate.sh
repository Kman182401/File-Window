set -euo pipefail
. "$HOME/.profile" 2>/dev/null || true
. "$HOME/M5-Trader/.venv/bin/activate"

# Skip if pipeline is running to avoid clientId collision
if pgrep -f 'run_pipeline.py' >/dev/null; then
  echo "nightly_validate: pipeline active; skipping"
  exit 0
fi

python - <<'PY'
import os, datetime as dt
from ib_insync import IB, Future, util
from utils.persist_market_data import persist_bars

host = os.getenv("IBKR_HOST","127.0.0.1")
port = int(os.getenv("IBKR_PORT","4002"))
cid  = int(os.getenv("IBKR_CLIENT_ID", "9002"))
ib = IB(); ib.connect(host, port, clientId=cid, timeout=10)
for sym in ("ES","NQ"):
    try:
        cds = ib.reqContractDetails(Future(sym,'',exchange='CME'))
        if not cds:
            print(f"nightly_validate: no contracts for {sym}")
            continue
        today = dt.datetime.utcnow().strftime('%Y%m%d')
        contracts = sorted([cd.contract for cd in cds if (cd.contract.lastTradeDateOrContractMonth or '0000') >= today],
                           key=lambda c: c.lastTradeDateOrContractMonth or '9999')
        c = contracts[0]
        bars = ib.reqHistoricalData(c, endDateTime='', durationStr='900 S', barSizeSetting='1 min',
                                    whatToShow='TRADES', useRTH=False, formatDate=1)
        df = util.df(bars)
        if df is not None and len(df):
            persist_bars(f"{sym}1!", df)
            print(f"nightly_validate: {sym} persisted {len(df)} rows")
    except Exception as e:
        print(f"nightly_validate error for {sym}: {e}")
ib.disconnect()
PY
