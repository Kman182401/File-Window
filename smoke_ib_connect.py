from ib_insync import IB, Contract
import os, sys

HOST = os.getenv("IBKR_HOST","127.0.0.1")
PORT = int(os.getenv("IBKR_PORT","4002"))
CID  = int(os.getenv("IBKR_CLIENT_ID","9002"))

try:
    ib = IB()
    ib.connect(HOST, PORT, clientId=CID, timeout=30)
    now = ib.reqCurrentTime()
    es = Contract(symbol='ES', secType='FUT', exchange='CME', currency='USD')
    cds = ib.reqContractDetails(es)
    assert cds, "No ES contracts found"
    front = sorted(
        [cd.contract for cd in cds if cd.contract.lastTradeDateOrContractMonth],
        key=lambda c: c.lastTradeDateOrContractMonth
    )[0]
    print(f"✅ Connected. IB time: {now}. Front ES: {front.localSymbol} {front.lastTradeDateOrContractMonth}")
    # Emit a heartbeat row so Grafana "IB Gateway" turns green
    try:
        import subprocess, json as _json, os as _os
        payload = _json.dumps({"run_id":"CONNECT_SMOKE","phase":"ib_connect_ok","pnl":0,"drawdown":0,"latency_ms":50})
        subprocess.run(["python3", str(_os.path.expanduser("~/monitoring/client/log_heartbeat_cli.py"))],
                       input=payload.encode("utf-8"), check=False)
    except Exception:
        pass

    ib.disconnect()
except Exception as e:
    print(f"❌ Smoke failed: {e}", file=sys.stderr)
    sys.exit(1)
