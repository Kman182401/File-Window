#!/usr/bin/env python3
import os, json, sys
from datetime import datetime

try:
    from ib_insync import IB
except Exception as e:
    err = {
        "connected": False,
        "error": "ib_insync not available",
        "detail": str(e),
        "hint": "Use .venv/bin/pip install ib-insync or run via .venv/bin/python"
    }
    print(json.dumps(err, indent=2))
    sys.exit(1)

host = os.getenv('IBKR_HOST', '127.0.0.1')
port = int(os.getenv('IBKR_PORT', '4002'))
client_id = int(os.getenv('IBKR_CLIENT_ID', '9001'))
timeout = float(os.getenv('IBKR_TIMEOUT', '5'))

ib = IB()
info = {"host": host, "port": port, "clientId": client_id, "ts": datetime.utcnow().isoformat()}

try:
    ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=True)
    info["connected"] = ib.isConnected()
    info["serverVersion"] = ib.client.serverVersion() if ib.client else None
    try:
        info["serverTime"] = ib.reqCurrentTime().isoformat()
    except Exception as e:
        info["serverTime_error"] = str(e)
    try:
        info["accounts"] = ib.managedAccounts()
    except Exception as e:
        info["accounts_error"] = str(e)
    try:
        info["openOrders"] = len(ib.reqOpenOrders())
    except Exception as e:
        info["openOrders_error"] = str(e)
    try:
        info["positionsCount"] = len(ib.positions())
    except Exception as e:
        info["positions_error"] = str(e)
except Exception as e:
    info["connected"] = False
    info["error"] = str(e)
finally:
    try:
        ib.disconnect()
    except Exception:
        pass

print(json.dumps(info, indent=2))
sys.exit(0 if info.get("connected") else 1)

