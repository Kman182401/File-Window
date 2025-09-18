#!/usr/bin/env python3
import os, sys, time
from ib_insync import IB, util

host = os.getenv("IBKR_HOST", "127.0.0.1")
port = int(os.getenv("IBKR_PORT", "4002"))
cid  = int(os.getenv("IBKR_CLIENT_ID", "9002"))

ib = IB()
util.startLoop()

# Try up to ~60 seconds (12 * 5s)
for i in range(12):
    try:
        ib.connect(host, port, clientId=cid, timeout=5)
        if ib.isConnected():
            print("IBGW_READY: True", "server:", ib.client.serverVersion())
            ib.disconnect()
            sys.exit(0)
    except Exception:
        pass
    time.sleep(5)

print("IBGW_READY: False")
sys.exit(1)

