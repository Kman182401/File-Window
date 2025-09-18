#!/usr/bin/env python3
import os, sys, json, argparse
from datetime import datetime, timedelta, timezone

try:
    from ib_insync import IB, ExecutionFilter
except Exception as e:
    print(json.dumps({"error": "ib_insync not available", "detail": str(e)}))
    sys.exit(1)


def connect(readonly=False):
    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '4002'))
    cid  = int(os.getenv('IBKR_CLIENT_ID', '9001'))
    to   = float(os.getenv('IBKR_TIMEOUT', '8'))
    ib   = IB()
    ib.connect(host, port, clientId=cid, timeout=to, readonly=readonly)
    return ib


def positions_json(ib):
    out = []
    for p in ib.positions():
        c = p.contract
        out.append({
            "account": p.account,
            "conId": c.conId,
            "symbol": getattr(c, "symbol", None),
            "localSymbol": getattr(c, "localSymbol", None),
            "secType": getattr(c, "secType", None),
            "exchange": getattr(c, "exchange", None),
            "lastTradeDateOrContractMonth": getattr(c, "lastTradeDateOrContractMonth", None),
            "position": p.position,
            "avgCost": p.avgCost
        })
    return out


def open_orders_json(ib):
    out = []
    for t in ib.openTrades():
        c, o, s = t.contract, t.order, t.orderStatus
        out.append({
            "conId": c.conId,
            "symbol": getattr(c, "symbol", None),
            "localSymbol": getattr(c, "localSymbol", None),
            "secType": getattr(c, "secType", None),
            "orderId": getattr(o, 'orderId', None),
            "action": getattr(o, 'action', None),
            "totalQuantity": getattr(o, 'totalQuantity', None),
            "orderType": getattr(o, 'orderType', None),
            "lmtPrice": getattr(o, 'lmtPrice', None),
            "tif": getattr(o, 'tif', None),
            "status": getattr(s, 'status', None),
            "filled": getattr(s, 'filled', None),
            "remaining": getattr(s, 'remaining', None)
        })
    return out


def accounts_json(ib):
    return ib.managedAccounts()


def account_values_json(ib):
    return [vars(v) for v in ib.accountValues()]


def executions_json(ib, since_iso):
    filt = ExecutionFilter()
    if since_iso:
        try:
            dt = datetime.fromisoformat(since_iso.replace("Z","+00:00")).astimezone(timezone.utc)
        except ValueError:
            print(json.dumps({"error": "Bad --since format; use ISO8601"}))
            sys.exit(1)
    else:
        dt = datetime.now(timezone.utc) - timedelta(days=1)
    filt.time = dt.strftime("%Y%m%d-%H:%M:%S")
    ex = ib.reqExecutions(filt)
    out = []
    for e in ex:
        o = e.execution
        c = e.contract
        out.append({
            "time": o.time,
            "symbol": getattr(c, "symbol", None),
            "localSymbol": getattr(c, "localSymbol", None),
            "secType": getattr(c, "secType", None),
            "side": o.side,
            "shares": o.shares,
            "price": o.price,
            "orderId": o.orderId,
            "execId": o.execId
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Dump IBG data as JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("accounts")
    sub.add_parser("positions")
    sub.add_parser("orders")
    av = sub.add_parser("account")
    av.add_argument("--all", action="store_true", help="Dump full account values")
    ex = sub.add_parser("executions")
    ex.add_argument("--since", help="ISO8601 UTC start time, default: now-24h")
    args = ap.parse_args()

    ib = connect(readonly=False)
    try:
        if args.cmd == "accounts":
            data = accounts_json(ib)
        elif args.cmd == "positions":
            data = positions_json(ib)
        elif args.cmd == "orders":
            data = open_orders_json(ib)
        elif args.cmd == "account":
            data = account_values_json(ib) if args.all else {"accounts": accounts_json(ib)}
        elif args.cmd == "executions":
            data = executions_json(ib, getattr(args, 'since', None))
        else:
            data = {"error": "unknown command"}
        print(json.dumps(data, indent=2, default=str))
        sys.exit(0)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

