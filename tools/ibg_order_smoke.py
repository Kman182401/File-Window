#!/usr/bin/env python3
import os, sys, json, argparse
from datetime import datetime, timezone

try:
    from ib_insync import IB, Stock, Forex, Future, ContFuture, MarketOrder, LimitOrder
except Exception as e:
    print(json.dumps({"error": "ib_insync not available", "detail": str(e)}))
    sys.exit(1)


def connect():
    host = os.getenv('IBKR_HOST', '127.0.0.1')
    port = int(os.getenv('IBKR_PORT', '4002'))
    cid  = int(os.getenv('IBKR_CLIENT_ID', '9001'))
    to   = float(os.getenv('IBKR_TIMEOUT', '8'))
    ib   = IB()
    ib.connect(host, port, clientId=cid, timeout=to, readonly=False)
    return ib


def resolve_future_front_month(ib, symbol, exchange):
    cds = ib.reqContractDetails(ContFuture(symbol, exchange=exchange))
    if not cds:
        raise RuntimeError(f"No ContractDetails for {symbol} {exchange}")
    now = datetime.now(timezone.utc).strftime("%Y%m")
    cds_sorted = sorted(cds, key=lambda cd: cd.contract.lastTradeDateOrContractMonth or "999999")
    for cd in cds_sorted:
        m = cd.contract.lastTradeDateOrContractMonth or ""
        if m >= now:
            c = cd.contract
            c.exchange = exchange
            ib.qualifyContracts(c)
            return c
    c = cds_sorted[0].contract
    c.exchange = exchange
    ib.qualifyContracts(c)
    return c


def build_contract(ib, asset, symbol, exchange, expiry, resolve_front):
    if asset == "STK":
        c = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(c)
        return c
    if asset == "FX":
        c = Forex(symbol)
        ib.qualifyContracts(c)
        return c
    if asset == "FUT":
        if expiry and not resolve_front:
            c = Future(symbol, lastTradeDateOrContractMonth=expiry, exchange=exchange or "CME")
            ib.qualifyContracts(c)
            return c
        return resolve_future_front_month(ib, symbol, exchange or "CME")
    raise RuntimeError("asset must be STK|FUT|FX")


def build_order(order_type, side, qty, limit, tif, transmit, whatif):
    if order_type == "MKT":
        o = MarketOrder(side, qty)
    else:
        if limit is None:
            raise RuntimeError("Limit price required for LMT orders")
        o = LimitOrder(side, qty, limit)
    o.tif = tif
    o.transmit = bool(transmit)
    o.whatIf = bool(whatif)
    return o


def main():
    ap = argparse.ArgumentParser(description="Paper-safe order smoke (what-if by default)")
    ap.add_argument("--asset", choices=["STK","FUT","FX"], default="STK")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--expiry", help="YYYYMM for FUT (omit with --resolve-front-month)")
    ap.add_argument("--resolve-front-month", dest="resolve_front_month", action="store_true")
    ap.add_argument("--side", choices=["BUY","SELL"], default="BUY")
    ap.add_argument("--qty", type=float, default=1)
    ap.add_argument("--order", choices=["MKT","LMT"], default="LMT")
    ap.add_argument("--limit", type=float, help="Limit price for LMT")
    ap.add_argument("--tif", default="DAY")
    ap.add_argument("--whatif", action="store_true", help="Force what-if even if env allows transmit")
    ap.add_argument("--auto-cancel", dest="auto_cancel", action="store_true", help="Cancel real order immediately after submission")
    args = ap.parse_args()

    allow_orders = os.getenv("ALLOW_ORDERS","0") == "1"
    paper_only   = os.getenv("IBKR_PAPER","0") == "1"
    transmit_real = allow_orders and paper_only and not args.whatif

    result = {}
    ib = connect()
    try:
        contract = build_contract(
            ib,
            args.asset,
            args.symbol,
            ("CME" if args.asset == "FUT" and args.exchange == "SMART" else args.exchange),
            args.expiry,
            args.resolve_front_month,
        )

        order = build_order(args.order, args.side, args.qty, args.limit, args.tif, transmit_real, not transmit_real)

        trade = ib.placeOrder(contract, order)
        ib.sleep(1.0)
        state = {k: getattr(trade.orderState, k) for k in [
            "status","initMarginChange","maintMarginChange","equityWithLoanChange"
        ] if hasattr(trade.orderState, k)}

        result = {
            "transmitted": transmit_real,
            "whatIf": not transmit_real,
            "asset": args.asset,
            "symbol": args.symbol,
            "exchange": getattr(contract, "exchange", None),
            "localSymbol": getattr(contract, "localSymbol", None),
            "orderType": args.order,
            "side": args.side,
            "qty": args.qty,
            "limit": args.limit,
            "tif": args.tif,
            "orderId": getattr(trade.order, "orderId", None),
            "orderState": state,
        }
        if transmit_real and args.auto_cancel:
            try:
                ib.cancelOrder(trade.order)
                ib.sleep(0.5)
                result["autoCanceled"] = True
            except Exception as e:
                result["autoCanceledError"] = str(e)
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e), **result}, indent=2))
        sys.exit(1)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()

