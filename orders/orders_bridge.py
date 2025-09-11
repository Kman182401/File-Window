#!/usr/bin/env python3
import os, json, time, sys
from datetime import datetime

# Add parent directory to path to import paper_trading_executor
sys.path.insert(0, os.path.expanduser('~'))

try:
    from ib_insync import IB, Future, MarketOrder, LimitOrder, StopOrder, Contract, ContractDetails
except Exception as e:
    print("[orders_bridge] ib_insync not available:", e, file=sys.stderr)
    IB=None
    Future=None
    MarketOrder=None
    LimitOrder=None
    StopOrder=None
    Contract=None
    ContractDetails=None

try:
    from paper_trading_executor import PaperTradingExecutor
except Exception as e:
    print("[orders_bridge] cannot import PaperTradingExecutor:", e, file=sys.stderr)
    PaperTradingExecutor=None

IB_HOST=os.getenv("IBKR_HOST","127.0.0.1")
IB_PORT=int(os.getenv("IBKR_PORT","4002"))
IB_CID=int(os.getenv("IBKR_ORDERS_CLIENT_ID","9007"))
DRY_RUN=int(os.getenv("DRY_RUN","1"))
ALLOW_ORDERS=int(os.getenv("ALLOW_ORDERS","0"))
FORCE_DIRECT=int(os.getenv("FORCE_DIRECT","0"))
SYMBOL_ALLOWLIST=set(os.getenv("SYMBOL_ALLOWLIST","MES,MNQ,ES,NQ,6E,6B,6A,GC").split(","))
ORDER_QTY_LIMIT=int(os.getenv("ORDER_QTY_LIMIT","1"))
KEEP_CONNECTED=int(os.getenv("KEEP_CONNECTED","1"))
PING_INTERVAL=float(os.getenv("PING_INTERVAL","30"))
ORDER_MODE=os.getenv("ORDER_MODE","FUT")
ORDER_USE_BRACKET=int(os.getenv("ORDER_USE_BRACKET","1"))
STOP_TICKS=float(os.getenv("STOP_TICKS","8"))
TAKE_TICKS=float(os.getenv("TAKE_TICKS","16"))
ENTRY_TYPE=os.getenv("ENTRY_TYPE","MARKET")  # MARKET|LIMIT
LIMIT_OFFSET_TICKS=float(os.getenv("LIMIT_OFFSET_TICKS","0"))
TIF=os.getenv("TIF","DAY")
USE_MICRO=int(os.getenv("USE_MICRO","1"))  # 1=prefer MES/MNQ, 0=ES/NQ
VOLATILITY_HALT=float(os.getenv("VOLATILITY_HALT","0"))  # 0=off; else pct per min

AUDIT=os.path.expanduser("~/trade_audit_log.jsonl")

def parse_decision(rec):
    sym = rec.get("symbol") or rec.get("ticker") or rec.get("instrument")
    side = rec.get("side") or rec.get("action") or rec.get("direction")
    qty = rec.get("qty") or rec.get("size") or 1
    conf = rec.get("confidence") or rec.get("score")
    reason = rec.get("reason") or rec.get("strategy") or rec.get("source","decision")
    if not sym or not side: return None
    side = str(side).upper()
    if side not in ("BUY","SELL"): return None
    try: qty = int(qty)
    except: qty = 1
    return {"symbol":sym, "side":side, "qty":qty, "confidence":conf, "reason":reason}

def log_trade(payload):
    import subprocess, json as _j
    subprocess.run(["python3", os.path.expanduser("~/monitoring/client/log_trade_event.py")],
                   input=_j.dumps(payload).encode(), check=False)

def submit_order(dec):
    ts = datetime.utcnow().isoformat()
    status = "dry_run"
    order_id = None
    entry_price = None
    exit_price = None
    pnl = None
    dur = None
    
    if not DRY_RUN and ALLOW_ORDERS:
        if PaperTradingExecutor is not None and not FORCE_DIRECT and ORDER_MODE!='FUT':
            try:
                print('[orders_bridge] Using EXECUTOR path', flush=True)
                ex = PaperTradingExecutor(host=IB_HOST, port=IB_PORT, clientId=IB_CID, paper=True)
                res = ex.place_market_order(symbol=dec["symbol"], side=dec["side"], qty=dec["qty"])
                order_id = getattr(res, "orderId", None)
                status = "submitted"
            except Exception as e:
                # Fallback to direct placement on any executor error
                try:
                    print(f"[orders_bridge] Using DIRECT path for {dec['symbol']}", flush=True)
                    status, order_id = _place_order_direct_fut(dec, IB_HOST, IB_PORT, IB_CID)
                    print(f"[orders_bridge] Direct placement result: status={status}, order_id={order_id}", flush=True)
                except Exception as e2:
                    status = f"error:{e2}"
        else:
            try:
                print(f"[orders_bridge] Using DIRECT path for {dec['symbol']}", flush=True)
                status, order_id = _place_order_direct_fut(dec, IB_HOST, IB_PORT, IB_CID)
                print(f"[orders_bridge] Direct placement result: status={status}, order_id={order_id}", flush=True)
            except Exception as e:
                status = f"error:{e}"
                print(f"[orders_bridge] Direct placement failed: {e}", flush=True)
    # existing log_trade payload follows
    log_trade({
        "ts": ts,
        "symbol": dec["symbol"],
        "side": dec["side"],
        "qty": dec["qty"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl": pnl,
        "duration_sec": dur,
        "confidence": dec.get("confidence"),
        "status": status,
        "order_id": order_id,
        "reason": dec.get("reason")
    })

from ib_insync import IB, Future, MarketOrder

def _qualify_front_month(ib, symbol: str):
    exch = 'COMEX' if symbol in ('GC','MGC') else 'CME'
    c = Future(symbol=symbol, exchange=exch, currency='USD')
    cds = ib.reqContractDetails(c)
    futs = [cd.contract for cd in cds if getattr(cd.contract, 'lastTradeDateOrContractMonth', None)]
    if not futs:
        raise RuntimeError(f'No contract details for {symbol}')
    futs.sort(key=lambda k: k.lastTradeDateOrContractMonth)
    return futs[0]

def _place_order_direct(dec: dict, host: str, port: int, cid: int):
    sym = str(dec['symbol']).upper()
    if sym not in SYMBOL_ALLOWLIST:
        return ('blocked_symbol', None)
    qty = max(1, min(int(dec['qty']), ORDER_QTY_LIMIT))
    side = str(dec['side']).upper()
    action = 'BUY' if side == 'BUY' else 'SELL'
    ib = IB()
    ib.connect(host, port, clientId=cid, timeout=30)
    try:
        contract = _qualify_front_month(ib, sym)
        order = MarketOrder(action, qty)
        order.outsideRth=True
        trade = ib.placeOrder(contract, order)
        ib.sleep(1.0)
        status = getattr(trade.orderStatus, 'status', 'submitted')
        order_id = getattr(trade.order, 'orderId', None)
        return (status, order_id)
    finally:
        if not KEEP_CONNECTED:
            try: ib.disconnect()
            except: pass



_EXCH= {'GC':'COMEX'}  # special exchange routing; default CME
_MICRO_MAP = {'ES':'MES','NQ':'MNQ'}  # expand as needed

def _fut_contract_and_tick(ib: IB, sym_root: str, use_micro: bool=True):
    root = sym_root.upper()
    if use_micro and root in _MICRO_MAP:
        qsym = _MICRO_MAP[root]
    else:
        qsym = root
    exch = _EXCH.get(root, 'CME')
    c = Future(symbol=qsym, exchange=exch, currency='USD')
    cds = ib.reqContractDetails(c)
    futs = [cd for cd in cds if getattr(cd.contract,'lastTradeDateOrContractMonth',None)]
    if not futs:
        raise RuntimeError(f'No ContractDetails for {qsym} on {exch}')
    futs.sort(key=lambda cd: cd.contract.lastTradeDateOrContractMonth)
    front = futs[0]
    min_tick = getattr(front, 'minTick', None) or getattr(front, 'priceMagnifier', None) or 0.25
    return front.contract, float(min_tick)

def _last_trade(ib: IB, contract: Contract):
    t = ib.reqMktData(contract, '', False, False)
    ib.sleep(0.8)
    px = t.last or t.close or t.marketPrice() or 0
    if not px or px <= 0:
        bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='1 D',
                                    barSizeSetting='1 min', whatToShow='TRADES',
                                    useRTH=False, formatDate=2)
        px = bars[-1].close if bars else 0
    if not px or px <= 0:
        raise RuntimeError('last price unavailable')
    return float(px)

def _round_to_tick(px, tick):
    return round(px / tick) * tick

def _build_bracket(action: str, qty: float, last_px: float, tick: float,
                   entry_type='MARKET', limit_offset_ticks=0.0,
                   stop_ticks=8.0, take_ticks=16.0, tif='DAY'):
    side = action.upper()
    assert side in ('BUY','SELL')
    # entry
    if entry_type == 'LIMIT':
        off = (limit_offset_ticks or 0.0) * tick
        entry_px = last_px - off if side=='BUY' else last_px + off
        entry_px = _round_to_tick(entry_px, tick)
        parent = LimitOrder(side, qty, entry_px, tif=tif)
        parent.outsideRth=True
    else:
        parent = MarketOrder(side, qty, tif=tif)
        parent.outsideRth=True

    # protective stop
    stp_px = last_px - (stop_ticks*tick) if side=='BUY' else last_px + (stop_ticks*tick)
    stp_px = _round_to_tick(stp_px, tick)
    stop = StopOrder('SELL' if side=='BUY' else 'BUY', qty, stp_px, tif=tif)
    stop.outsideRth=True

    # take-profit
    tp_px = last_px + (take_ticks*tick) if side=='BUY' else last_px - (take_ticks*tick)
    tp_px = _round_to_tick(tp_px, tick)
    take = LimitOrder('SELL' if side=='BUY' else 'BUY', qty, tp_px, tif=tif)
    take.outsideRth=True

    # OCO bracket wiring (parent transmit False; children last transmit True)
    parent.transmit = False
    stop.parentId = 0  # will be set by IB; parent be first
    take.parentId = 0
    take.ocaGroup = f"OCA_{abs(hash((last_px,qty,side)))%10_000_000}"
    take.ocaType = 1
    stop.ocaGroup = take.ocaGroup
    stop.ocaType = 1
    stop.transmit = False
    take.transmit = True
    return parent, stop, take

def _place_order_direct_fut(dec: dict, host: str, port: int, cid: int):
    sym = str(dec['symbol']).upper()
    qty = max(1, min(int(dec.get('qty',1)), ORDER_QTY_LIMIT))
    side = 'BUY' if str(dec.get('side','BUY')).upper()=='BUY' else 'SELL'

    ib = _get_ib()
    try:
        # Qualify contract + get tick + last price
        contract, tick = _fut_contract_and_tick(ib, sym_root=sym, use_micro=bool(USE_MICRO))
        last_px = _last_trade(ib, contract)

        # Optional volatility halt (very simple: pct move last minute)
        if VOLATILITY_HALT > 0:
            bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='2 m',
                                        barSizeSetting='1 min', whatToShow='TRADES',
                                        useRTH=False, formatDate=2)
            if len(bars) >= 2:
                pct = abs((bars[-1].close - bars[-2].close) / bars[-2].close) * 100.0
                if pct >= VOLATILITY_HALT:
                    return ('halt_volatility', None)

        # Build orders
        if ORDER_USE_BRACKET:
            parent, stop, take = _build_bracket(side, qty, last_px, tick,
                                                entry_type=ENTRY_TYPE,
                                                limit_offset_ticks=LIMIT_OFFSET_TICKS,
                                                stop_ticks=STOP_TICKS,
                                                take_ticks=TAKE_TICKS, tif=TIF)
            # Place bracket
            tr_parent = ib.placeOrder(contract, parent)
            ib.sleep(0.2)
            stop.parentId = tr_parent.order.orderId
            take.parentId = tr_parent.order.orderId
            tr_stop = ib.placeOrder(contract, stop); ib.sleep(0.2)
            tr_take = ib.placeOrder(contract, take); ib.sleep(0.6)
            status = getattr(tr_parent.orderStatus, 'status', 'submitted')
            return (status, tr_parent.order.orderId)
        else:
            # Plain entry only
            if ENTRY_TYPE == 'LIMIT':
                off = (LIMIT_OFFSET_TICKS or 0.0) * tick
                px = last_px - off if side=='BUY' else last_px + off
                px = _round_to_tick(px, tick)
                ord = LimitOrder(side, qty, px, tif=TIF)
                ord.outsideRth=True
            else:
                ord = MarketOrder(side, qty, tif=TIF)
                ord.outsideRth=True
            tr = ib.placeOrder(contract, ord)
            ib.sleep(0.6)
            status = getattr(tr.orderStatus, 'status', 'submitted')
            return (status, tr.order.orderId)
    finally:
        if not KEEP_CONNECTED:
            try: ib.disconnect()
            except: pass



import threading, time as _time
ORD_IB = None
IB_LOCK = threading.RLock()

def _get_ib():
    global ORD_IB
    with IB_LOCK:
        if ORD_IB is None:
            from ib_insync import IB
            ORD_IB = IB()
            ORD_IB.errorEvent += lambda *a, **k: None
        if not ORD_IB.isConnected():
            ORD_IB.connect(IB_HOST, IB_PORT, clientId=IB_CID, timeout=30)
        return ORD_IB

def _ping_loop():
    while True:
        try:
            ib = _get_ib()
            ib.reqCurrentTime()
        except Exception:
            # try to reconnect next tick
            pass
        _time.sleep(PING_INTERVAL)

# spawn pinger on startup if KEEP_CONNECTED
if KEEP_CONNECTED:
    threading.Thread(target=_ping_loop, daemon=True).start()

print(f"[orders_bridge] Starting: HOST={IB_HOST} PORT={IB_PORT} CLIENT_ID={IB_CID} DRY_RUN={DRY_RUN} ALLOW_ORDERS={ALLOW_ORDERS} FORCE_DIRECT={FORCE_DIRECT} ORDER_MODE={ORDER_MODE}", flush=True)
print(f"[orders_bridge] Watching: {AUDIT}", flush=True)
if not ALLOW_ORDERS:
    print(f"[orders_bridge] WARNING: ALLOW_ORDERS=0 - Orders will NOT be sent to IBKR", flush=True)

os.makedirs(os.path.dirname(AUDIT) or ".", exist_ok=True)
with open(AUDIT, "a+", buffering=1) as f:
    f.seek(0,2)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.5)
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        dec = parse_decision(rec)
        if not dec: 
            continue
        print(f"[orders_bridge] Decision parsed: {dec}", flush=True)
        submit_order(dec)
        print(f"[orders_bridge] Order processed for {dec['symbol']}", flush=True)

