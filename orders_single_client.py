from typing import Optional, Tuple
import os, time
import logging
from ib_insync import IB, MarketOrder, LimitOrder, StopOrder
from market_data_ibkr_adapter import get_ibkr_front_month_contract
from order_safety_wrapper import safe_place_order

logger = logging.getLogger(__name__)

_ib: Optional[IB] = None

def attach_ib(ib: IB) -> None:
    global _ib
    _ib = ib

def _resolve_contract(ib: IB, symbol: str):
    sym = symbol.upper()
    if sym in ("ES1!", "ES", "MES"):
        base, exch, curr = "ES", "CME", "USD"
    elif sym in ("NQ1!", "NQ", "MNQ"):
        base, exch, curr = "NQ", "CME", "USD"
    elif sym in ("XAUUSD", "GC", "MGC"):
        base, exch, curr = "GC", "COMEX", "USD"
    elif sym in ("EURUSD","6E"):
        base, exch, curr = "6E", "CME", "USD"
    elif sym in ("GBPUSD","6B"):
        base, exch, curr = "6B", "CME", "USD"
    elif sym in ("AUDUSD","6A"):
        base, exch, curr = "6A", "CME", "USD"
    else:
        base, exch, curr = sym, "CME", "USD"
    return get_ibkr_front_month_contract(ib, base, exch, curr)

def _get_min_tick(ib: IB, contract) -> float:
    cds = ib.reqContractDetails(contract) or []
    return getattr(cds[0], "minTick", 0.25) if cds else 0.25

def _get_market_price(ib: IB, contract) -> float:
    t = ib.reqMktData(contract, "", False, False)
    for _ in range(40):
        if t.last not in (None, float("inf")) and t.last > 0:
            return float(t.last)
        m = t.marketPrice()
        if m not in (None, float("inf")) and m > 0:
            return float(m)
        if t.close not in (None, float("inf")) and t.close > 0:
            return float(t.close)
        ib.sleep(0.1)
    return float(t.marketPrice() or 0)

def _tp_sl(entry: float, tick: float, take_ticks: int, stop_ticks: int, side: str) -> Tuple[float,float]:
    if side.upper() == "BUY":
        return (round(entry + take_ticks * tick, 10), round(entry - stop_ticks * tick, 10))
    else:
        return (round(entry - take_ticks * tick, 10), round(entry + stop_ticks * tick, 10))

def place_bracket_order(symbol: str, side: str, qty: int,
                        stop_ticks: int = 8, take_ticks: int = 16,
                        tif: str = "DAY", order_type: str = "MKT",
                        limit_offset_ticks: int = 0,
                        allow_env: str = "ALLOW_ORDERS", dry_env: str = "DRY_RUN") -> dict:
    assert _ib is not None and _ib.isConnected(), "IB not attached/connected"
    allow = os.getenv(allow_env, "0") == "1"
    dry = os.getenv(dry_env, "1") == "1"

    contract = _resolve_contract(_ib, symbol)
    _ib.qualifyContracts(contract)
    tick = _get_min_tick(_ib, contract)
    entry_px = _get_market_price(_ib, contract)
    if entry_px <= 0:
        raise RuntimeError("No market price available for entry")

    if order_type.upper() == "LIMIT" and limit_offset_ticks > 0:
        if side.upper() == "BUY":
            entry_px = entry_px - limit_offset_ticks * tick
        else:
            entry_px = entry_px + limit_offset_ticks * tick

    tp_px, sl_px = _tp_sl(entry_px, tick, take_ticks, stop_ticks, side)

    parent = MarketOrder(side.upper(), qty) if order_type.upper() == "MKT" else LimitOrder(side.upper(), qty, entry_px)
    parent.tif = tif

    tp_side = "SELL" if side.upper() == "BUY" else "BUY"
    take = LimitOrder(tp_side, qty, tp_px); take.tif = tif
    stop = StopOrder(tp_side, qty, sl_px);  stop.tif = tif

    if not allow or dry:
        return {"placed": False, "dry_run": True, "entry": entry_px, "tp": tp_px, "sl": sl_px,
                "symbol": symbol, "side": side, "qty": qty}

    # Wrap parent order with safety checks (this creates exposure)
    symbol_name = getattr(contract, "symbol", getattr(contract, "localSymbol", ""))
    trade = safe_place_order(_ib, contract, parent, 
                           symbol=symbol_name, side=side, quantity=qty, context="parent")
    
    if trade is None:
        logger.warning(f"[bracket_parent_blocked] {symbol_name} {side} x{qty}")
        return {"placed": False, "blocked": True, "reason": "safety_check_failed",
                "symbol": symbol, "side": side, "qty": qty}
    
    while not trade.orderStatus or not trade.orderStatus.status:
        _ib.sleep(0.05)
    parent_id = trade.order.orderId

    take.parentId = parent_id
    stop.parentId = parent_id
    take.ocaGroup = f"OCA_{int(time.time()*1000)}"
    stop.ocaGroup = take.ocaGroup

    # Children stay direct (they manage exposure, don't create it)
    # Could wrap with context="child_tp" and context="child_sl" for consistency
    _ib.placeOrder(contract, take)
    _ib.placeOrder(contract, stop)

    return {"placed": True, "orderId": parent_id, "entry": entry_px, "tp": tp_px, "sl": sl_px,
            "symbol": symbol, "side": side, "qty": qty}
