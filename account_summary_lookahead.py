# account_summary_lookahead.py
from typing import Dict, Tuple
from ib_insync import IB, AccountValue, util
import threading

_TAGS = [
    "NetLiquidation",
    "ExcessLiquidity",
    "AvailableFunds",
    "LookAheadExcessLiquidity",
    "LookAheadAvailableFunds",
    "LookAheadInitMarginReq",
    "LookAheadMaintMarginReq",
    "EquityWithLoanValue",
]

class AccountSummaryLookahead:
    """
    Lightweight cache of IB Account Summary, including LookAhead* fields.
    Non-blocking; safe to call from your trading loop.
    """
    def __init__(self, ib: IB, account_group: str = "All"):
        self.ib = ib
        self.account_group = account_group
        self._lock = threading.Lock()
        self._data: Dict[Tuple[str, str], float] = {}

        # Prime the cache and subscribe to updates
        self._subscribe()

    def _subscribe(self):
        # Request summary; ib-insync keeps results in ib.accountSummary()
        self.ib.reqAccountSummary(self.account_group)
        # Build initial map
        self._rebuild_map()

        # Live updates arrive via accountSummaryEvent; attach listener once
        self.ib.accountSummaryEvent += self._on_account_summary

    def _rebuild_map(self):
        m: Dict[Tuple[str, str], float] = {}
        for v in self.ib.accountSummary():
            if isinstance(v, AccountValue) and v.tag in _TAGS:
                key = (v.tag, (v.currency or "BASE"))
                try:
                    m[key] = float(v.value)
                except Exception:
                    # Non-numeric values are ignored
                    pass
        with self._lock:
            self._data = m

    def _on_account_summary(self, values):
        # values is a list of AccountValue
        changed = False
        with self._lock:
            for v in values:
                if v.tag in _TAGS:
                    key = (v.tag, (v.currency or "BASE"))
                    try:
                        self._data[key] = float(v.value)
                        changed = True
                    except Exception:
                        pass
        # No further action needed; readers will see latest snapshot

    def snapshot(self) -> Dict[str, float]:
        """
        Returns a flat dict using BASE currency where available.
        Missing keys are omitted.
        """
        with self._lock:
            m = dict(self._data)  # shallow copy

        def get(tag):
            # Prefer BASE consolidated; fall back to USD if present
            return m.get((tag, "BASE"), m.get((tag, "USD"), None))

        out = {}
        for tag in _TAGS:
            val = get(tag)
            if val is not None:
                out[tag] = val
        # Derived convenience metrics
        if "ExcessLiquidity" in out and "NetLiquidation" in out and out["NetLiquidation"] > 0:
            out["ExcessLiquidityPct"] = out["ExcessLiquidity"] / out["NetLiquidation"]
        if "LookAheadExcessLiquidity" in out and "NetLiquidation" in out and out["NetLiquidation"] > 0:
            out["LookAheadExcessLiquidityPct"] = out["LookAheadExcessLiquidity"] / out["NetLiquidation"]
        return out

