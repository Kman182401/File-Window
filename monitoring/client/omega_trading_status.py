import json
import os
import statistics
import tempfile
import time
from collections import deque
from typing import Dict, Any

OUT_PATH = os.path.expanduser("~/.local/share/omega/trading_status.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def _write_atomic_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


class TradingTelemetry:
    """Track rolling trading stats and persist to a JSON file for Polybar."""

    def __init__(self, sharpe_window: int = 300, risk_free: float = 0.0) -> None:
        self.risk_free = float(risk_free)
        self.rets = deque(maxlen=int(sharpe_window))
        self.equity = 1.0
        self.peak = 1.0
        self.max_drawdown = 0.0
        self.pnls: Dict[str, float] = {}

    def push_return(self, r: float) -> None:
        self.rets.append(float(r))
        self.equity *= 1.0 + float(r)
        self.peak = max(self.peak, self.equity)
        if self.peak > 0:
            drawdown = (self.equity - self.peak) / self.peak
            self.max_drawdown = min(self.max_drawdown, drawdown)

    def set_pnl(self, key: str, value: float) -> None:
        self.pnls[key] = float(value)

    def write(self) -> None:
        sharpe = None
        if len(self.rets) >= 5:
            excess = [r - self.risk_free for r in self.rets]
            mean = sum(excess) / len(excess)
            stdev = statistics.pstdev(excess)
            if stdev > 1e-16:
                sharpe = mean / stdev

        data = {
            "ts": int(time.time()),
            "sharpe": None if sharpe is None else round(sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "pnl": {k: round(v, 2) for k, v in self.pnls.items()},
        }

        _write_atomic_json(OUT_PATH, data)
