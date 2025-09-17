import time
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple, Any


class PaceGate:
    """
    Token-bucket gate with per-key rate and duplicate suppression window.
    key: tuple like (conId, whatToShow) or (contract_key, tickType)
    """

    def __init__(self, rate_per_sec: float = 3.0, burst: int = 3, dupe_cooldown_s: float = 15.0):
        self.rate = float(rate_per_sec)
        self.tokens: Dict[Any, float] = defaultdict(lambda: float(burst))
        self.updated: Dict[Any, float] = defaultdict(lambda: time.monotonic())
        self.lock = threading.Lock()
        self.dupes: Dict[Any, Deque[Tuple[float, str]]] = defaultdict(lambda: deque(maxlen=128))
        self.dupe_cooldown_s = float(dupe_cooldown_s)

    def _refill(self, key: Any) -> None:
        now = time.monotonic()
        elapsed = now - self.updated[key]
        add = elapsed * self.rate
        # cap bucket to avoid unbounded bursts after idle
        self.tokens[key] = min(self.tokens[key] + add, 10.0)
        self.updated[key] = now

    def acquire(self, key: Any, req_fingerprint: str) -> None:
        """
        Blocks until a token is available for this key and the request isn't a recent duplicate.
        req_fingerprint: stable string describing the exact historical request.
        """
        while True:
            with self.lock:
                self._refill(key)

                # purge aged dupes
                cutoff = time.monotonic() - self.dupe_cooldown_s
                dq = self.dupes[key]
                while dq and dq[0][0] < cutoff:
                    dq.popleft()

                # reject if identical request still "hot"
                is_dupe = any(fp == req_fingerprint for _, fp in dq)
                if not is_dupe and self.tokens[key] >= 1.0:
                    self.tokens[key] -= 1.0
                    dq.append((time.monotonic(), req_fingerprint))
                    return

            time.sleep(0.2)

