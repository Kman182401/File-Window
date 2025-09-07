import os, time, json, datetime
from typing import Optional, Dict, Any
import psycopg2
import psycopg2.extras

ENV_FILE = os.path.expanduser("~/monitoring/.env")

def _load_env(path: str) -> Dict[str, str]:
    env = {}
    with open(path, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line: 
                continue
            k,v = line.split("=",1)
            env[k.strip()] = v.strip()
    return env

class HeartbeatLogger:
    def __init__(self, env_path: str = ENV_FILE, retries: int = 10, backoff: float = 0.5):
        self.env = _load_env(env_path)
        self._conn = None
        self.retries = retries
        self.backoff = backoff

    def _connect(self):
        if self._conn and not self._conn.closed:
            return
        last_err = None
        for i in range(self.retries):
            try:
                self._conn = psycopg2.connect(
                    dbname=self.env["PG_DB"],
                    user=self.env["PG_USER"],
                    password=self.env["PG_PASSWORD"],
                    host="127.0.0.1",
                    port=5432,
                )
                self._conn.autocommit = True
                return
            except Exception as e:
                last_err = e
                time.sleep(self.backoff * (2 ** i))
        raise last_err

    def log(self,
            run_id: str,
            phase: str,
            pnl: Optional[float] = None,
            drawdown: Optional[float] = None,
            latency_ms: Optional[int] = None,
            orders_placed: Optional[int] = None,
            news_ingested: Optional[int] = None,
            positions: Optional[Dict[str, Any]] = None):
        self._connect()
        payload = {
            "ts": datetime.datetime.utcnow(),
            "run_id": run_id,
            "phase": phase,
            "pnl": pnl,
            "drawdown": drawdown,
            "latency_ms": latency_ms,
            "orders_placed": orders_placed,
            "news_ingested": news_ingested,
            "positions": json.dumps(positions) if positions is not None else None,
        }
        with self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                INSERT INTO heartbeats (ts, run_id, phase, pnl, drawdown, latency_ms, orders_placed, news_ingested, positions)
                VALUES (%(ts)s, %(run_id)s, %(phase)s, %(pnl)s, %(drawdown)s, %(latency_ms)s, %(orders_placed)s, %(news_ingested)s,
                        CASE WHEN %(positions)s IS NULL THEN NULL ELSE %(positions)s::jsonb END)
            """, payload)
        return True
