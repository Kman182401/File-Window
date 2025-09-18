#!/usr/bin/env python3
"""
Ingest Monitor — checks IBKR connectivity, pipeline process, and Parquet partitions.

Outputs a concise status report for live and historical market data ingestion.
Safe by default: uses clientId=9001 for smoke probes and reads Parquet under DATA_DIR.
"""
from __future__ import annotations
import os, sys, json, time, socket, subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def check_port(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def check_ibkr_connectivity(host: str, port: int, client_id: int = 9001, timeout: int = 7) -> Dict[str, Any]:
    out: Dict[str, Any] = {"host": host, "port": port, "client_id": client_id, "reachable": False}
    try:
        from ib_insync import IB, util
    except Exception as e:
        out.update({"error": f"ib_insync import failed: {e}"})
        return out
    try:
        ib = IB()
        util.startLoop()
        ib.connect(host, port, clientId=client_id, timeout=timeout)
        out.update({
            "reachable": ib.isConnected(),
            "server": getattr(ib.client, "serverVersion", lambda: None)(),
            "accounts": ib.managedAccounts(),
        })
    except Exception as e:
        out.update({"error": str(e)})
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass
    return out


def list_latest_parquets(data_dir: Path, symbols: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    import pandas as pd
    now = datetime.now(timezone.utc)
    for sym in symbols:
        base = data_dir / f"symbol={sym}"
        latest: Optional[Path] = None
        try:
            if not base.exists():
                rows.append({"symbol": sym, "status": "missing", "detail": f"no partition at {base}"})
                continue
            cands = sorted(base.rglob("bars.parquet"))
            if not cands:
                rows.append({"symbol": sym, "status": "empty", "detail": "no bars.parquet"})
                continue
            latest = cands[-1]
            st = latest.stat()
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            age_min = (now - mtime).total_seconds() / 60.0
            # lightweight read for row count
            try:
                nrows = len(pd.read_parquet(latest))
            except Exception as e:
                nrows = -1
                rows.append({"symbol": sym, "status": "error", "detail": f"read_parquet failed: {e}", "path": str(latest), "age_min": round(age_min,2)})
                continue
            status = "fresh" if age_min <= 60 else ("stale" if age_min <= 1440 else "ancient")
            rows.append({"symbol": sym, "status": status, "rows": nrows, "path": str(latest), "age_min": round(age_min, 2)})
        except Exception as e:
            rows.append({"symbol": sym, "status": "error", "detail": str(e), "path": str(latest) if latest else None})
    return rows


def pipeline_pids() -> List[int]:
    try:
        out = subprocess.check_output(["bash", "-lc", "ps -eo pid,cmd | rg -n 'run_pipeline.py|src/rl_trading_pipeline.py' | awk '{print $1}'"], text=True)
        return [int(x) for x in out.split() if x.strip().isdigit()]
    except Exception:
        return []


def main() -> None:
    host = _env("IBKR_HOST", "127.0.0.1")
    port = int(_env("IBKR_PORT", "4002"))
    data_dir = Path(_env("DATA_DIR", str(Path.home() / ".local/share/m5_trader/data")))

    # Symbols: prefer adapter canonical set
    symbols = ["ES1!", "NQ1!", "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD"]

    report: Dict[str, Any] = {"ts": datetime.now(timezone.utc).isoformat()}

    report["gateway_listening"] = check_port(host, port)
    report["ibkr"] = check_ibkr_connectivity(host, port, client_id=9001, timeout=7)
    report["pipeline_pids"] = pipeline_pids()
    report["data_dir"] = str(data_dir)
    report["parquets"] = list_latest_parquets(data_dir, symbols)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

