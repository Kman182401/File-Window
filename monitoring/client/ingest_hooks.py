import json, subprocess

def log_ingest(symbol: str, bars: int, gaps: int = 0, lat_ms: int | None = None) -> None:
    payload = json.dumps({"symbol": symbol, "bars_ingested": int(bars), "gaps_found": int(gaps), "latency_ms": (None if lat_ms is None else int(lat_ms))})
    subprocess.run(
        ["python3", "/home/karson/monitoring/client/log_ingest_event.py"],
        input=payload.encode("utf-8"),
        check=True
    )
