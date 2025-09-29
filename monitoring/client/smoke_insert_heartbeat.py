from datetime import datetime, UTC
from heartbeat import HeartbeatLogger

if __name__ == "__main__":
    hb = HeartbeatLogger()
    run_id = f"SMOKE_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
    ok = hb.log(
        run_id=run_id,
        phase="smoke",
        pnl=0.0,
        drawdown=0.0,
        latency_ms=123,
        orders_placed=0,
        news_ingested=0,
        positions={"ES": 0, "NQ": 0}
    )
    print(run_id if ok else "FAILED")
