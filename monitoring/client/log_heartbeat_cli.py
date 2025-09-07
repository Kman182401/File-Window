#!/usr/bin/env python3
"""CLI wrapper for heartbeat logging - accepts JSON via stdin"""
import sys
import json
from heartbeat import HeartbeatLogger

if __name__ == "__main__":
    try:
        # Read JSON from stdin
        data = json.loads(sys.stdin.read())
        
        # Create logger and emit heartbeat
        hb = HeartbeatLogger()
        ok = hb.log(
            run_id=data.get("run_id", "UNKNOWN"),
            phase=data.get("phase", "unknown"),
            pnl=data.get("pnl", 0),
            drawdown=data.get("drawdown", 0),
            latency_ms=data.get("latency_ms", 0),
            orders_placed=data.get("orders_placed", 0),
            news_ingested=data.get("news_ingested", 0),
            positions=data.get("positions", {})
        )
        
        # Exit with appropriate code
        sys.exit(0 if ok else 1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)