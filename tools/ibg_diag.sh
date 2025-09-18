#!/usr/bin/env bash
set -euo pipefail
HOST="${IBKR_HOST:-127.0.0.1}"
PORT="${IBKR_PORT:-4002}"
echo "=== IB Gateway process ==="
pgrep -fa ibgateway || true
echo
echo "=== Listener on $HOST:$PORT ==="
ss -ltnp | grep -E ":${PORT}\b" || true
echo
echo "=== Established connections on port $PORT ==="
ss -tanp | grep -E ":${PORT}\b" || true
echo
LOGDIR="$HOME/M5-Trader/ibg_logs"
if [ -d "$LOGDIR" ]; then
  LATEST=$(ls -1t "$LOGDIR"/*.log 2>/dev/null | head -n 1 || true)
  if [ -n "${LATEST:-}" ]; then
    echo "=== Tail of $LATEST ==="
    tail -n 120 "$LATEST" || true
  else
    echo "No *.log files found under $LOGDIR"
  fi
else
  echo "Log directory $LOGDIR not found; run tools/update_ibg_log_symlink.sh"
fi

