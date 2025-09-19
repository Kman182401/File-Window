#!/usr/bin/env bash
set -euo pipefail
TARGET="$HOME/File-Window/ibg_logs"
CANDIDATE=""
if [ -d "$HOME/Jts" ]; then
  C=$(ls -1dt "$HOME"/Jts/ibgateway/*/logs 2>/dev/null | head -n 1 || true)
  [ -n "${C:-}" ] && CANDIDATE="$C"
fi
if [ -z "${CANDIDATE:-}" ] && [ -d "$HOME/IBJts" ]; then
  C=$(ls -1dt "$HOME"/IBJts/ibgateway/*/logs 2>/dev/null | head -n 1 || true)
  [ -n "${C:-}" ] && CANDIDATE="$C"
fi
if [ -z "${CANDIDATE:-}" ]; then
  echo "No IB Gateway logs directory found" >&2
  exit 1
fi
ln -sfn "$CANDIDATE" "$TARGET"
echo "Linked $TARGET -> $CANDIDATE"
