#!/usr/bin/env bash
set -u
REPO="$HOME/File-Window"
CMD="$REPO/bin/m5-sync.sh"
"$CMD" --if-needed >/dev/null 2>&1 || true
inotifywait -m -r -e modify,close_write,move,create,delete --exclude '(^|/)\.git(/|$)' "$REPO" |
while read -r _; do
  "$CMD" --if-needed >/dev/null 2>&1 || true
  sleep 2
done
