#!/usr/bin/env bash
set -u
REPO="$HOME/M5-Trader"
cd "$REPO" || exit 1
LOG_DIR="$HOME/logs/m5_autosync"
mkdir -p "$LOG_DIR"
NOW="$(date -Iseconds)"
HOST="$(hostname -s || echo host)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
exec 9>"$REPO/.git/m5_autosync.lock"
flock -n 9 || exit 0
if [ "${M5_AUTOSYNC:-1}" != "1" ]; then
  exit 0
fi
git fetch origin >/dev/null 2>&1 || true
if ! git rebase "origin/$BRANCH" >/dev/null 2>&1; then
  git rebase --abort >/dev/null 2>&1 || true
  echo "$NOW rebase-conflict $BRANCH" >> "$LOG_DIR/events.log"
  exit 1
fi
CHANGES=0
git diff --quiet || CHANGES=1
UNTRACKED="$(git ls-files --others --exclude-standard | head -n 1 || true)"
if [ -n "$UNTRACKED" ]; then
  CHANGES=1
fi
if [ "${1:-}" = "--if-needed" ] && [ "$CHANGES" -eq 0 ]; then
  exit 0
fi
if [ "$CHANGES" -eq 1 ]; then
  git add -A
  git commit -m "autosync: $HOST $NOW" >/dev/null 2>&1 || true
fi
if ! git push origin "$BRANCH" >/dev/null 2>&1; then
  git fetch origin >/dev/null 2>&1 || true
  if ! git rebase "origin/$BRANCH" >/dev/null 2>&1; then
    git rebase --abort >/dev/null 2>&1 || true
    echo "$NOW push-failed-rebase-conflict $BRANCH" >> "$LOG_DIR/events.log"
    exit 1
  fi
  if ! git push origin "$BRANCH" >/dev/null 2>&1; then
    echo "$NOW push-failed $BRANCH" >> "$LOG_DIR/events.log"
    exit 1
  fi
fi
echo "$NOW synced $BRANCH" >> "$LOG_DIR/events.log"
exit 0

