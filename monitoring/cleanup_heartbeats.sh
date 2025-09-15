#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/bin:/usr/bin
PGUSER=$(awk -F= '/^PG_USER/{print $2}' ~/monitoring/.env)
PGDB=$(awk -F= '/^PG_DB/{print $2}'  ~/monitoring/.env)
docker compose -f ~/monitoring/docker-compose.yml exec -T db psql -U "$PGUSER" -d "$PGDB" \
  -c "DELETE FROM heartbeats WHERE ts < now() - interval '30 days';"
