#!/usr/bin/env bash
set -euo pipefail

# Set PATH for cron environment
export PATH=/usr/local/bin:/usr/bin:/bin

PGUSER=$(awk -F= '/^PG_USER/{print $2}' ~/monitoring/.env)
PGDB=$(awk -F= '/^PG_DB/{print $2}' ~/monitoring/.env)

/usr/bin/docker compose -f ~/monitoring/docker-compose.yml exec -T db psql -U "$PGUSER" -d "$PGDB" -c "DELETE FROM market_ingest_events WHERE ts < now() - interval '30 days';"
