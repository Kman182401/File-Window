import sys, json, os
import psycopg2, psycopg2.extras

ENV = os.path.expanduser("~/monitoring/.env")

def load_env(path):
    env={}
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#') or '=' not in line: 
                continue
            k,v=line.split('=',1); env[k]=v
    return env

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        print(f"ERR: invalid JSON on stdin: {e}", file=sys.stderr)
        sys.exit(2)

    # Required + optional fields (match table schema exactly)
    symbol = payload.get("symbol")
    bars_ingested = payload.get("bars_ingested")
    gaps_found = payload.get("gaps_found", 0)
    latency_ms = payload.get("latency_ms", None)

    if symbol is None or bars_ingested is None:
        print("ERR: required fields: symbol, bars_ingested", file=sys.stderr)
        sys.exit(2)

    env = load_env(ENV)
    conn = psycopg2.connect(
        dbname=env["PG_DB"],
        user=env["PG_USER"],
        password=env["PG_PASSWORD"],
        host="127.0.0.1",
        port=5432
    )
    conn.autocommit = True
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute("""
            INSERT INTO market_ingest_events
                (ts, symbol, bars_ingested, gaps_found, latency_ms)
            VALUES (now(), %s, %s, %s, %s)
        """, (symbol, int(bars_ingested), int(gaps_found), None if latency_ms is None else int(latency_ms)))
    print("OK")

if __name__ == "__main__":
    main()
