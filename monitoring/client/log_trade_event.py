#!/usr/bin/env python3
import sys, os, json, subprocess, shlex

env_path = os.path.expanduser('~/monitoring/.env')
pg = {}
with open(env_path) as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        k,v=line.split('=',1); pg[k]=v

def lit(x):
    if x is None: return "NULL"
    if isinstance(x,(int,float)): return str(x)
    s = str(x).replace("'","''")
    return f"'{s}'"

payload = json.loads(sys.stdin.read() or '{}')
cols = ["ts","symbol","side","qty","entry_price","exit_price","pnl","duration_sec","confidence","status","order_id","reason"]
vals = [payload.get(c) for c in cols]
sql = "INSERT INTO trades (" + ",".join(cols) + ") VALUES (" + ",".join(lit(v) for v in vals) + ");"
cmd = ["docker","exec","-e",f"PGPASSWORD={pg.get('PG_PASSWORD','')}",
       "monitoring-db-1","psql","-U",pg["PG_USER"],"-d",pg["PG_DB"],"-c",sql]
subprocess.run(cmd, check=False)
print("OK")