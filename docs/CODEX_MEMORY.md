M5-Trader Ops — IBG Client IDs, Ingest/Parquet, and Repo Autosync

Scope: This project now standardizes IB Gateway client IDs, stabilizes ingest (historical bars → Parquet), fixes the training gate to detect Parquet (not CSV), and enables automatic Git pushes on file change with systemd + inotify.
Stable IBG Client IDs & Heartbeat

    Client IDs:

        9002 → pipeline (long-lived).

        9001 → smokes/diagnostics (short-lived tests).

    Why: Prevents collisions/stale sessions. Keeps the pipeline’s socket “sticky.”

    Heartbeat: Pipeline sends a 30s heartbeat and has auto-reconnect; smokes are one-shot.

Codex actions (when to use):

    If the pipeline fails to connect or times out, first free 9002 (stale session), then prove connectivity with a 3× heartbeat smoke.

Commands (copy/paste):

ss -ltnp | grep 4002 || echo "No 4002 listener"

ps -ef | awk '/run_pipeline\.py|src\/rl_trading_pipeline\.py/{print $2}' | xargs -r kill

cd ~/M5-Trader

. .venv/bin/activate

python - <<'PY'
import os, time
from ib_insync import IB
ib = IB()
ib.connect(os.getenv("IBKR_HOST","127.0.0.1"), int(os.getenv("IBKR_PORT","4002")), clientId=9002, timeout=10)
print("connected?", ib.isConnected(), "server:", ib.client.serverVersion(), "accounts:", ib.managedAccounts())
for i in range(3):
    ib.reqCurrentTime(); print("heartbeat", i+1); time.sleep(30)
ib.disconnect()
PY

Pass criteria: 3 heartbeats without disconnects or resets.
Historical Ingest → Parquet (canonical) + One-Shot Proof

    Adapter behavior: IBKRIngestor.fetch_data() resolves the contract, paces requests, and persists bars as Parquet under partitioned paths symbol=/date=/bars.parquet.

    Symbols allowed: ES1!, NQ1!, XAUUSD, EURUSD, GBPUSD, AUDUSD. Others are rejected by design.

Codex actions (when to use):

    Use a one-shot ingest to verify fetch + persistence if the pipeline reports “no data.”

Commands (copy/paste):

cd ~/M5-Trader

. .venv/bin/activate

python - <<'PY'
from market_data_ibkr_adapter import IBKRIngestor
ing = IBKRIngestor()
df = ing.fetch_data("ES1!", duration="30 min", barSize="1 min", whatToShow="TRADES")
print("rows:", 0 if df is None else len(df))
PY

python - <<'PY'
from pathlib import Path; import pandas as pd, os
base = Path(os.getenv("DATA_DIR", str(Path.home()/".local/share/m5_trader/data")))
paths = sorted((base/"symbol=ES1!").rglob("bars.parquet"))
print("files:", len(paths))
if paths: print("latest:", paths[-1], "rows:", len(pd.read_parquet(paths[-1])))
PY

Pass criteria: Non-zero rows and a new Parquet file.
Note: Off-hours, retry with whatToShow="MIDPOINT" or larger duration.
Training Gate Fixed: Detect Parquet (not CSV)

    Problem fixed: new_data_available() originally looked for *_TRADES.csv, falsely reporting “no new data” while Parquet grew.

    Now: Function scans the latest bars.parquet under symbol=/date= and compares mtime with last_check_time.

Codex actions (when to use):

    If retraining doesn’t trigger despite new ingests, confirm this function is the Parquet-aware version.

Run Pipeline Foreground (observe live requests)

    Why: See contract requests, bar counts, and reconnect/heartbeat messages in real time.

    Client ID: 9002.

Commands (copy/paste):

cd ~/M5-Trader

. .venv/bin/activate

PYTHONUNBUFFERED=1 LOG_LEVEL=INFO IBKR_CLIENT_ID=9002 python -u run_pipeline.py 2>&1 | tee -a ~/logs/pipeline_run.out

Expect in log:

    [IBKRIngestor] connecting to 127.0.0.1:4002 clientId=9002 → Connected successfully …

    Requesting contract: … → bar counts → Parquet written under symbol=/date=.

    Heartbeat / auto-reconnect messages at ~30s cadence.

Git Autosync: Real-Time Commits & Pushes (inotify + systemd)

    What it is:

        bin/m5-watch.sh → recursive watcher (inotify).

        bin/m5-sync.sh → one-shot safe sync: fetch → rebase → add -A → commit-if-needed → push, guarded by flock.

        m5trader-watch.service → runs watcher (restart-always).

        m5trader-scan.timer → periodic sweep every 2 minutes (catch missed events).

        loginctl enable-linger → services persist across logout/reboot.

    Why: Keeps GitHub in lockstep with local edits; avoids race conditions and missed subdirs.

Codex actions (when to use):

    Assume autosync is on by default; use it to publish small code edits automatically.

    Pause/resume when doing large local experiments (see below).

Operate / pause / resume (copy/paste):

systemctl --user status m5trader-watch.service

systemctl --user list-timers | grep m5trader-scan || true

systemctl --user stop m5trader-watch.service m5trader-scan.timer

systemctl --user start m5trader-watch.service m5trader-scan.timer

Guardrails:

    Keep .gitignore strict (no datasets/models/secrets).

    Autosync commits with a standard message; no force-push; conflicts are logged to ~/logs/m5_autosync/events.log.

Acceptance Checklist (Codex self-check before reporting “healthy”)

    IB Gateway (Paper) running on 127.0.0.1:4002; API enabled; Trusted IP includes 127.0.0.1; “Download open orders on connection” ON.

    Exactly one pipeline process is connected with clientId=9002; smokes use 9001 only.

    Ingest calls resolve canonical tickers only: ES1!, NQ1!, XAUUSD, EURUSD, GBPUSD, AUDUSD.

    Logs show Requesting contract and bar counts; Parquet partitions grow under symbol=/date=.

    Heartbeats & auto-reconnect present (9002 stays up).

    new_data_available() is Parquet-aware, and retrain gates reflect new data.

    Autosync services active; recent code edits appear on GitHub without manual pushes.

Triage Hints (Codex playbook)

    Intermittent timeouts / resets on 9002: Likely stale session or ID collision. Free 9002 (kill stray process or restart IBG) and re-run the 3× heartbeat smoke.

    Zero bars off-hours: Retry with whatToShow="MIDPOINT" or longer duration.

    Pipeline says “no new data”: Recheck new_data_available() (must scan Parquet).

    Autosync didn’t push: See ~/logs/m5_autosync/events.log; run periodic sweep via m5trader-scan.timer.
