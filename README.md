# File-Window System Files

## Sponsor: AI Workstation Upgrade

<div align="center">

<a href="https://cash.app/$REDACTED_CASH_APP" target="_blank">
  <img src="https://img.shields.io/badge/Cash%20App-REDACTED_CASH_APP-00C244?style=for-the-badge" alt="Donate via Cash App" />
</a>
<a href="https://github.com/Kman182401/ai-trading-system#sponsor-ai-workstation-upgrade" target="_blank">
  <img src="https://img.shields.io/badge/Sponsor-AI%20Workstation%20Upgrade-ff69b4?style=for-the-badge&logo=github" alt="Sponsor via GitHub links" />
</a>

</div>

Help me retire the 2016 high school budget PC and move this system onto a real AI workstation for faster training, lower latency, and above all, greater profits.

- Primary: Cash App tag: `[REDACTED_CASH_APP_TAG]` → https://cash.app/$REDACTED_CASH_APP
- Prefer email for larger contributions or hardware offers: [REDACTED_EMAIL]

## Sync Workflow

The `bin/clone` command keeps GitHub identical to the live trading system on this workstation. It copies any external folders listed in `.clone.toml` into the repository tree, auto-commits local edits, and pushes whenever changes are detected.

Common calls:

- Preview: `clone --dry-run`
- Sync + push: `clone --push` (or just `clone`)
- Include oversized files once: `clone --allow-large --push`
- Review branch: `clone --pr --push`
- Opt out of auto-commit: `clone --no-auto-commit --force`

`.clone.toml` maps external directories (e.g., `~/orders`) into subfolders inside the repo. Global and per-source excludes keep logs/caches out of Git. A cron job runs `clone --push --scan-secrets` every 15 minutes so ChatGPT can analyze this GitHub repo and get the full, current system without needing any extra context.

## IBKR Historical Data Backfill & Validation

New helper scripts live in `tools/` and rely on the existing `market_data_ibkr_adapter` so pacing, contract mapping, and persistence stay consistent.

1. Set shared environment variables (e.g. in your shell profile or systemd unit):
   ```bash
   export DATA_DIR="$HOME/.local/share/m5_trader/data"
   export LOCAL_FALLBACK_DIR="$DATA_DIR"
   export IBKR_HOST="127.0.0.1"
   export IBKR_PORT="4002"
   export IBKR_CLIENT_ID="9002"
   ```
   Pointing `LOCAL_FALLBACK_DIR` at `DATA_DIR` lets historical fallbacks read the Parquet cache without additional copies.

2. Backfill up to two years of 1-minute TRADES bars per symbol while respecting IBKR pacing limits:
   ```bash
   python3 tools/backfill_ibkr_history.py
   ```
   Override defaults with `LOOKBACK_DAYS`, `BACKFILL_WINDOW_DAYS`, or `BACKFILL_SLEEP_SECS` as needed. The helper automatically downsizes requests so each call stays within the "few thousand bars" guidance from IBKR's [historical data table](https://interactivebrokers.github.io/tws-api/historical_limitations.html) and enforces the published pacing rules (≤60 requests/10 min, ≤6 per 2 s, ≥15 s between identical requests).

3. Materialize and inspect continuous datasets for walk-forward testing:
   ```bash
   python3 tools/validate_and_materialize.py ES1! NQ1! XAUUSD EURUSD GBPUSD AUDUSD
   ```
   The script merges partitioned Parquet files, flags gaps larger than one minute (excluding the daily CME Globex maintenance window), counts how many gaps fall inside the expected break, and writes `ibkr_continuous/<SYMBOL>_continuous_1min.csv` under `$DATA_DIR`.

Both scripts assume `pyarrow` (or `fastparquet`) is available in the Python environment to read parquet files. For quick reference, the underlying API call is [`IB.reqHistoricalData`](https://ib-insync.readthedocs.io/api.html#ib_insync.ib.IB.reqHistoricalData).
