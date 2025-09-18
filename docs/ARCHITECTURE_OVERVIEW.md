Architecture Overview
======================

Purpose
- Give contributors and ChatGPT a concise mental model of M5‑Trader’s structure and how to run it.

Entry Points
- `run_pipeline.py`: Resolves and runs `src.rl_trading_pipeline.main()` (fallback to legacy import if needed).
- `run_adaptive_trading.py`: Runs the adaptive pipeline using `src.rl_trading_pipeline.RLTradingPipeline`.
- `tools/pipeline_monitor_5loops.py`: Runs 5 loop checks using `RLTradingPipeline` and `default_config`.

Canonical Modules (under `src/`)
- `src/rl_trading_pipeline.py`: The main trading pipeline orchestrator (IBKR ingest, feature engineering, decisions, execution).
- `src/market_data_ibkr_adapter.py`: IBKR adapter for contracts, subscriptions, and data ingestion.
- `src/feature_engineering.py`, `src/purged_cv.py`, etc.: Feature generation and utilities used by the pipeline.

Supporting Modules
- `utils/`: IO helpers, Parquet utilities, GPU metrics, pacing, health checks.
- `monitoring/`: Alerting, troubleshooting, client hooks, 5‑loop monitor; optional Grafana assets.
- `orders/`: Bridge and related order utilities.

Configuration
- Runtime env: copy `config/env.example` to `~/.config/m5_trader/env.local` and adjust.
- Trading parameters: prefer `configs/market_data_config.py` (env‑aware) for symbols/limits.
- Some modules still import `market_data_config.py` (top‑level) for compatibility; new work should use `configs.market_data_config`.

Data & Logs
- Data and logs live outside the repo by default:
  - Data: `~/.local/share/m5_trader/data`
  - Cache: `~/.cache/m5_trader`
  - Logs: `~/logs`
- The repo `.gitignore` excludes logs, caches, datasets, and models.

How To Run (paper)
- Ensure IB Gateway (Paper) is running and API enabled on `127.0.0.1:4002`.
- Create env: `python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt`
- Configure: `mkdir -p ~/.config/m5_trader && cp config/env.example ~/.config/m5_trader/env.local && edit`
- Run pipeline: `python run_pipeline.py`

Notes
- Use SSH remotes for Git to avoid auth prompts.
- Avoid committing secrets; update `config/env.example` and docs when adding new settings.
- Prefer `src/` package imports; legacy top‑level modules will be phased out over time.

