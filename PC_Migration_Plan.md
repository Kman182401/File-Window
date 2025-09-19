# M5‑Trader Migration Plan — EC2 m5.large → Local PC

This document is the authoritative plan to migrate and tune the system from the prior AWS EC2 m5.large (2 vCPU / 8 GB RAM / no GPU) to this PC. All changes must respect the Comfortable Operation Constraint.

Comfortable Operation Constraint
- Any and all changes must allow the system to run comfortably on this PC without causing instability, resource exhaustion, or user disruption.
- Target budgets: CPU ≤ 80% sustained, RAM ≤ 12 GB used by stack, GPU VRAM ≤ 2.4 GB (80% of 3 GB), disk I/O favors sequential writes.

PC Specs Summary (from `~/PC_Specs` and live checks)
- CPU: Intel i7‑6700K (4C/8T, 4.0–4.2 GHz)
- RAM: 16 GB DDR4, swap 4 GB
- GPU: NVIDIA GTX 1060 3 GB, driver 535.247.01, CUDA 12.2 toolchain present
- Storage: 1 TB HDD (WD Blue); root ext4; ample free space
- OS: Ubuntu 24.04.3 LTS, kernel 6.14.0‑29‑generic

Repo State and Inventory
- Repo path: `File-Window` (renamed from M5-Trader). Git remote: `origin https://github.com/Kman182401/File-Window.git`.
- Local is behind origin; latest remote objects have been fetched, but working tree not updated to preserve local changes.
- Local changes detected (kept intact): `AGENTS.md`, `audit_logging_utils.py`, `comprehensive_system_monitor.py`, `enhanced_resource_monitor.py`, `performance_monitoring_system.py`, `requirements.txt`, `utils/gpu_metrics.py`.
- No submodules; no Git LFS patterns in `.gitattributes`.

High‑Level Goals
- Replace EC2 CPU‑only constraints with GPU‑aware but VRAM‑bounded configs.
- Remove EC2‑specific hard‑coded paths and home dirs; use portable locations.
- Keep paper‑trading and safety guards as defaults until local baseline is validated.
- Shift caches and data formats to Parquet/append‑only to respect HDD I/O.
- Add short/medium test suites; keep 120 s smokes for dev loops.

Subsystem Plans (What and Why)

1) Paths, Layout, and Portability
- Why: Many files hard‑code `/home/karson`. On this PC (`/home/karson`) that breaks logging, models, and scripts.
- What to change (replace hard‑coded paths with `Path.home()` or env vars):
  - `rl_trading_pipeline.py:15` → drop ``; rely on package‑relative imports.
  - `order_management_system.py:23` → drop ``.
  - `export_ibkr_data_to_csv.py:12` → `DATA_DIR = os.getenv('DATA_DIR', str(Path.home()/"data"))`.
  - `test_minimal_pipeline.py:28` → `Path.home()/"logs"/"minimal_pipeline.log"`.
  - `monitoring/client/ingest_hooks.py:6` → use `Path(__file__).resolve().parent/"log_ingest_event.py"`.
  - `config/master_config.py:...` (default config dir) → replace `/home/karson/config` with `Path.home()/".config"/"m5_trader"`.
  - All model paths under `src/rl_trading_pipeline.py` and `rl_trading_pipeline.py` where `/home/karson/models/...` occurs (see: `src/rl_trading_pipeline.py:838, 851, 1116, 1534, 1536, 1538`; `rl_trading_pipeline.py:1416, 1418, 1859, 2710`) → use `MODELS_DIR = Path(os.getenv('MODELS_DIR', Path.home()/"models"))`.
- Guard: Ensure directory creation with `mkdir(parents=True, exist_ok=True)` before writing.

2) GPU Enablement (bounded)
- Why: Prior system avoided GPU; this PC has GTX 1060 (3 GB). Enable CUDA carefully.
- PyTorch: Replace hard‑coded `'device': 'cpu'` with auto‑select in RL agents.
  - `recurrent_ppo_agent.py` and `sac_trading_agent.py`: set `device = 'cuda' if torch.cuda.is_available() else 'cpu'`; keep networks small; keep gradients clipped.
- RL batch sizing (VRAM‑aware): start `num_envs=4`, `n_steps=512`, `batch_size=32–64k logical via gradient accumulation if needed` (3 GB cap). Enable AMP autocast for inference; test stability for training.
- XGBoost/LightGBM: keep LightGBM CPU (`num_threads=6`); XGBoost not used currently. If added, test `tree_method=gpu_hist` with small DMatrix; fall back to `hist` on OOM.

3) Concurrency and Threading
- Why: 4C/8T permits more parallelism than EC2; avoid starving the desktop.
- Set defaults:
  - Classical ML/Data processing: prefer `n_jobs=6`, LightGBM `num_threads=6`.
  - ThreadPoolExecutors (e.g., `parallel_data_pipeline.py` and monitors): tune `max_workers=6`.
  - PyTorch DataLoader (if/where introduced): `num_workers=4–6`, `pin_memory=True`, `persistent_workers=True`.
- Keep 2 threads headroom for UI and background services.

4) Data Format, Caching, and I/O
- Why: HDD random I/O is slower than EC2 gp3; use sequential, columnar formats.
- Use Parquet for datasets and caches (already present in `ib_single_socket.py`, `market_aware_data_manager.py`, `parallel_data_pipeline.py`).
- Layout:
  - `DATA_DIR = ~/.local/share/m5_trader/data` (override via env).
  - `CACHE_DIR = ~/.cache/m5_trader` for temp/model caches.
  - `MODELS_DIR = ~/models` (or `~/.local/share/m5_trader/models`).
- Patterns: append‑only Parquet per symbol per day/week; avoid tiny CSVs; use mmap for large arrays (already supported in utils).

5) IBKR Connectivity
- Keep existing env discipline: `IBKR_HOST=127.0.0.1`, `IBKR_PORT=4002` (paper), `IBKR_CLIENT_ID=9002`.
- Ensure all IBKR entrypoints use env defaults from `configs/market_data_config.py:18–20` (they already do).
- Prefer single long‑lived socket (`ib_single_socket.py`); keep the optional connect tracer (file rotates into `~/logs`).

6) Monitoring, Logging, and Rotation
- Use existing monitors (`comprehensive_system_monitor.py`, `performance_monitoring_system.py`, `enhanced_resource_monitor.py`) with GPU metrics via `utils/gpu_metrics.py` (NVML or `nvidia‑smi` fallback).
- Log targets: redirect any `/home/karson/logs` to `~/logs` (create if missing). Keep `logrotate` to 14 days with size caps.
- Grafana/Postgres docker compose (monitoring/docker‑compose.yml) is fine for localhost; keep ports bound to `127.0.0.1`.

7) Tests and CI‑like Routines (local)
- Keep 120 s smoke tests (`tests/smoke_symbol_resolution.py`, `smoke_ib_connect.py`) as fast checks.
- Add short (≤5 min) nightly ingestion + RL update test; keep DRY_RUN and order guards.
- Use `pytest -q` locally; pin a “nightly” script to run 5–10 min RL/ingestion validations.

8) Safety and Run Modes
- Defaults: `ALLOW_ORDERS=0`, `DRY_RUN=1`, and paper trading on `127.0.0.1:4002` until baseline is validated.
- Reserve client IDs to avoid collisions: 9001 (smokes), 9002 (pipeline), 9007 (orders bridge) as documented in scripts.

9) Dependencies and CUDA Alignment
- Torch: current pin `torch==2.0.1` likely CPU‑only for this box; for CUDA, install compatible wheel (e.g., PyTorch 2.4.x cu121) and verify `torch.cuda.is_available()`.
- Keep JAX CPU wheels (GPU jaxlib is optional; GPU memory is tight on 3 GB).
- Validate `nvidia-smi` driver (535.247.01) matches CUDA runtime reported by PyTorch; prefer env isolation in `venv`.

10) S3 and Cloud Integrations
- Many modules have S3 hooks (`parallel_data_pipeline.py`, `market_aware_data_manager.py`, `src/rl_trading_pipeline.py`). On this PC, prefer local mode by default; keep S3 code paths gated by config/env flags.
- Document AWS credentials optional use; default to local storage for all datasets and models.

11) Duplicate Modules in `src/` vs root
- Prefer root modules (per `AGENTS.md`).
- Action: consolidate and remove legacy duplicates after migration validation to reduce drift (e.g., `src/rl_trading_pipeline.py` vs `rl_trading_pipeline.py`).

Concrete File‑Level Changes (surgical list)
- `rl_trading_pipeline.py:15` — remove `` and ensure imports resolve via project root.
- `order_management_system.py:23` — remove ``.
- `export_ibkr_data_to_csv.py:12` — make `DATA_DIR` env/`Path.home()` based.
- `test_minimal_pipeline.py:28` — log path to `Path.home()/"logs"/"minimal_pipeline.log"`.
- `monitoring/client/ingest_hooks.py:6` — call `python3` on a repo‑relative script via `Path(__file__).parent`.
- `config/master_config.py` — change default config dir from `/home/karson/config` to `Path.home()/".config"/"m5_trader"`.
- `recurrent_ppo_agent.py` and `sac_trading_agent.py` — set device auto‑select (`cuda` if available), keep networks compact; respect memory limits already in code.
- `requirements.txt` — consider upgrading PyTorch to a CUDA‑compatible wheel only after verifying driver/toolchain; keep CPU fallback path.

Tuning Knobs for This PC (defaults after migration)
- RL PPO:
  - `device`: auto
  - `num_envs`: 4
  - `n_steps`: 512
  - `batch_size`: 32–64k logical (use grad accumulation if VRAM pressure)
  - `policy_net`: [64,64] baseline; optionally [128,128] if stable
- LightGBM: `num_threads=6`; XGBoost: CPU `hist` unless GPU proven stable
- Thread pools: `max_workers=6` for pipeline/monitors
- Data: Parquet append‑only, weekly files per symbol; mmap for large arrays

Validation Plan (post‑change)
1. Environment sanity:
   - Verify `nvidia-smi` and `python -c 'import torch; print(torch.cuda.is_available())'`.
   - `pytest -q` plus smoke tests with `DRY_RUN=1` and IB Gateway running.
2. Functional baseline:
   - Run `run_adaptive_trading.py` with paper account for 30–60 minutes; confirm zero fatal errors, stable latencies, and GPU VRAM < 2.4 GB.
3. Performance checks:
   - Grafana dashboards available at `http://127.0.0.1:3000`; confirm ingestion latencies and GPU metrics.
4. Storage/I/O:
   - Confirm Parquet caches under `~/.local/share/m5_trader/data` grow append‑only; HDD usage stays low on random I/O.
5. Rollback:
   - All changes are env‑guarded; CPU paths remain. Revert to CPU by setting `CUDA_VISIBLE_DEVICES=""` or forcing `device='cpu'`.

Deferred Enhancements (optional but high impact)
- Add a SATA/NVMe SSD for datasets and caches (500 GB+). This is the single biggest responsiveness win on this hardware.
- Try a small transformer for sentiment (Distil‑RoBERTa finance) with batched GPU inference; fall back to CPU on OOM.
- Re‑enable drift checks with batched, low‑frequency tests after retraining cadence stabilizes.

Notes
- Paper‑trading, env guardrails (`DRY_RUN`, `ALLOW_ORDERS`) remain defaults until we re‑baseline locally.
- All path changes use env‑first with sensible defaults under the user home directory for portability.
