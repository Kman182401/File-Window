# Repository Guidelines

## Project Structure & Module Organization
- Source lives in `src/` (e.g., `rl_trading_pipeline.py`, `feature_engineering.py`).
- Orchestration and entry points: `start_trading_system.py`, `train_and_trade_pipeline.py`.
- Configuration in `config/` (e.g., `master_config.py`, `secrets_manager.py`); utilities in `utils/`; monitoring in `monitoring/`; CI in `.github/workflows/`.
- Tests in `tests/` (pytest + Hypothesis). Additional smoke/integration scripts exist at repo root and in `tools/`.
- Cross‑AI mirror: `gpt-files-repo/` stays in sync via `./sync_gpt_files.sh`.

## Build, Test, and Development Commands
- `make lint` — Run Ruff linting and imports.
- `make type` — Run MyPy type checks.
- `make test` — Execute pytest (configured to `tests/`).
- `make cov` — Tests with coverage summary.
- `make all` — Lint, type, and test.
- Run locally: `python start_trading_system.py` or `python train_and_trade_pipeline.py`.

## Coding Style & Naming Conventions
- Python 3.10; line length 100; target cyclomatic complexity ≤ 12.
- 4‑space indentation; add type hints for new/changed public functions.
- Naming: files/modules `snake_case.py`; functions/vars `snake_case`; classes `PascalCase`; constants `UPPER_SNAKE`.
- Tools: Ruff and MyPy configured in `pyproject.toml`. Optional hooks: `pip install pre-commit && pre-commit install && pre-commit run -a`.

## Testing Guidelines
- Framework: pytest (quiet mode) with property tests via Hypothesis where helpful.
- Place tests under `tests/` and name `test_*.py`; prefer small, deterministic units plus focused integration tests.
- Run: `make test`; coverage: `make cov`.
- IBKR integration: prefer dry‑run/paper; never arm orders in tests.

## Commit & Pull Request Guidelines
- History includes automated “Auto-sync” commits. For human changes, use imperative subjects (e.g., "Improve IBKR reconnect logic").
- Include scope when useful (e.g., `ibkr:`, `rl:`, `infra:`) and a concise body.
- PRs must include: purpose, linked issues, test plan/output, relevant logs/screenshots for pipeline changes, and risk/rollback notes.

## Security & Agent-Specific Tips
- Never commit secrets (`.env`, keys). Use `config/secrets_manager.py` and environment variables.
- IBKR single‑client mode: reserve `clientId=9002`; gate orders with `ENABLE_ORDER_EXEC`, `ALLOW_ORDERS`, `DRY_RUN` (keep disabled for tests).
- Keep `gpt-files-repo/` synchronized (`./sync_gpt_files.sh`) after system changes to support cross‑AI collaboration.

