Agent Notes (M5‑Trader)
=======================

Scope: Entire repository

Guidance for agents (Codex/ChatGPT) working in this repo:

- Always push: When making changes/updates to this GitHub repo, always push to the remote upon successful completion so updates show up on GitHub when the process completes.
- Secrets: Never commit real secrets or `~/.config/m5_trader/env.local`. Update `config/env.example` and docs instead.
- Canonical imports: Prefer modules under `src/` (e.g., `src.rl_trading_pipeline`). Provide graceful fallbacks only where needed for compatibility.
- Ignore artifacts: Logs, data, caches, models, and `*.bak*` must remain untracked (see `.gitignore`).
- Minimal, surgical edits: Keep changes focused and consistent with existing style. Avoid renaming files unless part of an explicit cleanup or consolidation.

