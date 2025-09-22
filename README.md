# File-Window System Files

## Sponsor: AI Workstation Upgrade

<div align="center">

<a href="https://cash.app/$YaBoiBroke7567" target="_blank">
  <img src="https://img.shields.io/badge/Cash%20App-%24YaBoiBroke7567-00C244?style=for-the-badge" alt="Donate via Cash App" />
</a>
<a href="https://github.com/Kman182401/ai-trading-system#sponsor-ai-workstation-upgrade" target="_blank">
  <img src="https://img.shields.io/badge/Sponsor-AI%20Workstation%20Upgrade-ff69b4?style=for-the-badge&logo=github" alt="Sponsor via GitHub links" />
</a>

</div>

Help me retire the 2016 high school budget PC and move this system onto a real AI workstation for faster training, lower latency, and above all, greater profits.

- Primary: Cash App tag: `$YaBoiBroke7567` → https://cash.app/$YaBoiBroke7567
- Prefer email for larger contributions or hardware offers: komanderkody18@gmail.com

## Syncing for ChatGPT

The `bin/file_window_sync` utility mirrors the live trading system into this repository without touching source locations. Key pieces:

- Configuration lives in `.file-window-sync.yml`; an annotated sample is available at `examples/.file-window-sync.example.yml`.
- Default mirrors land under `mirror/` alongside a generated `MANIFEST_SYNC.json` so ChatGPT can locate every file with checksums.
- Run a dry-run audit: `./bin/file_window_sync`.
- Apply changes and push: `./bin/file_window_sync --push` or the shorthand wrapper `./bin/fw-sync`.
- Optional safety rails: `--scan-secrets`, `--allow-large`, `--tag`, `--pr`, `--manifest/--no-manifest`, and `--update-metadata` for `GPT_SYNC_VERSION.md`. Mark rarely used sources with `optional: true` to skip missing directories.
- A `Makefile` target (`make sync`) and cron/example snippets in `docs/file_window_sync.md` provide additional automation options.

The script never edits the original sources—only the mirrored copies living inside this repository.

## Clone Workflow

Run `bin/clone` to mirror local sources into `mirror/` and push:

- Preview changes: `./bin/clone --dry-run`
- Apply and push: `./bin/clone --push`
- Include large files: `./bin/clone --allow-large --push`
- Review branch: `./bin/clone --pr --push`
- Skip auto-commit safeguards: `./bin/clone --no-auto-commit --force`

By default the command auto-commits any dirty working tree before mirroring. Configuration lives in `.clone.toml`; manifests land in `mirror/MANIFEST_CLONE.json`.
