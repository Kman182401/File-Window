# File-Window Sync Automation

This document captures the operational playbook for mirroring the local AI trading
system into the `File-Window` repository so that ChatGPT sees the exact,
current state of the environment.

## Script Overview

- Entry point: `bin/file_window_sync`
- Wrapper: `bin/fw-sync` → runs `file_window_sync --push`
- Configuration: `.file-window-sync.yml`
- Mirror target: `mirror/`
- Manifest: `mirror/MANIFEST_SYNC.json`
- Metadata: `GPT_SYNC_VERSION.md` (updated between `<!-- SYNC METADATA -->` markers)

## Command Cheat Sheet

- Dry run: `./bin/file_window_sync`
- Apply only: `./bin/file_window_sync --apply`
- Apply + push: `./bin/fw-sync`
- Push on review branch: `./bin/file_window_sync --apply --pr`
- Include large files: `./bin/file_window_sync --apply --allow-large`
- Secret scan: `./bin/file_window_sync --apply --scan-secrets`
- Force manifest toggle: `./bin/file_window_sync --apply --no-manifest`
- Tag release: `./bin/file_window_sync --push --tag`

All commands respect the max file size cap configured in
`.file-window-sync.yml` unless `--allow-large` is passed.

## Makefile Target

A convenience target is available at the repository root:

```bash
make sync
```

This runs `bin/file_window_sync --push` so you get the full end-to-end update in
one command.

## Shell Alias

For shells that source `~/.bashrc`, add the following snippet to create the
`fw-sync` alias globally:

```bash
alias fw-sync="$HOME/File-Window/bin/file_window_sync --push"
```

The repository already ships with `bin/fw-sync`, so using the alias is optional.

## Cron / Systemd Automation Example

Schedule a sync every hour (adjust cadence to taste):

```
0 * * * * /home/karson/File-Window/bin/file_window_sync --push --scan-secrets >> /home/karson/logs/file_window_sync.log 2>&1
```

Ensure the log directory exists (`mkdir -p ~/logs`) and that the GitHub SSH key
can authenticate non-interactively.

## Pull Request Mode

Use `--pr` to route changes through a dedicated branch instead of `main`:

```bash
./bin/file_window_sync --apply --pr --branch sync/review-$(date +%Y%m%d)
```

The script creates the branch, stages, commits, and pushes it. Open the actual
GitHub PR manually after the run finishes.

## Tagging Releases

Add `--tag` (optionally with `--tag-prefix`) to stamp an annotated tag after a
successful commit. Tags push automatically when `--push` is supplied.

## Manifest and Metadata

- `mirror/MANIFEST_SYNC.json` lists every mirrored file with SHA256 checksums and
  sizes so external agents can diff quickly.
- `GPT_SYNC_VERSION.md` receives a dynamic summary between
  `<!-- SYNC METADATA START -->` and `<!-- SYNC METADATA END -->` each time the
  script runs with metadata updates enabled.

## Safety Defaults

- Working tree must be clean unless `--force` is supplied.
- `rsync` runs in dry-run mode first, summarising adds/changes/deletes.
- The script never mutates the original source directories.
- Large files are skipped by default; the user opts in via `--allow-large`.

## Failure Recovery

If the sync fails mid-run, simply fix the underlying issue and re-run the
command. The mirror directory is deterministic; re-running will converge without
manual cleanup.

## Optional Sources

Mark rarely used or machine-specific paths with `optional: true` inside
`.file-window-sync.yml`. Missing optional sources are silently skipped, so the
workflow stays clean even when you temporarily remove a directory.

