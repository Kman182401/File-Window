# Clone Command Workflow

This script mirrors selected local paths directly into the repository tree so that
GitHub reflects the exact state of your workstation files while keeping the sources
untouched.

## Configuration

- Edit `.clone.toml` to declare sources, excludes, size caps, and behaviour flags.
- Each `[[source]]` entry copies a local path into `<dest>` inside the repository.
- `optional = true` lets you skip missing paths without failing the run.
- `global.global_excludes` applies to every source (logs, caches, etc.).
- `max_file_size_mb` limits file size unless `--allow-large` is passed.
- `delete_on_repo = true` prunes mirror files removed locally.

## Usage

```bash
./bin/clone --dry-run           # show planned rsync changes only
./bin/clone --push              # copy, commit, push to origin/main
./bin/clone --rebase --push     # fetch + rebase before syncing
./bin/clone --pr --push         # commit to sync/review-<ts> branch
./bin/clone --allow-large --push
./bin/clone --no-push           # keep commit local
./bin/clone --scan-secrets      # simple grep-based secret check
./bin/clone --no-auto-commit --force  # skip auto-commit and proceed even if tree is dirty
```

Add an alias for convenience:

```bash
echo 'alias clone="$HOME/File-Window/bin/clone"' >> ~/.bashrc
```

## Outputs

- Mirrored files land inside the repository grouped by `dest` values.
- `MANIFEST_CLONE.json` lists every mirrored file with SHA256 hashes.
- Commits use the message prefix `clone(sync)`.

## Automation

Schedule periodic refresh with cron:

```
0 * * * * /home/karson/File-Window/bin/clone --push --scan-secrets >> /home/karson/logs/clone.log 2>&1
```

Tail logs when needed:

```
tail -f ~/logs/clone.log
```

## Safety

- By default, the command auto-commits any dirty tree before mirroring so the cron job never stops.
- Use `--no-auto-commit` (optionally with `--force`) if you want to manage commits manually.
- Never modifies the original source directories; only writes inside the repository tree and Git metadata.
- Honors excludes and size caps to avoid leaking logs/secrets or gigantic binaries.
