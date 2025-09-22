# Clone Command Workflow

This script mirrors selected local paths into the repository under `mirror/` so that
GitHub reflects the exact state of your workstation files while keeping the sources
untouched.

## Configuration

- Edit `.clone.toml` to declare sources, excludes, size caps, and behaviour flags.
- Each `[[source]]` entry copies a local path into `mirror/<dest>/...`.
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
```

Add an alias for convenience:

```bash
echo 'alias clone="$HOME/File-Window/bin/clone"' >> ~/.bashrc
```

## Outputs

- Mirrored files live under `mirror/` grouped by `dest`.
- `mirror/MANIFEST_CLONE.json` lists every mirrored file with SHA256 hashes.
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

- Refuses to run on a dirty working tree or staged index.
- Never modifies original source directories; only writes inside `mirror/` and Git metadata.
- Honors excludes and size caps to avoid leaking logs/secrets or gigantic binaries.
