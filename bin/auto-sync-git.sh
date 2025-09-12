set -euo pipefail
REPO_DIR="$HOME/projects/ai-trading-system"
if [ ! -d "$REPO_DIR/.git" ]; then
  exit 0
fi
cd "$REPO_DIR"
git fetch --prune
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "auto-sync: $(date -u +'%Y-%m-%d %H:%M:%S UTC')"
  git pull --rebase --autostash
  git push
fi
