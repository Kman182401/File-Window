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

## Sync Workflow

The `bin/clone` command keeps GitHub identical to the live trading system on this workstation. It copies any external folders listed in `.clone.toml` into the repository tree, auto-commits local edits, and pushes whenever changes are detected.

Common calls:

- Preview: `clone --dry-run`
- Sync + push: `clone --push` (or just `clone`)
- Include oversized files once: `clone --allow-large --push`
- Review branch: `clone --pr --push`
- Opt out of auto-commit: `clone --no-auto-commit --force`

`.clone.toml` maps external directories (e.g., `~/orders`) into subfolders inside the repo. Global and per-source excludes keep logs/caches out of Git. A cron job runs `clone --push --scan-secrets` every 15 minutes so ChatGPT can analyze this GitHub repo and get the full, current system without needing any extra context.
