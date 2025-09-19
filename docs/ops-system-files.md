System Ops Files (Local Runtime)

This folder captures the local system files used to run and maintain the AI day trading system on this PC. These are for reference and provisioning; do not commit secrets.

Contents
- ops/systemd/user/*.service, *.timer — user services/timers for IB Gateway, autosync, nightly checks, and logrotation
- ops/logrotate/m5_trader.conf — logrotate policy for ~/logs/*.log
- ops/ibc/config.ini.template — IBC config template (do not commit real credentials)
- ops/Jts/jts.ini.sample — example IB Gateway settings

Install (user mode)
1) systemd units
   - mkdir -p ~/.config/systemd/user
   - cp -a ops/systemd/user/*.service ops/systemd/user/*.timer ~/.config/systemd/user/
   - systemctl --user daemon-reload
   - systemctl --user enable --now m5trader-scan.timer m5trader-nightly.timer m5-ibg-logs-link.timer logrotate-m5-trader.timer
   - Optional (GUI vs headless): enable one of ibgateway.service or ibgateway-headless.service

2) logrotate
   - mkdir -p ~/.config/logrotate.d ~/.cache/m5_trader
   - cp ops/logrotate/m5_trader.conf ~/.config/logrotate.d/m5_trader.conf
   - Ensure script is available: cp bin/logrotate_m5_trader.sh ~/bin/ (or adjust the unit to point into repo)

3) IBC and JTS
   - Place your real IBC config at ~/ibc/config.ini (never commit); use ops/ibc/config.ini.template as a guide
   - IB Gateway settings: edit via GUI, or copy ops/Jts/jts.ini.sample to ~/Jts/jts.ini and adjust

Notes
- Do not commit real credentials or machine-specific paths. .gitignore blocks ops/ibc/config.ini.
- The ibg_logs symlink in repo is updated locally by tools/update_ibg_log_symlink.sh; it is machine-specific and does not need to be committed.

## Current Systemd State (Snapshot)
- Captured: 2025-09-18T22:40:52-04:00
- User: karson

Enabled/disabled
- ibgateway.service: disabled
- ibgateway-headless.service: masked
- m5trader-watch.service: enabled
- m5trader-scan.service: static
- m5trader-nightly.service: static
- m5-ibg-logs-link.service: static
- logrotate-m5-trader.service: static
- m5trader-scan.timer: enabled
- m5trader-nightly.timer: enabled
- m5-ibg-logs-link.timer: enabled
- logrotate-m5-trader.timer: enabled

Active state summary
- ibgateway-desktop.service: active (running)
- ibgateway.service: failed (not in use; use desktop or headless variant)
- m5trader-watch.service: active (running)
- m5trader-scan.service: activating (on timer)
- m5trader-nightly.service: failed (last run failed; check `scripts/nightly_validate.sh`)
- m5-ibg-logs-link.service: inactive (runs via timer)
- logrotate-m5-trader.service: inactive (oneshot; runs via timer)
- Timers active: logrotate-m5-trader.timer, m5-ibg-logs-link.timer, m5trader-nightly.timer, m5trader-scan.timer

Notes
- ibgateway-desktop.service appears to be a separate user unit present on this system for GUI sessions.
- If you intend headless operation, enable `ibgateway-headless.service` instead and disable the desktop variant.
- Nightly validation last failed; run `systemctl --user status m5trader-nightly.service` and check logs under `~/logs/`.
