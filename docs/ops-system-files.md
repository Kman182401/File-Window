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

