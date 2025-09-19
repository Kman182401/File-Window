#!/usr/bin/env bash
/usr/sbin/logrotate -s "$HOME/.cache/m5_trader/logrotate.state" "$HOME/.config/logrotate.d/m5_trader.conf"

