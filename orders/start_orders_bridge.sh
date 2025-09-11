#!/usr/bin/env bash
set -euo pipefail
mkdir -p ~/logs
exec bash -lc 'python3 ~/orders/orders_bridge.py 2>&1 | tee -a ~/logs/orders_bridge.log'