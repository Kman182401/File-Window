#!/usr/bin/env bash
set -euo pipefail
pkill -15 -f "Xvfb|x11vnc|fluxbox|Jts/ibgateway" 2>/dev/null || true
sleep 2
pkill -9  -f "Xvfb|x11vnc|fluxbox|Jts/ibgateway" 2>/dev/null || true
Xvfb :1 -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 & 
sleep 0.5
DISPLAY=:1 fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :1 -rfbport 5900 -localhost -forever -shared -repeat -ncache 10 >/tmp/x11vnc.log 2>&1 &
tmux kill-session -t ibgw 2>/dev/null || true
tmux new-session -d -s ibgw -n ibgw "bash -lc 'DISPLAY=:1 ~/Jts/ibgateway/1039/ibgateway'"
echo
echo "Now do this in VNC (localhost:5901):"
echo "  1) Configure → API → Precautions → Apply → OK"
echo "  2) Configure → API → Settings → Apply; flip 4002→4003→Apply→4002→Apply→OK"
echo "Then run: IBKR_CLIENT_ID=9003 python3 ~/smoke_ib_connect.py"