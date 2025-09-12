# EC2 Trading System Status and Configuration
*Last Updated: 2025-09-12*

## Purpose of This Repository
**IMPORTANT**: This GPT-files repository is specifically designed for ChatGPT to analyze and gain a complete understanding of the EC2 trading system. All files here should be kept current to enable effective cross-AI collaboration.

## System Overview
- **Instance**: AWS EC2 m5.large (2 vCPU, 8GB RAM)
- **OS**: Ubuntu Linux 6.8.0-1035-aws
- **Primary Function**: AI-powered algorithmic trading via Interactive Brokers

## VNC Remote Access Configuration

### Current Setup (Secure, SSH Tunneled)
- **VNC Server**: x11vnc running on localhost:5900 (bound to 127.0.0.1 only)
- **Display**: :1 (Xvfb virtual display at 1920x1080x24)
- **Window Manager**: Fluxbox
- **Security**: localhost-only binding, requires SSH tunnel for access

### Client Connection Instructions
1. **From Windows/WSL**, establish SSH tunnel:
   ```bash
   ssh -N -L 5901:127.0.0.1:5900 -o ExitOnForwardFailure=yes \
       -o ServerAliveInterval=30 -o ServerAliveCountMax=3 aws-ib
   ```

2. **In VNC Viewer**, connect to: `localhost:5901`

3. **Verification** (Windows PowerShell):
   ```powershell
   Test-NetConnection -ComputerName localhost -Port 5901
   ```

### VNC Process Details
- **Xvfb PID**: 1825231
- **Fluxbox PID**: 1825308
- **x11vnc PID**: 1832439
- **Status**: All processes running and healthy

## IBKR Gateway Configuration

### Connection Settings
```bash
export IBKR_HOST=127.0.0.1
export IBKR_PORT=4002  # Paper trading
export IBKR_CLIENT_ID=9002
```

### Gateway Process
- **Java Process**: Running (PID: 1825664)
- **Memory Usage**: ~365MB
- **Status**: Requires manual re-arm via VNC

### Re-arm Procedure
1. Connect to VNC (localhost:5901)
2. In IB Gateway interface:
   - Configure → API → Precautions → Apply → OK
   - Configure → API → Settings → Apply
   - Port flip if needed: 4002→4003→Apply→4002→Apply→OK

### Quick Restart Script
```bash
~/bin/restart_ibgw_and_rearm.sh
```

## Trading System Components

### Core Files
- **run_adaptive_trading.py**: Main trading pipeline orchestrator
- **market_data_ibkr_adapter.py**: IBKR integration for market data
- **feature_engineering.py**: Technical indicators and features
- **rl_trading_pipeline.py**: Reinforcement learning models
- **market_data_config.py**: Central configuration

### Active Services
- **Port 5900**: VNC server (localhost only)
- **Port 4002**: IBKR API (when re-armed)
- **Port 8888**: Jupyter/Python service
- **Port 3000**: Application service

## Tmux Sessions
```
ibgw    - IB Gateway
omega   - Trading pipeline (4 windows)
jupyter - Jupyter services
vnc     - VNC backup session
```

## System Health Checks

### Check VNC Status
```bash
ss -tlnp | grep :5900
ps aux | grep -E 'Xvfb|fluxbox|x11vnc'
```

### Check IBKR Connection
```bash
IBKR_CLIENT_ID=9003 python3 ~/smoke_ib_connect.py
```

### Check Trading Pipeline
```bash
tmux attach -t omega
# Check pipeline.log for recent activity
tail -f ~/logs/pipeline.log
```

## Important Notes
1. **Security**: VNC is bound to localhost only - SSH tunnel required
2. **Gateway**: Requires manual re-arm after restarts
3. **Memory**: Keep usage under 6GB (system has 8GB total)
4. **Logs**: Check ~/logs/ for all system logs

## Repository Maintenance
This repository should be kept synchronized with the actual system state to enable effective ChatGPT analysis. Run sync script regularly:
```bash
~/sync_gpt_files.sh
```