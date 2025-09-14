# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## AUTOMATIC AGENT INVOCATION RULES

**CRITICAL**: Claude Code MUST proactively invoke specialized agents WITHOUT waiting for user requests when:
- ANY file matching agent specialization is accessed
- ANY error related to agent domain occurs  
- ANY operation within agent expertise is attempted
- Context suggests agent knowledge would help

**NO USER KEYWORDS OR COMMANDS REQUIRED** - Agents activate automatically based on context.

### Automatic Agent Triggers:
- **Trading Strategy Architect**: RL algorithm files, model training, convergence issues
- **Market Analysis Expert**: Market data access, feature engineering, technical indicators
- **IBKR Integration Specialist**: Connection errors, TWS/Gateway issues, order execution
- **Security Compliance Auditor**: Deployment, credentials, audit logs, compliance
- **Data Pipeline Engineer**: S3 operations, data ingestion, ETL processes
- **Infrastructure Optimizer**: Memory >5GB, performance issues, AWS/Docker operations

## GPT-Files Repository

**IMPORTANT**: The `gpt-files-repo` directory is specifically maintained for ChatGPT to analyze and gain a complete understanding of this EC2 trading system. Claude Code must ensure this repository stays up-to-date with current system configuration, code, and documentation to enable effective cross-AI collaboration. Run `~/sync_gpt_files.sh` regularly to synchronize files.

## Core Architecture

This is an AI-powered algorithmic trading system that integrates with Interactive Brokers (IBKR) for live market data and trading execution. The system uses reinforcement learning (PPO) for trading decisions and operates on futures contracts (ES, NQ, currency futures, gold).

### Key Components

- **run_adaptive_trading.py**: Main pipeline orchestrating the entire trading workflow - data ingestion, feature engineering, model training/inference, and trade execution
- **market_data_ibkr_adapter.py**: IBKR TWS/Gateway integration for real-time market data and futures contract management 
- **feature_engineering.py**: Technical indicator calculation, S3 data processing, and advanced market microstructure features
- **market_data_config.py**: Central configuration for IBKR connection, risk limits, and symbol mappings
- **news_ingestion_*.py**: Multi-source news data ingestion (MarketAux, Alpha Vantage, IBKR) with sentiment analysis
- **audit_logging_utils.py**: Trade audit logging and compliance tracking

### Data Flow

1. Market data flows from IBKR → feature engineering → ML model
2. News data from multiple sources → sentiment analysis → feature store
3. Model predictions → risk management → order execution via IBKR
4. All trades logged for audit compliance

## Environment Setup

### IBKR Connection
Set environment variables:
```bash
export IBKR_PORT=4002  # 4001 for live, 4002 for paper trading (CORRECTED)
export IBKR_CLIENT_ID=9002
```

### IBKR Connection Testing
To verify IBKR Gateway connection:
```bash
python - <<'PY'
from market_data_ibkr_adapter import IBKRIngestor
import os

# Use your environment variables / defaults
host = os.getenv("IBKR_HOST", "127.0.0.1")
port = int(os.getenv("IBKR_PORT", "4002"))
clientId = int(os.getenv("IBKR_CLIENT_ID", "9002"))

try:
    ing = IBKRIngestor(host=host, port=port, clientId=clientId)
    print(f"✅ Connected successfully to IB Gateway at {host}:{port} (clientId={clientId})")
    
    # Try fetching sample data as proof of connection
    df = ing.fetch_data("ES1!", duration="1 D", barSize="1 hour", whatToShow="TRADES", useRTH=False)
    print(f"✅ Data fetch succeeded: {df.shape[0]} rows, columns={list(df.columns)}")
    
    ing.disconnect()
    print("✅ Disconnected cleanly from IB Gateway")
except Exception as e:
    print(f"❌ Connection or data fetch failed: {e}")
PY
```

### IBKR Gateway Setup Reference
Connection setup details are stored in: "C:\Users\Karson\.ssh\IB Gateway ↔ EC2 Setup.txt"

### IBKR Gateway Troubleshooting
**CRITICAL: If IB Gateway connection fails with TimeoutError, follow these steps:**

#### 1. Quick Restart Solution (RECOMMENDED)
```bash
# One-command restart of IB Gateway (non-Docker, running on EC2)
~/bin/restart_ibgw_and_rearm.sh

# After running, manually re-arm in VNC (localhost:5901):
# 1) Configure → API → Precautions → Apply → OK
# 2) Configure → API → Settings → Apply; flip 4002→4003→Apply→4002→Apply→OK
```

#### 2. Quick Diagnosis
```bash
# Check if desktop processes are running
pgrep -a Xvfb; pgrep -a fluxbox; pgrep -a x11vnc

# Verify display is ready
ls -l /tmp/.X11-unix/X1 && echo "DISPLAY :1 ready"

# Check IB Gateway tmux session
tmux ls | grep ibgw
```

#### 3. Manual Restart (if script fails)
```bash
# Kill all processes
pkill -15 -f "Xvfb|x11vnc|fluxbox|Jts/ibgateway" 2>/dev/null || true
sleep 2
pkill -9  -f "Xvfb|x11vnc|fluxbox|Jts/ibgateway" 2>/dev/null || true

# Start desktop
Xvfb :1 -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
sleep 0.5
DISPLAY=:1 fluxbox >/tmp/fluxbox.log 2>&1 &
x11vnc -display :1 -rfbport 5900 -localhost -forever -shared -repeat -ncache 10 >/tmp/x11vnc.log 2>&1 &

# Start IB Gateway
tmux kill-session -t ibgw 2>/dev/null || true
tmux new-session -d -s ibgw -n ibgw "bash -lc 'DISPLAY=:1 ~/Jts/ibgateway/1039/ibgateway'"

# Wait 15-20 seconds for full initialization
sleep 20

# Test connection
python - <<'PY'
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1001, timeout=15)
print('connected:', ib.isConnected(), 'serverVersion:', ib.client.serverVersion() if ib.isConnected() else None)
ib.disconnect()
PY
```

#### 3. Port Configuration Verification
If connection still fails, verify socat port forwarding:
```bash
# Check current socat configuration
docker exec ibkr-ibkr-gateway-1 ps aux | grep socat

# Should show: socat -d -d TCP-LISTEN:8888,fork TCP:127.0.0.1:4002
# If showing different port, fix it:
docker exec ibkr-ibkr-gateway-1 bash -c "
pkill socat
nohup socat -d -d TCP-LISTEN:8888,fork TCP:127.0.0.1:4002 > /tmp/socat.log 2>&1 &
"
```

#### 4. Expected Success Indicators
- Container logs show: "Login has completed"
- Connection test returns: `connected: True serverVersion: 176`
- Market data fetch succeeds for ES1! futures

#### 5. Daily Startup (No 2FA Required for Paper Trading)
Paper trading does NOT require 2FA authentication. If connection fails:
1. Restart container: `docker compose restart`
2. Wait 20 seconds for initialization
3. Test connection as shown above

## 🔑 Single-Client IBKR Mode (Source of Truth)

**Architecture**: One process, one socket, clientId=9002 performs market data + orders. Never launch a second client for orders.

**Attach point**: After IBKRIngestor() is created in the pipeline, call `attach_ib(self.market_data_adapter.ib)` once, then (if PIPELINE_KEEPALIVE=1) start one re-entrant keepalive thread (ping reqCurrentTime(), reconnect on drop).

**Client-ID policy**:
- 9002 = pipeline (market data + orders)
- 9102 = temporary off-process tests only (disconnect immediately after)
- Never run two processes with the same clientId

**Order execution gate**: Orders fire only when ENABLE_ORDER_EXEC=1 and ALLOW_ORDERS=1 and DRY_RUN=0. Otherwise, decision logs only.

**Decision→order call site**: Exactly one call right after the pipeline's TRADING_DECISION log, to `_execute_order_single_client(symbol, side, qty)`. Keep this single, guarded call to avoid duplicates.

## 🟢 Golden Re-Arm & Quick Port Flip (When Attach Fails)

1. Close any open dialogs in Gateway
2. Configure → API → Precautions → Apply → OK
3. Configure → API → Settings → Apply
4. If second/next attach still fails: 4002 → Apply → 4003 → Apply → 4002 → Apply

**Acceptance**:
- `ss -tanp | grep 4002 | grep ESTAB` → one Python PID (the pipeline)
- VNC header: "API Client: 1 connected (9002)"
- pipeline.log shows data processing and, when armed, order lines

## ✅ Paper-Trading Acceptance (Run Every Time Before Arming)

1. **No sidecar present**: No orders_bridge.py in tmux or launcher
2. **One socket**: ss shows one pipeline PID connected to 127.0.0.1:4002
3. **Sanity tests** (as needed): Run the IBKR/pipeline tests from CLAUDE.md
4. **Symbols map to futures only**: Use documented mapping (ES/NQ/6E/6B/6A/GC)
5. **When armed**: pipeline.log emits [single_client_order] and Submitted/PreSubmitted lines; no `connectAsync was never awaited` warnings

## 🧪 In-Pipeline Smoke Hooks (No Second Client)

**Dry smoke (no orders)**: Leave DRY_RUN=1 or ENABLE_ORDER_EXEC=0. Ensure data/feature generation completes in the logs.

**One-shot paper smoke**: Optional, guard with TEST_ORDER_ON_START=1 to submit a far-away LIMIT (stays Submitted) and auto-cancel; disable itself after one run.

## 📉 Risk, Limits & Learning Modes

### Learning Mode Presets
**Learning (paper)**: 
- MAX_TRADES_PER_DAY=100
- MAX_ORDER_SIZE=1  
- Small tick-based brackets
- Focus: Sample efficiency for RL training

**Production (paper)**:
- Revert to documented defaults (20/day, size per config)
- Ensure rollback gates stay active

### AWS S3 Storage
The system uses S3 bucket "omega-singularity-ml" for:
- Historical market data storage
- Model artifacts (models/ directory)
- Feature store persistence

### Dependencies
Install with: `pip install -r requirements.txt`
Key dependencies: boto3, pandas, scikit-learn, ib_insync, stable-baselines3

## Testing Commands

### Pipeline Tests
- **Full pipeline test**: `python test_pipeline_simplified.py`
- **IBKR connection test**: `python test_ib_connection.py` 
- **Single iteration test**: `python test_single_iteration.py`
- **Data fetch test**: `python test_data_fetch.py`

### Monitoring
- **5-loop monitor**: `python tools/pipeline_monitor_5loops.py`
- **Smoke tests**: `python tests/smoke_symbol_resolution.py`

### Manual Testing
Individual test files are designed to be run with: `timeout 60 python3 test_pipeline_simplified.py`

## Configuration

### Risk Management (configs/market_data_config.py)
- MAX_DAILY_LOSS_PCT: 2% daily loss limit
- MAX_TRADES_PER_DAY: 20 trade limit
- MAX_POSITION_EXPOSURE: 3 contract position limit
- MAX_ORDER_SIZE: 2 contract order limit

### Symbol Mapping
The system trades futures as proxies for traditional FX/index symbols:
- ES1! → ES futures (S&P 500 E-mini)
- NQ1! → NQ futures (Nasdaq 100 E-mini) 
- GBPUSD → 6B futures (CME British Pound)
- EURUSD → 6E futures (CME Euro)
- AUDUSD → 6A futures (CME Australian Dollar)
- XAUUSD → GC futures (COMEX Gold)

## Logging and Monitoring

### Log Locations
- Pipeline logs: `logs/` directory with timestamped files
- IBKR logs: `ibkr_market_data.log`
- Trade audit: `trade_audit_log.jsonl`

### Key Monitoring Points
- IBKR connection health via `ib_health_check.py`
- Resource monitoring enabled in pipeline
- CloudWatch integration available via `setup_cloudwatch_agent.sh`

## Infrastructure Constraints

### EC2 Instance Specifications (m5.large)
**CRITICAL: All system changes must consider these hardware limitations**

- **CPU**: 2 vCPUs (1 core, 2 threads) Intel Xeon Platinum 8259CL @ 2.50 GHz
- **Memory**: 8 GiB DDR4 total (typically ~5.9 GiB available after OS overhead)
- **Storage**: 100 GB NVMe SSD (97 GB usable root partition, ~49 GB free)
- **Network**: ENA (Elastic Network Adapter)
- **Architecture**: x86_64, full AVX/AVX2/AVX512 instruction set support

### Resource Usage Guidelines
- **Memory**: Keep total memory usage under 6 GB to avoid swap (no swap configured)
- **CPU**: Single-threaded workloads optimal; limited benefit from >2 parallel threads
- **Disk**: Monitor disk usage; system already at 50% capacity (48/97 GB used)
- **ML Training**: Consider memory constraints for model size and batch processing

### Performance Considerations
- Cache hierarchy: L1: 32 KiB, L2: 1 MiB, L3: 35.8 MiB
- Hyper-Threading enabled but limited to 2 logical cores total
- NUMA single-node configuration
- No dedicated GPU; CPU-only ML processing

## Development Notes

### IBC Integration
The repository includes Interactive Brokers Controller (IBC) for automated TWS/Gateway management in `ibc/` and `IBC/` directories.

### Lambda Deployment
Docker containerization available for AWS Lambda deployment:
- Build: `docker build -t trading-lambda .`
- Deploy script: `deploy_lambda_container.sh`

### Data Sources
- Real-time: IBKR TWS/Gateway API
- Historical: S3 bucket with multi-year data via `multi_year_yf_ingestor.py`
- News: MarketAux, Alpha Vantage, IBKR fundamental data
- IB Gateway IS Running in Docker
- run_adaptive_trading.py is the main trading system
- never test with mock data only ever test with true data from IBKR
- Enable Anti-Dilution and Anti-Drift to ensure and any all changes made to the system do not dilute the system or drift from the core logic of the system.

## 🐞 Troubleshooting Patterns Claude Code Must Recognize

**Half-armed API (timeouts / second client blocked)**: Close dialogs → Precautions→Apply→OK → Settings→Apply → (if needed) 4002↔4003 flip; re-check ss.

**Ghost clientId after crash**: Resolved by the same golden re-arm; never reuse clientId across processes.

**Async misuse**: Never call IB.connectAsync without awaiting; in non-async code use `ib.connect(...)` or `util.run(...)`. Run the "ib_async_warnings" test if warnings appear.

**Resource limits on m5.large**: Keep total memory <6GB and accept lower parallelism; Claude should respect this.

## 🧭 Monitoring & Where to Look

**Logs**: 
- `~/logs/pipeline.log` for decisions/orders
- `trade_audit_log.jsonl` for audit

**Gateway header**: 1 connected (9002)

**Socket test**: `ss -tanp | grep 4002 | grep ESTAB`

**Tests**: `test_phase4a_fix.py` - run when you suspect regressions in async/connectivity/perf. The test's critical gates are listed (parallel perf, ib async warnings, latency target).

## PHASE 3 CRITICAL REQUIREMENTS - MANDATORY COMPLIANCE

### ABSOLUTE REQUIREMENTS (NEVER VIOLATE)

**The system MUST ALWAYS maintain these capabilities:**
1. **RL Implementation** - At least one working RL algorithm (SAC/RecurrentPPO/PPO/A2C)
2. **Trading Environment** - Gymnasium environment with proper reset()/step() methods
3. **Decision Making** - Ability to generate buy/sell/hold actions
4. **Learning Capability** - Ability to learn from trading results
5. **Paper Trading** - Real IBKR integration for paper trading execution

### MANDATORY VERIFICATION PROTOCOL

**After ANY system change, Claude Code MUST execute:**

```bash
# Step 1: Core Intelligence Verification (MUST PASS 6/6)
python3 verify_trading_intelligence.py
# If <6/6 tests pass: IMMEDIATELY ROLLBACK ALL CHANGES

# Step 2: Production Readiness Test (MUST PASS 100%)
python3 test_production_readiness.py
# Verifies: Decisions, Learning, Paper Trading, Memory, Fallbacks

# Step 3: Integration Test (MUST COMPLETE)
python3 demo_complete_trading_intelligence.py
# Must complete full trading cycle without errors

# Step 4: Paper Trading Connection (MUST CONNECT)
python3 run_adaptive_trading.py --mode paper_trading --test
# Must show: "Trading brain active"
```

### VERIFICATION REQUIREMENTS

**1. RL Implementation Verification**
```bash
# MUST return results for ALL:
grep -r "model.predict\|agent.predict" --include="*.py"
grep -r "agent.learn\|agent.train\|model.learn" --include="*.py"
grep -r "class.*Agent\|create.*agent" --include="*.py"
```

**2. Trading Environment Verification**
```python
# MUST have these methods/attributes:
- EnhancedTradingEnvironment.reset()
- EnhancedTradingEnvironment.step()
- EnhancedTradingEnvironment.observation_space
- EnhancedTradingEnvironment.action_space
```

**3. Decision Making Verification**
```python
# Test decision capability:
obs = env.reset()
action = agent.predict(obs)
assert action is not None
assert len(action) > 0
# Action must be executable by paper trading system
```

**4. Learning Capability Verification**
```python
# MUST have at least one:
- agent.learn() method
- agent.train() method
- agent.learn_from_result() method
# Must track performance metrics
```

**5. Memory Constraint Verification**
```bash
free -h
# Available memory MUST be >2GB
# Total system usage MUST be <6GB
# If exceeded: DISABLE newest components
```

### ANTI-DRIFT PROTECTION

**NEVER do any of the following:**
- ❌ Remove existing RL algorithms (SAC, RecurrentPPO, PPO, A2C)
- ❌ Break IBKR integration or paper trading capability
- ❌ Disable or remove fallback mechanisms
- ❌ Remove the rule-based trading agent fallback
- ❌ Modify core pipeline without verification
- ❌ Test with mock/simulated data instead of real IBKR data
- ❌ Exceed 6GB memory usage
- ❌ Increase decision latency beyond 100ms

### PHASE 3 IMPLEMENTATION REQUIREMENTS

**Every Phase 3 component MUST:**
1. Be independently toggleable via feature flags
2. Have its own circuit breaker for failure isolation
3. Log all decisions for audit compliance
4. Gracefully fallback to Phase 2 functionality if it fails
5. Not increase decision latency by more than 100ms
6. Pass ALL verification tests before being marked complete

**Feature Flag Structure:**
```python
PHASE3_FEATURES = {
    'ensemble_enabled': False,      # Enable only after full testing
    'online_learning_enabled': False,  # Enable only after ensemble works
    'meta_learning_enabled': False,    # Enable only after online learning works
    'lightgbm_validator_enabled': False,  # Optional enhancement
    'transformer_features_enabled': False  # Optional enhancement
}
```

### ROLLBACK PROTOCOL

**Automatic rollback triggers:**
```python
if any([
    memory_usage_mb > 6000,
    decision_latency_ms > 100,
    verification_tests_passed < 6,
    paper_trading_connection == False,
    rl_algorithms_working == 0
]):
    rollback_to_last_working_state()
    disable_all_phase3_features()
    alert_user("System rolled back to maintain production readiness")
```

### PRODUCTION READINESS GATES

**Before deploying ANY Phase 3 component:**
1. ✅ All Phase 2 tests must pass (baseline established)
2. ✅ Component must work in isolation
3. ✅ Component must work with existing system
4. ✅ Memory usage must stay under limits
5. ✅ Fallback mechanism must be tested
6. ✅ 1-hour paper trading test must succeed

### FINAL VALIDATION PROTOCOL

**Before marking Phase 3 complete:**
1. Run 24-hour paper trading test
2. Must execute minimum 10 trades
3. Must show learning improvement over time
4. Zero system crashes or rollbacks
5. All verification tests passing continuously

### GUARANTEE TO USER

**This system WILL:**
- ✅ Make trading decisions continuously
- ✅ Learn from every paper trade
- ✅ Execute real paper trades via IBKR
- ✅ Stay within memory constraints
- ✅ Maintain production readiness
- ✅ Have multiple fallback layers
- ✅ Improve performance over time

**Any change that breaks these guarantees will be automatically rolled back.**
- From now on when restarting the Gateway only restart the Gateway and leave the VNC display untouched unless explicitlly told otherwise and if you feel restarting the display is ever necessary let me know because I am not opposed to it at all id just like to avoind it if possible.
<!-- START: VNC_SSH_IBGW_PLAYBOOK -->
## 🛠️ VNC/SSH Tunnel & IB Gateway — Incident Playbook (Last updated: 2025-09-13)

**Scope:** Fix recurring `VNC connection refused` by ensuring SSH tunnel is up, the remote VNC service is actually listening, and IB Gateway can be operated after maintenance. This codifies the steps that worked in production on m5.large.

### Invariants & Known-Good Defaults
- OS user: `ubuntu`
- **IBKR Paper** API port: **4002** (do **not** default to 4004).
- Canonical clientId: **9002**
- VNC stack on EC2: `Xvfb :1`, `fluxbox`, `x11vnc -rfbport 5900 -localhost ...`
- IB Gateway run in tmux session `ibgw` (manual re-arm via VNC).
- Safe launcher: `~/bin/run_pipeline_safely.sh` (probes/backoff, prevents socket storms).
- CPU guards: `OMP/MKL/OPENBLAS=1`, `nice -n 5`, `taskset -c 0-1`.

### Symptom
- VNC viewer shows **"connection refused by computer"** OR SSH tunnel fails/hangs.

### Root-Cause Patterns (observed)
1) **Wrong EC2 public IP** after stop/start (no Elastic IP).
2) **Security Group** does not allow inbound TCP 22 from **current** client IP.
3) **x11vnc not listening** on EC2 (`127.0.0.1:5900`).
4) During **IBKR nightly maintenance** (~late US evening), IBGW login loops with "server error, will retry…".
5) Shell panes dying due to `set -euo pipefail` + fragile commands (fixed via safe wrappers).

### Decision Tree
```text
If VNC says "refused":
  └─> First prove SSH works to correct public IP on 22
      ├─ Laptop: nc -vz <EC2_PUBLIC_IP> 22  → success? yes → open tunnel
      │                                       no  → fix SG/NACL or network egress
      └─ If SSH OK but VNC refused → fix x11vnc on EC2 (ensure LISTEN 127.0.0.1:5900)
```

### Laptop — SSH tunnel (run locally, keep terminal open)
```bash
# Current EC2 public IPv4 (confirmed working 2025-09-13): 18.116.196.223
ssh -v -o ExitOnForwardFailure=yes \
    -L 5901:127.0.0.1:5900 \
    -i ~/.ssh/Kody.pem \
    ubuntu@18.116.196.223
# If 5901 busy: use -L 5905:127.0.0.1:5900 and VNC → localhost:5905
# Note: Update IP if instance is stopped/started without Elastic IP
```

### EC2 — Verify VNC server (x11vnc) is listening
```bash
ss -tlnp | grep ":5900" || {
  pkill -f x11vnc 2>/dev/null || true
  x11vnc -display :1 -rfbport 5900 -localhost -forever -shared -repeat -ncache 10 -o /tmp/x11vnc.log &
  sleep 2
  ss -tlnp | grep ":5900" || tail -n 100 /tmp/x11vnc.log
}
# If needed:
pkill -f "Xvfb|fluxbox" 2>/dev/null || true
Xvfb :1 -screen 0 1920x1080x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &
sleep 1
DISPLAY=:1 fluxbox >/tmp/fluxbox.log 2>&1 &
sleep 1
```

### IB Gateway re-arm (Paper)
1. In VNC: Configure → **API → Settings** → Enable ActiveX/Socket, **Port 4002**, Trusted IP `127.0.0.1`, SSL off, Read-Only off → **Apply → OK**
2. Configure → **API → Precautions** → **Apply → OK** (close modals)
3. If odd, do the quick **port flip**: 4002→Apply→4003→Apply→4002→Apply→OK
4. Probe attach (safe):
```bash
python3 - <<'PY'
from ib_insync import IB; ib=IB()
print('Connected:', bool(ib.connect('127.0.0.1',4002,clientId=9002,timeout=12)))
ib.disconnect()
PY
```

### IBKR nightly maintenance behavior
- Gateway may loop with **"Attempt X: server error, will retry in 10 seconds…"**.
- Action: wait 10–20 minutes; once back, re-arm (Settings & Precautions → Apply/OK), then probe attach.

### Prevent it from recurring
- **Elastic IP**: Allocate + associate to the instance so public IP doesn't change.
- **Security Group**: allow inbound TCP 22 from current client IP (/32), not just historic IP.
- Keep `x11vnc` restricted to `-localhost`.
- Keep code defaults at **4002** for Paper; change ports via env only.

### Shell stability (why terminals used to die)
- Use safe wrappers (no exit 1 on no-match): `~/bin/safe_commands.sh`, `~/bin/safe_aliases.sh`.
- Avoid ending scripts with fragile `grep`/`ss` pipelines under `set -e`; add `|| true` or use `awk` counters.
- Diagnostic once: `~/bin/diagnose_exit.sh` (ERR trap + set -x) to locate a failing line.

### Post-maintenance acceptance (5 lines)
```bash
ss -tlnp | egrep '(:4002)\b' || echo "no listener"
ss -tanp | awk '/:4002/ {a[$1]++} END{for(k in a) printf "%-12s %d\n",k,a[k]; if(!NR) print "no sockets"}'
tail -n 120 ~/logs/pipeline.log | egrep -i 'Connected|clientId|TRADING_DECISION|Sharpe|CAGR|drift|PSI' | tail -n 40
free -h
ps aux --sort=-%cpu | head -8
```

### What Claude Code must ask the user to do (cannot do autonomously)
- Confirm the **current EC2 public IP** in AWS Console if port 22 hangs.
- Update the **Security Group** to allow SSH from the user's **current** public IP.
- If the network blocks 22, try a different network (e.g., mobile hotspot).
- Allocate an **Elastic IP** to avoid future IP changes.
- In VNC, manually re-arm API after maintenance (Settings/Precautions → Apply/OK).

### Do-Not-Do list
- Do **not** hardcode 4004 as default; Paper = 4002 (env override allowed).
- Do **not** kill `tmux server` from within a pane (use targeted `tmux kill-session -t ibgw`).
- Do **not** hammer `reqHistoricalData` without 200–300 ms jitter across tickers.

<!-- END: VNC_SSH_IBGW_PLAYBOOK -->
- Any time an IB Gateway API connection fails do not move on or try another approach, pause the process and prompt the user for a port reset to see if that fixes the issue.  Also whenever an API connection to IB Gateway is successful always run another test ~30 seconds-1 minute after the successful connection to ensure the connection is persisting if intended.