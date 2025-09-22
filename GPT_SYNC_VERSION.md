# GPT Files Repository Sync Version

<!-- SYNC METADATA START -->
- Last sync: not yet run with file_window_sync
- Files mirrored: unknown
- Total mirror bytes: unknown
- Added files: 0
- Changed files: 0
- Deleted files: 0
<!-- SYNC METADATA END -->

## Current Version: 2025.09.13.001

### Last Sync Information
- **Date**: September 13, 2025
- **Time**: UTC
- **Sync Method**: sync_gpt_files.sh
- **Total Files**: 40+ core system files

### Updated Components
- ✅ Core trading pipeline (run_adaptive_trading.py)
- ✅ IBKR integration modules
- ✅ RL agents (SAC, RecurrentPPO, PPO)
- ✅ Feature engineering system
- ✅ Risk management components
- ✅ Order safety wrappers
- ✅ Test suites and verification scripts
- ✅ Configuration files
- ✅ CLAUDE.md documentation

### Key Changes in This Version
1. Updated order safety wrapper with enhanced validation
2. Improved single-client order execution logic
3. Refined RL trading pipeline structure
4. Updated market data configuration
5. Enhanced IB Gateway restart scripts
6. Added hypothesis testing for order safety

### System Capabilities Verified
- RL-based trading decisions ✅
- Paper trading via IBKR ✅
- Multi-algorithm support (SAC, PPO, RecurrentPPO) ✅
- Risk management and safety checks ✅
- Audit logging and compliance ✅

### Memory & Performance
- System optimized for m5.large EC2 instance
- Memory usage kept under 6GB threshold
- Decision latency < 100ms

### Notes for ChatGPT Analysis
This repository contains the complete EC2 trading system implementation.
All files are synchronized from the production environment.
The system uses reinforcement learning for trading decisions and executes via Interactive Brokers API.
