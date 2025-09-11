# AI Trading System - Complete Codebase

This repository contains the complete AI-powered algorithmic trading system that integrates with Interactive Brokers (IBKR) for live market data and trading execution.

## System Overview

An advanced reinforcement learning-based trading system featuring:
- Multiple RL algorithms (SAC, RecurrentPPO, PPO, A2C)
- Real-time IBKR integration for paper/live trading
- Advanced feature engineering with technical indicators
- Multi-source news sentiment analysis
- Comprehensive risk management
- Production-ready monitoring and logging

## Core Components

### Main Pipeline
- `run_adaptive_trading.py` - Main orchestrator for the trading workflow
- `rl_trading_pipeline.py` - Core RL training and inference pipeline
- `phase3_enhanced_system.py` - Phase 3 system enhancements

### IBKR Integration
- `market_data_ibkr_adapter.py` - IBKR TWS/Gateway connection handler
- `ibkr_paper_broker.py` - Paper trading broker implementation
- `paper_trading_executor.py` - Trade execution engine
- `order_management_system.py` - Order management and tracking
- `orders/orders_bridge.py` - Order execution bridge

### Machine Learning
- `algorithm_selector.py` - Smart algorithm selection
- `sac_trading_agent.py` - Soft Actor-Critic implementation
- `recurrent_ppo_agent.py` - Recurrent PPO agent
- `enhanced_trading_environment.py` - Gymnasium trading environment
- `ensemble_rl_coordinator.py` - Ensemble model coordination
- `online_learning_system.py` - Online learning capabilities
- `meta_learning_selector.py` - Meta-learning for algorithm selection

### Data Processing
- `feature_engineering.py` - Technical indicator calculation
- `neural_feature_extractor.py` - Neural network feature extraction
- `jax_technical_indicators.py` - JAX-optimized indicators
- `jax_advanced_features.py` - Advanced JAX features

### Risk & Monitoring
- `advanced_risk_management.py` - Comprehensive risk controls
- `comprehensive_system_monitor.py` - System health monitoring
- `memory_management_system.py` - Memory optimization
- `error_handling_system.py` - Error handling and recovery
- `audit_logging_utils.py` - Trade audit and compliance

### News & Sentiment
- `news_ingestion_marketaux.py` - MarketAux news integration
- `news_ingestion_ibkr.py` - IBKR news integration
- `news_data_utils.py` - News processing utilities

### Configuration
- `market_data_config.py` - Central configuration
- `system_optimization_config.py` - System optimization settings
- `configs/` - Additional configuration files

### Testing & Verification
- `verify_trading_intelligence.py` - Core intelligence verification
- `test_production_readiness.py` - Production readiness tests
- `demo_complete_trading_intelligence.py` - Full system demo
- `tools/pipeline_monitor_5loops.py` - 5-loop monitoring tool
- `tests/` - Test suite

### Utilities
- `bin/restart_ibgw_and_rearm.sh` - IB Gateway restart script
- `tools/` - Various utility scripts
- `monitoring/` - Monitoring configurations

## Environment Setup

### Required Environment Variables
```bash
export IBKR_PORT=4002        # 4002 for paper, 4001 for live
export IBKR_CLIENT_ID=9002   # Client ID for IBKR connection
export IBKR_HOST=127.0.0.1   # IBKR Gateway host
```

### AWS Configuration
- S3 Bucket: `omega-singularity-ml`
- Region: Configured via AWS CLI
- Used for: Historical data, model artifacts, feature store

### Dependencies
Install with: `pip install -r requirements.txt`

Key dependencies:
- ib_insync - IBKR API wrapper
- stable-baselines3 - RL algorithms
- jax/jaxlib - High-performance computing
- pandas/polars - Data processing
- boto3 - AWS integration
- lightgbm - Signal validation

## System Architecture

### Data Flow
1. Market data: IBKR → Feature Engineering → ML Models
2. News data: Multiple sources → Sentiment Analysis → Feature Store
3. Predictions: ML Models → Risk Management → Order Execution
4. Audit: All trades → Audit Logs → Compliance Tracking

### Trading Symbols
The system trades futures as proxies:
- ES1! → S&P 500 E-mini futures
- NQ1! → Nasdaq 100 E-mini futures
- 6B → British Pound futures (GBPUSD)
- 6E → Euro futures (EURUSD)
- 6A → Australian Dollar futures (AUDUSD)
- GC → Gold futures (XAUUSD)

## Running the System

### Production Mode
```bash
python run_adaptive_trading.py --mode paper_trading
```

### Testing
```bash
# Verify core intelligence
python verify_trading_intelligence.py

# Test production readiness
python test_production_readiness.py

# Run complete demo
python demo_complete_trading_intelligence.py

# Monitor 5 iterations
python tools/pipeline_monitor_5loops.py
```

### IBKR Connection Test
```bash
python smoke_ib_connect.py
```

## Risk Limits

Configured in `market_data_config.py`:
- Max daily loss: 2%
- Max trades per day: 20
- Max position exposure: 3 contracts
- Max order size: 2 contracts

## System Constraints

Running on AWS EC2 m5.large:
- CPU: 2 vCPUs
- Memory: 8 GiB (target <6GB usage)
- Storage: 100 GB NVMe SSD

## Phase 3 Features

All Phase 3 enhancements are toggleable via feature flags:
- Ensemble learning coordination
- Online learning capabilities
- Meta-learning selection
- LightGBM signal validation
- Advanced JAX features

## Documentation

- `CLAUDE.md` - Detailed system documentation and instructions
- Code is self-documenting with comprehensive docstrings
- Audit logs provide execution transparency

## Support Files

This repository is designed to give ChatGPT and other AI assistants complete visibility into the trading system architecture and implementation, enabling them to provide accurate assistance and modifications.

Last Updated: $(date +%Y-%m-%d)