#!/usr/bin/env python3
"""
Optimized Trading System Configuration
Implements targeted, low-risk optimizations for learning upgrades
while maintaining CPU efficiency on m5.large instance
"""

import os
from typing import Dict, List, Any

# =============================================================================
# 1. RUNTIME OPTIMIZATION - Single Agent (PPO) with VecNormalize
# =============================================================================

RUNTIME_CONFIG = {
    "agent_type": "PPO",  # Single agent only, no ensemble during runtime
    "use_vecnormalize": True,  # Enable VecNormalize for stable training
    "vecnormalize_config": {
        "norm_obs": True,
        "norm_reward": True,
        "clip_obs": 10.0,
        "clip_reward": 10.0,
        "gamma": 0.99
    },
    "ppo_config": {
        "n_steps": 2048,  # Reduced for memory efficiency
        "batch_size": 64,  # Small batch to reduce memory
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "policy": "MlpPolicy",  # Simple MLP, no LSTM
        "policy_kwargs": {
            "net_arch": [64, 64],  # Small network
            "activation_fn": "tanh"
        }
    },
    "disable_stats_update_in_eval": True,  # No stats update during evaluation
    "single_order_path": True,  # Unified order execution path
}

# Off-hours RecurrentPPO config (optional)
OFFHOURS_RECURRENT_CONFIG = {
    "enabled": False,  # Only enable during off-hours
    "hidden_size": 32,  # Tiny LSTM
    "sequence_length": 32,
    "max_steps": 200000
}

# =============================================================================
# 2. FEATURE BUDGET CONSTRAINTS (15-20 features per symbol)
# =============================================================================

FEATURE_CONFIG = {
    "max_features_per_symbol": 20,
    "core_features": [
        # Price-based (5)
        "ret_1",     # 1-period return
        "ret_5",     # 5-period return
        "ret_15",    # 15-period return (optional, can be removed)
        "close",     # Current price
        "vwap",      # Volume-weighted average price

        # Volatility (3)
        "atr",       # Average True Range
        "bb_width",  # Bollinger Band width
        "zscore_20", # Z-score for mean reversion

        # Momentum (4)
        "rsi",       # RSI(14)
        "macd_diff", # MACD divergence
        "stoch_k",   # Stochastic K
        "efficiency_ratio",  # Market efficiency

        # Volume (2)
        "volume",    # Current volume
        "vol_spike", # Volume spike indicator

        # Microstructure (3)
        "minute_of_day",  # Time of day
        "hour",           # Hour
        "day_of_week",    # Day of week

        # Memory/State (3)
        "last_action",    # Previous action taken
        "unrealized_pnl", # Current position P&L
        "position_held"   # Current position
    ],
    "use_fold_wise_scaling": True,  # Per-fold StandardScaler
    "feature_selection_method": "mutual_info",  # Off-hours only
    "feature_selection_k": 15  # Keep top-k features
}

# =============================================================================
# 3. EVENT LABELS & EMBARGO (Triple Barrier)
# =============================================================================

EVENT_LABEL_CONFIG = {
    "method": "triple_barrier",
    "profit_taking_multiple": 2.0,  # pt = 2x costs
    "stop_loss_multiple": 2.0,      # sl = 2x costs
    "max_horizon": 20,               # Maximum holding period
    "min_return_threshold": 0.0002, # 2 bps minimum move
    "sample_weight_method": "magnitude",  # Weight by |move|
    "embargo_periods": "dynamic",   # Embargo = max_horizon
    "purged_cv": True,              # Use purged cross-validation
}

# =============================================================================
# 4. OOF STACKING & AGREEMENT GATE
# =============================================================================

STACKING_CONFIG = {
    "enabled": True,
    "use_oof_only": True,  # Meta uses OOF probabilities only
    "agreement_gate": {
        "base_threshold": 0.5,
        "meta_agree_threshold": 0.55,  # Meta must agree >0.55 for long
        "meta_disagree_threshold": 0.45,  # Meta must agree <0.45 for short
        "strong_signal_threshold": 0.65,  # Strong signals bypass agreement
        "weak_signal_threshold": 0.35,    # Strong short signals
        "enabled": True
    },
    "escape_clause": {
        "meta_override_high": 0.65,  # Meta can override if very confident
        "meta_override_low": 0.35,
        "enabled": True
    }
}

# =============================================================================
# 5. POST-COST OOS SELECTION
# =============================================================================

OOS_SELECTION_CONFIG = {
    "enabled": True,
    "metrics": ["sharpe_post_cost", "calmar_ratio", "trade_count", "turnover"],
    "selection_method": "calmar_weighted",  # CAGR/MaxDD ratio
    "min_sharpe": 0.5,
    "min_calmar": 0.3,
    "min_trades": 20,
    "max_turnover": 10.0,
    "use_final_fold_only": True,  # Never average train folds
    "include_transaction_costs": True,
    "transaction_cost_bps": 2  # 2 basis points per trade
}

# =============================================================================
# 6. CONFIDENCE-BASED SIZING
# =============================================================================

SIZING_CONFIG = {
    "method": "confidence_vol_inverse",
    "base_size": 1,
    "vol_floor": 0.001,  # 10 bps floor
    "vol_lookback": 20,
    "confidence_threshold": 0.6,  # Only trade if confidence > 60%
    "size_formula": "confidence * (0.01 / max(vol, vol_floor))",
    "max_position_size": 1,  # Hard cap at 1 micro during validation
    "min_trade_size": 1,
    "scale_by_kelly": False  # Disable Kelly criterion for now
}

# =============================================================================
# 7. ATTACH/PACING HYGIENE
# =============================================================================

IBKR_CONFIG = {
    "connection": {
        "host": "127.0.0.1",
        "port": int(os.getenv("IBKR_PORT", "4002")),
        "client_id": int(os.getenv("IBKR_CLIENT_ID", "9002")),
        "timeout": 30,
        "max_retries": 3,
        "exponential_backoff": True,
        "backoff_base": 2,
        "backoff_max": 60
    },
    "pacing": {
        "historical_data_delay_ms": 250,  # 200-300ms jitter
        "historical_data_jitter_ms": 50,
        "max_requests_per_second": 3,
        "max_messages_per_second": 50,
        "order_delay_ms": 100,
        "market_data_delay_ms": 50
    },
    "symbols": {
        "start_lean": ["ES1!", "NQ1!"],  # Start with 2 symbols only
        "full_list": ["ES1!", "NQ1!", "6E", "6B", "6A", "GC"],
        "add_incrementally": True,
        "increment_after_minutes": 5
    },
    "safe_launcher": {
        "probe_first": True,
        "probe_timeout": 5,
        "wait_after_probe": 2,
        "start_with_dry_run": True,
        "dry_run_duration": 60
    }
}

# =============================================================================
# 8. RESOURCE FOOTPRINT CONSTRAINTS
# =============================================================================

RESOURCE_CONFIG = {
    "cpu": {
        "omp_num_threads": 1,
        "mkl_num_threads": 1,
        "openblas_num_threads": 1,
        "nice_level": 5,
        "cpu_affinity": "0-1",  # Pin to cores 0-1
        "max_parallel_tasks": 2
    },
    "memory": {
        "max_memory_gb": 5.5,  # Leave headroom on 8GB system
        "warning_threshold_gb": 5.0,
        "dataframe_max_rows": 100000,
        "cache_size_mb": 100,
        "gc_threshold": 0.8  # Trigger GC at 80% memory
    },
    "logging": {
        "format": "jsonl",
        "rolling": True,
        "max_size_mb": 100,
        "compress": True,
        "retention_days": 7
    },
    "async": {
        "max_concurrent_tasks": 2,  # Limited for m5.large
        "semaphore_limit": 3,
        "queue_size": 10
    },
    "display": {
        "disable_x11vnc_when_idle": True,
        "idle_timeout_minutes": 30
    }
}

# =============================================================================
# 9. DRIFT DETECTION (Lightweight)
# =============================================================================

DRIFT_CONFIG = {
    "enabled": False,  # DISABLED per baseline policy in CLAUDE.md
    "check_frequency_bars": 60,  # Check every hour (when enabled)
    "methods": ["psi", "rolling_zscore"],
    "key_features": ["ret_1", "atr", "volume", "efficiency_ratio"],
    "psi_threshold": 0.2,
    "zscore_threshold": 3.0,
    "rolling_window": 100,
    "on_drift_detected": "log_only",  # Just log, don't halt pipeline
    "alert_on_drift": False,
    "disable_heavy_retraining_hours": [9, 10, 11, 12, 13, 14, 15],  # Market hours
}

# =============================================================================
# INTEGRATION FLAGS
# =============================================================================

FEATURE_FLAGS = {
    # Core optimizations (always enabled)
    "use_optimized_config": True,
    "single_agent_runtime": True,
    "feature_budget": True,
    "event_labels": True,
    "oof_stacking": True,
    "post_cost_selection": True,
    "confidence_sizing": True,
    "resource_limits": True,
    "drift_detection": False,  # DISABLED per baseline policy

    # Phase 3 features (disabled for now)
    "ensemble_enabled": False,
    "online_learning_enabled": False,
    "meta_learning_enabled": False,
    "lightgbm_validator_enabled": False,
    "transformer_features_enabled": False,

    # Safety features
    "dry_run": bool(int(os.getenv("DRY_RUN", "1"))),
    "enable_order_exec": bool(int(os.getenv("ENABLE_ORDER_EXEC", "0"))),
    "allow_orders": bool(int(os.getenv("ALLOW_ORDERS", "0"))),
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def apply_resource_limits():
    """Apply CPU and memory constraints"""
    import os

    # Set thread limits
    os.environ["OMP_NUM_THREADS"] = str(RESOURCE_CONFIG["cpu"]["omp_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(RESOURCE_CONFIG["cpu"]["mkl_num_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(RESOURCE_CONFIG["cpu"]["openblas_num_threads"])

    # Set nice level
    try:
        os.nice(RESOURCE_CONFIG["cpu"]["nice_level"])
    except:
        pass

    # CPU affinity
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([0, 1])
    except:
        pass

def get_feature_list(symbol: str = None) -> List[str]:
    """Get the feature list for a given symbol"""
    features = FEATURE_CONFIG["core_features"].copy()

    # Add symbol-specific features if needed
    if symbol and symbol in ["6E", "6B", "6A"]:
        # FX pairs might need currency-specific features
        features.append("roll_corr_open")

    # Limit to max features
    return features[:FEATURE_CONFIG["max_features_per_symbol"]]

def should_trade(confidence: float, volatility: float) -> bool:
    """Determine if we should trade based on confidence and volatility"""
    if confidence < SIZING_CONFIG["confidence_threshold"]:
        return False

    vol_adjusted_size = confidence * (0.01 / max(volatility, SIZING_CONFIG["vol_floor"]))
    return vol_adjusted_size >= SIZING_CONFIG["min_trade_size"]

def calculate_position_size(confidence: float, volatility: float) -> int:
    """Calculate position size based on confidence and volatility"""
    if not should_trade(confidence, volatility):
        return 0

    vol_adjusted_size = confidence * (0.01 / max(volatility, SIZING_CONFIG["vol_floor"]))
    size = min(vol_adjusted_size, SIZING_CONFIG["max_position_size"])
    return max(int(size), SIZING_CONFIG["min_trade_size"])

# Auto-apply resource limits on import
if FEATURE_FLAGS["resource_limits"]:
    apply_resource_limits()

print("Optimized trading configuration loaded successfully")