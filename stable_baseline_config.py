#!/usr/bin/env python3
"""
STABLE BASELINE CONFIGURATION
Minimal, proven configuration that meets operational requirements
"""

import os

# ===========================================================================
# PROVEN STABLE SETTINGS - DO NOT MODIFY WITHOUT TESTING
# ===========================================================================

BASELINE_CONFIG = {
    # Connection (verified working)
    "ibkr": {
        "host": "127.0.0.1",
        "port": 4002,  # Paper trading port
        "client_id": 9002,
        "timeout": 30,
        "reconnect_delay": 5,
        "max_retries": 3,
    },

    # Symbols (start minimal)
    "symbols": {
        "initial": ["ES1!", "NQ1!"],  # Start with 2 only
        "add_after_minutes": 30,  # Add more only after stability proven
        "maximum": 4,  # Never exceed 4 on m5.large
    },

    # Features (pre-selected, no runtime selection)
    "features": [
        # Core 10 features only
        "ret_1", "ret_5",  # Returns
        "atr", "bb_width",  # Volatility
        "rsi", "macd_diff",  # Momentum
        "volume",  # Volume
        "hour", "minute_of_day",  # Time
        "position_held",  # State
    ],

    # Resource Limits (proven stable)
    "resources": {
        "max_memory_mb": 5500,  # Hard limit
        "warning_memory_mb": 5000,
        "cpu_threads": 1,
        "nice_level": 5,
    },

    # Trading Rules (conservative)
    "trading": {
        "max_position": 1,
        "max_trades_per_day": 20,
        "confidence_threshold": 0.65,  # High threshold
        "enable_ml": False,  # Start with rules only
        "dry_run": True,  # Always start dry
    },

    # Monitoring
    "monitoring": {
        "log_level": "INFO",
        "health_check_seconds": 30,
        "socket_check_seconds": 60,
        "memory_check_seconds": 60,
    },

    # DISABLED FEATURES (causing instability)
    "disabled": {
        "drift_detection": False,
        "ensemble": False,
        "online_learning": False,
        "feature_selection": False,
        "oof_stacking": False,
        "meta_learning": False,
    }
}

# ===========================================================================
# DECISION LATENCY OPTIMIZATION
# ===========================================================================

LATENCY_CONFIG = {
    "cache_features": True,  # Cache unchanged features
    "batch_symbols": False,  # Process sequentially for now
    "skip_unnecessary": True,  # Skip features not in top 10
    "precompute_indicators": True,  # Compute once per bar
}

# ===========================================================================
# ERROR RECOVERY
# ===========================================================================

ERROR_RECOVERY = {
    "on_timeout": "reconnect_once",  # Don't retry infinitely
    "on_drift": "log_only",  # Never halt
    "on_memory_high": "reduce_symbols",  # Drop to ES1! only
    "on_socket_leak": "restart_connection",
    "circuit_breaker": {
        "enabled": True,
        "max_errors_per_minute": 3,
        "cooldown_seconds": 60,
    }
}

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def apply_baseline_settings():
    """Apply all baseline settings"""
    # Thread limits
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Stability flags
    os.environ["DRIFT_ENABLED"] = "0"
    os.environ["ENSEMBLE_ENABLED"] = "0"
    os.environ["DRY_RUN"] = "1"

    # Nice level
    try:
        os.nice(5)
    except:
        pass

    print("✅ Baseline settings applied")

def check_prerequisites():
    """Verify system is ready"""
    checks = []

    # Check IB Gateway
    import socket
    try:
        s = socket.socket()
        s.settimeout(2)
        s.connect(("127.0.0.1", 4002))
        s.close()
        checks.append("✅ IB Gateway listening on 4002")
    except:
        checks.append("❌ IB Gateway not accessible on 4002")

    # Check memory
    import psutil
    mem = psutil.virtual_memory()
    if mem.available > 2e9:  # 2GB minimum
        checks.append(f"✅ Memory available: {mem.available/1e9:.1f}GB")
    else:
        checks.append(f"❌ Insufficient memory: {mem.available/1e9:.1f}GB")

    # Check no leaked sockets
    import subprocess
    result = subprocess.run(
        ["ss", "-tanp"],
        capture_output=True,
        text=True
    )
    close_wait = result.stdout.count("CLOSE-WAIT")
    if close_wait < 5:
        checks.append(f"✅ Socket health OK ({close_wait} CLOSE-WAIT)")
    else:
        checks.append(f"⚠️  Socket leak detected ({close_wait} CLOSE-WAIT)")

    return checks

if __name__ == "__main__":
    print("=" * 60)
    print("STABLE BASELINE CONFIGURATION")
    print("=" * 60)

    # Apply settings
    apply_baseline_settings()

    # Run checks
    print("\nPrerequisite Checks:")
    for check in check_prerequisites():
        print(f"  {check}")

    print("\nConfiguration Summary:")
    print(f"  • Symbols: {BASELINE_CONFIG['symbols']['initial']}")
    print(f"  • Features: {len(BASELINE_CONFIG['features'])} selected")
    print(f"  • Memory Limit: {BASELINE_CONFIG['resources']['max_memory_mb']}MB")
    print(f"  • Drift Detection: DISABLED")
    print(f"  • Mode: DRY RUN")
    print("=" * 60)