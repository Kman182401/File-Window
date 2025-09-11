"""
System Optimization Configuration for m5.large EC2 Instance
Based on comprehensive system analysis performed on 2025-08-25

This configuration is specifically tailored for:
- Intel Xeon Platinum 8259CL @ 2.50GHz (2 vCPUs with HyperThreading)
- 8GB RAM (7.6GB usable, ~3.6GB typically available)
- AVX512 instruction set support
- 35.8 MiB L3 cache
- No GPU/CUDA available
- IB Gateway consuming ~500MB RAM
"""

import os
from typing import Dict, Any

# ============== SYSTEM SPECIFICATIONS ==============
SYSTEM_SPECS = {
    "cpu": {
        "model": "Intel Xeon Platinum 8259CL",
        "cores": 1,
        "threads": 2,
        "frequency_ghz": 2.50,
        "cache": {
            "l1d_kb": 32,
            "l1i_kb": 32,
            "l2_mb": 1,
            "l3_mb": 35.8
        },
        "features": [
            "avx512f", "avx512dq", "avx512cd", "avx512bw", "avx512vl",
            "avx2", "avx", "fma", "sse4_2", "sse4_1"
        ]
    },
    "memory": {
        "total_gb": 8.0,
        "usable_gb": 7.6,
        "typical_available_gb": 3.6,
        "page_size_bytes": 4096,
        "no_swap": True
    },
    "storage": {
        "total_gb": 97,
        "available_gb": 63
    },
    "network": {
        "interface": "ens5",
        "mtu": 9001,  # Jumbo frames enabled
        "ena_adapter": True
    }
}

# ============== MEMORY ALLOCATION STRATEGY ==============
MEMORY_ALLOCATION = {
    # Conservative allocation leaving 2GB for OS and other processes
    "total_budget_mb": 6000,
    
    # Component allocations
    "ib_gateway": 500,          # IB Gateway Java process
    "rl_models": 1000,           # RL models and training
    "gradient_boosting": 500,    # LightGBM/XGBoost models
    "data_buffers": 1500,        # Market data and features
    "jax_operations": 800,       # JAX JIT compilation cache
    "ray_plasma": 500,           # Ray object store
    "feature_store": 300,        # Real-time feature storage
    "emergency_buffer": 900      # Safety margin
}

# ============== JAX OPTIMIZATION ==============
JAX_CONFIG = {
    # CPU-specific optimizations
    "platform": "cpu",
    "enable_x64": True,  # Use 64-bit precision for financial data
    
    # Memory management
    "preallocate": False,  # Don't preallocate memory
    "mem_fraction": 0.3,   # Use max 30% of available memory
    
    # JIT compilation settings
    "jit_config": {
        "inline_threshold": 10000,  # Aggressive inlining for L3 cache
        "enable_xla_fusion": True,   # Enable operation fusion
        "opt_level": 3,              # Maximum optimization
    },
    
    # Threading (use both vCPUs)
    "cpu_device_count": 2,
    "xla_cpu_multi_thread": True,
    
    # AVX512 optimizations
    "enable_avx512": True,
    "vectorize_threshold": 100,  # Vectorize ops with >100 elements
}

# ============== PYTORCH LIGHTNING OPTIMIZATION ==============
LIGHTNING_CONFIG = {
    # Mixed precision for memory efficiency
    "precision": 16,  # Use FP16 (saves 50% memory)
    
    # Gradient accumulation for larger effective batch sizes
    "accumulate_grad_batches": 8,
    
    # Memory management
    "gradient_clip_val": 0.5,
    "max_memory_gb": 2.0,  # Limit to 2GB for training
    
    # CPU-specific settings
    "accelerator": "cpu",
    "devices": 1,  # Single device (multi-threading handled internally)
    "num_workers": 1,  # DataLoader workers (avoid process overhead)
    
    # Checkpointing
    "enable_checkpointing": True,
    "checkpoint_frequency": 100,
}

# ============== STABLE-BASELINES3 OPTIMIZATION ==============
SB3_CONFIG = {
    # SAC configuration optimized for m5.large
    "sac": {
        "buffer_size": 50000,      # Reduced from 1M default
        "batch_size": 64,          # Small batches for memory
        "learning_rate": 3e-4,
        "gradient_steps": 1,
        "train_freq": 1,
        "use_sde": True,           # State-dependent exploration
        "policy_kwargs": {
            "net_arch": [128, 128],  # Smaller network for CPU
        }
    },
    
    # PPO configuration
    "ppo": {
        "n_steps": 1024,           # Reduced for memory
        "batch_size": 32,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "clip_range": 0.2,
    },
    
    # Common settings
    "device": "cpu",
    "verbose": 1,
    "tensorboard_log": "./tensorboard_logs/",
}

# ============== RAY CONFIGURATION ==============
RAY_CONFIG = {
    # Resource allocation
    "num_cpus": 2,
    "num_gpus": 0,
    
    # Object store memory (conservative)
    "object_store_memory": 500_000_000,  # 500MB
    
    # Plasma store settings
    "plasma_directory": "/tmp",
    
    # Ray Tune settings
    "tune": {
        "max_concurrent_trials": 1,  # Sequential trials to avoid OOM
        "checkpoint_freq": 10,
        "checkpoint_at_end": True,
    },
    
    # Scheduling
    "scheduler": "FIFO",  # Simple scheduling for limited resources
}

# ============== LIGHTGBM OPTIMIZATION ==============
LIGHTGBM_CONFIG = {
    # Core parameters
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    
    # CPU optimization
    "num_threads": 2,  # Use both vCPUs
    "device_type": "cpu",
    
    # Memory optimization
    "max_bin": 127,  # Reduced from 255 for memory
    "histogram_pool_size": 256,  # MB for histogram pool
    
    # Training efficiency
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    
    # Cache optimization (utilize L3 cache)
    "two_round": True,  # Use two-round loading for large datasets
}

# ============== MODIN/PANDAS OPTIMIZATION ==============
MODIN_CONFIG = {
    "engine": "ray",  # Use Ray backend
    "num_cpus": 2,
    
    # Memory settings
    "memory_limit": "2GB",
    "enable_cache": True,
    
    # Chunk size for operations
    "chunk_size": 100000,
}

# ============== DATA PIPELINE OPTIMIZATION ==============
DATA_PIPELINE_CONFIG = {
    # Batch processing
    "batch_size": 1000,
    "prefetch_batches": 2,
    
    # Streaming settings
    "stream_buffer_size_mb": 100,
    "enable_compression": True,
    
    # Feature engineering
    "parallel_features": True,
    "feature_cache_size_mb": 200,
    
    # I/O optimization
    "use_memory_map": True,
    "read_chunk_size": 50000,
}

# ============== TRADING SYSTEM OPTIMIZATION ==============
TRADING_CONFIG = {
    # Risk management aligned with memory
    "max_concurrent_positions": 3,  # Limit to reduce memory
    "max_orders_per_minute": 10,
    
    # Decision engine
    "decision_batch_size": 1,  # Process one symbol at a time
    "enable_caching": True,
    "cache_ttl_seconds": 60,
    
    # Feature store
    "feature_store_max_rows": 2000,  # Reduced from 3000
    "feature_store_max_memory_mb": 200,  # Reduced from 300
    
    # Model inference
    "inference_batch_size": 1,
    "enable_model_quantization": True,  # Reduce model size
}

# ============== MONITORING & PROFILING ==============
MONITORING_CONFIG = {
    # Memory monitoring
    "enable_memory_profiling": True,
    "memory_check_interval_seconds": 60,
    "memory_warning_threshold_pct": 80,
    "memory_critical_threshold_pct": 90,
    
    # Performance monitoring
    "enable_cpu_profiling": False,  # Disable in production
    "log_level": "INFO",
    
    # Metrics collection
    "metrics_interval_seconds": 30,
    "enable_prometheus": True,
}

# ============== ENVIRONMENT VARIABLES ==============
def set_optimization_env_vars():
    """Set environment variables for optimal performance."""
    
    # OpenMP settings for numerical libraries
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    
    # JAX settings
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    
    # Python settings
    os.environ["PYTHONHASHSEED"] = "0"  # Reproducibility
    
    # Memory settings
    os.environ["MALLOC_ARENA_MAX"] = "2"  # Reduce memory fragmentation

# ============== HELPER FUNCTIONS ==============
def get_available_memory_mb() -> int:
    """Get currently available memory in MB."""
    import psutil
    return int(psutil.virtual_memory().available / (1024 * 1024))

def should_use_reduced_mode() -> bool:
    """Check if system should run in reduced mode due to memory pressure."""
    available_mb = get_available_memory_mb()
    return available_mb < 2000  # Less than 2GB available

def get_optimal_batch_size(model_type: str) -> int:
    """Get optimal batch size based on available memory."""
    available_mb = get_available_memory_mb()
    
    if model_type == "rl_training":
        if available_mb > 3000:
            return 64
        elif available_mb > 2000:
            return 32
        else:
            return 16
    
    elif model_type == "gradient_boosting":
        if available_mb > 2500:
            return 10000
        elif available_mb > 1500:
            return 5000
        else:
            return 1000
    
    elif model_type == "inference":
        return 1  # Always use batch size of 1 for real-time inference
    
    return 32  # Default

def get_worker_count(task_type: str) -> int:
    """Get optimal worker count for different tasks."""
    if task_type == "data_loading":
        return 1  # Avoid process overhead
    elif task_type == "feature_engineering":
        return 2  # Use both cores
    elif task_type == "model_training":
        return 1  # Single process, multi-threaded
    return 1

# ============== ADAPTIVE CONFIGURATION ==============
class AdaptiveConfig:
    """Dynamically adjust configuration based on system state."""
    
    def __init__(self):
        self.base_config = {
            "jax": JAX_CONFIG,
            "lightning": LIGHTNING_CONFIG,
            "sb3": SB3_CONFIG,
            "ray": RAY_CONFIG,
            "lightgbm": LIGHTGBM_CONFIG,
            "trading": TRADING_CONFIG,
        }
    
    def get_config(self, component: str) -> Dict[str, Any]:
        """Get configuration adjusted for current system state."""
        import psutil
        
        config = self.base_config.get(component, {}).copy()
        
        # Adjust based on available memory
        mem_available_gb = psutil.virtual_memory().available / (1024**3)
        
        if mem_available_gb < 2.0:
            # Low memory mode
            if component == "sb3":
                config["sac"]["buffer_size"] = 10000
                config["sac"]["batch_size"] = 32
            elif component == "lightgbm":
                config["max_bin"] = 63
                config["histogram_pool_size"] = 128
            elif component == "trading":
                config["feature_store_max_rows"] = 1000
                config["feature_store_max_memory_mb"] = 100
        
        # Adjust based on CPU load
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            # High CPU load
            if component == "lightgbm":
                config["num_threads"] = 1
            elif component == "ray":
                config["num_cpus"] = 1
        
        return config

# ============== INITIALIZATION ==============
def initialize_optimized_environment():
    """Initialize the environment with optimal settings."""
    print("Initializing optimized environment for m5.large...")
    
    # Set environment variables
    set_optimization_env_vars()
    
    # Check system state
    import psutil
    mem_available_gb = psutil.virtual_memory().available / (1024**3)
    cpu_count = psutil.cpu_count()
    
    print(f"System Status:")
    print(f"  Available Memory: {mem_available_gb:.1f} GB")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  AVX512 Support: Yes")
    print(f"  L3 Cache: 35.8 MB")
    
    # Recommendations
    if mem_available_gb < 2.0:
        print("⚠️  Low memory detected. Running in reduced mode.")
    
    print("✓ Optimization configuration loaded successfully!")
    
    return AdaptiveConfig()

if __name__ == "__main__":
    # Test configuration
    config = initialize_optimized_environment()
    
    print("\nSample configurations:")
    print(f"SB3 Config: {config.get_config('sb3')['sac']}")
    print(f"Optimal batch size (RL): {get_optimal_batch_size('rl_training')}")
    print(f"Worker count (features): {get_worker_count('feature_engineering')}")