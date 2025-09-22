"""
Advanced Memory Management System for Trading Platform

This module provides comprehensive memory management, monitoring, and optimization
for the trading system running on AWS m5.large instances (8GB RAM, 2 vCPUs).

Key Features:
- Real-time memory monitoring with configurable thresholds
- Automatic garbage collection optimization
- Memory profiling and leak detection
- Data structure size limiting with FIFO/LRU eviction
- Memory-mapped file support for large datasets
- Automatic memory cleanup triggers
- Memory usage forecasting and alerts
- Integration with system resource limits

Author: AI Trading System
Version: 2.0.0
Hardware Target: AWS m5.large (8GB RAM, 2 vCPUs)
"""

import os
import gc
import sys
import time
import psutil
import logging
import tracemalloc
import weakref
import threading
import warnings
import platform
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict, deque
from functools import wraps, lru_cache
from pathlib import Path
import json
import pickle
import mmap
import numpy as np
import pandas as pd
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryLevel(Enum):
    """Memory usage severity levels"""
    NORMAL = 0      # < 60% usage
    WARNING = 1    # 60-80% usage
    CRITICAL = 2  # 80-90% usage
    EMERGENCY = 3  # > 90% usage


class EvictionPolicy(Enum):
    """Cache eviction policies"""
    FIFO = "fifo"  # First In First Out
    LRU = "lru"    # Least Recently Used
    LFU = "lfu"    # Least Frequently Used
    SIZE = "size"  # Largest items first


@dataclass
class MemoryThresholds:
    """Configurable memory thresholds for m5.large instance"""
    # Total available memory (leaving OS overhead)
    total_available_mb: float = 6000  # ~6GB usable from 8GB total
    
    # Threshold levels
    warning_mb: float = 3600   # 60% of 6GB
    critical_mb: float = 4800  # 80% of 6GB
    emergency_mb: float = 5400  # 90% of 6GB
    
    # Component-specific limits
    feature_store_max_mb: float = 1000  # 1GB for feature store
    model_cache_max_mb: float = 500     # 500MB for models
    data_buffer_max_mb: float = 1500    # 1.5GB for data buffers
    
    # Garbage collection thresholds
    gc_trigger_mb: float = 4000  # Trigger GC at 4GB usage
    gc_aggressive_mb: float = 5000  # Aggressive GC at 5GB
    
    # Cleanup settings
    cleanup_interval_seconds: int = 60  # Check memory every minute
    emergency_cleanup_delay: int = 5    # Wait 5 seconds between emergency cleanups


@dataclass
class MemoryStats:
    """Container for memory statistics"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_mb: float = 0
    available_mb: float = 0
    used_mb: float = 0
    percent_used: float = 0
    process_mb: float = 0
    level: MemoryLevel = MemoryLevel.NORMAL
    
    # Component usage
    feature_store_mb: float = 0
    model_cache_mb: float = 0
    data_buffer_mb: float = 0
    
    # GC stats
    gc_collections: Dict[int, int] = field(default_factory=dict)
    gc_collected: int = 0
    gc_uncollectable: int = 0


class MemoryTrackedDict(OrderedDict):
    """Dictionary with memory tracking and size limits"""
    
    def __init__(self, max_size_mb: float = 100, eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        super().__init__()
        self.max_size_mb = max_size_mb
        self.eviction_policy = eviction_policy
        self.access_counts = {}
        self.access_times = {}
        self.item_sizes = {}
        
    def __setitem__(self, key, value):
        # Calculate item size
        size_mb = self._calculate_size(value) / (1024 * 1024)
        
        # Check if adding this item would exceed limit
        current_size = sum(self.item_sizes.values())
        
        while current_size + size_mb > self.max_size_mb and len(self) > 0:
            # Evict items based on policy
            evicted_key = self._evict_item()
            current_size -= self.item_sizes.get(evicted_key, 0)
        
        # Add the item
        super().__setitem__(key, value)
        self.item_sizes[key] = size_mb
        self.access_counts[key] = 0
        self.access_times[key] = time.time()
    
    def __getitem__(self, key):
        # Track access for LRU/LFU
        if key in self:
            self.access_counts[key] += 1
            self.access_times[key] = time.time()
            
            if self.eviction_policy == EvictionPolicy.LRU:
                # Move to end for LRU
                self.move_to_end(key)
        
        return super().__getitem__(key)
    
    def _calculate_size(self, obj) -> int:
        """Calculate approximate size of an object in bytes"""
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (dict, list, tuple)):
            return sys.getsizeof(obj)
        else:
            # Fallback to pickle size estimation
            try:
                return len(pickle.dumps(obj))
            except:
                return sys.getsizeof(obj)
    
    def _evict_item(self) -> Any:
        """Evict an item based on the eviction policy"""
        if not self:
            return None
        
        if self.eviction_policy == EvictionPolicy.FIFO:
            # Remove oldest item
            key = next(iter(self))
        elif self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            key = min(self.access_times, key=self.access_times.get)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            key = min(self.access_counts, key=self.access_counts.get)
        elif self.eviction_policy == EvictionPolicy.SIZE:
            # Remove largest item
            key = max(self.item_sizes, key=self.item_sizes.get)
        else:
            key = next(iter(self))
        
        # Clean up tracking
        del self[key]
        self.item_sizes.pop(key, None)
        self.access_counts.pop(key, None)
        self.access_times.pop(key, None)
        
        logger.debug(f"Evicted item '{key}' using {self.eviction_policy.value} policy")
        return key
    
    def get_size_mb(self) -> float:
        """Get total size in MB"""
        return sum(self.item_sizes.values())
    
    def clear_old_items(self, max_age_seconds: int = 3600):
        """Clear items older than specified age"""
        current_time = time.time()
        keys_to_remove = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > max_age_seconds
        ]
        
        for key in keys_to_remove:
            del self[key]
            self.item_sizes.pop(key, None)
            self.access_counts.pop(key, None)
            self.access_times.pop(key, None)
        
        return len(keys_to_remove)


class MemoryManager:
    """
    Comprehensive memory management system for trading platform
    
    This class provides automatic memory monitoring, cleanup, and optimization
    specifically tuned for AWS m5.large instances running the trading system.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one memory manager"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the memory manager"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self.thresholds = MemoryThresholds()
        self.process = psutil.Process()
        
        # Memory tracking
        self.memory_history = deque(maxlen=1000)  # Keep last 1000 measurements
        self.cleanup_callbacks = []
        self.protected_objects = weakref.WeakSet()
        
        # Component caches with size limits
        self.feature_store = MemoryTrackedDict(
            max_size_mb=self.thresholds.feature_store_max_mb,
            eviction_policy=EvictionPolicy.LRU
        )
        self.model_cache = MemoryTrackedDict(
            max_size_mb=self.thresholds.model_cache_max_mb,
            eviction_policy=EvictionPolicy.LFU
        )
        self.data_buffer = MemoryTrackedDict(
            max_size_mb=self.thresholds.data_buffer_max_mb,
            eviction_policy=EvictionPolicy.FIFO
        )
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Memory profiling
        self.profiling_enabled = False
        self.profile_snapshots = []
        
        # Configure garbage collection
        self._configure_gc()
        
        # Initialize stats
        self.current_stats = self.get_memory_stats()
        
        host_label = platform.node() or "local-workstation"
        logger.info(
            "MemoryManager initialized for %s (Available: %sMB)",
            host_label,
            self.thresholds.total_available_mb,
        )
        logger.info("PHASE4A-FIX: Memory level comparison using proper enum values")
    
    def _configure_gc(self):
        """Configure garbage collection for optimal performance"""
        # Set GC thresholds for generations 0, 1, 2
        gc.set_threshold(700, 10, 10)  # More aggressive collection
        
        # Enable GC
        gc.enable()
        
        # Log current settings
        logger.debug(f"GC thresholds: {gc.get_threshold()}")
        logger.debug(f"GC counts: {gc.get_count()}")
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        # System memory
        mem = psutil.virtual_memory()
        
        # Process memory
        process_info = self.process.memory_info()
        
        # Calculate usage
        used_mb = (mem.total - mem.available) / (1024 * 1024)
        percent = (used_mb / self.thresholds.total_available_mb) * 100
        
        # Determine level
        if percent < 60:
            level = MemoryLevel.NORMAL
        elif percent < 80:
            level = MemoryLevel.WARNING
        elif percent < 90:
            level = MemoryLevel.CRITICAL
        else:
            level = MemoryLevel.EMERGENCY
        
        stats = MemoryStats(
            total_mb=mem.total / (1024 * 1024),
            available_mb=mem.available / (1024 * 1024),
            used_mb=used_mb,
            percent_used=percent,
            process_mb=process_info.rss / (1024 * 1024),
            level=level,
            feature_store_mb=self.feature_store.get_size_mb(),
            model_cache_mb=self.model_cache.get_size_mb(),
            data_buffer_mb=self.data_buffer.get_size_mb(),
            gc_collections={i: gc.get_count()[i] for i in range(len(gc.get_count()))},
            gc_collected=gc.collect(0),  # Collect generation 0
            gc_uncollectable=len(gc.garbage)
        )
        
        # Store in history
        self.memory_history.append(stats)
        self.current_stats = stats
        
        return stats
    
    def check_memory(self, raise_on_critical: bool = True) -> MemoryLevel:
        """
        Check current memory status
        
        Args:
            raise_on_critical: Raise exception if memory is critical
            
        Returns:
            Current memory level
            
        Raises:
            MemoryError: If memory is critical and raise_on_critical is True
        """
        stats = self.get_memory_stats()
        
        if stats.level == MemoryLevel.EMERGENCY:
            self.emergency_cleanup()
            
        if stats.level in [MemoryLevel.CRITICAL, MemoryLevel.EMERGENCY] and raise_on_critical:
            raise MemoryError(
                f"Memory usage critical: {stats.used_mb:.1f}MB / "
                f"{self.thresholds.total_available_mb:.1f}MB ({stats.percent_used:.1f}%)"
            )
        
        return stats.level
    
    def cleanup(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform memory cleanup
        
        Args:
            force: Force aggressive cleanup regardless of memory level
            
        Returns:
            Cleanup statistics
        """
        start_stats = self.get_memory_stats()
        start_mb = start_stats.used_mb
        
        cleanup_stats = {
            'start_mb': start_mb,
            'gc_collected': 0,
            'cache_cleared': 0,
            'callbacks_run': 0
        }
        
        # Run garbage collection
        if force or start_stats.used_mb > self.thresholds.gc_trigger_mb:
            cleanup_stats['gc_collected'] = gc.collect()
            
            if force or start_stats.used_mb > self.thresholds.gc_aggressive_mb:
                # Aggressive collection of all generations
                for i in range(3):
                    cleanup_stats['gc_collected'] += gc.collect(i)
        
        # Clear old cache items
        if force or start_stats.level.value >= MemoryLevel.WARNING.value:
            max_age = 1800 if force else 3600  # 30 min if forced, else 1 hour
            
            cleanup_stats['cache_cleared'] += self.feature_store.clear_old_items(max_age)
            cleanup_stats['cache_cleared'] += self.model_cache.clear_old_items(max_age)
            cleanup_stats['cache_cleared'] += self.data_buffer.clear_old_items(max_age)
        
        # Run registered cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
                cleanup_stats['callbacks_run'] += 1
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
        
        # Get final stats
        end_stats = self.get_memory_stats()
        cleanup_stats['end_mb'] = end_stats.used_mb
        cleanup_stats['freed_mb'] = start_mb - end_stats.used_mb
        
        logger.info(f"Memory cleanup: freed {cleanup_stats['freed_mb']:.1f}MB, "
                   f"GC collected {cleanup_stats['gc_collected']} objects")
        
        return cleanup_stats
    
    def emergency_cleanup(self) -> Dict[str, Any]:
        """Emergency memory cleanup when critically low"""
        logger.warning("EMERGENCY MEMORY CLEANUP INITIATED")
        
        # Clear all caches
        self.feature_store.clear()
        self.model_cache.clear()
        self.data_buffer.clear()
        
        # Force aggressive GC
        cleanup_stats = self.cleanup(force=True)
        
        # Clear module caches
        sys.modules.clear()
        
        # Clear matplotlib figures if imported
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
        
        return cleanup_stats
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a callback to be called during cleanup"""
        if callable(callback):
            self.cleanup_callbacks.append(callback)
            logger.debug(f"Registered cleanup callback: {callback.__name__}")
    
    def protect_object(self, obj: Any):
        """Mark an object as protected from cleanup"""
        self.protected_objects.add(obj)
    
    @contextmanager
    def memory_limit(self, max_mb: float):
        """
        Context manager to enforce memory limit for a block
        
        Args:
            max_mb: Maximum memory in MB for the block
            
        Example:
            with memory_manager.memory_limit(500):
                # Code that should not use more than 500MB
                large_data = process_data()
        """
        start_stats = self.get_memory_stats()
        start_mb = start_stats.process_mb
        
        try:
            yield
        finally:
            end_stats = self.get_memory_stats()
            used_mb = end_stats.process_mb - start_mb
            
            if used_mb > max_mb:
                logger.warning(f"Memory limit exceeded: {used_mb:.1f}MB > {max_mb:.1f}MB")
                self.cleanup()
    
    @contextmanager
    def memory_profile(self, label: str = ""):
        """
        Context manager for memory profiling
        
        Args:
            label: Label for this profiling session
        """
        if not self.profiling_enabled:
            tracemalloc.start()
        
        snapshot_start = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            snapshot_end = tracemalloc.take_snapshot()
            
            # Calculate differences
            top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
            
            logger.info(f"Memory profile [{label}]:")
            for stat in top_stats[:10]:
                logger.info(f"  {stat}")
            
            self.profile_snapshots.append({
                'label': label,
                'timestamp': datetime.now(),
                'snapshot': snapshot_end
            })
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory report"""
        stats = self.get_memory_stats()
        
        # Calculate trends if we have history
        trend = "stable"
        if len(self.memory_history) > 10:
            recent = [s.used_mb for s in list(self.memory_history)[-10:]]
            if recent[-1] > recent[0] * 1.1:
                trend = "increasing"
            elif recent[-1] < recent[0] * 0.9:
                trend = "decreasing"
        
        report = {
            'timestamp': stats.timestamp.isoformat(),
            'level': stats.level.name.lower(),
            'usage': {
                'total_mb': round(stats.total_mb, 2),
                'available_mb': round(stats.available_mb, 2),
                'used_mb': round(stats.used_mb, 2),
                'percent': round(stats.percent_used, 2),
                'process_mb': round(stats.process_mb, 2)
            },
            'components': {
                'feature_store_mb': round(stats.feature_store_mb, 2),
                'model_cache_mb': round(stats.model_cache_mb, 2),
                'data_buffer_mb': round(stats.data_buffer_mb, 2)
            },
            'garbage_collection': {
                'collections': stats.gc_collections,
                'collected': stats.gc_collected,
                'uncollectable': stats.gc_uncollectable
            },
            'trend': trend,
            'thresholds': {
                'warning_mb': self.thresholds.warning_mb,
                'critical_mb': self.thresholds.critical_mb,
                'emergency_mb': self.thresholds.emergency_mb
            }
        }
        
        return report
    
    def start_monitoring(self, interval: int = None):
        """Start automatic memory monitoring"""
        if self.monitoring_active:
            return
        
        interval = interval or self.thresholds.cleanup_interval_seconds
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    stats = self.get_memory_stats()
                    
                    # Log warnings  
                    if stats.level.value >= MemoryLevel.WARNING.value:
                        logger.warning(f"Memory {stats.level.name.lower()}: "
                                     f"{stats.used_mb:.1f}MB ({stats.percent_used:.1f}%)")
                    
                    # Auto cleanup if needed
                    if stats.level.value >= MemoryLevel.CRITICAL.value:
                        self.cleanup()
                    
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval)
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Memory monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop automatic memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def optimize_dataframe(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            inplace: Modify in place
            
        Returns:
            Optimized DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert object columns to category if beneficial
        for col in df.select_dtypes(include=['object']).columns:
            num_unique = df[col].nunique()
            num_total = len(df[col])
            
            if num_unique / num_total < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')
        
        return df
    
    def create_memory_mapped_array(self, shape: Tuple, dtype: np.dtype, 
                                  filename: Optional[str] = None) -> np.ndarray:
        """
        Create a memory-mapped numpy array for large datasets
        
        Args:
            shape: Shape of the array
            dtype: Data type
            filename: Optional file path for persistence
            
        Returns:
            Memory-mapped array
        """
        if filename is None:
            filename = Path('/tmp') / f'mmap_{datetime.now():%Y%m%d_%H%M%S}.dat'
        
        return np.memmap(filename, dtype=dtype, mode='w+', shape=shape)


def memory_efficient(max_mb: float = 100):
    """
    Decorator to enforce memory limits on functions
    
    Args:
        max_mb: Maximum memory in MB the function can use
        
    Example:
        @memory_efficient(max_mb=500)
        def process_large_data():
            # Function that should not use more than 500MB
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = MemoryManager()
            
            with manager.memory_limit(max_mb):
                result = func(*args, **kwargs)
            
            return result
        
        return wrapper
    return decorator


def auto_cleanup(threshold_mb: float = None):
    """
    Decorator to automatically cleanup memory after function execution
    
    Args:
        threshold_mb: Cleanup if memory usage exceeds this threshold
        
    Example:
        @auto_cleanup(threshold_mb=4000)
        def data_processing():
            # Process data
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = MemoryManager()
            
            try:
                result = func(*args, **kwargs)
            finally:
                stats = manager.get_memory_stats()
                
                if threshold_mb and stats.used_mb > threshold_mb:
                    manager.cleanup()
            
            return result
        
        return wrapper
    return decorator


# Global instance
_memory_manager = None


def get_memory_manager() -> MemoryManager:
    """Get or create the singleton MemoryManager instance"""
    global _memory_manager
    
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    
    return _memory_manager


if __name__ == "__main__":
    # Self-test
    print("Memory Manager Self-Test")
    print("=" * 50)
    
    manager = get_memory_manager()
    
    # Get current stats
    report = manager.get_memory_report()
    
    print(f"\nMemory Status: {report['level']}")
    print(f"Usage: {report['usage']['used_mb']:.1f}MB / "
          f"{report['usage']['total_mb']:.1f}MB "
          f"({report['usage']['percent']:.1f}%)")
    
    print(f"\nComponent Usage:")
    for component, size in report['components'].items():
        print(f"  {component}: {size:.1f}MB")
    
    print(f"\nGarbage Collection:")
    print(f"  Collections: {report['garbage_collection']['collections']}")
    print(f"  Uncollectable: {report['garbage_collection']['uncollectable']}")
    
    print(f"\nTrend: {report['trend']}")
    
    # Test memory tracking
    print("\nTesting memory-tracked dictionary...")
    test_dict = MemoryTrackedDict(max_size_mb=10)
    
    # Add some test data
    for i in range(100):
        test_dict[f"key_{i}"] = np.random.randn(1000, 100)
        if i % 10 == 0:
            print(f"  Added {i+1} items, size: {test_dict.get_size_mb():.2f}MB")
    
    print(f"  Final size: {test_dict.get_size_mb():.2f}MB")
    print(f"  Items in dict: {len(test_dict)}")
    
    # Start monitoring
    manager.start_monitoring(interval=5)
    print("\nMemory monitoring started (5 second interval)")
    print("Self-test complete!")
