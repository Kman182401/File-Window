#!/usr/bin/env python3
"""
Memory Management System for AI Trading Platform

Comprehensive memory monitoring, limiting, and optimization system
designed for m5.large EC2 instance (8GB RAM, 2 vCPUs).

Features:
- Real-time memory monitoring
- Automatic garbage collection
- Component-wise memory limits
- Memory pressure detection and response
- Circuit breakers for OOM prevention
"""

import psutil
import gc
import os
import sys
import threading
import time
import logging
import tracemalloc
import warnings
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from functools import wraps
from collections import deque
import json
import signal

# Import system configuration
from system_optimization_config import (
    MEMORY_ALLOCATION,
    get_available_memory_mb,
    should_use_reduced_mode
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Real-time memory monitoring with alerts and automatic actions.
    """
    
    def __init__(self, check_interval: int = 30):
        """
        Initialize memory monitor.
        
        Args:
            check_interval: Seconds between memory checks
        """
        self.check_interval = check_interval
        self.is_running = False
        self.monitor_thread = None
        
        # Memory thresholds (MB)
        self.warning_threshold_mb = 2000  # Warn when <2GB available
        self.critical_threshold_mb = 1000  # Critical when <1GB available
        self.emergency_threshold_mb = 500  # Emergency when <500MB available
        
        # Memory history
        self.memory_history = deque(maxlen=100)
        self.alert_callbacks = []
        
        # Statistics
        self.stats = {
            'checks_performed': 0,
            'warnings_triggered': 0,
            'critical_alerts': 0,
            'emergency_actions': 0,
            'gc_collections': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Memory monitor initialized")
    
    def start(self):
        """Start monitoring in background thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop(self):
        """Stop monitoring."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                self._check_memory()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
    
    def _check_memory(self):
        """Check current memory status and take actions if needed."""
        # Get memory info
        mem = psutil.virtual_memory()
        available_mb = mem.available / (1024 * 1024)
        used_mb = mem.used / (1024 * 1024)
        percent_used = mem.percent
        
        # Record history
        self.memory_history.append({
            'timestamp': datetime.now(),
            'available_mb': available_mb,
            'used_mb': used_mb,
            'percent_used': percent_used
        })
        
        self.stats['checks_performed'] += 1
        
        # Check thresholds and take actions
        if available_mb < self.emergency_threshold_mb:
            self._handle_emergency(available_mb)
        elif available_mb < self.critical_threshold_mb:
            self._handle_critical(available_mb)
        elif available_mb < self.warning_threshold_mb:
            self._handle_warning(available_mb)
        
        # Log status every 10 checks
        if self.stats['checks_performed'] % 10 == 0:
            logger.info(f"Memory status: {available_mb:.0f}MB available ({percent_used:.1f}% used)")
    
    def _handle_warning(self, available_mb: float):
        """Handle warning level memory pressure."""
        self.stats['warnings_triggered'] += 1
        logger.warning(f"Memory warning: Only {available_mb:.0f}MB available")
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback('warning', available_mb)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _handle_critical(self, available_mb: float):
        """Handle critical memory pressure."""
        self.stats['critical_alerts'] += 1
        logger.critical(f"Memory critical: Only {available_mb:.0f}MB available")
        
        # Force garbage collection
        self._force_gc()
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback('critical', available_mb)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _handle_emergency(self, available_mb: float):
        """Handle emergency memory situation."""
        self.stats['emergency_actions'] += 1
        logger.critical(f"EMERGENCY: Only {available_mb:.0f}MB available - taking emergency actions")
        
        # Emergency actions
        self._force_gc()
        self._clear_caches()
        self._reduce_memory_usage()
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback('emergency', available_mb)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _force_gc(self):
        """Force garbage collection."""
        self.stats['gc_collections'] += 1
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
    
    def _clear_caches(self):
        """Clear various caches to free memory."""
        try:
            # Clear Python caches
            gc.collect(2)  # Full collection
            
            # Clear NumPy/JAX caches if available
            try:
                import jax
                jax.clear_caches()
            except:
                pass
            
            # Clear PyTorch caches if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            logger.info("Caches cleared")
        except Exception as e:
            logger.error(f"Cache clearing error: {e}")
    
    def _reduce_memory_usage(self):
        """Attempt to reduce memory usage across the system."""
        # This would trigger system-wide memory reduction
        # Implementation depends on specific components
        logger.info("Triggered system-wide memory reduction")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for memory alerts."""
        self.alert_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        mem = psutil.virtual_memory()
        return {
            'available_mb': mem.available / (1024 * 1024),
            'used_mb': mem.used / (1024 * 1024),
            'percent_used': mem.percent,
            'stats': self.stats,
            'is_monitoring': self.is_running
        }


class ComponentMemoryLimiter:
    """
    Enforce memory limits for individual components.
    """
    
    def __init__(self):
        """Initialize component memory limiter."""
        self.limits = MEMORY_ALLOCATION.copy()
        self.usage = {component: 0 for component in self.limits}
        self.components = {}
        
        logger.info(f"Component memory limiter initialized with budget: {self.limits}")
    
    def register_component(self, name: str, component: Any, limit_mb: Optional[int] = None):
        """
        Register a component for memory tracking.
        
        Args:
            name: Component name
            component: Component object
            limit_mb: Memory limit in MB (uses config default if None)
        """
        if limit_mb is not None:
            self.limits[name] = limit_mb
        
        self.components[name] = component
        logger.info(f"Registered component '{name}' with limit {self.limits.get(name, 'unlimited')}MB")
    
    def check_component_memory(self, name: str) -> Dict[str, Any]:
        """Check memory usage of a specific component."""
        if name not in self.components:
            return {'error': f'Component {name} not registered'}
        
        # Get component memory usage (implementation depends on component type)
        usage_mb = self._estimate_component_memory(self.components[name])
        self.usage[name] = usage_mb
        
        limit_mb = self.limits.get(name, float('inf'))
        
        return {
            'component': name,
            'usage_mb': usage_mb,
            'limit_mb': limit_mb,
            'percent_used': (usage_mb / limit_mb * 100) if limit_mb != float('inf') else 0,
            'within_limit': usage_mb <= limit_mb
        }
    
    def _estimate_component_memory(self, component: Any) -> float:
        """Estimate memory usage of a component."""
        # This is a simplified estimation
        # In production, use more sophisticated tracking
        
        import sys
        
        def get_size(obj, seen=None):
            """Recursively find size of objects."""
            size = sys.getsizeof(obj)
            if seen is None:
                seen = set()
            
            obj_id = id(obj)
            if obj_id in seen:
                return 0
            
            seen.add(obj_id)
            
            if isinstance(obj, dict):
                size += sum([get_size(v, seen) for v in obj.values()])
                size += sum([get_size(k, seen) for k in obj.keys()])
            elif hasattr(obj, '__dict__'):
                size += get_size(obj.__dict__, seen)
            elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
                size += sum([get_size(i, seen) for i in obj])
            
            return size
        
        try:
            size_bytes = get_size(component)
            return size_bytes / (1024 * 1024)
        except:
            return 0
    
    def enforce_limits(self) -> List[str]:
        """
        Enforce memory limits, returning list of components over limit.
        """
        over_limit = []
        
        for name, component in self.components.items():
            status = self.check_component_memory(name)
            
            if not status.get('within_limit', True):
                over_limit.append(name)
                logger.warning(f"Component '{name}' exceeds memory limit: "
                             f"{status['usage_mb']:.1f}MB > {status['limit_mb']}MB")
                
                # Trigger component-specific memory reduction
                if hasattr(component, 'reduce_memory'):
                    component.reduce_memory()
        
        return over_limit


class MemoryOptimizer:
    """
    Automatic memory optimization strategies.
    """
    
    def __init__(self):
        """Initialize memory optimizer."""
        self.optimization_strategies = {
            'data_pruning': self._prune_old_data,
            'model_quantization': self._quantize_models,
            'cache_clearing': self._clear_caches,
            'batch_size_reduction': self._reduce_batch_sizes,
            'feature_reduction': self._reduce_features
        }
        
        self.optimization_history = []
        logger.info("Memory optimizer initialized")
    
    def optimize(self, target_free_mb: int = 2000) -> bool:
        """
        Run optimization to free up memory.
        
        Args:
            target_free_mb: Target free memory in MB
            
        Returns:
            True if target achieved
        """
        initial_available = get_available_memory_mb()
        logger.info(f"Starting optimization: {initial_available}MB available, target: {target_free_mb}MB")
        
        for strategy_name, strategy_func in self.optimization_strategies.items():
            if get_available_memory_mb() >= target_free_mb:
                break
            
            try:
                freed_mb = strategy_func()
                self.optimization_history.append({
                    'timestamp': datetime.now(),
                    'strategy': strategy_name,
                    'freed_mb': freed_mb
                })
                logger.info(f"Strategy '{strategy_name}' freed {freed_mb:.1f}MB")
            except Exception as e:
                logger.error(f"Optimization strategy '{strategy_name}' failed: {e}")
        
        final_available = get_available_memory_mb()
        success = final_available >= target_free_mb
        
        logger.info(f"Optimization complete: {final_available}MB available "
                   f"(freed {final_available - initial_available:.1f}MB)")
        
        return success
    
    def _prune_old_data(self) -> float:
        """Prune old data from memory."""
        initial = get_available_memory_mb()
        
        # Implementation would prune old market data, features, etc.
        # This is a placeholder
        gc.collect()
        
        return get_available_memory_mb() - initial
    
    def _quantize_models(self) -> float:
        """Quantize models to reduce memory usage."""
        initial = get_available_memory_mb()
        
        # Implementation would quantize PyTorch/TensorFlow models
        # This is a placeholder
        
        return get_available_memory_mb() - initial
    
    def _clear_caches(self) -> float:
        """Clear various caches."""
        initial = get_available_memory_mb()
        
        gc.collect(2)
        
        # Clear framework-specific caches
        try:
            import jax
            jax.clear_caches()
        except:
            pass
        
        return get_available_memory_mb() - initial
    
    def _reduce_batch_sizes(self) -> float:
        """Reduce batch sizes for training/inference."""
        # This would signal components to use smaller batch sizes
        # Implementation depends on specific components
        return 0
    
    def _reduce_features(self) -> float:
        """Reduce number of features to save memory."""
        # This would reduce feature dimensionality
        # Implementation depends on feature engineering pipeline
        return 0


def memory_limit(limit_mb: int):
    """
    Decorator to enforce memory limit on functions.
    
    Args:
        limit_mb: Maximum memory usage in MB
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check available memory before execution
            available = get_available_memory_mb()
            if available < limit_mb:
                raise MemoryError(f"Insufficient memory: {available}MB < {limit_mb}MB required")
            
            # Track memory usage during execution
            tracemalloc.start()
            
            try:
                result = func(*args, **kwargs)
                
                # Check peak memory usage
                current, peak = tracemalloc.get_traced_memory()
                peak_mb = peak / (1024 * 1024)
                
                if peak_mb > limit_mb:
                    logger.warning(f"Function {func.__name__} exceeded memory limit: "
                                 f"{peak_mb:.1f}MB > {limit_mb}MB")
                
                return result
            finally:
                tracemalloc.stop()
        
        return wrapper
    return decorator


class MemoryCircuitBreaker:
    """
    Circuit breaker to prevent OOM crashes.
    """
    
    def __init__(self, threshold_mb: int = 500):
        """
        Initialize circuit breaker.
        
        Args:
            threshold_mb: Memory threshold to trip breaker
        """
        self.threshold_mb = threshold_mb
        self.is_tripped = False
        self.trip_count = 0
        self.last_trip_time = None
        
        # Register signal handler for emergency shutdown
        signal.signal(signal.SIGUSR1, self._emergency_shutdown)
        
        logger.info(f"Memory circuit breaker initialized with {threshold_mb}MB threshold")
    
    def check(self) -> bool:
        """
        Check if circuit breaker should trip.
        
        Returns:
            True if breaker is tripped
        """
        available = get_available_memory_mb()
        
        if available < self.threshold_mb:
            self.trip()
            return True
        
        # Reset if memory recovered
        if self.is_tripped and available > self.threshold_mb * 2:
            self.reset()
        
        return self.is_tripped
    
    def trip(self):
        """Trip the circuit breaker."""
        if not self.is_tripped:
            self.is_tripped = True
            self.trip_count += 1
            self.last_trip_time = datetime.now()
            
            logger.critical(f"Memory circuit breaker TRIPPED! "
                          f"Available: {get_available_memory_mb()}MB < {self.threshold_mb}MB")
            
            # Take emergency actions
            self._emergency_actions()
    
    def reset(self):
        """Reset the circuit breaker."""
        if self.is_tripped:
            self.is_tripped = False
            logger.info("Memory circuit breaker reset")
    
    def _emergency_actions(self):
        """Take emergency actions when breaker trips."""
        # Stop non-critical processes
        # Dump memory profile
        # Clear all caches
        # Reduce all batch sizes to minimum
        
        gc.collect(2)
        
        # Save critical state before potential crash
        self._save_emergency_state()
    
    def _save_emergency_state(self):
        """Save critical system state."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'memory_available_mb': get_available_memory_mb(),
                'trip_count': self.trip_count,
                'pid': os.getpid()
            }
            
            with open('/tmp/emergency_state.json', 'w') as f:
                json.dump(state, f)
            
            logger.info("Emergency state saved")
        except Exception as e:
            logger.error(f"Failed to save emergency state: {e}")
    
    def _emergency_shutdown(self, signum, frame):
        """Emergency shutdown handler."""
        logger.critical("Emergency shutdown initiated!")
        self._save_emergency_state()
        sys.exit(1)


class MemoryManagementSystem:
    """
    Comprehensive memory management system integrating all components.
    """
    
    def __init__(self):
        """Initialize complete memory management system."""
        self.monitor = MemoryMonitor(check_interval=30)
        self.limiter = ComponentMemoryLimiter()
        self.optimizer = MemoryOptimizer()
        self.circuit_breaker = MemoryCircuitBreaker(threshold_mb=500)
        
        # Register callbacks
        self.monitor.register_alert_callback(self._handle_memory_alert)
        
        logger.info("Memory management system initialized")
    
    def start(self):
        """Start memory management system."""
        self.monitor.start()
        logger.info("Memory management system started")
    
    def stop(self):
        """Stop memory management system."""
        self.monitor.stop()
        logger.info("Memory management system stopped")
    
    def _handle_memory_alert(self, level: str, available_mb: float):
        """Handle memory alerts from monitor."""
        if level == 'emergency':
            # Check circuit breaker
            if self.circuit_breaker.check():
                logger.critical("Circuit breaker tripped - initiating emergency procedures")
            
            # Run aggressive optimization
            self.optimizer.optimize(target_free_mb=2000)
            
        elif level == 'critical':
            # Run optimization
            self.optimizer.optimize(target_free_mb=1500)
            
        elif level == 'warning':
            # Light optimization
            gc.collect()
    
    def get_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        return {
            'monitor': self.monitor.get_status(),
            'circuit_breaker': {
                'is_tripped': self.circuit_breaker.is_tripped,
                'trip_count': self.circuit_breaker.trip_count,
                'last_trip': self.circuit_breaker.last_trip_time
            },
            'optimizer': {
                'optimizations_run': len(self.optimizer.optimization_history)
            },
            'system': {
                'total_memory_mb': psutil.virtual_memory().total / (1024 * 1024),
                'available_memory_mb': get_available_memory_mb(),
                'reduced_mode': should_use_reduced_mode()
            }
        }
    
    def register_component(self, name: str, component: Any, limit_mb: Optional[int] = None):
        """Register a component for memory management."""
        self.limiter.register_component(name, component, limit_mb)


# Global instance
_memory_manager = None

def get_memory_manager() -> MemoryManagementSystem:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManagementSystem()
    return _memory_manager


def test_memory_management():
    """Test memory management system."""
    print("Testing Memory Management System")
    print("=" * 50)
    
    # Initialize system
    manager = get_memory_manager()
    
    # Start monitoring
    print("\n1. Starting memory monitoring...")
    manager.start()
    print("   ✓ Monitoring started")
    
    # Check status
    print("\n2. System status:")
    status = manager.get_status()
    print(f"   Available memory: {status['system']['available_memory_mb']:.0f} MB")
    print(f"   Reduced mode: {status['system']['reduced_mode']}")
    print(f"   Circuit breaker: {'TRIPPED' if status['circuit_breaker']['is_tripped'] else 'OK'}")
    
    # Test memory limit decorator
    print("\n3. Testing memory limit decorator...")
    
    @memory_limit(100)
    def memory_intensive_function():
        # Allocate some memory
        data = [0] * (10 * 1024 * 1024)  # ~40MB
        return len(data)
    
    try:
        result = memory_intensive_function()
        print(f"   Function executed successfully")
    except MemoryError as e:
        print(f"   Memory limit enforced: {e}")
    
    # Test optimization
    print("\n4. Testing memory optimization...")
    initial = get_available_memory_mb()
    success = manager.optimizer.optimize(target_free_mb=initial + 100)
    print(f"   Optimization {'succeeded' if success else 'failed'}")
    
    # Stop monitoring
    print("\n5. Stopping monitoring...")
    manager.stop()
    print("   ✓ Monitoring stopped")
    
    print("\n✓ Memory management system test completed!")


if __name__ == "__main__":
    test_memory_management()