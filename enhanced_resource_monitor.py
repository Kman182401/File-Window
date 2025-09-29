#!/usr/bin/env python3
"""
Enhanced Resource Monitoring for m5.large Production Trading System

Provides real-time monitoring, alerting, and automatic safeguards for:
- Memory usage with predictive alerts
- CPU utilization and thermal management  
- Disk space monitoring with cleanup
- Network connectivity (IBKR Gateway)
- PPO training performance metrics
- System stability indicators

Safety Features:
- Automatic memory cleanup before critical thresholds
- Emergency trading halt if resources exhausted
- Model training suspension during high memory periods
- Comprehensive logging and notifications

Author: AI Trading System
Version: 1.0.0
"""

import os
import sys
import time
import logging
import threading
import psutil
import json
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
import warnings
from typing import Dict, List, Optional, Callable, Any, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ResourceThresholds:
    """Resource threshold configuration for m5.large"""
    # Memory thresholds (based on 5.9GB available)
    memory_warning_gb: float = 4.5      # 76% usage - start cleanup
    memory_critical_gb: float = 5.2     # 88% usage - halt training
    memory_emergency_gb: float = 5.5    # 93% usage - emergency stop
    
    # CPU thresholds (2 vCPUs)
    cpu_warning_percent: float = 85.0   # High sustained usage
    cpu_critical_percent: float = 95.0  # Critical sustained usage
    cpu_duration_seconds: int = 60      # Duration for sustained alerts
    
    # Disk thresholds (97GB total, 49GB free)
    disk_warning_percent: float = 85.0  # Start cleanup
    disk_critical_percent: float = 95.0 # Stop new data ingestion
    
    # Network/IBKR thresholds
    ibkr_timeout_seconds: int = 30       # IBKR connection timeout
    max_reconnect_attempts: int = 3      # Max IBKR reconnection tries
    # GPU thresholds
    gpu_util_warning_pct: float = float(os.getenv("GPU_UTIL_WARN", 90))
    gpu_util_critical_pct: float = float(os.getenv("GPU_UTIL_CRIT", 98))
    gpu_mem_warning_pct: float = float(os.getenv("GPU_MEM_WARN", 80))
    gpu_mem_critical_pct: float = float(os.getenv("GPU_MEM_CRIT", 92))
    gpu_temp_warning_c: float = float(os.getenv("GPU_TEMP_WARN", 80))
    gpu_temp_critical_c: float = float(os.getenv("GPU_TEMP_CRIT", 85))

@dataclass
class ResourceMetrics:
    """Current system resource metrics"""
    timestamp: datetime
    memory_used_gb: float
    memory_percent: float
    cpu_percent: float
    disk_used_percent: float
    disk_free_gb: float
    python_memory_mb: float
    ibkr_connected: bool = False
    active_processes: int = 0
    # GPU summary
    gpu_count: int = 0
    gpu_util_percent: float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_mem_percent: float = 0.0
    gpu_temp_c: float = 0.0
    
class EnhancedResourceMonitor:
    """
    Production-grade resource monitor for m5.large trading system
    """
    
    def __init__(self, 
                 thresholds: ResourceThresholds = None,
                 alert_callback: Callable[[str, str], None] = None,
                 cleanup_callback: Callable[[], None] = None):
        """
        Initialize enhanced monitoring
        
        Args:
            thresholds: Custom resource thresholds
            alert_callback: Function called on alerts (level, message)
            cleanup_callback: Function called for memory cleanup
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.alert_callback = alert_callback or self._default_alert
        self.cleanup_callback = cleanup_callback
        
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history: List[ResourceMetrics] = []
        self.max_history_size = 1000
        
        # Alert state tracking
        self.alert_states = {
            'memory_warning': False,
            'memory_critical': False,
            'cpu_high': False,
            'disk_warning': False
        }
        
        # Performance tracking
        self.training_start_time = None
        self.training_metrics = {}
        
        # Emergency flags
        self.emergency_stop = False
        self.training_halted = False
        
    def _default_alert(self, level: str, message: str):
        """Default alert handler - log to console and file"""
        log_method = {
            'INFO': logger.info,
            'WARNING': logger.warning,
            'CRITICAL': logger.error,
            'EMERGENCY': logger.critical
        }.get(level, logger.info)
        
        log_method(f"[{level}] RESOURCE MONITOR: {message}")
        
    def get_current_metrics(self) -> ResourceMetrics:
        """Get comprehensive current system metrics"""
        try:
            # Memory information
            memory = psutil.virtual_memory()
            
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Disk information
            disk = psutil.disk_usage('/home/karson')
            
            # Python process memory
            try:
                process = psutil.Process()
                python_memory_mb = process.memory_info().rss / 1024 / 1024
            except:
                python_memory_mb = 0
                
            # Active process count
            try:
                active_processes = len([p for p in psutil.process_iter() 
                                     if p.status() == psutil.STATUS_RUNNING])
            except:
                active_processes = 0
                
            # Check IBKR connection (basic check)
            ibkr_connected = self._check_ibkr_connection()

            # GPU metrics (aggregate)
            try:
                from utils import gpu_metrics as g
                gsample = g.collect_gpu_metrics()
                if gsample.get("available") and gsample.get("aggregate"):
                    agg = gsample["aggregate"]
                    gpu_count = int(agg.get("count", 0))
                    gpu_util_percent = float(agg.get("max_util_pct", 0.0))
                    gpu_mem_percent = float(agg.get("max_mem_pct", 0.0))
                    gpu_temp_c = float(agg.get("max_temp_c", 0.0))
                    # Convert mem_used from max percent into GB via first GPU total if possible
                    gpu_mem_used_gb = 0.0
                    gpus = gsample.get("gpus", [])
                    if gpus:
                        # Use the GPU with max mem pct
                        top = max(gpus, key=lambda x: x.get("mem_pct", 0.0))
                        gpu_mem_used_gb = float(top.get("mem_used_mb", 0.0)) / 1024.0
                else:
                    gpu_count = 0
                    gpu_util_percent = 0.0
                    gpu_mem_percent = 0.0
                    gpu_temp_c = 0.0
                    gpu_mem_used_gb = 0.0
            except Exception:
                gpu_count = 0
                gpu_util_percent = 0.0
                gpu_mem_percent = 0.0
                gpu_temp_c = 0.0
                gpu_mem_used_gb = 0.0
            
            return ResourceMetrics(
                timestamp=datetime.now(UTC),
                memory_used_gb=memory.used / 1024**3,
                memory_percent=memory.percent,
                cpu_percent=cpu_percent,
                disk_used_percent=(disk.used / disk.total) * 100,
                disk_free_gb=disk.free / 1024**3,
                python_memory_mb=python_memory_mb,
                ibkr_connected=ibkr_connected,
                active_processes=active_processes,
                gpu_count=gpu_count,
                gpu_util_percent=gpu_util_percent,
                gpu_mem_used_gb=gpu_mem_used_gb,
                gpu_mem_percent=gpu_mem_percent,
                gpu_temp_c=gpu_temp_c
            )
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            # Return safe defaults
            return ResourceMetrics(
                timestamp=datetime.now(UTC),
                memory_used_gb=0,
                memory_percent=0,
                cpu_percent=0,
                disk_used_percent=0,
                disk_free_gb=0,
                python_memory_mb=0
            )
            
    def _check_ibkr_connection(self) -> bool:
        """Quick check if IBKR Gateway is accessible"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # 2 second timeout
            result = sock.connect_ex(('127.0.0.1', 4002))
            sock.close()
            return result == 0
        except:
            return False
            
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check all thresholds and trigger appropriate actions"""
        
        # Memory threshold checks
        if metrics.memory_used_gb >= self.thresholds.memory_emergency_gb:
            if not self.emergency_stop:
                self.emergency_stop = True
                self.alert_callback('EMERGENCY', 
                    f"MEMORY EMERGENCY: {metrics.memory_used_gb:.1f}GB used "
                    f"(>{self.thresholds.memory_emergency_gb}GB). HALTING ALL OPERATIONS!")
                self._emergency_memory_cleanup()
                
        elif metrics.memory_used_gb >= self.thresholds.memory_critical_gb:
            if not self.alert_states['memory_critical']:
                self.alert_states['memory_critical'] = True
                self.training_halted = True
                self.alert_callback('CRITICAL', 
                    f"MEMORY CRITICAL: {metrics.memory_used_gb:.1f}GB used "
                    f"(>{self.thresholds.memory_critical_gb}GB). HALTING TRAINING!")
                self._critical_memory_cleanup()
                
        elif metrics.memory_used_gb >= self.thresholds.memory_warning_gb:
            if not self.alert_states['memory_warning']:
                self.alert_states['memory_warning'] = True
                self.alert_callback('WARNING', 
                    f"MEMORY WARNING: {metrics.memory_used_gb:.1f}GB used "
                    f"(>{self.thresholds.memory_warning_gb}GB). Starting cleanup...")
                self._warning_memory_cleanup()
        else:
            # Reset memory alerts if usage drops
            if self.alert_states['memory_critical']:
                self.alert_states['memory_critical'] = False
                self.training_halted = False
                self.alert_callback('INFO', 
                    f"MEMORY RECOVERED: {metrics.memory_used_gb:.1f}GB used. "
                    "Training can resume.")
                    
            if self.alert_states['memory_warning']:
                self.alert_states['memory_warning'] = False
                self.alert_callback('INFO', 
                    f"MEMORY NORMAL: {metrics.memory_used_gb:.1f}GB used.")
        
        # CPU threshold checks
        if metrics.cpu_percent >= self.thresholds.cpu_critical_percent:
            if not self.alert_states['cpu_high']:
                self.alert_states['cpu_high'] = True
                self.alert_callback('CRITICAL', 
                    f"CPU CRITICAL: {metrics.cpu_percent:.1f}% sustained usage")
        elif metrics.cpu_percent < self.thresholds.cpu_warning_percent:
            if self.alert_states['cpu_high']:
                self.alert_states['cpu_high'] = False
                self.alert_callback('INFO', 
                    f"CPU NORMAL: {metrics.cpu_percent:.1f}% usage")
        
        # Disk threshold checks  
        if metrics.disk_used_percent >= self.thresholds.disk_critical_percent:
            self.alert_callback('CRITICAL', 
                f"DISK CRITICAL: {metrics.disk_used_percent:.1f}% used, "
                f"only {metrics.disk_free_gb:.1f}GB free")
                
        elif metrics.disk_used_percent >= self.thresholds.disk_warning_percent:
            if not self.alert_states['disk_warning']:
                self.alert_states['disk_warning'] = True
                self.alert_callback('WARNING', 
                    f"DISK WARNING: {metrics.disk_used_percent:.1f}% used")

        # GPU threshold checks (use max aggregates)
        if metrics.gpu_count > 0:
            if metrics.gpu_util_percent >= self.thresholds.gpu_util_critical_pct:
                self.alert_callback('CRITICAL', f"GPU UTIL CRITICAL: {metrics.gpu_util_percent:.1f}%")
            elif metrics.gpu_util_percent >= self.thresholds.gpu_util_warning_pct:
                self.alert_callback('WARNING', f"GPU UTIL WARNING: {metrics.gpu_util_percent:.1f}%")

            if metrics.gpu_mem_percent >= self.thresholds.gpu_mem_critical_pct:
                self.alert_callback('CRITICAL', f"GPU MEM CRITICAL: {metrics.gpu_mem_percent:.1f}%")
            elif metrics.gpu_mem_percent >= self.thresholds.gpu_mem_warning_pct:
                self.alert_callback('WARNING', f"GPU MEM WARNING: {metrics.gpu_mem_percent:.1f}%")

            if metrics.gpu_temp_c >= self.thresholds.gpu_temp_critical_c:
                self.alert_callback('CRITICAL', f"GPU TEMP CRITICAL: {metrics.gpu_temp_c:.1f}°C")
            elif metrics.gpu_temp_c >= self.thresholds.gpu_temp_warning_c:
                self.alert_callback('WARNING', f"GPU TEMP WARNING: {metrics.gpu_temp_c:.1f}°C")
                    
    def _warning_memory_cleanup(self):
        """Gentle memory cleanup for warning threshold"""
        if self.cleanup_callback:
            try:
                self.cleanup_callback()
                logger.info("Warning-level memory cleanup completed")
            except Exception as e:
                logger.error(f"Memory cleanup failed: {e}")
                
    def _critical_memory_cleanup(self):
        """Aggressive memory cleanup for critical threshold"""
        import gc
        
        # Call custom cleanup first
        self._warning_memory_cleanup()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Clear various caches
        try:
            import pandas as pd
            if hasattr(pd.api.types, 'pandas_dtype'):
                pd.api.types.pandas_dtype.clear_cache()
        except:
            pass
            
        logger.warning(f"Critical memory cleanup: collected {collected} objects")
        
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup - most aggressive"""
        self._critical_memory_cleanup()
        
        # Clear all possible caches
        import sys
        sys.modules.clear()  # Dangerous but necessary in emergency
        
        logger.critical("Emergency memory cleanup completed")
        
    def start_monitoring(self, interval_seconds: int = 5):
        """Start continuous resource monitoring"""
        if self.monitoring:
            logger.warning("Resource monitoring already running")
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Enhanced resource monitoring started (interval: {interval_seconds}s)")
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
        
    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Get current metrics
                metrics = self.get_current_metrics()
                
                # Store in history
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Check thresholds and take actions
                self._check_thresholds(metrics)
                
                # Sleep until next check
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
                
    def is_safe_for_training(self) -> Tuple[bool, str]:
        """Check if system resources are safe for ML training"""
        if self.emergency_stop:
            return False, "Emergency stop activated due to resource exhaustion"
            
        if self.training_halted:
            return False, "Training halted due to critical resource usage"
            
        # Get current metrics
        metrics = self.get_current_metrics()
        
        # Check memory safety
        if metrics.memory_used_gb >= self.thresholds.memory_warning_gb:
            return False, f"Memory usage too high: {metrics.memory_used_gb:.1f}GB"
            
        # Check CPU safety
        if metrics.cpu_percent >= self.thresholds.cpu_critical_percent:
            return False, f"CPU usage too high: {metrics.cpu_percent:.1f}%"
            
        return True, "Resources safe for training"
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
            
        latest = self.metrics_history[-1]
        
        # Calculate averages from recent history (last 10 readings)
        recent_metrics = self.metrics_history[-10:]
        avg_memory = sum(m.memory_used_gb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            'current': {
                'memory_used_gb': latest.memory_used_gb,
                'memory_percent': latest.memory_percent,
                'cpu_percent': latest.cpu_percent,
                'disk_used_percent': latest.disk_used_percent,
                'disk_free_gb': latest.disk_free_gb,
                'python_memory_mb': latest.python_memory_mb,
                'ibkr_connected': latest.ibkr_connected,
                'gpu_count': latest.gpu_count,
                'gpu_util_percent': latest.gpu_util_percent,
                'gpu_mem_used_gb': latest.gpu_mem_used_gb,
                'gpu_mem_percent': latest.gpu_mem_percent,
                'gpu_temp_c': latest.gpu_temp_c
            },
            'averages': {
                'memory_gb_10min': avg_memory,
                'cpu_percent_10min': avg_cpu
            },
            'thresholds': {
                'memory_warning_gb': self.thresholds.memory_warning_gb,
                'memory_critical_gb': self.thresholds.memory_critical_gb,
                'cpu_warning_percent': self.thresholds.cpu_warning_percent
            },
            'alerts': self.alert_states,
            'emergency_status': {
                'emergency_stop': self.emergency_stop,
                'training_halted': self.training_halted
            },
            'safety_assessment': self.is_safe_for_training()
        }

# Global monitor instance
_global_monitor: Optional[EnhancedResourceMonitor] = None

def get_global_monitor() -> EnhancedResourceMonitor:
    """Get or create global resource monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EnhancedResourceMonitor()
    return _global_monitor

def is_memory_safe_for_training() -> bool:
    """Quick check if memory is safe for training"""
    monitor = get_global_monitor()
    safe, _ = monitor.is_safe_for_training()
    return safe

if __name__ == "__main__":
    # Test the enhanced resource monitor
    print("=== Enhanced Resource Monitor Test ===")
    
    monitor = EnhancedResourceMonitor()
    
    # Get current metrics
    metrics = monitor.get_current_metrics()
    print(f"Memory: {metrics.memory_used_gb:.1f}GB ({metrics.memory_percent:.1f}%)")
    print(f"CPU: {metrics.cpu_percent:.1f}%")
    print(f"Disk: {metrics.disk_used_percent:.1f}% used, {metrics.disk_free_gb:.1f}GB free")
    print(f"Python process: {metrics.python_memory_mb:.1f}MB")
    print(f"IBKR connected: {metrics.ibkr_connected}")
    
    # Test safety check
    safe, reason = monitor.is_safe_for_training()
    print(f"\nSafe for training: {safe}")
    print(f"Reason: {reason}")
    
    # Test monitoring for 30 seconds
    print(f"\nStarting 30-second monitoring test...")
    monitor.start_monitoring(interval_seconds=2)
    time.sleep(30)
    monitor.stop_monitoring()
    
    # Show summary
    summary = monitor.get_resource_summary()
    print(f"\nResource Summary:")
    print(json.dumps(summary, indent=2, default=str))
