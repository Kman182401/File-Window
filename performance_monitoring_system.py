"""
PERFORMANCE MONITORING AND ALERTING SYSTEM - Phase 3
====================================================

Comprehensive monitoring system tracking pipeline performance metrics
Real-time alerting, dashboard metrics, and performance optimization recommendations
"""

import time
import os
import threading
import logging
import json
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from contextlib import contextmanager
from enum import Enum
import queue
import statistics
import asyncio
from functools import wraps
from utils import gpu_metrics as gpu

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE METRICS DEFINITIONS
# ============================================================================

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class PerformanceMetric:
    """Individual performance metric with thresholds"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class PerformanceThreshold:
    """Performance threshold configuration"""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: Optional[float] = None
    direction: str = "above"  # "above" or "below"
    window_seconds: int = 60
    min_samples: int = 3

@dataclass
class Alert:
    """Performance alert"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False

# ============================================================================
# REAL-TIME METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """High-performance metrics collection system"""
    
    def __init__(self, max_history: int = 10000):
        self.metrics_history = defaultdict(lambda: deque(maxlen=max_history))
        self.current_metrics = {}
        self.lock = threading.RLock()
        self.collection_stats = {
            'total_metrics': 0,
            'collection_errors': 0,
            'last_collection': None
        }
        
        # System monitoring
        self.process = psutil.Process()
        self.system_start_time = time.time()
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        try:
            with self.lock:
                self.metrics_history[metric.name].append(metric)
                self.current_metrics[metric.name] = metric
                self.collection_stats['total_metrics'] += 1
                self.collection_stats['last_collection'] = time.time()
                
        except Exception as e:
            self.collection_stats['collection_errors'] += 1
            logger.error(f"Failed to record metric {metric.name}: {e}")
    
    def get_metric_history(self, 
                          metric_name: str, 
                          window_seconds: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metric history within time window"""
        with self.lock:
            history = list(self.metrics_history[metric_name])
            
            if window_seconds:
                cutoff_time = time.time() - window_seconds
                history = [m for m in history if m.timestamp >= cutoff_time]
            
            return history
    
    def get_current_metric(self, metric_name: str) -> Optional[PerformanceMetric]:
        """Get current value of a metric"""
        with self.lock:
            return self.current_metrics.get(metric_name)
    
    def get_metric_stats(self, 
                        metric_name: str, 
                        window_seconds: int = 300) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        history = self.get_metric_history(metric_name, window_seconds)
        
        if not history:
            return {}
        
        values = [m.value for m in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'current': values[-1] if values else 0.0
        }
    
    def collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            # Memory metrics
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            self.record_metric(PerformanceMetric(
                name="memory_usage_mb",
                value=memory_info.rss / 1024 / 1024,
                unit="MB",
                tags={"component": "system"}
            ))
            
            self.record_metric(PerformanceMetric(
                name="memory_percent",
                value=memory_percent,
                unit="%",
                tags={"component": "system"}
            ))
            
            # CPU metrics
            cpu_percent = self.process.cpu_percent()
            self.record_metric(PerformanceMetric(
                name="cpu_usage_percent",
                value=cpu_percent,
                unit="%",
                tags={"component": "system"}
            ))
            
            # System-wide memory
            system_memory = psutil.virtual_memory()
            self.record_metric(PerformanceMetric(
                name="system_memory_available_mb",
                value=system_memory.available / 1024 / 1024,
                unit="MB",
                tags={"component": "system"}
            ))
            
            # Thread count
            thread_count = threading.active_count()
            self.record_metric(PerformanceMetric(
                name="active_threads",
                value=thread_count,
                unit="count",
                tags={"component": "system"}
            ))
            
            # Uptime
            uptime_seconds = time.time() - self.system_start_time
            self.record_metric(PerformanceMetric(
                name="uptime_seconds",
                value=uptime_seconds,
                unit="seconds",
                tags={"component": "system"}
            ))
            
            # GPU metrics (aggregates + per GPU flat keys)
            try:
                gsample = gpu.collect_gpu_metrics()
                if gsample.get("available"):
                    agg = gsample.get("aggregate", {})
                    self.record_metric(PerformanceMetric(
                        name="gpu.max.util_pct", value=float(agg.get("max_util_pct", 0.0)), unit="%"
                    ))
                    self.record_metric(PerformanceMetric(
                        name="gpu.max.mem_pct", value=float(agg.get("max_mem_pct", 0.0)), unit="%"
                    ))
                    self.record_metric(PerformanceMetric(
                        name="gpu.max.temp_c", value=float(agg.get("max_temp_c", 0.0)), unit="C"
                    ))
                    self.record_metric(PerformanceMetric(
                        name="gpu.count", value=float(agg.get("count", 0)), unit="count"
                    ))
                    for g in gsample.get("gpus", [])[:8]:
                        idx = int(g.get("index", 0))
                        self.record_metric(PerformanceMetric(
                            name=f"gpu.{idx}.util_pct", value=float(g.get("util_pct", 0.0)), unit="%"
                        ))
                        self.record_metric(PerformanceMetric(
                            name=f"gpu.{idx}.mem_used_mb", value=float(g.get("mem_used_mb", 0.0)), unit="MB"
                        ))
                        self.record_metric(PerformanceMetric(
                            name=f"gpu.{idx}.mem_pct", value=float(g.get("mem_pct", 0.0)), unit="%"
                        ))
                        self.record_metric(PerformanceMetric(
                            name=f"gpu.{idx}.temp_c", value=float(g.get("temp_c", 0.0)), unit="C"
                        ))
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            self.collection_stats['collection_errors'] += 1

# ============================================================================
# PERFORMANCE ALERTING SYSTEM
# ============================================================================

class AlertManager:
    """Intelligent alerting system with threshold management"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.thresholds = {}
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks = []
        self.lock = threading.RLock()
        
        # Default thresholds for trading system
        self._setup_default_thresholds()
        
        # Alert suppression to prevent spam
        self.alert_cooldown = {}
        self.default_cooldown_seconds = 60
        
    def _setup_default_thresholds(self):
        """Setup default performance thresholds for trading system"""
        thresholds = [
            # Latency thresholds
            PerformanceThreshold(
                metric_name="pipeline_latency_ms",
                warning_threshold=800,
                critical_threshold=1000,
                emergency_threshold=2000,
                direction="above"
            ),
            
            PerformanceThreshold(
                metric_name="ticker_processing_latency_ms",
                warning_threshold=400,
                critical_threshold=600,
                emergency_threshold=1000,
                direction="above"
            ),
            
            # Memory thresholds
            PerformanceThreshold(
                metric_name="memory_usage_mb",
                warning_threshold=4000,
                critical_threshold=6000,
                emergency_threshold=7500,
                direction="above"
            ),
            
            PerformanceThreshold(
                metric_name="memory_percent",
                warning_threshold=75,
                critical_threshold=90,
                emergency_threshold=95,
                direction="above"
            ),
            
            # System availability
            PerformanceThreshold(
                metric_name="system_memory_available_mb",
                warning_threshold=2000,
                critical_threshold=1000,
                emergency_threshold=500,
                direction="below"
            ),
            
            # Error rates
            PerformanceThreshold(
                metric_name="error_rate_percent",
                warning_threshold=5,
                critical_threshold=10,
                emergency_threshold=25,
                direction="above"
            ),
            
            # IBKR connection health
            PerformanceThreshold(
                metric_name="ibkr_connection_failures",
                warning_threshold=1,
                critical_threshold=3,
                emergency_threshold=5,
                direction="above"
            ),
            
            # S3 performance
            PerformanceThreshold(
                metric_name="s3_operation_latency_ms",
                warning_threshold=5000,
                critical_threshold=10000,
                emergency_threshold=30000,
                direction="above"
            )
        ]
        
        for threshold in thresholds:
            self.add_threshold(threshold)
        
        # GPU aggregate thresholds (env-driven)
        self.add_threshold(PerformanceThreshold(
            metric_name="gpu.max.util_pct",
            warning_threshold=float(os.getenv("GPU_UTIL_WARN", 90)),
            critical_threshold=float(os.getenv("GPU_UTIL_CRIT", 98)),
            emergency_threshold=None,
            direction="above",
        ))
        self.add_threshold(PerformanceThreshold(
            metric_name="gpu.max.mem_pct",
            warning_threshold=float(os.getenv("GPU_MEM_WARN", 80)),
            critical_threshold=float(os.getenv("GPU_MEM_CRIT", 92)),
            emergency_threshold=None,
            direction="above",
        ))
        self.add_threshold(PerformanceThreshold(
            metric_name="gpu.max.temp_c",
            warning_threshold=float(os.getenv("GPU_TEMP_WARN", 80)),
            critical_threshold=float(os.getenv("GPU_TEMP_CRIT", 85)),
            emergency_threshold=None,
            direction="above",
        ))
    
    def add_threshold(self, threshold: PerformanceThreshold):
        """Add a performance threshold"""
        with self.lock:
            self.thresholds[threshold.metric_name] = threshold
            logger.info(f"Added threshold for {threshold.metric_name}: "
                       f"warn={threshold.warning_threshold}, "
                       f"crit={threshold.critical_threshold}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
    
    def check_thresholds(self):
        """Check all thresholds and generate alerts"""
        current_time = time.time()
        
        with self.lock:
            for metric_name, threshold in self.thresholds.items():
                try:
                    self._check_single_threshold(metric_name, threshold, current_time)
                except Exception as e:
                    logger.error(f"Error checking threshold for {metric_name}: {e}")
    
    def _check_single_threshold(self, 
                               metric_name: str, 
                               threshold: PerformanceThreshold, 
                               current_time: float):
        """Check a single threshold and generate alerts if needed"""
        
        # Get recent metric history
        history = self.metrics_collector.get_metric_history(
            metric_name, threshold.window_seconds
        )
        
        if len(history) < threshold.min_samples:
            return  # Not enough samples
        
        # Calculate metric statistics
        recent_values = [m.value for m in history[-threshold.min_samples:]]
        avg_value = statistics.mean(recent_values)
        current_value = recent_values[-1]
        
        # Determine alert level
        alert_level = self._determine_alert_level(current_value, threshold)
        
        if alert_level:
            alert_key = f"{metric_name}_{alert_level.value}"
            
            # Check alert cooldown
            if self._is_alert_suppressed(alert_key, current_time):
                return
            
            # Create alert
            alert = Alert(
                level=alert_level,
                message=self._generate_alert_message(metric_name, current_value, threshold, alert_level),
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=self._get_threshold_value(alert_level, threshold),
                tags={"avg_value": avg_value, "samples": len(recent_values)}
            )
            
            self._trigger_alert(alert, alert_key, current_time)
        
        else:
            # Check if we need to resolve existing alerts
            self._check_alert_resolution(metric_name, current_value, threshold, current_time)
    
    def _determine_alert_level(self, value: float, threshold: PerformanceThreshold) -> Optional[AlertLevel]:
        """Determine alert level based on value and threshold"""
        if threshold.direction == "above":
            if threshold.emergency_threshold and value >= threshold.emergency_threshold:
                return AlertLevel.EMERGENCY
            elif value >= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif value >= threshold.warning_threshold:
                return AlertLevel.WARNING
        else:  # direction == "below"
            if threshold.emergency_threshold and value <= threshold.emergency_threshold:
                return AlertLevel.EMERGENCY
            elif value <= threshold.critical_threshold:
                return AlertLevel.CRITICAL
            elif value <= threshold.warning_threshold:
                return AlertLevel.WARNING
        
        return None
    
    def _get_threshold_value(self, alert_level: AlertLevel, threshold: PerformanceThreshold) -> float:
        """Get threshold value for specific alert level"""
        if alert_level == AlertLevel.EMERGENCY and threshold.emergency_threshold:
            return threshold.emergency_threshold
        elif alert_level == AlertLevel.CRITICAL:
            return threshold.critical_threshold
        elif alert_level == AlertLevel.WARNING:
            return threshold.warning_threshold
        return 0.0
    
    def _generate_alert_message(self, 
                               metric_name: str, 
                               current_value: float, 
                               threshold: PerformanceThreshold,
                               alert_level: AlertLevel) -> str:
        """Generate human-readable alert message"""
        threshold_value = self._get_threshold_value(alert_level, threshold)
        direction_text = "above" if threshold.direction == "above" else "below"
        
        return (f"{alert_level.value.upper()}: {metric_name} is {current_value:.2f} "
                f"({direction_text} threshold of {threshold_value:.2f})")
    
    def _is_alert_suppressed(self, alert_key: str, current_time: float) -> bool:
        """Check if alert is suppressed due to cooldown"""
        last_alert_time = self.alert_cooldown.get(alert_key, 0)
        return (current_time - last_alert_time) < self.default_cooldown_seconds
    
    def _trigger_alert(self, alert: Alert, alert_key: str, current_time: float):
        """Trigger an alert"""
        # Store alert
        self.active_alerts[alert_key] = alert
        self.alert_history.append(alert)
        self.alert_cooldown[alert_key] = current_time
        
        # Log alert
        logger.log(
            logging.CRITICAL if alert.level == AlertLevel.EMERGENCY else
            logging.ERROR if alert.level == AlertLevel.CRITICAL else
            logging.WARNING,
            alert.message
        )
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _check_alert_resolution(self, 
                               metric_name: str, 
                               current_value: float, 
                               threshold: PerformanceThreshold,
                               current_time: float):
        """Check if existing alerts should be resolved"""
        alert_keys_to_resolve = []
        
        for alert_key, alert in self.active_alerts.items():
            if alert.metric_name == metric_name and not alert.resolved:
                # Check if current value is now within acceptable range
                if not self._determine_alert_level(current_value, threshold):
                    alert.resolved = True
                    alert_keys_to_resolve.append(alert_key)
                    
                    logger.info(f"Alert resolved: {alert.message}")
        
        # Remove resolved alerts from active alerts
        for key in alert_keys_to_resolve:
            self.active_alerts.pop(key, None)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        with self.lock:
            return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status"""
        with self.lock:
            active_alerts = self.get_active_alerts()
            alert_counts = defaultdict(int)
            
            for alert in active_alerts:
                alert_counts[alert.level.value] += 1
            
            return {
                'active_alerts_count': len(active_alerts),
                'alert_counts_by_level': dict(alert_counts),
                'total_thresholds': len(self.thresholds),
                'alert_history_count': len(self.alert_history),
                'suppressed_alerts': len(self.alert_cooldown)
            }

# ============================================================================
# PERFORMANCE MONITORING DECORATORS
# ============================================================================

def monitor_performance(metric_name: str, tags: Dict[str, str] = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                success = True
                error = None
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                # Record performance metric
                latency_ms = (time.time() - start_time) * 1000
                
                metric_tags = {"function": func.__name__, "success": str(success)}
                if tags:
                    metric_tags.update(tags)
                if error:
                    metric_tags["error"] = error[:100]  # Truncate long errors
                
                metric = PerformanceMetric(
                    name=metric_name,
                    value=latency_ms,
                    unit="ms",
                    tags=metric_tags
                )
                
                # Get global metrics collector
                if hasattr(wrapper, '_metrics_collector'):
                    wrapper._metrics_collector.record_metric(metric)
            
            return result
        
        return wrapper
    return decorator

# ============================================================================
# COMPREHENSIVE MONITORING SYSTEM
# ============================================================================

class PerformanceMonitoringSystem:
    """Comprehensive performance monitoring and alerting system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Background monitoring thread
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_interval = 10  # seconds
        
        # Performance recommendations
        self.performance_analyzer = PerformanceAnalyzer(self.metrics_collector)
        
        # Setup default alert callbacks
        self._setup_alert_callbacks()
        
        logger.info("Performance monitoring system initialized")
    
    def _setup_alert_callbacks(self):
        """Setup default alert callbacks"""
        def log_alert(alert: Alert):
            logger.critical(f"PERFORMANCE ALERT: {alert.message}")
        
        self.alert_manager.add_alert_callback(log_alert)
    
    def start_monitoring(self):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Background monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.metrics_collector.collect_system_metrics()
                
                # Check thresholds and generate alerts
                self.alert_manager.check_thresholds()
                
                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def record_pipeline_latency(self, latency_ms: float, ticker: str = None):
        """Record pipeline latency metric"""
        tags = {}
        if ticker:
            tags["ticker"] = ticker
        
        self.metrics_collector.record_metric(PerformanceMetric(
            name="pipeline_latency_ms",
            value=latency_ms,
            unit="ms",
            tags=tags
        ))
    
    def record_ticker_processing_time(self, ticker: str, processing_time_ms: float):
        """Record individual ticker processing time"""
        self.metrics_collector.record_metric(PerformanceMetric(
            name="ticker_processing_latency_ms",
            value=processing_time_ms,
            unit="ms",
            tags={"ticker": ticker}
        ))
    
    def record_error_rate(self, success_count: int, total_count: int):
        """Record error rate"""
        error_rate = ((total_count - success_count) / total_count * 100) if total_count > 0 else 0
        
        self.metrics_collector.record_metric(PerformanceMetric(
            name="error_rate_percent",
            value=error_rate,
            unit="%",
            tags={"success_count": str(success_count), "total_count": str(total_count)}
        ))
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive performance dashboard data"""
        return {
            'timestamp': time.time(),
            'system_metrics': {
                'memory_usage': self.metrics_collector.get_metric_stats('memory_usage_mb'),
                'cpu_usage': self.metrics_collector.get_metric_stats('cpu_usage_percent'),
                'available_memory': self.metrics_collector.get_metric_stats('system_memory_available_mb'),
                'gpu_max_util': self.metrics_collector.get_metric_stats('gpu.max.util_pct'),
                'gpu_max_mem': self.metrics_collector.get_metric_stats('gpu.max.mem_pct'),
                'gpu_max_temp': self.metrics_collector.get_metric_stats('gpu.max.temp_c')
            },
            'pipeline_metrics': {
                'pipeline_latency': self.metrics_collector.get_metric_stats('pipeline_latency_ms'),
                'ticker_processing': self.metrics_collector.get_metric_stats('ticker_processing_latency_ms'),
                'error_rate': self.metrics_collector.get_metric_stats('error_rate_percent')
            },
            'alert_summary': self.alert_manager.get_alert_summary(),
            'active_alerts': self.alert_manager.get_active_alerts(),
            'recommendations': self.performance_analyzer.get_recommendations()
        }

# ============================================================================
# PERFORMANCE ANALYSIS AND RECOMMENDATIONS
# ============================================================================

class PerformanceAnalyzer:
    """Analyze performance metrics and provide optimization recommendations"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
    
    def get_recommendations(self) -> List[Dict[str, str]]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        # Memory usage analysis
        memory_stats = self.metrics_collector.get_metric_stats('memory_usage_mb')
        if memory_stats and memory_stats.get('mean', 0) > 4000:
            recommendations.append({
                'type': 'memory',
                'priority': 'high',
                'message': f"High memory usage detected (avg: {memory_stats['mean']:.1f}MB). "
                          "Consider enabling aggressive garbage collection or reducing batch sizes."
            })
        
        # Latency analysis
        latency_stats = self.metrics_collector.get_metric_stats('pipeline_latency_ms')
        if latency_stats and latency_stats.get('p95', 0) > 1000:
            recommendations.append({
                'type': 'latency',
                'priority': 'critical',
                'message': f"Pipeline latency P95 is {latency_stats['p95']:.1f}ms (>1000ms target). "
                          "Consider enabling parallel processing optimization."
            })
        
        # Error rate analysis
        error_stats = self.metrics_collector.get_metric_stats('error_rate_percent')
        if error_stats and error_stats.get('mean', 0) > 5:
            recommendations.append({
                'type': 'reliability',
                'priority': 'high',
                'message': f"High error rate detected ({error_stats['mean']:.1f}%). "
                          "Review error logs and consider implementing additional retry logic."
            })
        
        return recommendations

# Global performance monitoring system instance
performance_monitor = PerformanceMonitoringSystem()

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Start monitoring
    performance_monitor.start_monitoring()
    
    # Simulate some metrics
    for i in range(10):
        performance_monitor.record_pipeline_latency(800 + i * 50, f"TEST{i}")
        performance_monitor.record_ticker_processing_time(f"TEST{i}", 200 + i * 20)
        time.sleep(1)
    
    # Get dashboard
    dashboard = performance_monitor.get_performance_dashboard()
    print(json.dumps(dashboard, indent=2, default=str))
    
    # Stop monitoring
    performance_monitor.stop_monitoring()
