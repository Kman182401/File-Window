"""
Comprehensive Performance Tracking System for Trading Platform

This module provides detailed performance monitoring, metrics collection, and
analysis for all aspects of the trading system including latency, throughput,
model performance, and system resource utilization.

Key Features:
- Real-time performance metrics collection
- Latency tracking for all operations
- Throughput monitoring for data pipelines
- Model inference performance tracking
- Trading performance analytics
- System resource utilization monitoring
- Performance anomaly detection
- Historical performance analysis
- Performance report generation

Author: AI Trading System
Version: 2.0.0
"""

import os
import time
import json
import logging
import threading
import numpy as np
import pandas as pd
import uuid
import psutil
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
import statistics
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    SUCCESS_RATE = "success_rate"
    RESOURCE_USAGE = "resource_usage"
    MODEL_PERFORMANCE = "model_performance"
    TRADING_PERFORMANCE = "trading_performance"
    CUSTOM = "custom"


class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"    # Top 10% performance
    GOOD = "good"             # Top 25% performance
    NORMAL = "normal"         # Average performance
    DEGRADED = "degraded"     # Below average
    POOR = "poor"             # Bottom 10% performance


@dataclass
class PerformanceMetric:
    """Container for a single performance metric"""
    timestamp: datetime = field(default_factory=datetime.now)
    name: str = ""
    metric_type: MetricType = MetricType.CUSTOM
    value: float = 0.0
    unit: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thread_id: Optional[str] = field(default_factory=lambda: str(threading.get_ident()))
    symbol: Optional[str] = None
    component: Optional[str] = None


@dataclass
class OperationMetrics:
    """Metrics for a single operation"""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    memory_used_mb: float = 0.0
    cpu_percent: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    data_processed_mb_per_second: float = 0.0
    
    # Success/Error rates
    success_rate: float = 0.0
    error_rate: float = 0.0
    
    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Component-specific metrics
    component_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class PerformanceTracker:
    """
    Comprehensive performance tracking system
    
    Tracks and analyzes performance metrics across all system components
    with support for real-time monitoring and historical analysis.
    """
    
    def __init__(self, history_size: int = 10000, 
                 persist_interval: int = 300,
                 persist_path: Optional[str] = None):
        """
        Initialize performance tracker with Phase 4A enhancements
        
        Args:
            history_size: Number of metrics to keep in memory
            persist_interval: Seconds between persistence to disk
            persist_path: Path to persist metrics
        """
        self.history_size = history_size
        self.persist_interval = persist_interval
        self.persist_path = persist_path or Path.home() / "logs" / "performance"
        
        # Metric storage
        self.metrics_history = deque(maxlen=history_size)
        self.operation_history = deque(maxlen=history_size)
        self.snapshots = deque(maxlen=1000)
        
        # Active operations tracking with correlation IDs
        self.active_operations = {}
        self.correlation_map = defaultdict(list)  # Map correlation IDs to operations
        
        # Aggregated metrics by component and symbol
        self.component_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.symbol_metrics = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance baselines for anomaly detection
        self.baselines = {}
        
        # Phase 4A specific: Performance regression detection
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.regression_thresholds = {
            'latency_degradation': 1.5,  # 50% increase
            'throughput_degradation': 0.7,  # 30% decrease
            'memory_increase': 1.3,  # 30% increase
            'error_rate_increase': 2.0  # 100% increase
        }
        
        # Alert system
        self.alert_callbacks = []
        self.alert_thresholds = {
            'critical_latency_ms': 1000,
            'critical_memory_mb': 6000,  # 6GB limit from EC2 constraints
            'critical_error_rate': 10.0,  # 10% error rate
            'critical_cpu_percent': 90.0
        }
        
        # Symbol-specific processing times
        self.symbol_processing_times = defaultdict(lambda: deque(maxlen=1000))
        
        # Callbacks for performance events
        self.performance_callbacks = []
        self.anomaly_callbacks = []
        
        # Threading
        self._lock = threading.Lock()
        self.monitoring_active = False
        self.persist_thread = None
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Database for audit compliance
        self.db_path = self.persist_path / "performance_audit.db"
        self._init_audit_database()
        
        # Ensure persist directory exists
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing metrics if available
        self._load_persisted_metrics()
        
        logger.info(f"PerformanceTracker initialized with history_size={history_size} (Phase 4A Enhanced)")
    
    def _init_audit_database(self):
        """Initialize SQLite database for audit compliance"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_audit (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        correlation_id TEXT,
                        component TEXT,
                        operation TEXT,
                        symbol TEXT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        unit TEXT,
                        tags TEXT,
                        metadata TEXT,
                        thread_id TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance_regressions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        baseline_value REAL NOT NULL,
                        current_value REAL NOT NULL,
                        degradation_percent REAL NOT NULL,
                        severity TEXT NOT NULL,
                        detected_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TEXT,
                        resolution_notes TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimization_changes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        change_type TEXT NOT NULL,
                        component TEXT,
                        description TEXT NOT NULL,
                        before_metrics TEXT,
                        after_metrics TEXT,
                        impact_assessment TEXT,
                        rollback_info TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.debug("Audit database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audit database: {e}")
    
    def create_correlation_id(self) -> str:
        """Create a new correlation ID for linking related operations"""
        return str(uuid.uuid4())
    
    def record_symbol_processing_time(self, symbol: str, processing_time_ms: float, 
                                    correlation_id: Optional[str] = None):
        """Record processing time for a specific symbol"""
        with self._lock:
            self.symbol_processing_times[symbol].append({
                'timestamp': datetime.now(),
                'processing_time_ms': processing_time_ms,
                'correlation_id': correlation_id or self.create_correlation_id()
            })
        
        # Record as metric
        self.record_metric(
            f"symbol.{symbol}.processing_time",
            processing_time_ms,
            MetricType.LATENCY,
            "ms",
            tags={'symbol': symbol, 'component': 'symbol_processing'},
            metadata={'correlation_id': correlation_id}
        )
    
    def detect_performance_regression(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """Detect if current performance represents a regression"""
        with self._lock:
            history = self.performance_history[metric_name]
            
            if len(history) < 10:  # Need sufficient history
                history.append(current_value)
                return None
            
            # Calculate baseline (median of recent history)
            baseline = statistics.median(list(history))
            
            # Check for regression based on metric type
            regression_detected = False
            degradation_percent = 0.0
            
            if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
                # Higher is worse for latency/time metrics
                if current_value > baseline * self.regression_thresholds['latency_degradation']:
                    regression_detected = True
                    degradation_percent = ((current_value - baseline) / baseline) * 100
            
            elif 'throughput' in metric_name.lower() or 'rate' in metric_name.lower():
                # Lower is worse for throughput metrics
                if current_value < baseline * self.regression_thresholds['throughput_degradation']:
                    regression_detected = True
                    degradation_percent = ((baseline - current_value) / baseline) * 100
            
            elif 'memory' in metric_name.lower():
                # Higher is worse for memory metrics
                if current_value > baseline * self.regression_thresholds['memory_increase']:
                    regression_detected = True
                    degradation_percent = ((current_value - baseline) / baseline) * 100
            
            elif 'error' in metric_name.lower():
                # Higher is worse for error metrics
                if current_value > baseline * self.regression_thresholds['error_rate_increase']:
                    regression_detected = True
                    degradation_percent = ((current_value - baseline) / baseline) * 100
            
            # Update history
            history.append(current_value)
            
            if regression_detected:
                severity = 'CRITICAL' if degradation_percent > 100 else 'HIGH' if degradation_percent > 50 else 'MEDIUM'
                
                regression_info = {
                    'timestamp': datetime.now().isoformat(),
                    'metric_name': metric_name,
                    'baseline_value': baseline,
                    'current_value': current_value,
                    'degradation_percent': degradation_percent,
                    'severity': severity
                }
                
                # Log to audit database
                self._log_regression_to_db(regression_info)
                
                logger.warning(f"Performance regression detected: {metric_name} degraded by {degradation_percent:.1f}% (severity: {severity})")
                return regression_info
            
            return None
    
    def _log_regression_to_db(self, regression_info: Dict[str, Any]):
        """Log performance regression to audit database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO performance_regressions 
                    (timestamp, metric_name, baseline_value, current_value, 
                     degradation_percent, severity)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    regression_info['timestamp'],
                    regression_info['metric_name'],
                    regression_info['baseline_value'],
                    regression_info['current_value'],
                    regression_info['degradation_percent'],
                    regression_info['severity']
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log regression to database: {e}")
    
    def trigger_performance_alert(self, metric_name: str, current_value: float, threshold: float, severity: str = "HIGH"):
        """Trigger performance alert when thresholds are exceeded"""
        alert_info = {
            'timestamp': datetime.now().isoformat(),
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'severity': severity,
            'alert_id': str(uuid.uuid4())
        }
        
        logger.warning(f"PERFORMANCE ALERT [{severity}]: {metric_name}={current_value} exceeds threshold {threshold}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_info)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        return alert_info
    
    def check_alert_thresholds(self, metric_name: str, value: float):
        """Check if metric value exceeds alert thresholds"""
        alerts = []
        
        # Check critical thresholds
        if 'latency' in metric_name.lower() and value > self.alert_thresholds['critical_latency_ms']:
            alerts.append(self.trigger_performance_alert(metric_name, value, self.alert_thresholds['critical_latency_ms'], 'CRITICAL'))
        
        elif 'memory' in metric_name.lower() and value > self.alert_thresholds['critical_memory_mb']:
            alerts.append(self.trigger_performance_alert(metric_name, value, self.alert_thresholds['critical_memory_mb'], 'CRITICAL'))
        
        elif 'error' in metric_name.lower() and value > self.alert_thresholds['critical_error_rate']:
            alerts.append(self.trigger_performance_alert(metric_name, value, self.alert_thresholds['critical_error_rate'], 'HIGH'))
        
        elif 'cpu' in metric_name.lower() and value > self.alert_thresholds['critical_cpu_percent']:
            alerts.append(self.trigger_performance_alert(metric_name, value, self.alert_thresholds['critical_cpu_percent'], 'HIGH'))
        
        return alerts
    
    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def get_symbol_performance_stats(self, symbol: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance statistics for a specific symbol"""
        with self._lock:
            processing_times = list(self.symbol_processing_times[symbol])
            
            if time_window:
                cutoff = datetime.now() - time_window
                processing_times = [pt for pt in processing_times if pt['timestamp'] > cutoff]
            
            if not processing_times:
                return {}
            
            times = [pt['processing_time_ms'] for pt in processing_times]
            
            return {
                'symbol': symbol,
                'count': len(times),
                'mean_processing_time_ms': statistics.mean(times),
                'median_processing_time_ms': statistics.median(times),
                'p95_processing_time_ms': np.percentile(times, 95),
                'p99_processing_time_ms': np.percentile(times, 99),
                'min_processing_time_ms': min(times),
                'max_processing_time_ms': max(times),
                'stdev_processing_time_ms': statistics.stdev(times) if len(times) > 1 else 0
            }
    
    def log_optimization_change(self, change_type: str, component: str, description: str,
                              before_metrics: Optional[Dict[str, float]] = None,
                              after_metrics: Optional[Dict[str, float]] = None,
                              impact_assessment: Optional[str] = None,
                              rollback_info: Optional[str] = None):
        """Log optimization changes for audit compliance"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO optimization_changes 
                    (timestamp, change_type, component, description, before_metrics, 
                     after_metrics, impact_assessment, rollback_info)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    change_type,
                    component,
                    description,
                    json.dumps(before_metrics) if before_metrics else None,
                    json.dumps(after_metrics) if after_metrics else None,
                    impact_assessment,
                    rollback_info
                ))
                conn.commit()
            
            logger.info(f"Logged optimization change: {change_type} in {component}: {description}")
        except Exception as e:
            logger.error(f"Failed to log optimization change: {e}")
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.CUSTOM,
                     unit: str = "", tags: Optional[Dict[str, Any]] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     correlation_id: Optional[str] = None,
                     symbol: Optional[str] = None,
                     component: Optional[str] = None):
        """
        Record a performance metric with Phase 4A enhancements
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            unit: Unit of measurement
            tags: Tags for categorization
            metadata: Additional metadata
            correlation_id: Correlation ID for linking related operations
            symbol: Symbol being processed
            component: Component generating the metric
        """
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Extract component and symbol from tags if not provided
        tags = tags or {}
        if component:
            tags['component'] = component
        if symbol:
            tags['symbol'] = symbol
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            unit=unit,
            tags=tags,
            metadata=metadata or {},
            correlation_id=correlation_id,
            symbol=symbol,
            component=component or tags.get('component')
        )
        
        with self._lock:
            self.metrics_history.append(metric)
            
            # Store by component if tagged
            if tags and 'component' in tags:
                self.component_metrics[tags['component']].append(metric)
            
            # Store by symbol if tagged
            if symbol or (tags and 'symbol' in tags):
                symbol_key = symbol or tags['symbol']
                self.symbol_metrics[symbol_key].append(metric)
            
            # Map correlation ID
            if correlation_id:
                self.correlation_map[correlation_id].append(metric)
        
        # Log to audit database
        self._log_metric_to_db(metric)
        
        # Check for performance regression
        regression = self.detect_performance_regression(name, value)
        
        # Check alert thresholds
        self.check_alert_thresholds(name, value)
        
        # Check for anomalies
        if self._is_anomaly(name, value):
            self._trigger_anomaly_callbacks(metric)
        
        logger.debug(f"Recorded metric: {name}={value}{unit} [correlation_id={correlation_id[:8]}...]")
    
    def _log_metric_to_db(self, metric: PerformanceMetric):
        """Log metric to audit database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO performance_audit 
                    (timestamp, correlation_id, component, symbol, metric_name, 
                     metric_value, unit, tags, metadata, thread_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(),
                    metric.correlation_id,
                    metric.component,
                    metric.symbol,
                    metric.name,
                    metric.value,
                    metric.unit,
                    json.dumps(metric.tags) if metric.tags else None,
                    json.dumps(metric.metadata) if metric.metadata else None,
                    metric.thread_id
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to log metric to database: {e}")
    
    def start_operation(self, operation_name: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking an operation
        
        Args:
            operation_name: Name of the operation
            metadata: Optional metadata
            
        Returns:
            Operation ID for tracking
        """
        operation_id = f"{operation_name}_{datetime.now().timestamp()}"
        
        operation = OperationMetrics(
            operation_name=operation_name,
            start_time=datetime.now(),
            custom_metrics=metadata or {}
        )
        
        with self._lock:
            self.active_operations[operation_id] = operation
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True,
                     error_message: Optional[str] = None,
                     custom_metrics: Optional[Dict[str, Any]] = None):
        """
        End tracking an operation
        
        Args:
            operation_id: Operation ID from start_operation
            success: Whether operation succeeded
            error_message: Error message if failed
            custom_metrics: Additional metrics to record
        """
        with self._lock:
            if operation_id not in self.active_operations:
                logger.warning(f"Unknown operation ID: {operation_id}")
                return
            
            operation = self.active_operations[operation_id]
            operation.end_time = datetime.now()
            operation.duration_ms = (operation.end_time - operation.start_time).total_seconds() * 1000
            operation.success = success
            operation.error_message = error_message
            
            if custom_metrics:
                operation.custom_metrics.update(custom_metrics)
            
            # Move to history
            self.operation_history.append(operation)
            del self.active_operations[operation_id]
        
        # Record as metrics
        self.record_metric(
            f"{operation.operation_name}_latency",
            operation.duration_ms,
            MetricType.LATENCY,
            "ms",
            tags={'operation': operation.operation_name}
        )
        
        if not success:
            self.record_metric(
                f"{operation.operation_name}_error",
                1,
                MetricType.ERROR_RATE,
                tags={'operation': operation.operation_name, 'error': error_message}
            )
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager for tracking operations
        
        Args:
            operation_name: Name of the operation
            
        Example:
            with tracker.track_operation("data_fetch"):
                # Perform operation
                data = fetch_data()
        """
        operation_id = self.start_operation(operation_name)
        success = True
        error_message = None
        
        try:
            yield operation_id
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            self.end_operation(operation_id, success, error_message)
    
    def record_latency(self, component: str, operation: str, latency_ms: float):
        """
        Record latency for a specific operation
        
        Args:
            component: Component name
            operation: Operation name
            latency_ms: Latency in milliseconds
        """
        self.record_metric(
            f"{component}.{operation}.latency",
            latency_ms,
            MetricType.LATENCY,
            "ms",
            tags={'component': component, 'operation': operation}
        )
    
    def record_throughput(self, component: str, items_processed: int, 
                         time_seconds: float):
        """
        Record throughput for a component
        
        Args:
            component: Component name
            items_processed: Number of items processed
            time_seconds: Time taken in seconds
        """
        throughput = items_processed / time_seconds if time_seconds > 0 else 0
        
        self.record_metric(
            f"{component}.throughput",
            throughput,
            MetricType.THROUGHPUT,
            "items/sec",
            tags={'component': component}
        )
    
    def record_model_performance(self, model_name: str, metrics: Dict[str, float]):
        """
        Record model performance metrics
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
        """
        for metric_name, value in metrics.items():
            self.record_metric(
                f"model.{model_name}.{metric_name}",
                value,
                MetricType.MODEL_PERFORMANCE,
                tags={'model': model_name, 'metric': metric_name}
            )
    
    def record_trading_performance(self, metrics: Dict[str, float]):
        """
        Record trading performance metrics
        
        Args:
            metrics: Dictionary of trading metrics (pnl, sharpe, win_rate, etc.)
        """
        for metric_name, value in metrics.items():
            self.record_metric(
                f"trading.{metric_name}",
                value,
                MetricType.TRADING_PERFORMANCE,
                tags={'metric': metric_name}
            )
    
    def get_latency_stats(self, component: Optional[str] = None,
                         time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """
        Get latency statistics
        
        Args:
            component: Optional component filter
            time_window: Optional time window filter
            
        Returns:
            Dictionary of latency statistics
        """
        with self._lock:
            # Filter operations
            operations = list(self.operation_history)
            
            if component:
                operations = [op for op in operations 
                            if component in op.operation_name]
            
            if time_window:
                cutoff = datetime.now() - time_window
                operations = [op for op in operations 
                            if op.start_time > cutoff]
            
            if not operations:
                return {}
            
            latencies = [op.duration_ms for op in operations if op.duration_ms > 0]
            
            if not latencies:
                return {}
            
            return {
                'count': len(latencies),
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min': min(latencies),
                'max': max(latencies),
                'p50': np.percentile(latencies, 50),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            }
    
    def get_error_rate(self, component: Optional[str] = None,
                      time_window: Optional[timedelta] = None) -> float:
        """
        Calculate error rate
        
        Args:
            component: Optional component filter
            time_window: Optional time window filter
            
        Returns:
            Error rate as percentage
        """
        with self._lock:
            operations = list(self.operation_history)
            
            if component:
                operations = [op for op in operations 
                            if component in op.operation_name]
            
            if time_window:
                cutoff = datetime.now() - time_window
                operations = [op for op in operations 
                            if op.start_time > cutoff]
            
            if not operations:
                return 0.0
            
            errors = sum(1 for op in operations if not op.success)
            return (errors / len(operations)) * 100
    
    def get_throughput(self, component: str,
                      time_window: Optional[timedelta] = None) -> Dict[str, float]:
        """
        Get throughput statistics for a component
        
        Args:
            component: Component name
            time_window: Optional time window filter
            
        Returns:
            Throughput statistics
        """
        with self._lock:
            metrics = [m for m in self.metrics_history
                      if m.metric_type == MetricType.THROUGHPUT
                      and m.tags.get('component') == component]
            
            if time_window:
                cutoff = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp > cutoff]
            
            if not metrics:
                return {}
            
            values = [m.value for m in metrics]
            
            return {
                'current': values[-1] if values else 0,
                'average': statistics.mean(values),
                'peak': max(values),
                'minimum': min(values)
            }
    
    def create_snapshot(self) -> PerformanceSnapshot:
        """
        Create a performance snapshot
        
        Returns:
            Current performance snapshot
        """
        snapshot = PerformanceSnapshot()
        
        # Calculate latency stats
        latency_stats = self.get_latency_stats(time_window=timedelta(minutes=5))
        if latency_stats:
            snapshot.avg_latency_ms = latency_stats.get('mean', 0)
            snapshot.p50_latency_ms = latency_stats.get('p50', 0)
            snapshot.p95_latency_ms = latency_stats.get('p95', 0)
            snapshot.p99_latency_ms = latency_stats.get('p99', 0)
            snapshot.max_latency_ms = latency_stats.get('max', 0)
        
        # Calculate throughput
        recent_ops = [op for op in self.operation_history
                     if op.end_time and 
                     op.end_time > datetime.now() - timedelta(seconds=60)]
        
        if recent_ops:
            snapshot.requests_per_second = len(recent_ops) / 60
        
        # Calculate success/error rates
        snapshot.error_rate = self.get_error_rate(time_window=timedelta(minutes=5))
        snapshot.success_rate = 100 - snapshot.error_rate
        
        # Get resource usage
        try:
            import psutil
            snapshot.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            snapshot.memory_usage_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except:
            pass
        
        # Component-specific metrics
        for component in set(m.tags.get('component', '') 
                           for m in self.metrics_history 
                           if 'component' in m.tags):
            if component:
                component_latency = self.get_latency_stats(component, timedelta(minutes=5))
                component_throughput = self.get_throughput(component, timedelta(minutes=5))
                
                snapshot.component_metrics[component] = {
                    'latency_ms': component_latency.get('mean', 0) if component_latency else 0,
                    'throughput': component_throughput.get('current', 0) if component_throughput else 0,
                    'error_rate': self.get_error_rate(component, timedelta(minutes=5))
                }
        
        with self._lock:
            self.snapshots.append(snapshot)
        
        return snapshot
    
    def _is_anomaly(self, metric_name: str, value: float) -> bool:
        """
        Detect if a metric value is anomalous
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            True if anomalous
        """
        if metric_name not in self.baselines:
            return False
        
        baseline = self.baselines[metric_name]
        
        # Simple z-score based anomaly detection
        if 'mean' in baseline and 'stdev' in baseline:
            z_score = abs((value - baseline['mean']) / baseline['stdev']) if baseline['stdev'] > 0 else 0
            return z_score > 3  # 3 standard deviations
        
        return False
    
    def update_baselines(self):
        """Update performance baselines for anomaly detection"""
        with self._lock:
            # Group metrics by name
            metrics_by_name = defaultdict(list)
            
            for metric in self.metrics_history:
                metrics_by_name[metric.name].append(metric.value)
            
            # Calculate baselines
            for name, values in metrics_by_name.items():
                if len(values) >= 100:  # Need sufficient data
                    self.baselines[name] = {
                        'mean': statistics.mean(values),
                        'stdev': statistics.stdev(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values)
                    }
    
    def _trigger_anomaly_callbacks(self, metric: PerformanceMetric):
        """Trigger callbacks for anomalies"""
        for callback in self.anomaly_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Anomaly callback failed: {e}")
    
    def register_anomaly_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Register callback for performance anomalies"""
        self.anomaly_callbacks.append(callback)
    
    def get_performance_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            time_window: Optional time window for analysis
            
        Returns:
            Performance report dictionary
        """
        if not time_window:
            time_window = timedelta(hours=1)
        
        cutoff = datetime.now() - time_window
        
        # Filter recent data
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff]
        recent_operations = [op for op in self.operation_history 
                           if op.start_time > cutoff]
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff]
        
        # Calculate overall statistics
        overall_latency = self.get_latency_stats(time_window=time_window)
        overall_error_rate = self.get_error_rate(time_window=time_window)
        
        # Component performance
        component_performance = {}
        components = set(m.tags.get('component', '') for m in recent_metrics 
                       if 'component' in m.tags)
        
        for component in components:
            if component:
                component_performance[component] = {
                    'latency': self.get_latency_stats(component, time_window),
                    'throughput': self.get_throughput(component, time_window),
                    'error_rate': self.get_error_rate(component, time_window)
                }
        
        # Performance trends
        if recent_snapshots:
            latency_trend = [s.avg_latency_ms for s in recent_snapshots]
            throughput_trend = [s.requests_per_second for s in recent_snapshots]
            error_trend = [s.error_rate for s in recent_snapshots]
        else:
            latency_trend = []
            throughput_trend = []
            error_trend = []
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_window': str(time_window),
            'summary': {
                'total_operations': len(recent_operations),
                'total_metrics': len(recent_metrics),
                'active_operations': len(self.active_operations),
                'overall_error_rate': overall_error_rate,
                'overall_success_rate': 100 - overall_error_rate
            },
            'latency': overall_latency,
            'components': component_performance,
            'trends': {
                'latency_ms': latency_trend,
                'throughput_rps': throughput_trend,
                'error_rate': error_trend
            },
            'current_snapshot': asdict(recent_snapshots[-1]) if recent_snapshots else None
        }
        
        return report
    
    def _persist_metrics(self):
        """Persist metrics to disk"""
        try:
            # Prepare data for persistence
            data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': [asdict(m) for m in list(self.metrics_history)[-1000:]],
                'operations': [
                    {
                        'operation_name': op.operation_name,
                        'start_time': op.start_time.isoformat(),
                        'end_time': op.end_time.isoformat() if op.end_time else None,
                        'duration_ms': op.duration_ms,
                        'success': op.success,
                        'error_message': op.error_message
                    }
                    for op in list(self.operation_history)[-1000:]
                ],
                'baselines': self.baselines
            }
            
            # Save to file
            filename = self.persist_path / f"performance_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, default=str)
            
            logger.debug(f"Persisted metrics to {filename}")
            
            # Clean old files (keep last 7 days)
            cutoff = datetime.now() - timedelta(days=7)
            for file in self.persist_path.glob("performance_*.json"):
                try:
                    file_time = datetime.strptime(file.stem.split('_', 1)[1], "%Y%m%d_%H%M%S")
                    if file_time < cutoff:
                        file.unlink()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def _load_persisted_metrics(self):
        """Load previously persisted metrics"""
        try:
            # Find most recent metrics file
            files = sorted(self.persist_path.glob("performance_*.json"))
            if not files:
                return
            
            with open(files[-1], 'r') as f:
                data = json.load(f)
            
            # Load baselines
            self.baselines = data.get('baselines', {})
            
            logger.info(f"Loaded persisted metrics from {files[-1]}")
            
        except Exception as e:
            logger.debug(f"Could not load persisted metrics: {e}")
    
    def start_monitoring(self):
        """Start automatic monitoring and persistence"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def persist_loop():
            while self.monitoring_active:
                time.sleep(self.persist_interval)
                self._persist_metrics()
                self.update_baselines()
        
        self.persist_thread = threading.Thread(target=persist_loop, daemon=True)
        self.persist_thread.start()
        
        logger.info(f"Performance monitoring started (persist every {self.persist_interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring and persist final metrics"""
        self.monitoring_active = False
        
        if self.persist_thread:
            self.persist_thread.join(timeout=5)
        
        self._persist_metrics()
        logger.info("Performance monitoring stopped")
    
    def get_correlation_metrics(self, correlation_id: str) -> List[PerformanceMetric]:
        """Get all metrics associated with a correlation ID"""
        with self._lock:
            return list(self.correlation_map.get(correlation_id, []))
    
    def get_audit_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report for compliance"""
        if not time_window:
            time_window = timedelta(hours=24)
        
        cutoff = datetime.now() - time_window
        cutoff_str = cutoff.isoformat()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_window': str(time_window),
            'audit_summary': {},
            'performance_regressions': [],
            'optimization_changes': [],
            'symbol_performance': {},
            'correlation_analysis': {}
        }
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get audit summary
                cursor = conn.execute('''
                    SELECT COUNT(*) as total_metrics, 
                           COUNT(DISTINCT correlation_id) as unique_correlations,
                           COUNT(DISTINCT component) as components_monitored,
                           COUNT(DISTINCT symbol) as symbols_processed
                    FROM performance_audit 
                    WHERE timestamp > ?
                ''', (cutoff_str,))
                
                row = cursor.fetchone()
                if row:
                    report['audit_summary'] = {
                        'total_metrics': row[0],
                        'unique_correlations': row[1],
                        'components_monitored': row[2],
                        'symbols_processed': row[3]
                    }
                
                # Get performance regressions
                cursor = conn.execute('''
                    SELECT * FROM performance_regressions 
                    WHERE timestamp > ?
                    ORDER BY degradation_percent DESC
                    LIMIT 50
                ''', (cutoff_str,))
                
                report['performance_regressions'] = [
                    {
                        'timestamp': row[1],
                        'metric_name': row[2],
                        'baseline_value': row[3],
                        'current_value': row[4],
                        'degradation_percent': row[5],
                        'severity': row[6]
                    }
                    for row in cursor.fetchall()
                ]
                
                # Get optimization changes
                cursor = conn.execute('''
                    SELECT * FROM optimization_changes 
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 20
                ''', (cutoff_str,))
                
                report['optimization_changes'] = [
                    {
                        'timestamp': row[1],
                        'change_type': row[2],
                        'component': row[3],
                        'description': row[4],
                        'before_metrics': json.loads(row[5]) if row[5] else None,
                        'after_metrics': json.loads(row[6]) if row[6] else None,
                        'impact_assessment': row[7]
                    }
                    for row in cursor.fetchall()
                ]
                
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
        
        # Add symbol performance statistics
        for symbol in self.symbol_metrics.keys():
            report['symbol_performance'][symbol] = self.get_symbol_performance_stats(
                symbol, time_window
            )
        
        return report
    
    def cleanup_old_data(self, retention_days: int = 30):
        """Clean up old performance data beyond retention period"""
        cutoff = datetime.now() - timedelta(days=retention_days)
        cutoff_str = cutoff.isoformat()
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Clean old audit records
                result = conn.execute('''
                    DELETE FROM performance_audit WHERE timestamp < ?
                ''', (cutoff_str,))
                
                deleted_audit = result.rowcount
                
                # Clean old regression records (keep longer for trend analysis)
                old_cutoff = datetime.now() - timedelta(days=retention_days * 2)
                result = conn.execute('''
                    DELETE FROM performance_regressions WHERE timestamp < ?
                ''', (old_cutoff.isoformat(),))
                
                deleted_regressions = result.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_audit} audit records and {deleted_regressions} regression records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")


# Global instance
_performance_tracker = None


def get_performance_tracker() -> PerformanceTracker:
    """Get or create singleton PerformanceTracker instance"""
    global _performance_tracker
    
    if _performance_tracker is None:
        _performance_tracker = PerformanceTracker()
        _performance_tracker.start_monitoring()
    
    return _performance_tracker


def correlation_decorator(operation_name: str, component: str = None, symbol: str = None):
    """Decorator for automatic correlation ID tracking"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            correlation_id = tracker.create_correlation_id()
            
            # Add correlation_id to kwargs if function accepts it
            import inspect
            sig = inspect.signature(func)
            if 'correlation_id' in sig.parameters:
                kwargs['correlation_id'] = correlation_id
            
            with tracker.track_operation(f"{operation_name}"):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000
                    
                    tracker.record_metric(
                        f"{operation_name}.duration",
                        duration_ms,
                        MetricType.LATENCY,
                        "ms",
                        correlation_id=correlation_id,
                        component=component,
                        symbol=symbol
                    )
                    
                    return result
                except Exception as e:
                    duration_ms = (time.time() - start_time) * 1000
                    
                    tracker.record_metric(
                        f"{operation_name}.error",
                        1,
                        MetricType.ERROR_RATE,
                        "count",
                        correlation_id=correlation_id,
                        component=component,
                        symbol=symbol,
                        metadata={'error': str(e)}
                    )
                    
                    raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Self-test with Phase 4A enhancements
    print("Performance Tracker Self-Test (Phase 4A Enhanced)")
    print("=" * 60)
    
    tracker = PerformanceTracker()
    
    # Test operation tracking
    print("\nTesting operation tracking...")
    
    with tracker.track_operation("test_operation"):
        time.sleep(0.1)  # Simulate work
        tracker.record_metric("test_metric", 42, MetricType.CUSTOM, "units")
    
    # Test latency recording
    print("Recording sample latencies...")
    for i in range(10):
        tracker.record_latency("test_component", "operation", 10 + i * 2)
    
    # Test throughput recording
    print("Recording throughput...")
    tracker.record_throughput("test_component", 1000, 10)
    
    # Test model performance
    print("Recording model performance...")
    tracker.record_model_performance("test_model", {
        'accuracy': 0.95,
        'loss': 0.05,
        'inference_time_ms': 25
    })
    
    # Test trading performance
    print("Recording trading performance...")
    tracker.record_trading_performance({
        'pnl': 1500,
        'sharpe_ratio': 1.8,
        'win_rate': 0.65,
        'max_drawdown': -0.05
    })
    
    # Get statistics
    print("\nLatency Statistics:")
    stats = tracker.get_latency_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Create snapshot
    print("\nCreating performance snapshot...")
    snapshot = tracker.create_snapshot()
    print(f"  Success rate: {snapshot.success_rate:.1f}%")
    print(f"  Error rate: {snapshot.error_rate:.1f}%")
    print(f"  Avg latency: {snapshot.avg_latency_ms:.2f}ms")
    
    # Generate report
    print("\nGenerating performance report...")
    report = tracker.get_performance_report(timedelta(minutes=5))
    print(f"  Total operations: {report['summary']['total_operations']}")
    print(f"  Total metrics: {report['summary']['total_metrics']}")
    print(f"  Overall error rate: {report['summary']['overall_error_rate']:.1f}%")
    
    # Test Phase 4A features
    print("\nTesting Phase 4A features...")
    
    # Test symbol processing tracking
    for symbol in ['ES1!', 'NQ1!', 'EURUSD']:
        correlation_id = tracker.create_correlation_id()
        tracker.record_symbol_processing_time(symbol, 15 + hash(symbol) % 10, correlation_id)
        
        # Test symbol stats
        stats = tracker.get_symbol_performance_stats(symbol)
        print(f"  {symbol}: {stats.get('mean_processing_time_ms', 0):.2f}ms avg")
    
    # Test optimization change logging
    tracker.log_optimization_change(
        "feature_optimization",
        "feature_engineering",
        "Optimized technical indicator calculations",
        {'latency_ms': 50, 'memory_mb': 500},
        {'latency_ms': 35, 'memory_mb': 400},
        "30% latency improvement, 20% memory reduction",
        "Revert commit abc123"
    )
    
    # Test correlation decorator
    @correlation_decorator("test_operation", "test_component", "ES1!")
    def test_decorated_function(delay=0.05, correlation_id=None):
        print(f"  Running with correlation_id: {correlation_id[:8] if correlation_id else 'None'}...")
        time.sleep(delay)
        return "success"
    
    result = test_decorated_function()
    print(f"  Decorated function result: {result}")
    
    # Test regression detection
    print("\nTesting regression detection...")
    for i in range(15):
        # Simulate gradually increasing latency
        latency = 20 + i * 2
        regression = tracker.detect_performance_regression("api_latency", latency)
        if regression:
            print(f"  Regression detected at iteration {i}: {regression['degradation_percent']:.1f}% degradation")
    
    print("\nPhase 4A Self-test complete!")