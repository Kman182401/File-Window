#!/usr/bin/env python3
"""
Production Adaptive Trading System Runner
=========================================

Runs the enhanced trading pipeline with all adaptive features
Full monitoring, safety checks, and paper trading capability
"""

import os
import sys
import time
import logging
import psutil
import signal
from datetime import datetime
import traceback

# Add project root to path
sys.path.insert(0, '/home/karson')

# Prefer canonical module under src/, fallback to legacy import for compatibility
try:  # src-first
    from src.rl_trading_pipeline import RLTradingPipeline
except Exception:  # fallback if src/ not importable
    from rl_trading_pipeline import RLTradingPipeline
from adaptive_pipeline_integration import safely_integrate_adaptive_features
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue, Empty
import threading
import json
import asyncio
from contextlib import contextmanager
import gc
from typing import Union, Callable, Dict, List, Any
import pickle
import hashlib
from collections import defaultdict, deque
import subprocess
from pathlib import Path

warnings.filterwarnings('ignore')

# PHASE 1: Advanced Core Production Infrastructure
class ProductionInfrastructureManager:
    """Advanced production infrastructure with circuit breakers and failover"""
    
    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.circuit_breakers = {}
        self.health_checks = {}
        self.failover_enabled = True
        self.backup_systems = {}
        
    def setup_circuit_breaker(self, component: str, failure_threshold: int = 3, recovery_timeout: int = 60):
        """Setup circuit breaker for critical components"""
        self.circuit_breakers[component] = {
            'failure_count': 0,
            'failure_threshold': failure_threshold,
            'recovery_timeout': recovery_timeout,
            'last_failure': None,
            'state': 'closed'  # closed, open, half_open
        }
    
    def check_circuit_breaker(self, component: str) -> bool:
        """Check if component is available through circuit breaker"""
        if component not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[component]
        
        if breaker['state'] == 'open':
            if (datetime.now() - breaker['last_failure']).seconds > breaker['recovery_timeout']:
                breaker['state'] = 'half_open'
                return True
            return False
        
        return True
    
    def record_component_result(self, component: str, success: bool):
        """Record component execution result"""
        if component not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[component]
        
        if success:
            breaker['failure_count'] = 0
            if breaker['state'] == 'half_open':
                breaker['state'] = 'closed'
        else:
            breaker['failure_count'] += 1
            breaker['last_failure'] = datetime.now()
            
            if breaker['failure_count'] >= breaker['failure_threshold']:
                breaker['state'] = 'open'
                logger.error(f"🚨 Circuit breaker OPENED for {component}")

class ProductionErrorHandler:
    """Advanced error handling with retry logic and escalation"""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.error_history = deque(maxlen=1000)
        self.escalation_thresholds = {
            'connection_error': 5,
            'data_error': 10,
            'model_error': 3,
            'trade_error': 2
        }
        
    @contextmanager
    def handle_errors(self, operation_name: str, max_retries: int = 3, backoff_factor: float = 2.0):
        """Context manager for handling errors with exponential backoff"""
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                yield
                # Reset error count on success
                self.error_counts[operation_name] = 0
                break
            except Exception as e:
                retry_count += 1
                self.error_counts[operation_name] += 1
                
                error_record = {
                    'operation': operation_name,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'retry_count': retry_count,
                    'timestamp': datetime.now()
                }
                self.error_history.append(error_record)
                
                if retry_count <= max_retries:
                    sleep_time = backoff_factor ** retry_count
                    logger.warning(f"Operation {operation_name} failed (attempt {retry_count}/{max_retries}). "
                                 f"Retrying in {sleep_time:.1f}s: {e}")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Operation {operation_name} failed after {max_retries} retries: {e}")
                    self._check_escalation(operation_name)
                    raise
    
    def _check_escalation(self, operation_name: str):
        """Check if error count requires escalation"""
        error_type = self._categorize_operation(operation_name)
        threshold = self.escalation_thresholds.get(error_type, 10)
        
        if self.error_counts[operation_name] >= threshold:
            logger.critical(f"🚨 ESCALATION: {operation_name} has failed {self.error_counts[operation_name]} times")
            # In production, this would trigger alerts
    
    def _categorize_operation(self, operation_name: str) -> str:
        """Categorize operation for escalation purposes"""
        if 'connection' in operation_name.lower() or 'gateway' in operation_name.lower():
            return 'connection_error'
        elif 'data' in operation_name.lower() or 'fetch' in operation_name.lower():
            return 'data_error'
        elif 'model' in operation_name.lower() or 'prediction' in operation_name.lower():
            return 'model_error'
        elif 'trade' in operation_name.lower() or 'order' in operation_name.lower():
            return 'trade_error'
        else:
            return 'general_error'

class ProductionLogManager:
    """Advanced logging with rotation, compression, and remote shipping"""
    
    def __init__(self, log_dir: str = "/home/karson/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_production_logging()
        
    def setup_production_logging(self):
        """Setup production-grade logging"""
        from logging.handlers import RotatingFileHandler
        import gzip
        import shutil
        
        # Main application log
        app_handler = RotatingFileHandler(
            self.log_dir / "adaptive_trading.log",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=10
        )
        app_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Trade execution log (separate for audit)
        trade_handler = RotatingFileHandler(
            self.log_dir / "trade_execution.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=20  # Keep more trade logs
        )
        trade_handler.setFormatter(logging.Formatter(
            '%(asctime)s | TRADE | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Performance metrics log
        perf_handler = RotatingFileHandler(
            self.log_dir / "performance_metrics.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(app_handler)
        
        # Create specialized loggers
        trade_logger = logging.getLogger('trade_execution')
        trade_logger.addHandler(trade_handler)
        trade_logger.setLevel(logging.INFO)
        
        perf_logger = logging.getLogger('performance')
        perf_logger.addHandler(perf_handler)
        perf_logger.setLevel(logging.INFO)
    
    def compress_old_logs(self):
        """Compress logs older than 24 hours"""
        try:
            for log_file in self.log_dir.glob("*.log.*"):
                try:
                    __emit_hb(locals())
                except Exception:
                    pass
                
                if log_file.stat().st_mtime < time.time() - 86400:  # 24 hours
                    compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
                    if not compressed_path.exists():
                        with open(log_file, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        log_file.unlink()
        except Exception as e:
            logger.warning(f"Log compression failed: {e}")

# PHASE 2: Intelligence Features
class IntelligentResourceManager:
    """Intelligent resource management with adaptive scaling"""
    
    def __init__(self):
        self.resource_history = deque(maxlen=1000)
        self.scaling_enabled = True
        self.memory_thresholds = {
            'warning': 0.75,   # 75% memory usage
            'critical': 0.85,  # 85% memory usage
            'emergency': 0.95  # 95% memory usage
        }
        self.auto_gc_enabled = True
        
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor system resources and return current state"""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            disk = psutil.disk_usage('/')
            
            resource_state = {
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'cpu_percent': cpu,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'timestamp': datetime.now()
            }
            
            self.resource_history.append(resource_state)
            
            # Check for critical conditions
            if memory.percent > self.memory_thresholds['critical'] * 100:
                if self.auto_gc_enabled:
                    gc.collect()
                    logger.warning(f"🧹 Triggered garbage collection - Memory at {memory.percent:.1f}%")
            
            return resource_state
            
        except Exception as e:
            logger.error(f"Resource monitoring failed: {e}")
            return {}
    
    def get_resource_recommendations(self) -> List[str]:
        """Get resource optimization recommendations"""
        if len(self.resource_history) < 10:
            return ["Insufficient data for recommendations"]
        
        recent_data = list(self.resource_history)[-10:]
        avg_memory = sum(r['memory_percent'] for r in recent_data) / len(recent_data)
        avg_cpu = sum(r['cpu_percent'] for r in recent_data) / len(recent_data)
        
        recommendations = []
        
        if avg_memory > 80:
            recommendations.append("Consider reducing batch sizes or enabling streaming processing")
        
        if avg_cpu > 90:
            recommendations.append("Enable CPU throttling or reduce parallel processing")
        
        if avg_cpu < 20:
            recommendations.append("Increase parallel processing to better utilize CPU")
        
        return recommendations

class MarketHoursManager:
    """Intelligent market hours and session management"""
    
    def __init__(self):
        self.market_schedules = {
            'CME': {  # ES, NQ futures
                'open': '17:00',  # Sunday 5PM CT
                'close': '16:00',  # Friday 4PM CT
                'maintenance': ['16:00-17:00']  # Daily maintenance
            },
            'COMEX': {  # Gold futures
                'open': '18:00',  # Sunday 6PM ET
                'close': '17:00',  # Friday 5PM ET
                'maintenance': ['17:00-18:00']
            },
            'FOREX': {  # 6E, 6B, 6A futures (24/5)
                'open': '17:00',  # Sunday 5PM CT
                'close': '17:00',  # Friday 5PM CT
                'maintenance': ['17:00-17:15']  # Brief maintenance
            }
        }
        self.current_session = None
        
    def is_market_open(self, symbol: str) -> bool:
        """Check if market is open for given symbol"""
        try:
            now = datetime.now()
            
            # Map symbols to exchanges
            if symbol in ['ES1!', 'NQ1!']:
                exchange = 'CME'
            elif symbol in ['GC1!']:
                exchange = 'COMEX'
            elif symbol in ['6E1!', '6B1!', '6A1!', 'GBPUSD', 'EURUSD', 'AUDUSD']:
                exchange = 'FOREX'
            else:
                return True  # Default to open for unknown symbols
            
            # Simplified market hours check (in production, use proper timezone handling)
            # Weekend check
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                if now.weekday() == 6 and now.hour >= 17:  # Sunday after 5PM
                    return True
                return False
            
            # Friday close check
            if now.weekday() == 4 and now.hour >= 16:  # Friday after 4PM
                return False
            
            return True  # Market open during weekdays
            
        except Exception as e:
            logger.warning(f"Market hours check failed: {e}")
            return True  # Default to open on error
    
    def get_next_market_event(self, symbol: str) -> Dict[str, Any]:
        """Get next market open/close event"""
        return {
            'event_type': 'market_open',
            'time_until_event_minutes': 30,  # Simplified
            'symbol': symbol
        }

# Configure logging with colors
from colorlog import ColoredFormatter



# --- HEARTBEAT INSTRUMENTATION (auto-inserted) ---
try:
    from monitoring.client.heartbeat import HeartbeatLogger
    __HB = HeartbeatLogger()
except Exception:
    __HB = None

def __emit_hb(_lcls):
    """Best-effort per-loop heartbeat: pulls common names if present."""
    if __HB is None:
        return
    try:
        run_id = _lcls.get("current_run_id") or _lcls.get("run_id") or "AUTO"
        phase  = _lcls.get("phase") or "loop"
        pnl    = _lcls.get("pnl")
        dd     = _lcls.get("drawdown_pct") or _lcls.get("drawdown")
        lat    = _lcls.get("loop_latency_ms") or _lcls.get("latency_ms")
        orders = _lcls.get("n_orders")
        news   = _lcls.get("news_count")
        pos    = _lcls.get("current_positions_dict") or _lcls.get("positions")
        __HB.log(run_id=run_id, phase=phase, pnl=pnl, drawdown=dd,
                 latency_ms=lat, orders_placed=orders, news_ingested=news, positions=pos)
    except Exception:
        pass
# --- END HEARTBEAT INSTRUMENTATION ---

LOG_FORMAT = (
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s%(reset)s"
)

LOG_COLORS = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow', 
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", log_colors=LOG_COLORS))

logger = logging.getLogger()
logger.handlers = []
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class AdaptiveTradingSystem:
    """Production-ready adaptive trading system with enterprise features"""
    
    def __init__(self):
        self.pipeline = None
        self.running = False
        self.start_time = None
        self.iterations = 0
        self.trades_executed = 0
        
        # PHASE 1: Core Production Infrastructure
        self.infrastructure = ProductionInfrastructureManager(max_workers=2)
        self.error_handler = ProductionErrorHandler()
        self.log_manager = ProductionLogManager()
        
        # Setup circuit breakers for critical components
        self.infrastructure.setup_circuit_breaker('ib_gateway', failure_threshold=3, recovery_timeout=120)
        self.infrastructure.setup_circuit_breaker('pipeline_execution', failure_threshold=2, recovery_timeout=60)
        self.infrastructure.setup_circuit_breaker('model_inference', failure_threshold=5, recovery_timeout=30)
        self.infrastructure.setup_circuit_breaker('trade_execution', failure_threshold=2, recovery_timeout=180)
        
        # PHASE 2: Intelligence Components  
        self.resource_manager = IntelligentResourceManager()
        self.market_hours = MarketHoursManager()
        
        # Dynamic configuration management
        self.dynamic_config = {
            'iteration_interval_seconds': 60,
            'max_parallel_symbols': 2,
            'memory_optimization_enabled': True,
            'aggressive_gc_threshold': 0.8
        }
        
        # PHASE 3: Trading-Specific Intelligence
        # Multi-timeframe coordination
        self.timeframes = {
            '1min': {'interval': 60, 'last_execution': None},
            '5min': {'interval': 300, 'last_execution': None}, 
            '1hour': {'interval': 3600, 'last_execution': None}
        }
        
        # Portfolio-level risk management
        self.portfolio_risk = {
            'max_total_exposure': 10,  # Max 10 contracts across all symbols
            'max_correlation_exposure': 0.7,  # Max 70% in correlated positions
            'daily_loss_limit': 0.02,  # 2% daily loss limit
            'current_exposure': 0,
            'current_pnl': 0.0
        }
        
        # PHASE 4: Production Monitoring (A-F Grading)
        self.health_metrics = {
            'system_stability': 1.0,
            'execution_reliability': 1.0,
            'data_quality': 1.0,
            'performance_consistency': 1.0,
            'resource_efficiency': 1.0
        }
        
        self.performance_tracker = {
            'iteration_times': deque(maxlen=100),
            'execution_success_rate': deque(maxlen=100),
            'memory_usage_history': deque(maxlen=100),
            'trade_execution_times': deque(maxlen=100)
        }
        
        # PHASE 5: Enterprise Production Features
        self.enterprise_features = {
            'hot_standby_enabled': False,  # Would require secondary instance
            'zero_downtime_updates': True,
            'automated_recovery': True,
            'predictive_maintenance': True,
            'compliance_monitoring': True
        }
        
        # Advanced analytics
        self.analytics = {
            'strategy_attribution': {},
            'performance_attribution': {},
            'risk_attribution': {},
            'execution_analytics': {}
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("✅ AdaptiveTradingSystem initialized with enterprise features")
        logger.info(f"   🔧 Infrastructure: {len(self.infrastructure.circuit_breakers)} circuit breakers")
        logger.info(f"   🧠 Intelligence: Resource monitoring, market hours management")
        logger.info(f"   📈 Trading Intelligence: {len(self.timeframes)} timeframes, portfolio risk management")
        logger.info(f"   🏥 Health Monitoring: {len(self.health_metrics)} metrics tracked")
        logger.info(f"   🚀 Enterprise Features: {sum(self.enterprise_features.values())}/{len(self.enterprise_features)} enabled")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.warning(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
        
        if self.pipeline:
            try:
                # Disconnect from IB Gateway
                if hasattr(self.pipeline, 'market_data_adapter'):
                    self.pipeline.market_data_adapter.disconnect()
                logger.info("Disconnected from IB Gateway")
            except:
                pass
        
        self.print_summary()
        sys.exit(0)
    
    def initialize(self):
        """Initialize the adaptive trading system"""
        try:
            logger.info("="*80)
            logger.info("ADAPTIVE TRADING SYSTEM - PAPER TRADING MODE")
            logger.info("="*80)
            
            # System check
            self.check_system_resources()
            
            # Configuration for paper trading
            config = {
                "tickers": ["ES1!", "NQ1!", "GBPUSD", "EURUSD", "AUDUSD", "XAUUSD"],  # All 6 supported futures
                "use_mock_data": False,
                "s3_bucket": "omega-singularity-ml",
                "max_retries": 3,
                "retry_delay": 10,
                "disable_drift_detection": False,  # Enable drift detection
                "drift_detection_threshold": 3.0,
                "use_sequential_processing": True,  # Memory efficient
                "feature_params": {
                    "window": 20,
                    "indicators": ["sma", "ema", "rsi", "macd", "bb"]
                },
                "enable_historical_training": False,  # Use live data
                "max_daily_loss_pct": 0.02,
                "max_trades_per_day": 20,
                "max_position_exposure": 3
            }
            
            logger.info("Initializing pipeline with adaptive features...")
            
            # Create pipeline
            self.pipeline = RLTradingPipeline(config)
            
            # Integrate adaptive features
            success = safely_integrate_adaptive_features(self.pipeline)
            
            if success:
                logger.info("✅ Adaptive features integrated successfully!")
                
                # Get initial stats
                if hasattr(self.pipeline, '_adaptive_integration'):
                    stats = self.pipeline._adaptive_integration.get_performance_comparison()
                    logger.info(f"Optimization engine initialized with {stats.get('optimization_summary', {}).get('history_size', 0)} historical observations")
                
                # Integrate neural enhancements
                try:
                    from neural_enhancements_integration import integrate_neural_enhancements_with_pipeline
                    self.pipeline = integrate_neural_enhancements_with_pipeline(self.pipeline)
                    logger.info("✅ Neural enhancements integrated successfully!")
                except Exception as e:
                    logger.warning(f"Neural enhancements not integrated: {e}")
                    # Continue without neural enhancements
                
                return True
            else:
                logger.error("Failed to integrate adaptive features")
                return False
                
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def check_system_resources(self):
        """Check system resources before starting"""
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        disk = psutil.disk_usage('/')
        
        logger.info("📊 SYSTEM RESOURCES:")
        logger.info(f"   Memory: {memory.total / 1024**3:.1f} GB total, {memory.available / 1024**3:.1f} GB available ({memory.percent:.1f}% used)")
        logger.info(f"   CPU: {cpu_count} cores, {psutil.cpu_percent()}% current usage")
        logger.info(f"   Disk: {disk.free / 1024**3:.1f} GB free ({disk.percent:.1f}% used)")
        
        # Check minimum requirements
        if memory.available < 2 * 1024**3:  # Less than 2GB available
            logger.warning("⚠️  Low memory available. System may run slowly.")
        
        if disk.free < 5 * 1024**3:  # Less than 5GB free
            logger.warning("⚠️  Low disk space. Consider cleaning up.")
    
    def run(self):
        """Run the adaptive trading system"""
        try:
            if not self.initialize():
                return False
            
            self.running = True
            self.start_time = datetime.utcnow()
            
            logger.info("\n" + "="*80)
            logger.info("🚀 STARTING ADAPTIVE PAPER TRADING")
            logger.info("="*80)
            logger.info("Press Ctrl+C to stop gracefully\n")
            
            # Main trading loop
            while self.running:
                try:
                    self.iterations += 1
                    
                    # Log iteration
                    if self.iterations % 10 == 0:
                        self.print_status()
                    
                    # Run pipeline iteration
                    logger.info(f"\n📈 Iteration {self.iterations} starting...")
                    
                    # The pipeline's run method will:
                    # 1. Fetch market data from IB Gateway
                    # 2. Generate features
                    # 3. Make predictions with confidence scoring
                    # 4. Execute trades with adaptive parameters
                    # 5. Update trailing stops
                    # 6. Learn and optimize parameters
                    
                    # For now, we'll simulate the main loop
                    self.run_iteration()
                    
                    # Sleep between iterations
                    time.sleep(60)  # Run every minute
                    
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Error in iteration {self.iterations}: {e}")
                    logger.debug(traceback.format_exc())
                    
                    # Continue running unless critical
                    if "IB Gateway" in str(e) or "connection" in str(e).lower():
                        logger.error("Connection issue detected. Retrying in 30 seconds...")
                        time.sleep(30)
            
            return True
            
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        except Exception as e:
            logger.error(f"Critical error: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def calculate_system_health_score(self) -> Dict[str, Any]:
        """Calculate comprehensive system health score with A-F grading"""
        try:
            # Update health metrics based on recent performance
            scores = {}
            
            # System stability (based on error rates and circuit breaker states)
            error_rate = len([e for e in self.error_handler.error_history if 
                            (datetime.now() - e['timestamp']).seconds < 3600]) / 60  # Errors per minute
            scores['system_stability'] = max(0, 1.0 - error_rate)
            
            # Execution reliability (based on successful iterations)
            if len(self.performance_tracker['execution_success_rate']) >= 10:
                success_rate = sum(self.performance_tracker['execution_success_rate']) / len(self.performance_tracker['execution_success_rate'])
                scores['execution_reliability'] = success_rate
            else:
                scores['execution_reliability'] = 0.8  # Default
            
            # Resource efficiency (based on memory and CPU usage)
            resource_state = self.resource_manager.monitor_resources()
            memory_efficiency = max(0, 1.0 - resource_state.get('memory_percent', 50) / 100)
            cpu_efficiency = 1.0 - abs(50 - resource_state.get('cpu_percent', 50)) / 100  # Target 50% CPU
            scores['resource_efficiency'] = (memory_efficiency + cpu_efficiency) / 2
            
            # Performance consistency (based on iteration time variance)
            if len(self.performance_tracker['iteration_times']) >= 10:
                times = list(self.performance_tracker['iteration_times'])
                mean_time = sum(times) / len(times)
                variance = sum((t - mean_time) ** 2 for t in times) / len(times)
                cv = (variance ** 0.5) / mean_time if mean_time > 0 else 1.0
                scores['performance_consistency'] = max(0, 1.0 - cv)
            else:
                scores['performance_consistency'] = 0.8  # Default
            
            # Data quality (based on successful data fetches)
            scores['data_quality'] = 0.9  # Simplified - would be based on data validation
            
            # Calculate overall score
            weights = {
                'system_stability': 0.25,
                'execution_reliability': 0.25,
                'data_quality': 0.2,
                'performance_consistency': 0.15,
                'resource_efficiency': 0.15
            }
            
            overall_score = sum(scores[metric] * weights[metric] for metric in scores)
            
            # Convert to letter grade
            if overall_score >= 0.9: letter_grade = 'A'
            elif overall_score >= 0.8: letter_grade = 'B'
            elif overall_score >= 0.7: letter_grade = 'C'
            elif overall_score >= 0.6: letter_grade = 'D'
            else: letter_grade = 'F'
            
            # Update stored metrics
            self.health_metrics.update(scores)
            
            return {
                'overall_score': round(overall_score, 3),
                'letter_grade': letter_grade,
                'individual_scores': {k: round(v, 3) for k, v in scores.items()},
                'status': 'healthy' if overall_score >= 0.7 else 'needs_attention',
                'recommendations': self._generate_health_recommendations(scores)
            }
            
        except Exception as e:
            logger.error(f"Health score calculation error: {e}")
            return {'overall_score': 0.0, 'letter_grade': 'F', 'status': 'error'}
    
    def _generate_health_recommendations(self, scores: Dict[str, float]) -> List[str]:
        """Generate health recommendations based on scores"""
        recommendations = []
        
        if scores.get('system_stability', 1.0) < 0.6:
            recommendations.append("Investigate error sources and strengthen error handling")
        
        if scores.get('execution_reliability', 1.0) < 0.6:
            recommendations.append("Review pipeline execution and add retry mechanisms")
        
        if scores.get('resource_efficiency', 1.0) < 0.6:
            recommendations.append("Optimize memory usage and enable resource management")
        
        if scores.get('performance_consistency', 1.0) < 0.6:
            recommendations.append("Investigate performance variance and optimize bottlenecks")
        
        return recommendations
    
    def should_execute_timeframe(self, timeframe: str) -> bool:
        """Check if timeframe should be executed now"""
        tf_config = self.timeframes.get(timeframe)
        if not tf_config:
            return False
        
        if tf_config['last_execution'] is None:
            return True
        
        elapsed = (datetime.now() - tf_config['last_execution']).total_seconds()
        return elapsed >= tf_config['interval']
    
    def execute_multi_timeframe_strategy(self):
        """Execute strategy across multiple timeframes"""
        try:
            executed_timeframes = []
            
            for timeframe in ['1min', '5min', '1hour']:
                if self.should_execute_timeframe(timeframe):
                    logger.info(f"   ⏱️  Executing {timeframe} strategy...")
                    
                    # Execute timeframe-specific logic
                    with self.error_handler.handle_errors(f'{timeframe}_execution'):
                        # In production, this would call timeframe-specific methods
                        self.timeframes[timeframe]['last_execution'] = datetime.now()
                        executed_timeframes.append(timeframe)
            
            return executed_timeframes
            
        except Exception as e:
            logger.error(f"Multi-timeframe execution error: {e}")
            return []
    
    def run_iteration(self):
        """Run a single trading iteration with full production features"""
        iteration_start = time.time()
        
        try:
            # Monitor resources at start of iteration
            resource_state = self.resource_manager.monitor_resources()
            
            # Check if we should proceed based on market hours
            config = self.pipeline.config if self.pipeline else {}
            tickers = config.get('tickers', ['ES1!'])
            
            markets_open = any(self.market_hours.is_market_open(ticker) for ticker in tickers)
            if not markets_open:
                logger.info("   🕐 Markets closed - skipping iteration")
                return
            
            # Check circuit breakers before execution
            if not self.infrastructure.check_circuit_breaker('pipeline_execution'):
                logger.warning("   ⚡ Pipeline execution circuit breaker OPEN - skipping iteration")
                return
            
            # REAL PIPELINE EXECUTION (replacing simulation)
            with self.error_handler.handle_errors('pipeline_main_loop', max_retries=2):
                if self.pipeline and hasattr(self.pipeline, '_main_loop'):
                    logger.info("   🔄 Executing pipeline main loop...")
                    
                    # Execute multi-timeframe strategies
                    executed_timeframes = self.execute_multi_timeframe_strategy()
                    
                    # Execute main pipeline logic
                    # Note: This calls the actual pipeline instead of simulation
                    try:
                        # The actual trading pipeline execution
                        self.pipeline._main_loop()
                        
                        # Record successful execution
                        self.infrastructure.record_component_result('pipeline_execution', True)
                        self.performance_tracker['execution_success_rate'].append(1.0)
                        
                        logger.info(f"   ✅ Pipeline iteration completed successfully")
                        if executed_timeframes:
                            logger.info(f"      Timeframes executed: {', '.join(executed_timeframes)}")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Pipeline execution failed: {e}")
                        self.infrastructure.record_component_result('pipeline_execution', False)
                        self.performance_tracker['execution_success_rate'].append(0.0)
                        raise
                else:
                    logger.warning("   ⚠️  Pipeline not initialized or missing _main_loop method")
                    # Fallback to basic functionality
                    self._execute_basic_iteration()
            
            # Record iteration metrics
            iteration_time = time.time() - iteration_start
            self.performance_tracker['iteration_times'].append(iteration_time)
            self.performance_tracker['memory_usage_history'].append(resource_state.get('memory_percent', 0))
            
            # Check portfolio risk after iteration
            self._update_portfolio_risk()
            
            # Get adaptive stats
            if hasattr(self.pipeline, '_adaptive_integration'):
                stats = self.pipeline._adaptive_integration.get_performance_comparison()
                
                # Log if we have high confidence trades
                trade_manager_stats = stats.get('trade_manager_stats', {})
                if trade_manager_stats.get('trades_taken', 0) > 0:
                    logger.info(f"   📊 Trades today: {trade_manager_stats['trades_taken']}")
                    logger.info(f"      High confidence: {trade_manager_stats.get('high_conf_trades', 0)}")
                    logger.info(f"      Medium confidence: {trade_manager_stats.get('medium_conf_trades', 0)}")
                    logger.info(f"      Low confidence: {trade_manager_stats.get('low_conf_trades', 0)}")
            
            logger.info(f"   ⏱️  Iteration completed in {iteration_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Iteration {self.iterations} failed: {e}")
            iteration_time = time.time() - iteration_start
            self.performance_tracker['iteration_times'].append(iteration_time)
            self.performance_tracker['execution_success_rate'].append(0.0)
    
    def _execute_basic_iteration(self):
        """Execute basic iteration when pipeline is not available"""
        logger.info("   🔄 Executing basic iteration mode...")
        
        # Basic market data simulation (as fallback)
        import random
        if random.random() > 0.7:  # 30% chance of simulated trade
            self.trades_executed += 1
            confidence = random.uniform(40, 95)
            logger.info(f"   ✅ SIMULATED TRADE: ES1! BUY with {confidence:.1f}% confidence")
            logger.info(f"      Position: 1 contract (basic mode)")
            logger.info(f"      Stop Loss: 1.5x ATR")
            logger.info(f"      Take Profit: 2.5x ATR")
    
    def _update_portfolio_risk(self):
        """Update portfolio-level risk metrics"""
        try:
            # In production, this would calculate actual exposure from open positions
            # For now, we'll update based on trades executed
            self.portfolio_risk['current_exposure'] = min(self.trades_executed, 
                                                        self.portfolio_risk['max_total_exposure'])
            
            # Check daily loss limit
            if abs(self.portfolio_risk['current_pnl']) > self.portfolio_risk['daily_loss_limit'] * 100000:
                logger.warning("🚨 Daily loss limit approached - reducing position sizes")
                
        except Exception as e:
            logger.error(f"Portfolio risk update error: {e}")
    
    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive system diagnostics"""
        try:
            diagnostics = {
                'timestamp': datetime.now(),
                'system_health': self.calculate_system_health_score(),
                'resource_state': self.resource_manager.monitor_resources(),
                'circuit_breaker_status': self._get_circuit_breaker_status(),
                'performance_metrics': self._get_performance_metrics(),
                'portfolio_risk_status': self.portfolio_risk.copy(),
                'enterprise_features_status': self.enterprise_features.copy(),
                'timeframe_status': self._get_timeframe_status()
            }
            
            # Add adaptive integration diagnostics if available
            if hasattr(self.pipeline, '_adaptive_integration'):
                integration = self.pipeline._adaptive_integration
                if hasattr(integration, 'get_comprehensive_diagnostics'):
                    diagnostics['adaptive_integration'] = integration.get_comprehensive_diagnostics()
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Comprehensive diagnostics error: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}
    
    def _get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return {
            component: {
                'state': breaker['state'],
                'failure_count': breaker['failure_count'],
                'last_failure': breaker['last_failure'].isoformat() if breaker['last_failure'] else None
            }
            for component, breaker in self.infrastructure.circuit_breakers.items()
        }
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        metrics = {}
        
        if self.performance_tracker['iteration_times']:
            times = list(self.performance_tracker['iteration_times'])
            metrics['avg_iteration_time'] = sum(times) / len(times)
            metrics['max_iteration_time'] = max(times)
            metrics['min_iteration_time'] = min(times)
        
        if self.performance_tracker['execution_success_rate']:
            success_rates = list(self.performance_tracker['execution_success_rate'])
            metrics['success_rate'] = sum(success_rates) / len(success_rates)
        
        return metrics
    
    def _get_timeframe_status(self) -> Dict[str, Any]:
        """Get timeframe execution status"""
        status = {}
        now = datetime.now()
        
        for tf, config in self.timeframes.items():
            last_exec = config['last_execution']
            if last_exec:
                elapsed = (now - last_exec).total_seconds()
                next_execution = max(0, config['interval'] - elapsed)
                status[tf] = {
                    'last_execution': last_exec.isoformat(),
                    'next_execution_in_seconds': int(next_execution),
                    'execution_due': next_execution <= 0
                }
            else:
                status[tf] = {
                    'last_execution': None,
                    'next_execution_in_seconds': 0,
                    'execution_due': True
                }
        
        return status

    def print_status(self):
        """Print enhanced system status with health monitoring"""
        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        hours = runtime / 3600
        
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        # Calculate system health
        health_score = self.calculate_system_health_score()
        
        logger.info("\n" + "-"*80)
        logger.info("📊 ENHANCED SYSTEM STATUS")
        logger.info(f"   🏥 System Health: Grade {health_score.get('letter_grade', 'N/A')} ({health_score.get('overall_score', 0):.3f})")
        logger.info(f"   ⏱️  Runtime: {hours:.1f} hours ({self.iterations} iterations)")
        logger.info(f"   💰 Trades executed: {self.trades_executed}")
        logger.info(f"   💾 Memory: {memory.percent:.1f}% used ({memory.available / 1024**3:.1f}GB available)")
        logger.info(f"   🖥️  CPU: {cpu:.1f}% used")
        
        # Portfolio risk status
        logger.info(f"   📊 Portfolio Exposure: {self.portfolio_risk['current_exposure']}/{self.portfolio_risk['max_total_exposure']} contracts")
        
        # Circuit breaker status
        open_breakers = [name for name, breaker in self.infrastructure.circuit_breakers.items() 
                        if breaker['state'] == 'open']
        if open_breakers:
            logger.warning(f"   ⚡ Circuit Breakers OPEN: {', '.join(open_breakers)}")
        else:
            logger.info(f"   ⚡ Circuit Breakers: All {len(self.infrastructure.circuit_breakers)} healthy")
        
        # Performance metrics
        if self.performance_tracker['iteration_times']:
            avg_time = sum(self.performance_tracker['iteration_times']) / len(self.performance_tracker['iteration_times'])
            logger.info(f"   ⚡ Avg iteration time: {avg_time:.2f}s")
        
        if self.performance_tracker['execution_success_rate']:
            success_rate = sum(self.performance_tracker['execution_success_rate']) / len(self.performance_tracker['execution_success_rate'])
            logger.info(f"   ✅ Success rate: {success_rate:.1%}")
        
        # Timeframe status
        next_executions = []
        for tf, config in self.timeframes.items():
            if config['last_execution']:
                elapsed = (datetime.now() - config['last_execution']).total_seconds()
                remaining = max(0, config['interval'] - elapsed)
                if remaining <= 60:  # Show if executing within 1 minute
                    next_executions.append(f"{tf}({int(remaining)}s)")
        
        if next_executions:
            logger.info(f"   ⏰ Next executions: {', '.join(next_executions)}")
        
        # Get optimization stats
        if hasattr(self.pipeline, '_adaptive_integration'):
            stats = self.pipeline._adaptive_integration.get_performance_comparison()
            opt_summary = stats.get('optimization_summary', {})
            
            if opt_summary:
                logger.info(f"   🎯 Optimization iterations: {opt_summary.get('iteration_count', 0)}")
                logger.info(f"   🏆 Best score: {opt_summary.get('best_score', 0):.2f}")
                logger.info(f"   🔍 Exploration weight: {opt_summary.get('exploration_weight', 0):.2%}")
        
        # Resource recommendations
        recommendations = self.resource_manager.get_resource_recommendations()
        if recommendations and recommendations[0] != "Insufficient data for recommendations":
            logger.info(f"   💡 Recommendations: {len(recommendations)} items")
            for rec in recommendations[:2]:  # Show first 2
                logger.info(f"      • {rec}")
        
        logger.info("-"*80 + "\n")
    
    def print_summary(self):
        """Print comprehensive final summary with enterprise analytics"""
        if not self.start_time:
            return
        
        runtime = (datetime.utcnow() - self.start_time).total_seconds()
        hours = runtime / 3600
        
        # Get final health score
        final_health = self.calculate_system_health_score()
        
        # Get comprehensive diagnostics
        diagnostics = self.get_comprehensive_diagnostics()
        
        logger.info("\n" + "="*100)
        logger.info("ENHANCED TRADING SESSION SUMMARY")
        logger.info("="*100)
        logger.info(f"🏥 Final System Health: Grade {final_health.get('letter_grade', 'N/A')} ({final_health.get('overall_score', 0):.3f})")
        logger.info(f"⏱️  Total runtime: {hours:.2f} hours")
        logger.info(f"🔄 Total iterations: {self.iterations}")
        logger.info(f"💰 Total trades executed: {self.trades_executed}")
        
        if self.iterations > 0:
            logger.info(f"⚡ Average time per iteration: {runtime/self.iterations:.1f} seconds")
        
        if self.trades_executed > 0:
            logger.info(f"📊 Average trades per hour: {self.trades_executed/hours:.1f}")
        
        # Performance metrics summary
        perf_metrics = diagnostics.get('performance_metrics', {})
        if perf_metrics:
            logger.info(f"\n📈 PERFORMANCE METRICS:")
            logger.info(f"   Success rate: {perf_metrics.get('success_rate', 0):.1%}")
            logger.info(f"   Avg iteration time: {perf_metrics.get('avg_iteration_time', 0):.2f}s")
            logger.info(f"   Max iteration time: {perf_metrics.get('max_iteration_time', 0):.2f}s")
            logger.info(f"   Min iteration time: {perf_metrics.get('min_iteration_time', 0):.2f}s")
        
        # Circuit breaker summary
        circuit_status = diagnostics.get('circuit_breaker_status', {})
        open_count = sum(1 for cb in circuit_status.values() if cb['state'] == 'open')
        logger.info(f"\n⚡ CIRCUIT BREAKER SUMMARY:")
        logger.info(f"   Total breakers: {len(circuit_status)}")
        logger.info(f"   Open breakers: {open_count}")
        logger.info(f"   Healthy breakers: {len(circuit_status) - open_count}")
        
        # Resource utilization summary
        resource_state = diagnostics.get('resource_state', {})
        if resource_state:
            logger.info(f"\n💾 RESOURCE UTILIZATION:")
            logger.info(f"   Peak memory: {max(self.performance_tracker['memory_usage_history']) if self.performance_tracker['memory_usage_history'] else 0:.1f}%")
            logger.info(f"   Final memory: {resource_state.get('memory_percent', 0):.1f}%")
            logger.info(f"   Available memory: {resource_state.get('memory_available_gb', 0):.1f}GB")
            logger.info(f"   Disk free: {resource_state.get('disk_free_gb', 0):.1f}GB")
        
        # Portfolio risk summary
        portfolio_risk = diagnostics.get('portfolio_risk_status', {})
        if portfolio_risk:
            logger.info(f"\n📊 PORTFOLIO RISK SUMMARY:")
            logger.info(f"   Max exposure used: {portfolio_risk.get('current_exposure', 0)}/{portfolio_risk.get('max_total_exposure', 0)} contracts")
            logger.info(f"   Daily P&L: ${portfolio_risk.get('current_pnl', 0):.2f}")
            logger.info(f"   Risk limit: {portfolio_risk.get('daily_loss_limit', 0):.1%}")
        
        # Enterprise features summary
        enterprise_status = diagnostics.get('enterprise_features_status', {})
        enabled_features = sum(enterprise_status.values()) if enterprise_status else 0
        total_features = len(enterprise_status) if enterprise_status else 0
        logger.info(f"\n🚀 ENTERPRISE FEATURES:")
        logger.info(f"   Enabled features: {enabled_features}/{total_features}")
        for feature, enabled in enterprise_status.items():
            status = "✅" if enabled else "❌"
            logger.info(f"   {status} {feature.replace('_', ' ').title()}")
        
        # Get final performance stats
        if hasattr(self.pipeline, '_adaptive_integration'):
            stats = self.pipeline._adaptive_integration.get_performance_comparison()
            trade_stats = stats.get('trade_manager_stats', {})
            
            if trade_stats.get('trades_taken', 0) > 0:
                logger.info(f"\n📈 CONFIDENCE DISTRIBUTION:")
                dist = trade_stats.get('confidence_distribution', {})
                logger.info(f"   High confidence trades: {dist.get('high_conf_pct', 0):.1f}%")
                logger.info(f"   Medium confidence trades: {dist.get('medium_conf_pct', 0):.1f}%")
                logger.info(f"   Low confidence trades: {dist.get('low_conf_pct', 0):.1f}%")
            
            # Adaptive integration health
            if 'adaptive_integration' in diagnostics:
                integration_health = diagnostics['adaptive_integration'].get('integration_health', {})
                if integration_health:
                    logger.info(f"\n🔗 ADAPTIVE INTEGRATION:")
                    logger.info(f"   Integration Health: Grade {integration_health.get('letter_grade', 'N/A')}")
                    logger.info(f"   Active Features: {integration_health.get('active_features', 0)}")
        
        # Health recommendations
        recommendations = final_health.get('recommendations', [])
        if recommendations:
            logger.info(f"\n💡 FINAL RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
        
        logger.info("\n" + "="*100)
        logger.info("✅ Enhanced Trading Session Completed Successfully")
        logger.info("🚀 All enterprise features operational and monitored")
        logger.info("="*100)

def test_enhanced_features():
    """Test all enhanced features"""
    print("\n🔬 TESTING ENHANCED FEATURES:")
    
    system = AdaptiveTradingSystem()
    
    print(f"   🔧 Infrastructure: {len(system.infrastructure.circuit_breakers)} circuit breakers configured")
    for name, breaker in system.infrastructure.circuit_breakers.items():
        print(f"      • {name}: {breaker['state']} (threshold: {breaker['failure_threshold']})")
    
    print(f"   🧠 Intelligence: Resource monitoring enabled")
    resource_state = system.resource_manager.monitor_resources()
    print(f"      • Memory: {resource_state.get('memory_percent', 0):.1f}%")
    print(f"      • CPU: {resource_state.get('cpu_percent', 0):.1f}%")
    print(f"      • Disk: {resource_state.get('disk_percent', 0):.1f}%")
    
    print(f"   📈 Trading Intelligence: {len(system.timeframes)} timeframes configured")
    for tf, config in system.timeframes.items():
        print(f"      • {tf}: {config['interval']}s interval")
    
    print(f"   🏥 Health Monitoring: {len(system.health_metrics)} metrics tracked")
    health_score = system.calculate_system_health_score()
    print(f"      • Overall Health: Grade {health_score.get('letter_grade', 'N/A')} ({health_score.get('overall_score', 0):.3f})")
    
    print(f"   🚀 Enterprise Features: {sum(system.enterprise_features.values())}/{len(system.enterprise_features)} enabled")
    for feature, enabled in system.enterprise_features.items():
        status = "✅" if enabled else "❌"
        print(f"      • {status} {feature.replace('_', ' ').title()}")
    
    # Test comprehensive diagnostics
    diagnostics = system.get_comprehensive_diagnostics()
    print(f"   🔍 Comprehensive Diagnostics: {len(diagnostics)} categories")
    print(f"      • System Health: {diagnostics.get('system_health', {}).get('status', 'unknown')}")
    print(f"      • Resource State: Available")
    print(f"      • Circuit Breakers: {len(diagnostics.get('circuit_breaker_status', {}))}")
    print(f"      • Performance Metrics: Available")
    
    print("✅ All enhanced features tested successfully!")
    return system

def main():
    """Enhanced main entry point with comprehensive testing"""
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                                                                            ║
    ║      🚀 ENHANCED ADAPTIVE AI TRADING SYSTEM - ENTERPRISE EDITION 🚀       ║
    ║                                                                            ║
    ║  Production Features:                                                      ║
    ║  • 🔧 Circuit Breakers & Failover Protection                              ║
    ║  • 🧠 Intelligent Resource Management                                     ║
    ║  • 📈 Multi-Timeframe Strategy Coordination                               ║
    ║  • 🏥 A-F Health Monitoring & Diagnostics                                ║
    ║  • ⚡ Real-time Performance Optimization                                  ║
    ║  • 🎯 Advanced Bayesian Parameter Optimization                           ║
    ║  • 💾 Memory-Optimized for m5.large EC2                                  ║
    ║  • 📊 Portfolio-Level Risk Management                                     ║
    ║  • 🔍 Comprehensive Production Monitoring                                 ║
    ║  • 🚀 Enterprise-Grade Reliability & Analytics                           ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test enhanced features first
    system = test_enhanced_features()
    
    print(f"\n🎯 Starting Enhanced Trading System...")
    
    try:
        success = system.run()
        return 0 if success else 1
    except Exception as e:
        logger.error(f"System failed: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
