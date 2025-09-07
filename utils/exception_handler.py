"""
Advanced Exception Handling System for Trading Platform

This module provides comprehensive exception handling, error recovery, and 
resilience patterns for the trading system. It includes custom exception classes,
automatic retry logic, circuit breakers, and detailed error tracking.

Key Features:
- Custom exception hierarchy for trading-specific errors
- Automatic retry with exponential backoff
- Circuit breaker pattern for failing services
- Error context preservation and detailed logging
- Graceful degradation strategies
- Error rate monitoring and alerting
- Integration with monitoring systems

Author: AI Trading System  
Version: 2.0.0
"""

import sys
import time
import logging
import traceback
import functools
import threading
import warnings
from typing import Type, Callable, Any, Optional, Dict, List, Union, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import json
import inspect

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for prioritization and handling"""
    LOW = 1      # Informational, system continues normally
    MEDIUM = 2   # Warning, degraded performance possible
    HIGH = 3     # Error, feature disabled but system continues
    CRITICAL = 4 # Critical, immediate attention required  
    FATAL = 5    # Fatal, system must stop
    
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented
    
    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented
    
    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented
    
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented


class RecoveryStrategy(Enum):
    """Strategies for error recovery"""
    RETRY = "retry"              # Retry the operation
    FALLBACK = "fallback"        # Use fallback value/method
    CIRCUIT_BREAK = "circuit_break"  # Stop attempting temporarily
    GRACEFUL_DEGRADE = "degrade"     # Reduce functionality
    FAIL_FAST = "fail_fast"          # Fail immediately
    IGNORE = "ignore"                # Log and continue


@dataclass
class ErrorContext:
    """Context information for an error"""
    timestamp: datetime = field(default_factory=datetime.now)
    function_name: str = ""
    module_name: str = ""
    line_number: int = 0
    error_type: str = ""
    error_message: str = ""
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


# ============================================================================
# Custom Exception Classes
# ============================================================================

class TradingSystemError(Exception):
    """Base exception for all trading system errors"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()


class IBKRConnectionError(TradingSystemError):
    """IBKR connection-related errors"""
    
    def __init__(self, message: str, retry_count: int = 0, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, context)
        self.retry_count = retry_count


class DataFetchError(TradingSystemError):
    """Data fetching and ingestion errors"""
    
    def __init__(self, message: str, ticker: Optional[str] = None,
                 data_source: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, context)
        self.ticker = ticker
        self.data_source = data_source


class ModelTrainingError(TradingSystemError):
    """Model training and prediction errors"""
    
    def __init__(self, message: str, model_type: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, context)
        self.model_type = model_type


class OrderExecutionError(TradingSystemError):
    """Order execution and trading errors"""
    
    def __init__(self, message: str, order_id: Optional[str] = None,
                 ticker: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, context)
        self.order_id = order_id
        self.ticker = ticker


class RiskLimitExceededError(TradingSystemError):
    """Risk management limit violations"""
    
    def __init__(self, message: str, limit_type: str,
                 current_value: float, limit_value: float,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.CRITICAL, context)
        self.limit_type = limit_type
        self.current_value = current_value
        self.limit_value = limit_value


class ConfigurationError(TradingSystemError):
    """Configuration and setup errors"""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.HIGH, context)
        self.config_key = config_key


class DataValidationError(TradingSystemError):
    """Data validation and integrity errors"""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 expected_type: Optional[str] = None,
                 actual_value: Any = None,
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorSeverity.MEDIUM, context)
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    
    Prevents cascading failures by temporarily blocking calls to failing services
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function through circuit breaker
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception(f"Circuit breaker is OPEN (failures: {self.failure_count})")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        return (self.last_failure_time and 
                datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout))
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self.failure_count = 0
            self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
    
    def reset(self):
        """Manually reset the circuit breaker"""
        with self._lock:
            self.failure_count = 0
            self.last_failure_time = None
            self.state = CircuitBreakerState.CLOSED
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'last_failure': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


# ============================================================================
# Error Recovery Manager
# ============================================================================

class ErrorRecoveryManager:
    """
    Centralized error recovery and resilience management
    """
    
    def __init__(self):
        """Initialize error recovery manager"""
        self.error_history = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        self.fallback_values = {}
        self._lock = threading.Lock()
    
    def register_circuit_breaker(self, name: str, breaker: CircuitBreaker):
        """Register a circuit breaker"""
        self.circuit_breakers[name] = breaker
        logger.info(f"Registered circuit breaker: {name}")
    
    def register_recovery_strategy(self, error_type: Type[Exception],
                                  strategy: RecoveryStrategy,
                                  fallback_value: Any = None):
        """
        Register recovery strategy for an error type
        
        Args:
            error_type: Exception type to handle
            strategy: Recovery strategy to use
            fallback_value: Optional fallback value for FALLBACK strategy
        """
        self.recovery_strategies[error_type.__name__] = strategy
        if fallback_value is not None:
            self.fallback_values[error_type.__name__] = fallback_value
        
        logger.info(f"Registered recovery strategy for {error_type.__name__}: {strategy.value}")
    
    def record_error(self, error: Exception, context: Optional[ErrorContext] = None):
        """Record an error occurrence"""
        with self._lock:
            error_type = type(error).__name__
            self.error_counts[error_type] += 1
            
            if context is None:
                context = self._create_error_context(error)
            
            self.error_history.append(context)
            
            # Log based on severity
            if hasattr(error, 'severity'):
                if error.severity >= ErrorSeverity.CRITICAL:
                    logger.critical(f"{error_type}: {str(error)}")
                elif error.severity >= ErrorSeverity.HIGH:
                    logger.error(f"{error_type}: {str(error)}")
                else:
                    logger.warning(f"{error_type}: {str(error)}")
            else:
                logger.error(f"{error_type}: {str(error)}")
    
    def _create_error_context(self, error: Exception) -> ErrorContext:
        """Create error context from exception"""
        tb = traceback.extract_tb(error.__traceback__)
        
        if tb:
            last_frame = tb[-1]
            context = ErrorContext(
                function_name=last_frame.name,
                module_name=last_frame.filename,
                line_number=last_frame.lineno,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                severity=getattr(error, 'severity', ErrorSeverity.MEDIUM)
            )
        else:
            context = ErrorContext(
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                severity=getattr(error, 'severity', ErrorSeverity.MEDIUM)
            )
        
        return context
    
    def get_recovery_strategy(self, error: Exception) -> RecoveryStrategy:
        """Get recovery strategy for an error"""
        error_type = type(error).__name__
        return self.recovery_strategies.get(error_type, RecoveryStrategy.FAIL_FAST)
    
    def get_fallback_value(self, error: Exception) -> Any:
        """Get fallback value for an error"""
        error_type = type(error).__name__
        return self.fallback_values.get(error_type)
    
    def get_error_report(self) -> Dict[str, Any]:
        """Generate error report"""
        with self._lock:
            # Calculate error rate
            recent_errors = [e for e in self.error_history 
                           if e.timestamp > datetime.now() - timedelta(minutes=5)]
            
            # Group by severity
            severity_counts = defaultdict(int)
            for error in recent_errors:
                severity_counts[error.severity.name] += 1
            
            # Get top error types
            top_errors = sorted(self.error_counts.items(), 
                              key=lambda x: x[1], reverse=True)[:10]
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_errors': sum(self.error_counts.values()),
                'recent_errors_5min': len(recent_errors),
                'error_rate_per_min': len(recent_errors) / 5,
                'severity_distribution': dict(severity_counts),
                'top_error_types': dict(top_errors),
                'circuit_breakers': {
                    name: breaker.get_state() 
                    for name, breaker in self.circuit_breakers.items()
                }
            }
            
            return report


# ============================================================================
# Decorators for Error Handling
# ============================================================================

def retry_on_exception(max_retries: int = 3, delay: float = 1.0,
                       backoff: float = 2.0,
                       exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to retry on
        
    Example:
        @retry_on_exception(max_retries=3, delay=1, exceptions=(ConnectionError,))
        def fetch_data():
            # May raise ConnectionError
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                                     f"{func.__name__}: {str(e)}. Retrying in {current_delay:.1f}s...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            # Re-raise the last exception if all retries failed
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def fallback_on_error(fallback_value: Any = None, 
                      fallback_func: Optional[Callable] = None,
                      exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Decorator to provide fallback value on error
    
    Args:
        fallback_value: Static fallback value
        fallback_func: Function to generate fallback value
        exceptions: Exceptions to catch
        
    Example:
        @fallback_on_error(fallback_value=[], exceptions=(DataFetchError,))
        def fetch_market_data():
            # May raise DataFetchError
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(f"{func.__name__} failed: {str(e)}. Using fallback.")
                
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    return fallback_value
        
        return wrapper
    return decorator


def circuit_breaker(failure_threshold: int = 5, recovery_timeout: int = 60,
                   expected_exception: Type[Exception] = Exception):
    """
    Decorator to apply circuit breaker pattern
    
    Args:
        failure_threshold: Failures before opening circuit
        recovery_timeout: Seconds before recovery attempt
        expected_exception: Exception type to monitor
        
    Example:
        @circuit_breaker(failure_threshold=3, recovery_timeout=30)
        def connect_to_service():
            # May fail repeatedly
            pass
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        # Attach breaker for external access
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator


def log_exceptions(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  include_context: bool = True):
    """
    Decorator to log exceptions with context
    
    Args:
        severity: Default severity level
        include_context: Include function context in log
        
    Example:
        @log_exceptions(severity=ErrorSeverity.HIGH)
        def critical_operation():
            # Exceptions will be logged with HIGH severity
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Create context
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'timestamp': datetime.now().isoformat()
                }
                
                if include_context:
                    # Add function arguments (be careful with sensitive data)
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    
                    # Sanitize arguments
                    sanitized_args = {}
                    for key, value in bound_args.arguments.items():
                        if 'password' in key.lower() or 'key' in key.lower():
                            sanitized_args[key] = '***REDACTED***'
                        else:
                            sanitized_args[key] = str(value)[:100]  # Limit length
                    
                    context['arguments'] = sanitized_args
                
                # Log based on severity
                if severity >= ErrorSeverity.CRITICAL:
                    logger.critical(f"Exception in {func.__name__}: {str(e)}", 
                                  extra={'context': context})
                elif severity >= ErrorSeverity.HIGH:
                    logger.error(f"Exception in {func.__name__}: {str(e)}", 
                               extra={'context': context})
                else:
                    logger.warning(f"Exception in {func.__name__}: {str(e)}", 
                                 extra={'context': context})
                
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator


@contextmanager
def error_handler(strategy: RecoveryStrategy = RecoveryStrategy.FAIL_FAST,
                 fallback_value: Any = None,
                 exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """
    Context manager for error handling
    
    Args:
        strategy: Recovery strategy to use
        fallback_value: Value to return on error (for FALLBACK strategy)
        exceptions: Exceptions to handle
        
    Example:
        with error_handler(strategy=RecoveryStrategy.FALLBACK, fallback_value=[]):
            data = fetch_risky_data()
    """
    try:
        yield
    except exceptions as e:
        if strategy == RecoveryStrategy.RETRY:
            raise  # Let caller handle retry
        elif strategy == RecoveryStrategy.FALLBACK:
            return fallback_value
        elif strategy == RecoveryStrategy.IGNORE:
            logger.debug(f"Ignoring error: {str(e)}")
            return None
        else:
            raise


# ============================================================================
# Global Error Recovery Instance
# ============================================================================

_error_recovery_manager = None


def get_error_recovery_manager() -> ErrorRecoveryManager:
    """Get or create singleton ErrorRecoveryManager instance"""
    global _error_recovery_manager
    
    if _error_recovery_manager is None:
        _error_recovery_manager = ErrorRecoveryManager()
        
        # Register default recovery strategies
        _error_recovery_manager.register_recovery_strategy(
            IBKRConnectionError, RecoveryStrategy.RETRY
        )
        _error_recovery_manager.register_recovery_strategy(
            DataFetchError, RecoveryStrategy.FALLBACK, fallback_value=[]
        )
        _error_recovery_manager.register_recovery_strategy(
            ModelTrainingError, RecoveryStrategy.CIRCUIT_BREAK
        )
        _error_recovery_manager.register_recovery_strategy(
            OrderExecutionError, RecoveryStrategy.FAIL_FAST
        )
        _error_recovery_manager.register_recovery_strategy(
            RiskLimitExceededError, RecoveryStrategy.FAIL_FAST
        )
        
        logger.info("ErrorRecoveryManager initialized with default strategies")
    
    return _error_recovery_manager


if __name__ == "__main__":
    # Self-test
    print("Exception Handler Self-Test")
    print("=" * 50)
    
    # Test custom exceptions
    print("\nTesting custom exceptions...")
    try:
        raise IBKRConnectionError("Connection timeout", retry_count=3)
    except IBKRConnectionError as e:
        print(f"  Caught: {type(e).__name__} - {str(e)}")
        print(f"  Severity: {e.severity.name}")
        print(f"  Retry count: {e.retry_count}")
    
    # Test retry decorator
    print("\nTesting retry decorator...")
    attempt_count = 0
    
    @retry_on_exception(max_retries=2, delay=0.1, exceptions=(ValueError,))
    def failing_function():
        global attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}")
        if attempt_count < 3:
            raise ValueError("Simulated failure")
        return "Success!"
    
    try:
        result = failing_function()
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  Failed after retries: {e}")
    
    # Test circuit breaker
    print("\nTesting circuit breaker...")
    
    @circuit_breaker(failure_threshold=2, recovery_timeout=1)
    def unstable_service():
        import random
        if random.random() < 0.7:
            raise ConnectionError("Service unavailable")
        return "Connected!"
    
    for i in range(5):
        try:
            result = unstable_service()
            print(f"  Call {i+1}: {result}")
        except Exception as e:
            print(f"  Call {i+1}: {str(e)}")
        time.sleep(0.2)
    
    print(f"  Circuit state: {unstable_service.circuit_breaker.get_state()}")
    
    # Test error recovery manager
    print("\nTesting error recovery manager...")
    manager = get_error_recovery_manager()
    
    # Record some errors
    manager.record_error(DataFetchError("API timeout", ticker="AAPL"))
    manager.record_error(ModelTrainingError("Insufficient data"))
    manager.record_error(IBKRConnectionError("Gateway not responding"))
    
    # Get report
    report = manager.get_error_report()
    print(f"  Total errors: {report['total_errors']}")
    print(f"  Error types: {report['top_error_types']}")
    
    print("\nSelf-test complete!")