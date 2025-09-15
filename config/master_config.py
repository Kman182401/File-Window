"""
Master Configuration System for Trading Platform

Centralized configuration management with validation, environment-specific
overrides, hot-reload capability, and versioning support.

Author: Trading System
Version: 2.0.0
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    PAPER = "paper"


class ConfigSection(Enum):
    """Configuration sections"""
    TRADING = "trading"
    RISK = "risk"
    MODEL = "model"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"
    IBKR = "ibkr"
    AWS = "aws"


@dataclass
class ConfigValidation:
    """Configuration validation rules"""
    required: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    type_check: Optional[type] = None
    custom_validator: Optional[Callable] = None
    default_value: Any = None
    description: str = ""


class ConfigManager:
    """
    Centralized configuration management system
    
    Features:
    - Hierarchical configuration with inheritance
    - Environment-specific overrides
    - Runtime validation
    - Hot-reload capability
    - Change tracking and versioning
    - Thread-safe operations
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Determine environment
        self.environment = self._detect_environment()
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        self.config_metadata: Dict[str, ConfigValidation] = {}
        self.config_history: List[Dict[str, Any]] = []
        
        # File paths
        self.config_dir = Path("/home/ubuntu/config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files (in priority order)
        self.config_files = [
            self.config_dir / "defaults.yaml",
            self.config_dir / f"{self.environment.value}.yaml",
            self.config_dir / "local.yaml"  # Local overrides (not in git)
        ]
        
        # Change tracking
        self.config_version = 0
        self.last_modified = {}
        self.change_callbacks = defaultdict(list)
        
        # Hot-reload
        self.hot_reload_enabled = False
        self.reload_thread = None
        self.reload_interval = 60  # seconds
        
        # Initialize configuration
        self._initialize_defaults()
        self._load_configuration()
        
        logger.info(f"ConfigManager initialized for {self.environment.value} environment")
    
    def _detect_environment(self) -> ConfigEnvironment:
        """Detect current environment"""
        env = os.getenv('TRADING_ENVIRONMENT', 'development').lower()
        
        # Map environment strings to enum
        env_map = {
            'dev': ConfigEnvironment.DEVELOPMENT,
            'development': ConfigEnvironment.DEVELOPMENT,
            'staging': ConfigEnvironment.STAGING,
            'prod': ConfigEnvironment.PRODUCTION,
            'production': ConfigEnvironment.PRODUCTION,
            'test': ConfigEnvironment.TESTING,
            'testing': ConfigEnvironment.TESTING,
            'paper': ConfigEnvironment.PAPER
        }
        
        return env_map.get(env, ConfigEnvironment.DEVELOPMENT)
    
    def _initialize_defaults(self):
        """Initialize default configuration and validation rules"""
        
        # Trading configuration
        self.config_metadata.update({
            'trading.enabled': ConfigValidation(
                required=True,
                type_check=bool,
                default_value=False,
                description="Enable live trading"
            ),
            'trading.symbols': ConfigValidation(
                required=True,
                type_check=list,
                default_value=['ES1!', 'NQ1!', 'GBPUSD', 'EURUSD', 'AUDUSD', 'XAUUSD'],
                description="Trading symbols"
            ),
            'trading.max_trades_per_day': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1,
                max_value=100,
                default_value=20,
                description="Maximum trades per day"
            ),
            'trading.allow_orders': ConfigValidation(
                required=True,
                type_check=bool,
                default_value=False,
                description="Allow real order execution"
            ),
            'trading.dry_run': ConfigValidation(
                required=True,
                type_check=bool,
                default_value=True,
                description="Run in simulation mode"
            )
        })
        
        # Risk management configuration
        self.config_metadata.update({
            'risk.max_daily_loss_pct': ConfigValidation(
                required=True,
                type_check=float,
                min_value=0.1,
                max_value=10.0,
                default_value=2.0,
                description="Maximum daily loss percentage"
            ),
            'risk.max_position_exposure': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1,
                max_value=10,
                default_value=3,
                description="Maximum position exposure in contracts"
            ),
            'risk.max_order_size': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1,
                max_value=10,
                default_value=2,
                description="Maximum order size in contracts"
            ),
            'risk.stop_loss_pct': ConfigValidation(
                required=False,
                type_check=float,
                min_value=0.1,
                max_value=5.0,
                default_value=1.0,
                description="Stop loss percentage"
            ),
            'risk.take_profit_pct': ConfigValidation(
                required=False,
                type_check=float,
                min_value=0.1,
                max_value=10.0,
                default_value=2.0,
                description="Take profit percentage"
            ),
            'risk.max_drawdown_pct': ConfigValidation(
                required=True,
                type_check=float,
                min_value=5.0,
                max_value=50.0,
                default_value=20.0,
                description="Maximum drawdown percentage"
            )
        })
        
        # Model configuration
        self.config_metadata.update({
            'model.type': ConfigValidation(
                required=True,
                type_check=str,
                allowed_values=['PPO', 'A2C', 'DQN', 'SAC'],
                default_value='PPO',
                description="Reinforcement learning algorithm"
            ),
            'model.learning_rate': ConfigValidation(
                required=True,
                type_check=float,
                min_value=1e-6,
                max_value=1e-2,
                default_value=3e-4,
                description="Model learning rate"
            ),
            'model.batch_size': ConfigValidation(
                required=True,
                type_check=int,
                min_value=16,
                max_value=512,
                default_value=64,
                description="Training batch size"
            ),
            'model.n_steps': ConfigValidation(
                required=True,
                type_check=int,
                min_value=128,
                max_value=4096,
                default_value=2048,
                description="Number of steps per update"
            ),
            'model.gamma': ConfigValidation(
                required=True,
                type_check=float,
                min_value=0.9,
                max_value=0.999,
                default_value=0.99,
                description="Discount factor"
            ),
            'model.save_frequency': ConfigValidation(
                required=True,
                type_check=int,
                min_value=100,
                max_value=10000,
                default_value=1000,
                description="Model save frequency (iterations)"
            )
        })
        
        # Data configuration
        self.config_metadata.update({
            'data.lookback_days': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1,
                max_value=365,
                default_value=30,
                description="Historical data lookback period"
            ),
            'data.update_frequency': ConfigValidation(
                required=True,
                type_check=str,
                allowed_values=['1min', '5min', '15min', '30min', '1hour', '1day'],
                default_value='5min',
                description="Data update frequency"
            ),
            'data.feature_window': ConfigValidation(
                required=True,
                type_check=int,
                min_value=10,
                max_value=200,
                default_value=20,
                description="Feature calculation window"
            ),
            'data.cache_enabled': ConfigValidation(
                required=True,
                type_check=bool,
                default_value=True,
                description="Enable data caching"
            ),
            'data.cache_ttl': ConfigValidation(
                required=False,
                type_check=int,
                min_value=60,
                max_value=86400,
                default_value=3600,
                description="Cache time-to-live in seconds"
            )
        })
        
        # Infrastructure configuration
        self.config_metadata.update({
            'infrastructure.memory_limit_mb': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1000,
                max_value=8000,
                default_value=6000,
                description="Memory limit for m5.large instance"
            ),
            'infrastructure.cpu_limit_percent': ConfigValidation(
                required=True,
                type_check=float,
                min_value=10.0,
                max_value=100.0,
                default_value=80.0,
                description="CPU usage limit percentage"
            ),
            'infrastructure.log_level': ConfigValidation(
                required=True,
                type_check=str,
                allowed_values=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                default_value='INFO',
                description="Logging level"
            ),
            'infrastructure.monitoring_enabled': ConfigValidation(
                required=True,
                type_check=bool,
                default_value=True,
                description="Enable system monitoring"
            )
        })
        
        # IBKR configuration
        self.config_metadata.update({
            'ibkr.host': ConfigValidation(
                required=True,
                type_check=str,
                default_value='127.0.0.1',
                description="IBKR Gateway host"
            ),
            'ibkr.port': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1,
                max_value=65535,
                default_value=4002,
                description="IBKR Gateway port (4002 for paper, 4001 for live)"
            ),
            'ibkr.client_id': ConfigValidation(
                required=True,
                type_check=int,
                min_value=1,
                max_value=99999,
                default_value=9002,
                description="IBKR client ID"
            ),
            'ibkr.connection_timeout': ConfigValidation(
                required=False,
                type_check=int,
                min_value=5,
                max_value=60,
                default_value=15,
                description="Connection timeout in seconds"
            ),
            'ibkr.retry_attempts': ConfigValidation(
                required=False,
                type_check=int,
                min_value=1,
                max_value=10,
                default_value=3,
                description="Connection retry attempts"
            )
        })
        
        # AWS configuration
        self.config_metadata.update({
            'aws.s3_bucket': ConfigValidation(
                required=True,
                type_check=str,
                default_value='omega-singularity-ml',
                description="S3 bucket for storage"
            ),
            'aws.region': ConfigValidation(
                required=True,
                type_check=str,
                default_value='us-east-1',
                description="AWS region"
            ),
            'aws.model_prefix': ConfigValidation(
                required=False,
                type_check=str,
                default_value='models/',
                description="S3 prefix for models"
            ),
            'aws.data_prefix': ConfigValidation(
                required=False,
                type_check=str,
                default_value='data/',
                description="S3 prefix for data"
            )
        })
        
        # Monitoring configuration
        self.config_metadata.update({
            'monitoring.performance_tracking': ConfigValidation(
                required=True,
                type_check=bool,
                default_value=True,
                description="Enable performance tracking"
            ),
            'monitoring.health_check_interval': ConfigValidation(
                required=False,
                type_check=int,
                min_value=10,
                max_value=300,
                default_value=30,
                description="Health check interval in seconds"
            ),
            'monitoring.alert_email': ConfigValidation(
                required=False,
                type_check=str,
                default_value=None,
                description="Email for alerts"
            ),
            'monitoring.metrics_export_interval': ConfigValidation(
                required=False,
                type_check=int,
                min_value=60,
                max_value=3600,
                default_value=300,
                description="Metrics export interval in seconds"
            )
        })
    
    def _load_configuration(self):
        """Load configuration from files"""
        # Start with defaults
        for key, validation in self.config_metadata.items():
            if validation.default_value is not None:
                self._set_nested(self.config, key, validation.default_value)
        
        # Load from files in priority order
        for config_file in self.config_files:
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix == '.yaml':
                            file_config = yaml.safe_load(f) or {}
                        else:
                            file_config = json.load(f)
                    
                    # Merge configuration
                    self._merge_config(self.config, file_config)
                    self.last_modified[str(config_file)] = config_file.stat().st_mtime
                    
                    logger.info(f"Loaded configuration from {config_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {config_file}: {e}")
        
        # Load environment variables (highest priority)
        self._load_env_overrides()
        
        # Validate configuration
        self._validate_config()
        
        # Save snapshot
        self._save_config_snapshot()
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        # Environment variables format: TRADING_CONFIG_SECTION_KEY
        prefix = "TRADING_CONFIG_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Convert env key to config key
                config_key = env_key[len(prefix):].lower().replace('_', '.')
                
                # Parse value
                try:
                    # Try to parse as JSON first
                    value = json.loads(env_value)
                except:
                    # Fallback to string
                    value = env_value
                
                # Set in config
                self._set_nested(self.config, config_key, value)
                logger.debug(f"Override from env: {config_key} = {value}")
    
    def _merge_config(self, base: Dict, override: Dict):
        """Recursively merge configuration dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _set_nested(self, d: Dict, key: str, value: Any):
        """Set nested dictionary value using dot notation"""
        keys = key.split('.')
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    
    def _get_nested(self, d: Dict, key: str, default: Any = None) -> Any:
        """Get nested dictionary value using dot notation"""
        keys = key.split('.')
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
        return d if d is not None else default
    
    def _validate_config(self):
        """Validate configuration against rules"""
        errors = []
        
        for key, validation in self.config_metadata.items():
            value = self._get_nested(self.config, key)
            
            # Check required
            if validation.required and value is None:
                errors.append(f"{key}: Required configuration missing")
                continue
            
            if value is None:
                continue
            
            # Type check
            if validation.type_check and not isinstance(value, validation.type_check):
                errors.append(f"{key}: Expected {validation.type_check.__name__}, got {type(value).__name__}")
            
            # Min/max validation
            if validation.min_value is not None and value < validation.min_value:
                errors.append(f"{key}: Value {value} below minimum {validation.min_value}")
            
            if validation.max_value is not None and value > validation.max_value:
                errors.append(f"{key}: Value {value} above maximum {validation.max_value}")
            
            # Allowed values
            if validation.allowed_values and value not in validation.allowed_values:
                errors.append(f"{key}: Value {value} not in allowed values {validation.allowed_values}")
            
            # Custom validator
            if validation.custom_validator:
                try:
                    if not validation.custom_validator(value):
                        errors.append(f"{key}: Custom validation failed")
                except Exception as e:
                    errors.append(f"{key}: Validator error: {e}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _save_config_snapshot(self):
        """Save configuration snapshot for versioning"""
        snapshot = {
            'version': self.config_version,
            'timestamp': datetime.now().isoformat(),
            'environment': self.environment.value,
            'config': self.config.copy(),
            'checksum': self._calculate_checksum()
        }
        
        self.config_history.append(snapshot)
        self.config_version += 1
        
        # Keep only last 100 snapshots
        if len(self.config_history) > 100:
            self.config_history = self.config_history[-100:]
    
    def _calculate_checksum(self) -> str:
        """Calculate configuration checksum"""
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Args:
            key: Configuration key (dot notation)
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self._get_nested(self.config, key, default)
    
    def set(self, key: str, value: Any, persist: bool = False):
        """
        Set configuration value
        
        Args:
            key: Configuration key (dot notation)
            value: New value
            persist: Whether to persist to file
        """
        # Validate if rule exists
        if key in self.config_metadata:
            validation = self.config_metadata[key]
            
            # Run validation
            temp_config = self.config.copy()
            self._set_nested(temp_config, key, value)
            
            # Validate just this key
            if validation.type_check and not isinstance(value, validation.type_check):
                raise ValueError(f"Invalid type for {key}")
            
            if validation.min_value is not None and value < validation.min_value:
                raise ValueError(f"Value below minimum for {key}")
            
            if validation.max_value is not None and value > validation.max_value:
                raise ValueError(f"Value above maximum for {key}")
            
            if validation.allowed_values and value not in validation.allowed_values:
                raise ValueError(f"Value not allowed for {key}")
        
        # Set value
        old_value = self._get_nested(self.config, key)
        self._set_nested(self.config, key, value)
        
        # Trigger callbacks
        self._trigger_change_callbacks(key, old_value, value)
        
        # Persist if requested
        if persist:
            self.save_local_overrides()
        
        # Save snapshot
        self._save_config_snapshot()
        
        logger.info(f"Configuration updated: {key} = {value}")
    
    def _trigger_change_callbacks(self, key: str, old_value: Any, new_value: Any):
        """Trigger callbacks for configuration changes"""
        # Exact key callbacks
        for callback in self.change_callbacks.get(key, []):
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Change callback failed for {key}: {e}")
        
        # Section callbacks (e.g., 'trading.*')
        section = key.split('.')[0]
        for callback in self.change_callbacks.get(f"{section}.*", []):
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                logger.error(f"Section callback failed for {key}: {e}")
    
    def register_change_callback(self, key_pattern: str, 
                                callback: Callable[[str, Any, Any], None]):
        """
        Register callback for configuration changes
        
        Args:
            key_pattern: Key or pattern (e.g., 'trading.*')
            callback: Function to call on change
        """
        self.change_callbacks[key_pattern].append(callback)
        logger.debug(f"Registered change callback for {key_pattern}")
    
    def save_local_overrides(self):
        """Save current configuration as local overrides"""
        local_file = self.config_dir / "local.yaml"
        
        with open(local_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Saved local configuration to {local_file}")
    
    def reload(self):
        """Reload configuration from files"""
        logger.info("Reloading configuration...")
        
        # Check for file changes
        changed = False
        for config_file in self.config_files:
            if config_file.exists():
                mtime = config_file.stat().st_mtime
                if str(config_file) not in self.last_modified or \
                   mtime > self.last_modified[str(config_file)]:
                    changed = True
                    break
        
        if changed:
            old_config = self.config.copy()
            self._load_configuration()
            
            # Find and trigger callbacks for changes
            self._detect_changes(old_config, self.config)
            
            logger.info("Configuration reloaded successfully")
        else:
            logger.debug("No configuration changes detected")
    
    def _detect_changes(self, old_config: Dict, new_config: Dict, prefix: str = ""):
        """Detect and trigger callbacks for configuration changes"""
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            
            old_value = old_config.get(key)
            new_value = new_config.get(key)
            
            if isinstance(old_value, dict) and isinstance(new_value, dict):
                self._detect_changes(old_value, new_value, full_key)
            elif old_value != new_value:
                self._trigger_change_callbacks(full_key, old_value, new_value)
    
    def start_hot_reload(self, interval: int = 60):
        """
        Start hot-reload monitoring
        
        Args:
            interval: Check interval in seconds
        """
        if self.hot_reload_enabled:
            return
        
        self.hot_reload_enabled = True
        self.reload_interval = interval
        
        def reload_loop():
            while self.hot_reload_enabled:
                try:
                    self.reload()
                except Exception as e:
                    logger.error(f"Hot-reload failed: {e}")
                
                import time
                time.sleep(self.reload_interval)
        
        self.reload_thread = threading.Thread(target=reload_loop, daemon=True)
        self.reload_thread.start()
        
        logger.info(f"Hot-reload started (interval: {interval}s)")
    
    def stop_hot_reload(self):
        """Stop hot-reload monitoring"""
        self.hot_reload_enabled = False
        
        if self.reload_thread:
            self.reload_thread.join(timeout=5)
        
        logger.info("Hot-reload stopped")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._get_nested(self.config, section, {})
    
    def export_config(self, filepath: str, format: str = 'yaml'):
        """
        Export configuration to file
        
        Args:
            filepath: Export file path
            format: Export format ('yaml' or 'json')
        """
        with open(filepath, 'w') as f:
            if format == 'yaml':
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
        
        logger.info(f"Exported configuration to {filepath}")
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information and statistics"""
        return {
            'environment': self.environment.value,
            'version': self.config_version,
            'checksum': self._calculate_checksum(),
            'loaded_files': [str(f) for f in self.config_files if f.exists()],
            'total_keys': len(self.config_metadata),
            'hot_reload': self.hot_reload_enabled,
            'history_size': len(self.config_history),
            'callbacks_registered': sum(len(v) for v in self.change_callbacks.values())
        }


# Global instance
def get_config() -> ConfigManager:
    """Get or create singleton ConfigManager instance"""
    return ConfigManager()


# Convenience functions
def config_get(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config().get(key, default)


def config_set(key: str, value: Any, persist: bool = False):
    """Set configuration value"""
    get_config().set(key, value, persist)


if __name__ == "__main__":
    # Self-test
    print("Configuration Manager Self-Test")
    print("=" * 50)
    
    config = get_config()
    
    print(f"\nEnvironment: {config.environment.value}")
    print(f"Version: {config.config_version}")
    
    # Test getting values
    print("\nSample Configuration Values:")
    print(f"  Trading enabled: {config.get('trading.enabled')}")
    print(f"  Risk max daily loss: {config.get('risk.max_daily_loss_pct')}%")
    print(f"  Model type: {config.get('model.type')}")
    print(f"  IBKR port: {config.get('ibkr.port')}")
    
    # Test setting values
    print("\nTesting configuration updates...")
    old_value = config.get('trading.max_trades_per_day')
    config.set('trading.max_trades_per_day', 25)
    print(f"  Updated max_trades_per_day: {old_value} -> {config.get('trading.max_trades_per_day')}")
    
    # Test section retrieval
    print("\nRisk Configuration Section:")
    risk_config = config.get_section('risk')
    for key, value in risk_config.items():
        print(f"  {key}: {value}")
    
    # Configuration info
    print("\nConfiguration Info:")
    info = config.get_config_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nSelf-test complete!")
