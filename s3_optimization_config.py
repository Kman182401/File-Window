"""
S3 OPTIMIZATION CONFIGURATION - Phase 3 Enhancement
===================================================

Intelligent S3 storage strategy balancing JSON vs Parquet for optimal performance
Configures data access patterns and format selection for different use cases
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import json
import pandas as pd
import numpy as np
from io import BytesIO
import gzip
import pickle
import logging
from datetime import datetime, UTC

logger = logging.getLogger(__name__)

class DataFormat(Enum):
    """Supported data formats for S3 storage"""
    JSON = "json"
    JSON_COMPRESSED = "json.gz"
    PARQUET = "parquet"
    CSV = "csv"
    PICKLE = "pickle"
    FEATHER = "feather"

class AccessPattern(Enum):
    """Data access patterns for optimization decisions"""
    FREQUENT_SMALL = "frequent_small"     # Small data, accessed often
    FREQUENT_LARGE = "frequent_large"     # Large data, accessed often  
    INFREQUENT_SMALL = "infrequent_small" # Small data, rarely accessed
    INFREQUENT_LARGE = "infrequent_large" # Large data, rarely accessed
    STREAMING = "streaming"               # Real-time streaming data
    ANALYTICAL = "analytical"             # Analytics/reporting queries

@dataclass
class StorageConfig:
    """Configuration for specific data type storage"""
    primary_format: DataFormat
    fallback_format: DataFormat
    compression_enabled: bool = True
    cache_ttl_seconds: int = 300
    max_size_bytes: int = 10 * 1024 * 1024  # 10MB default
    partition_key: Optional[str] = None

class S3OptimizationStrategy:
    """Intelligent S3 optimization strategy based on data characteristics"""
    
    def __init__(self):
        # Performance thresholds (bytes)
        self.SMALL_DATA_THRESHOLD = 100 * 1024      # 100KB
        self.LARGE_DATA_THRESHOLD = 10 * 1024 * 1024 # 10MB
        self.COMPRESSION_THRESHOLD = 1024 * 1024     # 1MB
        
        # Format configurations by data type
        self.data_type_configs = {
            # Market data configurations
            'raw_market_data': StorageConfig(
                primary_format=DataFormat.PARQUET,
                fallback_format=DataFormat.CSV,
                compression_enabled=True,
                cache_ttl_seconds=3600,  # 1 hour
                max_size_bytes=100 * 1024 * 1024,  # 100MB
                partition_key='date'
            ),
            
            'features_engineered': StorageConfig(
                primary_format=DataFormat.PARQUET,
                fallback_format=DataFormat.FEATHER,
                compression_enabled=True,
                cache_ttl_seconds=1800,  # 30 minutes
                max_size_bytes=50 * 1024 * 1024,   # 50MB
                partition_key='ticker'
            ),
            
            'model_predictions': StorageConfig(
                primary_format=DataFormat.JSON_COMPRESSED,
                fallback_format=DataFormat.JSON,
                compression_enabled=True,
                cache_ttl_seconds=60,    # 1 minute
                max_size_bytes=1 * 1024 * 1024,    # 1MB
            ),
            
            'trade_execution_logs': StorageConfig(
                primary_format=DataFormat.JSON,
                fallback_format=DataFormat.CSV,
                compression_enabled=False,  # For real-time access
                cache_ttl_seconds=0,        # No caching
                max_size_bytes=5 * 1024 * 1024,    # 5MB
                partition_key='date'
            ),
            
            'model_artifacts': StorageConfig(
                primary_format=DataFormat.PICKLE,
                fallback_format=DataFormat.PICKLE,
                compression_enabled=True,
                cache_ttl_seconds=86400,    # 24 hours
                max_size_bytes=500 * 1024 * 1024,  # 500MB
            ),
            
            'configuration_metadata': StorageConfig(
                primary_format=DataFormat.JSON,
                fallback_format=DataFormat.JSON,
                compression_enabled=False,
                cache_ttl_seconds=3600,     # 1 hour
                max_size_bytes=1 * 1024 * 1024,    # 1MB
            ),
            
            'realtime_features': StorageConfig(
                primary_format=DataFormat.FEATHER,
                fallback_format=DataFormat.JSON_COMPRESSED,
                compression_enabled=False,  # Speed over compression
                cache_ttl_seconds=30,       # 30 seconds
                max_size_bytes=10 * 1024 * 1024,   # 10MB
            )
        }
        
        logger.info("S3 optimization strategy initialized with intelligent format selection")
    
    def determine_optimal_format(self, 
                               data: Any, 
                               data_type: str, 
                               access_pattern: AccessPattern) -> Tuple[DataFormat, Dict[str, Any]]:
        """Determine optimal storage format based on data characteristics"""
        
        # Get base configuration
        config = self.data_type_configs.get(data_type)
        if not config:
            # Default configuration for unknown data types
            config = StorageConfig(
                primary_format=DataFormat.JSON,
                fallback_format=DataFormat.JSON,
                compression_enabled=True,
                cache_ttl_seconds=300
            )
        
        # Analyze data characteristics
        data_size = self._estimate_data_size(data)
        data_complexity = self._analyze_data_complexity(data)
        
        # Format selection logic
        selected_format = self._select_format_by_rules(
            data, data_size, data_complexity, config, access_pattern
        )
        
        # Generate metadata
        metadata = {
            'data_type': data_type,
            'access_pattern': access_pattern.value,
            'estimated_size': data_size,
            'complexity_score': data_complexity,
            'selected_format': selected_format.value,
            'compression_enabled': config.compression_enabled,
            'cache_ttl': config.cache_ttl_seconds
        }
        
        logger.debug(f"Format selected for {data_type}: {selected_format.value} "
                    f"(size: {data_size} bytes, complexity: {data_complexity})")
        
        return selected_format, metadata
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate serialized data size"""
        if isinstance(data, pd.DataFrame):
            # Estimate DataFrame size
            return len(data) * len(data.columns) * 8  # Rough estimate
        
        elif isinstance(data, (dict, list)):
            # Estimate JSON size
            try:
                return len(json.dumps(data, default=str))
            except:
                return 1024  # Default estimate
        
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        
        elif isinstance(data, (int, float)):
            return 8
        
        else:
            # Try to serialize and measure
            try:
                return len(pickle.dumps(data))
            except:
                return 1024  # Default estimate
    
    def _analyze_data_complexity(self, data: Any) -> float:
        """Analyze data complexity (0-1 scale)"""
        complexity = 0.0
        
        if isinstance(data, pd.DataFrame):
            # DataFrame complexity based on shape and dtypes
            shape_complexity = min(1.0, (len(data) * len(data.columns)) / 1000000)
            dtype_complexity = len(set(str(dt) for dt in data.dtypes)) / 10
            complexity = (shape_complexity + dtype_complexity) / 2
        
        elif isinstance(data, dict):
            # Dictionary complexity based on nesting and key count
            def count_nested_levels(obj, level=0):
                if isinstance(obj, dict):
                    return max([count_nested_levels(v, level + 1) for v in obj.values()] + [level])
                elif isinstance(obj, list) and obj:
                    return max([count_nested_levels(item, level) for item in obj] + [level])
                return level
            
            nesting_complexity = min(1.0, count_nested_levels(data) / 5)
            size_complexity = min(1.0, len(str(data)) / 10000)
            complexity = (nesting_complexity + size_complexity) / 2
        
        elif isinstance(data, list):
            # List complexity
            complexity = min(1.0, len(data) / 1000)
        
        return complexity
    
    def _select_format_by_rules(self, 
                              data: Any, 
                              data_size: int, 
                              complexity: float,
                              config: StorageConfig, 
                              access_pattern: AccessPattern) -> DataFormat:
        """Apply format selection rules"""
        
        # Rule 1: Use primary format for normal cases
        if data_size <= config.max_size_bytes:
            primary_format = config.primary_format
            
            # Rule 2: Override based on access pattern
            if access_pattern == AccessPattern.FREQUENT_SMALL and data_size < self.SMALL_DATA_THRESHOLD:
                # Small, frequently accessed data -> JSON for speed
                return DataFormat.JSON
            
            elif access_pattern == AccessPattern.FREQUENT_LARGE and data_size > self.LARGE_DATA_THRESHOLD:
                # Large, frequently accessed data -> Parquet for efficiency
                if isinstance(data, pd.DataFrame):
                    return DataFormat.PARQUET
                else:
                    return DataFormat.JSON_COMPRESSED
            
            elif access_pattern == AccessPattern.ANALYTICAL:
                # Analytics queries -> Parquet for columnar access
                if isinstance(data, pd.DataFrame):
                    return DataFormat.PARQUET
                else:
                    return primary_format
            
            elif access_pattern == AccessPattern.STREAMING:
                # Real-time streaming -> fast formats
                if isinstance(data, pd.DataFrame):
                    return DataFormat.FEATHER
                else:
                    return DataFormat.JSON
            
            else:
                return primary_format
        
        # Rule 3: Use fallback for oversized data
        else:
            return config.fallback_format

# ============================================================================
# S3 PATH OPTIMIZATION
# ============================================================================

class S3PathOptimizer:
    """Optimize S3 paths for different access patterns"""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        
        # Path templates for different data types
        self.path_templates = {
            'raw_market_data': 'raw-data/market/{ticker}/year={year}/month={month:02d}/day={day:02d}/{filename}',
            'features_engineered': 'features/{ticker}/date={date}/{filename}',
            'model_predictions': 'predictions/{model_type}/{date}/{ticker}/{timestamp}_{filename}',
            'trade_execution_logs': 'trades/year={year}/month={month:02d}/day={day:02d}/{filename}',
            'model_artifacts': 'models/{model_type}/{version}/{filename}',
            'configuration_metadata': 'config/{environment}/{filename}',
            'realtime_features': 'realtime/{ticker}/{minute_bucket}/{filename}'
        }
    
    def generate_optimized_path(self, 
                              data_type: str, 
                              filename: str,
                              **kwargs) -> str:
        """Generate optimized S3 path based on data type and parameters"""
        
        template = self.path_templates.get(data_type, '{data_type}/{filename}')
        
        # Add default parameters
        now = datetime.now(UTC)
        
        default_params = {
            'data_type': data_type,
            'filename': filename,
            'year': now.year,
            'month': now.month,
            'day': now.day,
            'date': now.strftime('%Y-%m-%d'),
            'timestamp': now.strftime('%Y%m%d_%H%M%S'),
            'minute_bucket': f"{now.hour:02d}{now.minute//5*5:02d}"  # 5-minute buckets
        }
        
        # Merge with provided parameters
        params = {**default_params, **kwargs}
        
        try:
            path = template.format(**params)
            return path
        except KeyError as e:
            # Fallback path if template parameters are missing
            logger.warning(f"Path template missing parameter {e}, using fallback")
            return f"{data_type}/{filename}"

# ============================================================================
# USAGE EXAMPLES AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    strategy = S3OptimizationStrategy()
    path_optimizer = S3PathOptimizer("omega-singularity-ml")
    
    # Test DataFrame optimization
    test_df = pd.DataFrame({
        'close': np.random.random(1000),
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    format_result, metadata = strategy.determine_optimal_format(
        test_df, 
        'raw_market_data', 
        AccessPattern.FREQUENT_LARGE
    )
    
    print(f"Recommended format: {format_result}")
    print(f"Metadata: {metadata}")
    
    # Test path optimization
    optimized_path = path_optimizer.generate_optimized_path(
        'raw_market_data',
        'ES_trades.parquet',
        ticker='ES1!'
    )
    
    print(f"Optimized path: {optimized_path}")
