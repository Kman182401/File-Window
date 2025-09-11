"""
Historical Replay System - Ultra-optimized for m5.large
Memory-efficient, fast, and reliable market data replay
"""
import os
import json
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import boto3
from botocore.config import Config
import pickle
import lz4.frame
import psutil
from collections import deque
from tools.ultra_cache import get_cache, VectorCache

logger = logging.getLogger(__name__)


@dataclass
class ReplayConfig:
    """Configuration for historical replay."""
    symbols: List[str]
    start_date: str
    end_date: str
    s3_bucket: str = "omega-singularity-ml"
    s3_prefix: str = "market_data"
    speed_multiplier: float = 60.0  # 60x speed for testing
    cache_size_mb: int = 256  # Optimized for m5.large
    realism_mode: str = "realistic"  # realistic, perfect, or fast
    max_memory_mb: int = 512  # Hard limit for replay system
    prefetch_hours: int = 4  # Prefetch 4 hours of data
    

class ResourceGovernor:
    """Manages resources to ensure smooth operation on m5.large."""
    
    def __init__(self, max_memory_mb: int = 512, max_cpu_percent: int = 60):
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.throttle_level = 0  # 0=normal, 1=slow, 2=minimal
        
    def check_resources(self) -> Tuple[bool, str]:
        """Check if resources are within limits."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        # Memory check
        if mem.percent > 75:
            self.throttle_level = 2
            return False, f"Memory critical: {mem.percent:.1f}%"
        elif mem.percent > 65:
            self.throttle_level = 1
            return True, f"Memory high: {mem.percent:.1f}%"
            
        # CPU check
        if cpu > 80:
            self.throttle_level = 2
            return False, f"CPU critical: {cpu:.1f}%"
        elif cpu > 70:
            self.throttle_level = 1
            return True, f"CPU high: {cpu:.1f}%"
            
        self.throttle_level = 0
        return True, "Resources OK"
    
    def get_sleep_time(self) -> float:
        """Get sleep time based on throttle level."""
        if self.throttle_level == 0:
            return 0.0
        elif self.throttle_level == 1:
            return 0.01  # 10ms
        else:
            return 0.1  # 100ms


class HistoricalReplay:
    """
    Ultra-efficient historical data replay for training and backtesting.
    Optimized for m5.large with 2 vCPUs and 8GB RAM.
    """
    
    def __init__(self, config: ReplayConfig):
        self.config = config
        self.cache = get_cache()
        self.vector_cache = VectorCache(max_memory_mb=128)
        self.governor = ResourceGovernor(max_memory_mb=config.max_memory_mb)
        
        # S3 client with optimized config
        self.s3 = boto3.client('s3', config=Config(
            max_pool_connections=10,
            retries={'max_attempts': 2}
        ))
        
        # Time management
        self.current_time = pd.Timestamp(config.start_date)
        self.end_time = pd.Timestamp(config.end_date)
        self.last_real_time = time.time()
        
        # Data buffers (memory-efficient)
        self.data_buffers = {symbol: deque(maxlen=10000) for symbol in config.symbols}
        self.current_indices = {symbol: 0 for symbol in config.symbols}
        
        # Prefetch management
        self.last_prefetch = {}
        self.prefetch_queue = deque(maxlen=100)
        
        # Performance metrics
        self.bars_served = 0
        self.cache_hits = 0
        self.s3_loads = 0
        
        # Circuit breakers
        self.error_count = 0
        self.max_errors = 10
        
        # Initialize data
        self._prefetch_initial_data()
    
    def _prefetch_initial_data(self):
        """Prefetch initial data for all symbols."""
        logger.info(f"Prefetching data for {len(self.config.symbols)} symbols")
        
        for symbol in self.config.symbols:
            try:
                self._load_symbol_data(symbol, self.current_time, self.config.prefetch_hours)
            except Exception as e:
                logger.error(f"Failed to prefetch {symbol}: {e}")
                self.error_count += 1
                
    def _load_symbol_data(self, symbol: str, start_time: pd.Timestamp, hours: int):
        """Load data from S3 with caching and compression."""
        # Check cache first
        cache_key = f"data_{symbol}_{start_time.strftime('%Y%m%d_%H')}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            self.cache_hits += 1
            self.data_buffers[symbol].extend(cached_data)
            return
        
        # Check resources before S3 load
        ok, msg = self.governor.check_resources()
        if not ok:
            logger.warning(f"Skipping S3 load: {msg}")
            return
            
        # Load from S3
        try:
            # Build S3 key pattern
            date_str = start_time.strftime('%Y/%m/%d')
            prefix = f"{self.config.s3_prefix}/{symbol}/{date_str}/"
            
            # List objects
            response = self.s3.list_objects_v2(
                Bucket=self.config.s3_bucket,
                Prefix=prefix,
                MaxKeys=100  # Limit for memory
            )
            
            if 'Contents' not in response:
                logger.debug(f"No data found for {symbol} on {date_str}")
                return
                
            # Load and parse files
            all_data = []
            for obj in response['Contents'][:10]:  # Limit files
                try:
                    # Download to memory
                    response = self.s3.get_object(
                        Bucket=self.config.s3_bucket,
                        Key=obj['Key']
                    )
                    
                    # Parse CSV data
                    df = pd.read_csv(response['Body'], nrows=1000)  # Limit rows
                    
                    # Convert to lightweight format
                    bars = self._convert_to_bars(df)
                    all_data.extend(bars)
                    
                except Exception as e:
                    logger.debug(f"Error loading {obj['Key']}: {e}")
                    
            if all_data:
                # Cache the data
                self.cache.put(cache_key, all_data[:1000])  # Cache limited data
                self.data_buffers[symbol].extend(all_data)
                self.s3_loads += 1
                
        except Exception as e:
            logger.error(f"S3 load error for {symbol}: {e}")
            self.error_count += 1
            
    def _convert_to_bars(self, df: pd.DataFrame) -> List[Dict]:
        """Convert DataFrame to lightweight bar format."""
        bars = []
        
        for _, row in df.iterrows():
            bar = {
                'timestamp': row.get('timestamp', row.get('Datetime')),
                'open': float(row.get('open', row.get('Open', 0))),
                'high': float(row.get('high', row.get('High', 0))),
                'low': float(row.get('low', row.get('Low', 0))),
                'close': float(row.get('close', row.get('Close', 0))),
                'volume': int(row.get('volume', row.get('Volume', 0)))
            }
            
            # Add realism if configured
            if self.config.realism_mode == "realistic":
                bar = self._apply_realism(bar)
                
            bars.append(bar)
            
        return bars
    
    def _apply_realism(self, bar: Dict) -> Dict:
        """Apply realistic market microstructure."""
        # Add spread (optimized calculation)
        spread_bps = 1.0 + np.random.exponential(0.5)  # Fast approximation
        bar['spread_bps'] = min(spread_bps, 5.0)
        
        # Add slippage potential
        bar['slippage_ticks'] = 1 if np.random.random() > 0.7 else 0
        
        # Market impact estimate
        bar['market_impact_bps'] = 0.5 * (1 + np.random.random())
        
        return bar
    
    def get_next_bar(self, symbol: str) -> Optional[Dict]:
        """Get next bar for symbol."""
        # Check circuit breaker
        if self.error_count >= self.max_errors:
            logger.error("Circuit breaker triggered - too many errors")
            return None
            
        # Check resources
        if self.governor.throttle_level > 0:
            time.sleep(self.governor.get_sleep_time())
            
        # Check if we have data
        if symbol not in self.data_buffers or not self.data_buffers[symbol]:
            # Try to load more data
            self._load_symbol_data(symbol, self.current_time, 1)
            
        if self.data_buffers[symbol]:
            bar = self.data_buffers[symbol].popleft()
            self.bars_served += 1
            return bar
            
        return None
    
    def advance_time(self, minutes: int = 1):
        """Advance replay time."""
        self.current_time += timedelta(minutes=minutes)
        
        # Check if we need to prefetch
        for symbol in self.config.symbols:
            if symbol not in self.last_prefetch or \
               (self.current_time - self.last_prefetch.get(symbol, self.current_time)).hours >= 1:
                # Prefetch next hour
                self._load_symbol_data(symbol, self.current_time, 1)
                self.last_prefetch[symbol] = self.current_time
    
    def reset(self):
        """Reset replay to start."""
        self.current_time = pd.Timestamp(self.config.start_date)
        self.data_buffers = {symbol: deque(maxlen=10000) for symbol in self.config.symbols}
        self.current_indices = {symbol: 0 for symbol in self.config.symbols}
        self.error_count = 0
        self._prefetch_initial_data()
    
    def get_stats(self) -> Dict:
        """Get replay statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'current_time': self.current_time.isoformat(),
            'bars_served': self.bars_served,
            'cache_hits': self.cache_hits,
            's3_loads': self.s3_loads,
            'cache_hit_rate': self.cache_hits / max(self.bars_served, 1),
            'error_count': self.error_count,
            'throttle_level': self.governor.throttle_level,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cache_stats': cache_stats
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.cache.clear()
        self.data_buffers.clear()
        

class FastReplay(HistoricalReplay):
    """
    Faster version for rapid testing with less realism.
    Uses pre-computed features and minimal processing.
    """
    
    def __init__(self, config: ReplayConfig):
        config.realism_mode = "fast"
        super().__init__(config)
        
    def _apply_realism(self, bar: Dict) -> Dict:
        """Skip realism for speed."""
        bar['spread_bps'] = 1.0
        bar['slippage_ticks'] = 0
        bar['market_impact_bps'] = 0.5
        return bar
    
    def get_next_batch(self, symbol: str, size: int = 100) -> List[Dict]:
        """Get batch of bars for vectorized processing."""
        bars = []
        for _ in range(size):
            bar = self.get_next_bar(symbol)
            if bar:
                bars.append(bar)
            else:
                break
        return bars