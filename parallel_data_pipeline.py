"""
PARALLEL DATA PIPELINE ARCHITECTURE - Phase 3 Optimization
==========================================================

High-performance parallel processing pipeline designed for <1 second latency
Replaces sequential bottleneck with ThreadPoolExecutor and memory-optimized data flow
"""

import asyncio
import concurrent.futures
import threading
import time
import logging
import gc
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import pyarrow as pa
import polars as pl
import pandas as pd
import numpy as np
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from functools import wraps
import json
from io import BytesIO
import queue

logger = logging.getLogger(__name__)

# ============================================================================
# PERFORMANCE MONITORING SYSTEM
# ============================================================================

@dataclass
class PipelineMetrics:
    """Track pipeline performance metrics"""
    total_latency: float = 0.0
    ticker_latencies: Dict[str, float] = None
    memory_peak: float = 0.0
    memory_current: float = 0.0
    error_count: int = 0
    success_count: int = 0
    s3_upload_time: float = 0.0
    s3_download_time: float = 0.0
    worker_utilization: float = 0.0
    
    def __post_init__(self):
        if self.ticker_latencies is None:
            self.ticker_latencies = {}

class PerformanceMonitor:
    """Monitor and alert on pipeline performance"""
    
    def __init__(self):
        self.metrics = PipelineMetrics()
        self.process = psutil.Process()
        self.start_time = None
        
        # Performance thresholds
        self.LATENCY_WARNING = 800  # ms
        self.LATENCY_CRITICAL = 1000  # ms
        self.MEMORY_WARNING = 4.0  # GB
        self.MEMORY_CRITICAL = 6.0  # GB
        self.ERROR_RATE_WARNING = 0.05
        self.ERROR_RATE_CRITICAL = 0.10
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring individual operations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            self.metrics.success_count += 1
        except Exception as e:
            self.metrics.error_count += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            latency = (time.time() - start_time) * 1000
            current_memory = self._get_memory_usage()
            
            logger.info(f"{operation_name} completed in {latency:.1f}ms, memory: {current_memory:.1f}MB")
            
            # Update peak memory
            self.metrics.memory_peak = max(self.metrics.memory_peak, current_memory / 1024)
            self.metrics.memory_current = current_memory / 1024
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def start_pipeline(self):
        """Start pipeline monitoring"""
        self.start_time = time.time()
        self.metrics = PipelineMetrics()
        logger.info("Pipeline monitoring started")
    
    def finish_pipeline(self) -> PipelineMetrics:
        """Finish pipeline monitoring and return metrics"""
        if self.start_time:
            self.metrics.total_latency = (time.time() - self.start_time) * 1000
        
        # Calculate error rate
        total_operations = self.metrics.success_count + self.metrics.error_count
        error_rate = self.metrics.error_count / total_operations if total_operations > 0 else 0
        
        # Check thresholds and log warnings
        self._check_performance_thresholds(error_rate)
        
        logger.info(f"Pipeline complete: {self.metrics.total_latency:.1f}ms, "
                   f"success: {self.metrics.success_count}, errors: {self.metrics.error_count}")
        
        return self.metrics
    
    def _check_performance_thresholds(self, error_rate: float):
        """Check performance thresholds and log alerts"""
        if self.metrics.total_latency > self.LATENCY_CRITICAL:
            logger.critical(f"Pipeline latency CRITICAL: {self.metrics.total_latency:.1f}ms > {self.LATENCY_CRITICAL}ms")
        elif self.metrics.total_latency > self.LATENCY_WARNING:
            logger.warning(f"Pipeline latency WARNING: {self.metrics.total_latency:.1f}ms > {self.LATENCY_WARNING}ms")
        
        if self.metrics.memory_peak > self.MEMORY_CRITICAL:
            logger.critical(f"Memory usage CRITICAL: {self.metrics.memory_peak:.1f}GB > {self.MEMORY_CRITICAL}GB")
        elif self.metrics.memory_peak > self.MEMORY_WARNING:
            logger.warning(f"Memory usage WARNING: {self.metrics.memory_peak:.1f}GB > {self.MEMORY_WARNING}GB")
        
        if error_rate > self.ERROR_RATE_CRITICAL:
            logger.critical(f"Error rate CRITICAL: {error_rate:.1%} > {self.ERROR_RATE_CRITICAL:.1%}")
        elif error_rate > self.ERROR_RATE_WARNING:
            logger.warning(f"Error rate WARNING: {error_rate:.1%} > {self.ERROR_RATE_WARNING:.1%}")

# ============================================================================
# MEMORY MANAGEMENT SYSTEM
# ============================================================================

class MemoryPool:
    """Pre-allocated memory pool for efficient data processing"""
    
    def __init__(self, initial_size_mb: int = 500):
        self.pool_size_bytes = initial_size_mb * 1024 * 1024
        self.allocated_buffers = []
        self.free_buffers = queue.Queue()
        self.lock = threading.Lock()
        
        # Pre-allocate buffers
        self._preallocate_buffers()
        logger.info(f"Memory pool initialized with {initial_size_mb}MB")
    
    def _preallocate_buffers(self):
        """Pre-allocate memory buffers to avoid fragmentation"""
        buffer_count = 10
        buffer_size = self.pool_size_bytes // buffer_count
        
        for _ in range(buffer_count):
            buffer = np.empty(buffer_size // 8, dtype=np.float64)  # 8 bytes per float64
            self.free_buffers.put(buffer)
            self.allocated_buffers.append(buffer)
    
    @contextmanager
    def get_buffer(self, size_bytes: int):
        """Get a buffer from the pool"""
        try:
            buffer = self.free_buffers.get_nowait()
            if len(buffer) * 8 >= size_bytes:
                yield buffer[:size_bytes // 8]
            else:
                # Fallback to regular allocation for oversized requests
                yield np.empty(size_bytes // 8, dtype=np.float64)
        except queue.Empty:
            # Pool exhausted, allocate new buffer
            yield np.empty(size_bytes // 8, dtype=np.float64)
        finally:
            try:
                self.free_buffers.put_nowait(buffer)
            except:
                pass  # Buffer may have been replaced

# ============================================================================
# S3 OPTIMIZATION SYSTEM  
# ============================================================================

class S3OptimizedClient:
    """Optimized S3 client with connection pooling and intelligent format selection"""
    
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            's3',
            config=boto3.session.Config(
                max_pool_connections=20,  # Connection pooling
                retries={'max_attempts': 3}
            )
        )
        
        # Format selection thresholds (bytes)
        self.PARQUET_THRESHOLD = 1024 * 1024  # 1MB
        self.JSON_SMALL_THRESHOLD = 100 * 1024  # 100KB
        
        logger.info("S3 optimized client initialized")
    
    async def upload_data_async(self, data: Any, key: str, metadata: Dict[str, str] = None) -> float:
        """Async upload with intelligent format selection"""
        start_time = time.time()
        
        # Determine optimal format based on data characteristics
        file_format, serialized_data = self._select_optimal_format(data)
        
        # Add format to metadata
        if metadata is None:
            metadata = {}
        metadata['data_format'] = file_format
        metadata['data_size'] = str(len(serialized_data))
        
        # Async upload
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=serialized_data,
                Metadata=metadata
            )
        )
        
        upload_time = time.time() - start_time
        logger.info(f"S3 upload complete: {key} ({file_format}, {len(serialized_data)} bytes) in {upload_time:.3f}s")
        return upload_time
    
    def _select_optimal_format(self, data: Any) -> Tuple[str, bytes]:
        """Select optimal storage format based on data characteristics"""
        if isinstance(data, pd.DataFrame):
            if len(data) * len(data.columns) * 8 > self.PARQUET_THRESHOLD:
                # Large DataFrame -> Parquet for compression and performance
                buffer = BytesIO()
                data.to_parquet(buffer, index=False)
                return 'parquet', buffer.getvalue()
            else:
                # Small DataFrame -> CSV for simplicity
                return 'csv', data.to_csv(index=False).encode()
        
        elif isinstance(data, dict) or isinstance(data, list):
            serialized = json.dumps(data, default=str).encode()
            if len(serialized) < self.JSON_SMALL_THRESHOLD:
                return 'json', serialized
            else:
                # Large JSON -> try compression
                import gzip
                compressed = gzip.compress(serialized)
                return 'json.gz', compressed
        
        else:
            # Default to JSON for unknown types
            return 'json', json.dumps(data, default=str).encode()

# ============================================================================
# PARALLEL PROCESSING PIPELINE
# ============================================================================

class ParallelDataPipeline:
    """High-performance parallel data processing pipeline"""
    
    def __init__(self, market_data_adapter, max_workers: int = 2):
        self.market_data_adapter = market_data_adapter
        self.max_workers = max_workers
        self.monitor = PerformanceMonitor()
        self.memory_pool = MemoryPool()
        self.s3_client = S3OptimizedClient("omega-singularity-ml")
        
        # Dynamic batching parameters
        self.max_batch_size = 1000
        self.memory_per_ticker_mb = 50  # Estimated memory per ticker processing
        
        logger.info(f"Parallel pipeline initialized with {max_workers} workers")
    
    def calculate_optimal_batch_size(self, ticker_count: int) -> int:
        """Calculate optimal batch size based on available memory and ticker count"""
        # Get available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Calculate max tickers that fit in memory
        max_tickers_by_memory = int((available_memory_gb * 1024) // self.memory_per_ticker_mb)
        
        # Apply constraints
        optimal_batch_size = min(
            ticker_count,
            max_tickers_by_memory,
            self.max_batch_size
        )
        
        logger.info(f"Optimal batch size: {optimal_batch_size} "
                   f"(tickers: {ticker_count}, memory_limit: {max_tickers_by_memory})")
        
        return optimal_batch_size
    
    async def process_ticker_async(self, ticker: str) -> Dict[str, Any]:
        """Async processing of single ticker with error handling"""
        result = {
            'ticker': ticker,
            'success': False,
            'error': None,
            'data': None,
            'features': None,
            'labels': None,
            'processing_time': 0,
            'memory_used': 0
        }
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            with self.monitor.monitor_operation(f"fetch_{ticker}"):
                # Fetch data from IBKR
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None,
                    lambda: self.market_data_adapter.fetch_data(ticker)
                )
                
                if df is None or df.empty:
                    result['error'] = f"No data for {ticker}"
                    return result
                
                result['data'] = df
            
            with self.monitor.monitor_operation(f"features_{ticker}"):
                # Feature engineering in thread pool
                from feature_engineering import generate_features
                X, y = await loop.run_in_executor(
                    None,
                    lambda: generate_features(df)
                )
                
                result['features'] = X
                result['labels'] = y
                result['success'] = True
                
        except Exception as e:
            result['error'] = f"Processing failed for {ticker}: {str(e)}"
            logger.error(f"Ticker {ticker} processing error: {e}")
        
        finally:
            result['processing_time'] = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            result['memory_used'] = end_memory - start_memory
            
            self.monitor.metrics.ticker_latencies[ticker] = result['processing_time'] * 1000
        
        return result
    
    async def process_tickers_parallel_optimized(self, tickers: List[str]) -> Dict[str, Any]:
        """Optimized parallel processing with ThreadPoolExecutor and async coordination"""
        
        self.monitor.start_pipeline()
        
        # Calculate optimal batch size
        batch_size = self.calculate_optimal_batch_size(len(tickers))
        
        # Split into batches for memory management
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]
        
        logger.info(f"Processing {len(tickers)} tickers in {len(batches)} batches "
                   f"(batch_size={batch_size}, workers={self.max_workers})")
        
        all_results = {}
        
        for batch_idx, batch_tickers in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}: {batch_tickers}")
            
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(self.max_workers)
            
            async def process_with_semaphore(ticker):
                async with semaphore:
                    return await self.process_ticker_async(ticker)
            
            # Process batch concurrently
            with self.monitor.monitor_operation(f"batch_{batch_idx}"):
                tasks = [process_with_semaphore(ticker) for ticker in batch_tickers]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for ticker, result in zip(batch_tickers, batch_results):
                if isinstance(result, Exception):
                    all_results[ticker] = {
                        'ticker': ticker,
                        'success': False,
                        'error': str(result),
                        'data': None,
                        'features': None,
                        'labels': None,
                        'processing_time': 0
                    }
                else:
                    all_results[ticker] = result
            
            # Memory cleanup between batches
            if batch_idx < len(batches) - 1:  # Don't cleanup after last batch
                collected = gc.collect()
                logger.info(f"Batch {batch_idx + 1} complete, freed {collected} objects")
        
        # Final metrics
        metrics = self.monitor.finish_pipeline()
        
        success_count = sum(1 for r in all_results.values() if r['success'])
        total_time = sum(r['processing_time'] for r in all_results.values())
        avg_time = total_time / len(all_results) if all_results else 0
        
        logger.info(f"Parallel processing complete: {success_count}/{len(tickers)} successful")
        logger.info(f"Total time: {metrics.total_latency:.1f}ms, "
                   f"Avg per ticker: {avg_time*1000:.1f}ms")
        
        return all_results

# ============================================================================
# INTEGRATION INTERFACE
# ============================================================================

def timing_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} completed in {(end_time - start_time)*1000:.1f}ms")
        return result
    return wrapper

class PipelineOptimizer:
    """Main interface for pipeline optimization"""
    
    def __init__(self, market_data_adapter):
        self.market_data_adapter = market_data_adapter
        self.pipeline = ParallelDataPipeline(market_data_adapter)
    
    @timing_decorator
    def process_tickers_optimized(self, tickers: List[str]) -> Dict[str, Any]:
        """Main entry point for optimized ticker processing"""
        try:
            # Run async processing in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self.pipeline.process_tickers_parallel_optimized(tickers)
                )
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Pipeline optimization failed: {e}")
            # Fallback to sequential processing
            logger.info("Falling back to sequential processing")
            return self._fallback_sequential(tickers)
    
    def _fallback_sequential(self, tickers: List[str]) -> Dict[str, Any]:
        """Fallback sequential processing for error recovery"""
        results = {}
        for ticker in tickers:
            try:
                result = asyncio.run(self.pipeline.process_ticker_async(ticker))
                results[ticker] = result
            except Exception as e:
                results[ticker] = {
                    'ticker': ticker,
                    'success': False,
                    'error': str(e),
                    'data': None,
                    'features': None,
                    'labels': None,
                    'processing_time': 0
                }
        return results

# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================

def benchmark_pipeline(market_data_adapter, tickers: List[str], iterations: int = 3):
    """Benchmark the optimized pipeline"""
    optimizer = PipelineOptimizer(market_data_adapter)
    
    results = []
    for i in range(iterations):
        logger.info(f"Benchmark iteration {i+1}/{iterations}")
        start_time = time.time()
        
        processed_results = optimizer.process_tickers_optimized(tickers)
        
        end_time = time.time()
        iteration_time = (end_time - start_time) * 1000
        
        success_count = sum(1 for r in processed_results.values() if r['success'])
        
        results.append({
            'iteration': i + 1,
            'total_time_ms': iteration_time,
            'success_count': success_count,
            'total_tickers': len(tickers),
            'avg_time_per_ticker_ms': iteration_time / len(tickers)
        })
        
        logger.info(f"Iteration {i+1} complete: {iteration_time:.1f}ms, "
                   f"{success_count}/{len(tickers)} successful")
    
    # Calculate averages
    avg_time = sum(r['total_time_ms'] for r in results) / len(results)
    avg_success = sum(r['success_count'] for r in results) / len(results)
    
    logger.info(f"Benchmark complete: avg {avg_time:.1f}ms, avg success {avg_success:.1f}/{len(tickers)}")
    
    return results

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    # Mock adapter for testing
    class MockAdapter:
        def fetch_data(self, ticker):
            import time
            time.sleep(0.1)  # Simulate network delay
            import pandas as pd
            return pd.DataFrame({'close': [100, 101, 102]})
    
    adapter = MockAdapter()
    test_tickers = ['ES1!', 'NQ1!', 'EURUSD', 'GBPUSD']
    
    benchmark_pipeline(adapter, test_tickers)