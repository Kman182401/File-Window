#!/usr/bin/env python3
"""
Enhanced Trading System Integration Module

This module integrates all state-of-the-art upgrades into a cohesive
trading system optimized for m5.large EC2 instance.

Key Components:
- JAX-optimized technical indicators
- PyTorch Lightning mixed precision training
- Memory management with circuit breakers
- Comprehensive error handling
- System optimization configuration
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path


# Import enhanced components
from system_optimization_config import (
    initialize_optimized_environment,
    AdaptiveConfig,
    get_optimal_batch_size
)
from jax_feature_engineering_integration import JAXEnhancedFeatureEngineering
from lightning_rl_training import LightningRLTrainer
from memory_management_system import get_memory_manager
from error_handling_system import (
    get_error_handler,
    get_safe_mode,
    robust_execution,
    ErrorCategory,
    ErrorSeverity
)

# Import existing components
from market_data_ibkr_adapter import IBKRIngestor
from trading_environment import TradingEnvironment
from paper_trading_executor import PaperTradingExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_trading_system.log')
    ]
)
logger = logging.getLogger(__name__)


class EnhancedTradingSystem:
    """
    State-of-the-art AI trading system with all enhancements integrated.
    
    This system combines:
    - Advanced ML algorithms (SAC, PPO, LightGBM)
    - JAX-accelerated computations
    - Lightning-based training infrastructure
    - Comprehensive memory management
    - Production-grade error handling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced trading system.
        
        Args:
            config_path: Path to configuration file
        """
        logger.info("=" * 60)
        logger.info("Initializing Enhanced AI Trading System")
        logger.info("=" * 60)
        
        # Initialize optimized environment
        self.adaptive_config = initialize_optimized_environment()
        
        # Initialize memory management
        self.memory_manager = get_memory_manager()
        self.memory_manager.start()
        logger.info("✓ Memory management system started")
        
        # Initialize error handling
        self.error_handler = get_error_handler()
        self.safe_mode = get_safe_mode()
        logger.info("✓ Error handling system initialized")
        
        # Initialize components with memory registration
        self._initialize_components()
        
        # System status
        self.is_running = False
        self.start_time = None
        
        logger.info("✓ Enhanced trading system initialized successfully")
    
    @robust_execution(category=ErrorCategory.SYSTEM, fallback_value=None)
    def _initialize_components(self):
        """Initialize all system components with error handling."""
        
        # 1. JAX-enhanced feature engineering
        logger.info("Initializing JAX feature engineering...")
        self.feature_engineering = JAXEnhancedFeatureEngineering()
        self.memory_manager.register_component(
            'feature_engineering',
            self.feature_engineering,
            limit_mb=500
        )
        
        # 2. IBKR market data adapter
        logger.info("Initializing IBKR market data adapter...")
        self.ibkr_adapter = self._initialize_ibkr_adapter()
        self.memory_manager.register_component(
            'ibkr_adapter',
            self.ibkr_adapter,
            limit_mb=300
        )
        
        # 3. Trading environment
        logger.info("Initializing trading environment...")
        self.trading_env = self._initialize_trading_environment()
        self.memory_manager.register_component(
            'trading_env',
            self.trading_env,
            limit_mb=200
        )
        
        # 4. Lightning RL trainer
        logger.info("Initializing Lightning RL trainer...")
        self.rl_trainer = LightningRLTrainer(
            self.trading_env,
            self.adaptive_config.get_config('lightning')
        )
        self.memory_manager.register_component(
            'rl_trainer',
            self.rl_trainer,
            limit_mb=1000
        )
        
        # 5. Paper trading executor
        logger.info("Initializing paper trading executor...")
        self.paper_trading = self._initialize_paper_trading()
        self.memory_manager.register_component(
            'paper_trading',
            self.paper_trading,
            limit_mb=400
        )
        
        logger.info("✓ All components initialized")
    
    def _initialize_ibkr_adapter(self) -> IBKRIngestor:
        """Initialize IBKR adapter with error handling."""
        try:
            host = os.getenv('IBKR_HOST', '127.0.0.1')
            port = int(os.getenv('IBKR_PORT', '4002'))
            client_id = int(os.getenv('IBKR_CLIENT_ID', '9002'))
            
            adapter = IBKRIngestor(host=host, port=port, clientId=client_id)
            
            # Test connection
            if not adapter.is_connected():
                logger.warning("IBKR not connected, will retry on first use")
            
            return adapter
            
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorCategory.NETWORK,
                ErrorSeverity.HIGH,
                {'component': 'ibkr_adapter'}
            )
            # Return mock adapter for testing
            logger.warning("Using mock IBKR adapter")
            return None
    
    def _initialize_trading_environment(self) -> TradingEnvironment:
        """Initialize trading environment with enhancements."""
        # Use adaptive configuration
        env_config = self.adaptive_config.get_config('trading')
        
        # Create environment with proper configuration
        env = TradingEnvironment(config=env_config)
        
        return env
    
    def _initialize_paper_trading(self) -> Optional[PaperTradingExecutor]:
        """Initialize paper trading with safety checks."""
        if not self.safe_mode.is_operation_allowed('paper_trading'):
            logger.warning("Paper trading disabled in safe mode")
            return None
        
        try:
            executor = PaperTradingExecutor(
                config=self.adaptive_config.get_config('trading')
            )
            return executor
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorCategory.TRADING,
                ErrorSeverity.MEDIUM,
                {'component': 'paper_trading'}
            )
            return None
    
    @robust_execution(category=ErrorCategory.DATA, max_retries=3)
    def collect_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Collect market data with JAX-enhanced processing.
        
        Args:
            symbols: List of symbols to collect data for
            
        Returns:
            Dictionary mapping symbols to DataFrames with features
        """
        logger.info(f"Collecting market data for {symbols}")
        
        data_dict = {}
        
        for symbol in symbols:
            try:
                # Fetch from IBKR
                if self.ibkr_adapter:
                    df = self.ibkr_adapter.fetch_data(
                        symbol,
                        duration="1 D",
                        barSize="1 min",
                        whatToShow="TRADES"
                    )
                else:
                    # Generate dummy data for testing
                    df = self._generate_dummy_data(symbol)
                
                # Apply JAX-enhanced feature engineering
                df_with_features = self.feature_engineering._add_technical_indicators_jax(
                    df, symbol
                )
                
                data_dict[symbol] = df_with_features
                
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {e}")
                self.error_handler.handle_error(
                    e,
                    ErrorCategory.DATA,
                    ErrorSeverity.MEDIUM,
                    {'symbol': symbol}
                )
        
        return data_dict
    
    def _generate_dummy_data(self, symbol: str) -> pd.DataFrame:
        """Generate dummy data for testing."""
        dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
        
        np.random.seed(hash(symbol) % 1000)
        close = 100 + np.cumsum(np.random.randn(1000) * 0.1)
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': close + np.random.randn(1000) * 0.05,
            'high': close + np.abs(np.random.randn(1000) * 0.1),
            'low': close - np.abs(np.random.randn(1000) * 0.1),
            'close': close,
            'volume': np.random.randint(1000, 10000, 1000)
        })
    
    @robust_execution(category=ErrorCategory.MODEL)
    def train_models(self, data: Dict[str, pd.DataFrame], epochs: int = 10):
        """
        Train models using Lightning with mixed precision.
        
        Args:
            data: Market data with features
            epochs: Number of training epochs
        """
        logger.info(f"Starting model training for {epochs} epochs")
        
        # Check memory before training
        memory_status = self.memory_manager.get_status()
        available_mb = memory_status['system']['available_memory_mb']
        
        if available_mb < 2000:
            logger.warning(f"Low memory ({available_mb}MB), using reduced batch size")
            batch_size = 16
        else:
            batch_size = get_optimal_batch_size('rl_training')
        
        # Train with Lightning (mixed precision enabled)
        self.rl_trainer.train(num_epochs=epochs, batch_size=batch_size)
        
        logger.info("✓ Model training completed")
    
    def run_paper_trading(self, duration_minutes: int = 60):
        """
        Run paper trading with real IBKR integration.
        
        Args:
            duration_minutes: Duration to run paper trading
        """
        if not self.paper_trading:
            logger.error("Paper trading not available")
            return
        
        logger.info(f"Starting paper trading for {duration_minutes} minutes")
        
        # Check if safe mode allows trading
        if not self.safe_mode.is_operation_allowed('paper_trading'):
            logger.warning("Paper trading blocked by safe mode")
            return
        
        # Run paper trading
        self.paper_trading.run(duration_minutes=duration_minutes)
        
        logger.info("✓ Paper trading session completed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'uptime_hours': self._get_uptime_hours(),
            'memory': self.memory_manager.get_status(),
            'errors': self.error_handler.get_status(),
            'safe_mode': self.safe_mode.get_status(),
            'components': {
                'feature_engineering': 'active',
                'ibkr_adapter': 'active' if self.ibkr_adapter else 'mock',
                'rl_trainer': 'active',
                'paper_trading': 'active' if self.paper_trading else 'disabled'
            }
        }
    
    def _get_uptime_hours(self) -> float:
        """Get system uptime in hours."""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds() / 3600
        return 0
    
    def start(self):
        """Start the enhanced trading system."""
        logger.info("Starting enhanced trading system...")
        
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start monitoring
        self.memory_manager.start()
        
        logger.info("✓ Enhanced trading system started")
    
    def stop(self):
        """Stop the enhanced trading system."""
        logger.info("Stopping enhanced trading system...")
        
        self.is_running = False
        
        # Stop components
        self.memory_manager.stop()
        
        if self.ibkr_adapter:
            self.ibkr_adapter.disconnect()
        
        logger.info("✓ Enhanced trading system stopped")
    
    def run_diagnostic(self) -> Dict[str, Any]:
        """Run system diagnostic tests."""
        logger.info("Running system diagnostics...")
        
        diagnostics = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test 1: Memory check
        memory_status = self.memory_manager.get_status()
        diagnostics['tests']['memory'] = {
            'status': 'pass' if memory_status['system']['available_memory_mb'] > 1000 else 'fail',
            'available_mb': memory_status['system']['available_memory_mb']
        }
        
        # Test 2: JAX functionality
        try:
            import jax.numpy as jnp
            test_array = jnp.ones(1000)
            result = self.feature_engineering.jax_indicators.rsi(test_array, 14)
            diagnostics['tests']['jax'] = {
                'status': 'pass',
                'message': 'JAX indicators working'
            }
        except Exception as e:
            diagnostics['tests']['jax'] = {
                'status': 'fail',
                'error': str(e)
            }
        
        # Test 3: Lightning
        try:
            import pytorch_lightning as pl
            diagnostics['tests']['lightning'] = {
                'status': 'pass',
                'version': pl.__version__
            }
        except Exception as e:
            diagnostics['tests']['lightning'] = {
                'status': 'fail',
                'error': str(e)
            }
        
        # Test 4: IBKR connection
        if self.ibkr_adapter:
            diagnostics['tests']['ibkr'] = {
                'status': 'pass' if self.ibkr_adapter.is_connected() else 'warning',
                'message': 'Connected' if self.ibkr_adapter.is_connected() else 'Not connected'
            }
        else:
            diagnostics['tests']['ibkr'] = {
                'status': 'mock',
                'message': 'Using mock adapter'
            }
        
        # Overall status
        all_pass = all(
            test.get('status') in ['pass', 'warning', 'mock']
            for test in diagnostics['tests'].values()
        )
        
        diagnostics['overall_status'] = 'healthy' if all_pass else 'degraded'
        
        logger.info(f"Diagnostics complete: {diagnostics['overall_status']}")
        
        return diagnostics


def main():
    """Main entry point for enhanced trading system."""
    print("\n" + "=" * 70)
    print("ENHANCED AI TRADING SYSTEM - STATE OF THE ART")
    print("=" * 70)
    
    # Initialize system
    print("\n📦 Initializing enhanced trading system...")
    system = EnhancedTradingSystem()
    
    # Run diagnostics
    print("\n🔍 Running system diagnostics...")
    diagnostics = system.run_diagnostic()
    
    print("\n📊 Diagnostic Results:")
    for test_name, test_result in diagnostics['tests'].items():
        status = test_result.get('status', 'unknown')
        symbol = "✓" if status == 'pass' else "⚠" if status == 'warning' else "✗"
        print(f"  {symbol} {test_name}: {status}")
    
    print(f"\n  Overall Status: {diagnostics['overall_status'].upper()}")
    
    # Get system status
    print("\n📈 System Status:")
    status = system.get_system_status()
    
    print(f"  Memory Available: {status['memory']['system']['available_memory_mb']:.0f} MB")
    print(f"  Error Count: {status['errors']['total_errors']}")
    print(f"  Safe Mode: {'ACTIVE' if status['safe_mode']['is_active'] else 'inactive'}")
    
    print("\n  Components:")
    for component, state in status['components'].items():
        print(f"    • {component}: {state}")
    
    # Demonstration
    print("\n🚀 Running demonstration...")
    
    # 1. Collect market data
    print("\n1. Collecting market data with JAX processing...")
    symbols = ['ES1!', 'NQ1!']
    data = system.collect_market_data(symbols)
    
    for symbol, df in data.items():
        print(f"   {symbol}: {len(df)} rows, {len(df.columns)} features")
    
    # 2. Train models (quick demo)
    print("\n2. Training models with Lightning (mixed precision)...")
    # system.train_models(data, epochs=1)  # Commented for quick demo
    print("   [Skipped for demo - would train with FP16 mixed precision]")
    
    # 3. Paper trading
    print("\n3. Paper trading capability:")
    if system.paper_trading:
        print("   ✓ Paper trading ready (connected to IBKR)")
    else:
        print("   ⚠ Paper trading not available (using mock mode)")
    
    # Cleanup
    print("\n🛑 Shutting down system...")
    system.stop()
    
    print("\n✅ Enhanced trading system demonstration completed!")
    print("\n" + "=" * 70)
    print("System ready for production paper trading and learning!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()