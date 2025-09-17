"""
Adaptive Trading System - Main Orchestrator

This is the main entry point that orchestrates all components:
- RL training and model management
- Paper trading execution
- Performance monitoring
- Continuous learning and adaptation

This system can learn from paper trading and automatically improve its strategies.
"""

import os
import sys
import time
import logging
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import argparse

# Local imports


from rl_trainer import RLTrainer, create_sample_data
from trading_decision_engine import TradingDecisionEngine
from paper_trading_executor import PaperTradingExecutor
from model_manager import ModelManager
from market_data_ibkr_adapter import IBKRIngestor
from monitoring.alerting_system import get_alerting_system, AlertSeverity, AlertType
from configs.market_data_config import IBKR_SYMBOLS

logger = logging.getLogger(__name__)


class AdaptiveTradingSystem:
    """
    Main orchestrator for the adaptive trading system
    """
    
    def __init__(self, config_file: str = None):
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.alerting = get_alerting_system()
        self.model_manager = ModelManager(self.config.get('model_manager', {}))
        self.rl_trainer = None
        self.paper_trader = None
        
        # System state
        self.is_running = False
        self.training_thread = None
        self.monitoring_thread = None
        
        # Performance tracking
        self.start_time = None
        self.training_cycles = 0
        self.last_model_update = None
        
        # Market data
        self.ibkr_ingestor = None
        self.historical_data = {}
        
        logger.info("Adaptive Trading System initialized")
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'system': {
                'mode': 'paper_trading',  # 'training', 'paper_trading', 'live_trading'
                'symbols': IBKR_SYMBOLS,
                'data_lookback_days': 365,
                'training_frequency_hours': 24,  # Retrain every 24 hours
                'performance_review_hours': 6,   # Review performance every 6 hours
                'auto_model_updates': True,
                'min_training_samples': 1000,
            },
            'trading': {
                'initial_capital': 100000,
                'risk_per_trade': 0.02,  # 2% risk per trade
                'max_daily_trades': 20,
                'max_drawdown_threshold': 0.15,  # Stop trading at 15% drawdown
                'decision_interval': 60,  # Make decisions every 60 seconds
            },
            'rl_training': {
                'algorithm': 'PPO',
                'total_timesteps': 500000,
                'learning_rate': 3e-4,
                'lookback_window': 50,
                'validation_episodes': 50,
                'save_frequency': 50000,
            },
            'model_manager': {
                'models_dir': 'models',
                'auto_deploy_threshold': 0.05,  # Deploy if 5% better
                'max_models_to_keep': 20,
            },
            'monitoring': {
                'log_level': 'INFO',
                'save_results_frequency': 300,  # Save every 5 minutes
                'results_dir': 'results',
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge configurations (file config overrides defaults)
                def merge_dict(default, override):
                    result = default.copy()
                    for key, value in override.items():
                        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                            result[key] = merge_dict(result[key], value)
                        else:
                            result[key] = value
                    return result
                
                return merge_dict(default_config, file_config)
            
            except Exception as e:
                logger.error(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def start(self):
        """Start the adaptive trading system"""
        if self.is_running:
            logger.warning("System is already running")
            return
        
        logger.info(f"Starting Adaptive Trading System in {self.config['system']['mode']} mode")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Initialize market data connection
            if not self._initialize_market_data():
                raise RuntimeError("Failed to initialize market data")
            
            # Load historical data for training
            if not self._load_historical_data():
                logger.warning("Failed to load historical data, using sample data")
                self._create_sample_data()
            
            # Initialize RL trainer
            self._initialize_rl_trainer()
            
            # Start appropriate mode
            mode = self.config['system']['mode']
            
            if mode == 'training':
                self._start_training_mode()
            elif mode == 'paper_trading':
                self._start_paper_trading_mode()
            elif mode == 'live_trading':
                self._start_live_trading_mode()
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Start monitoring
            self._start_monitoring()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            # Send startup alert
            self.alerting.send_alert(self.alerting.create_alert(
                severity=AlertSeverity.INFO,
                alert_type=AlertType.SYSTEM_ERROR,
                title="Trading System Started",
                message=f"Adaptive Trading System started in {mode} mode",
                details={
                    'mode': mode,
                    'symbols': self.config['system']['symbols'],
                    'start_time': self.start_time.isoformat()
                }
            ))
            
            logger.info("System startup complete")
        
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop the adaptive trading system"""
        if not self.is_running:
            return
        
        logger.info("Stopping Adaptive Trading System...")
        
        self.is_running = False
        
        # Stop paper trading
        if self.paper_trader:
            self.paper_trader.stop_trading()
        
        # Wait for threads to finish
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=30)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        # Save final state
        self._save_final_results()
        
        # Send shutdown alert
        self.alerting.send_alert(self.alerting.create_alert(
            severity=AlertSeverity.INFO,
            alert_type=AlertType.SYSTEM_ERROR,
            title="Trading System Stopped",
            message="Adaptive Trading System stopped",
            details={
                'stop_time': datetime.now().isoformat(),
                'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
                'training_cycles': self.training_cycles
            }
        ))
        
        logger.info("System shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def _initialize_market_data(self) -> bool:
        """Initialize connection to market data"""
        try:
            self.ibkr_ingestor = IBKRIngestor()
            logger.info("Market data connection initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize market data: {e}")
            return False
    
    def _load_historical_data(self) -> bool:
        """Load historical data for training"""
        if not self.ibkr_ingestor:
            return False
        
        lookback_days = self.config['system']['data_lookback_days']
        
        for symbol in self.config['system']['symbols']:
            try:
                logger.info(f"Loading historical data for {symbol}...")
                
                data = self.ibkr_ingestor.fetch_data(
                    symbol=symbol,
                    duration=f"{lookback_days} D",
                    barSize="1 hour",
                    whatToShow="TRADES",
                    useRTH=False
                )
                
                if data is not None and len(data) > 0:
                    self.historical_data[symbol] = data
                    logger.info(f"Loaded {len(data)} bars for {symbol}")
                else:
                    logger.warning(f"No data retrieved for {symbol}")
            
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
        
        return len(self.historical_data) > 0
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        logger.info("Creating sample data for testing...")
        
        for symbol in self.config['system']['symbols'][:2]:  # Only first 2 symbols for testing
            sample_data = create_sample_data()
            self.historical_data[symbol] = sample_data
            logger.info(f"Created sample data for {symbol}: {len(sample_data)} bars")
    
    def _initialize_rl_trainer(self):
        """Initialize RL trainer with data"""
        if not self.historical_data:
            logger.error("No historical data available for training")
            return
        
        # Use the first symbol's data for training (can be extended to multi-symbol)
        first_symbol = list(self.historical_data.keys())[0]
        training_data = self.historical_data[first_symbol]
        
        # Initialize trainer
        self.rl_trainer = RLTrainer(self.config['rl_training'])
        self.rl_trainer.load_data(training_data, first_symbol)
        
        logger.info(f"RL trainer initialized with {len(training_data)} samples")
    
    def _start_training_mode(self):
        """Start pure training mode"""
        logger.info("Starting training mode...")
        
        if not self.rl_trainer:
            raise RuntimeError("RL trainer not initialized")
        
        # Create and start training
        self.rl_trainer.create_model()
        
        def training_loop():
            while self.is_running:
                try:
                    logger.info("Starting RL training cycle...")
                    training_stats = self.rl_trainer.train()
                    
                    # Evaluate model
                    eval_results = self.rl_trainer.evaluate()
                    
                    # Register model
                    training_data_hash = f"training_{self.training_cycles}"
                    model_id = self.model_manager.register_model(
                        model=self.rl_trainer.model,
                        model_type='RL',
                        algorithm=self.config['rl_training']['algorithm'],
                        training_data_hash=training_data_hash,
                        performance_metrics=eval_results,
                        hyperparameters=self.config['rl_training'],
                        training_info=training_stats
                    )
                    
                    self.training_cycles += 1
                    self.last_model_update = datetime.now()
                    
                    logger.info(f"Training cycle {self.training_cycles} completed. Model: {model_id}")
                    
                    # Sleep until next training cycle
                    if self.is_running:
                        sleep_hours = self.config['system']['training_frequency_hours']
                        time.sleep(sleep_hours * 3600)
                
                except Exception as e:
                    logger.error(f"Error in training loop: {e}")
                    time.sleep(600)  # Sleep 10 minutes on error
        
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
    
    def _start_paper_trading_mode(self):
        """Start paper trading mode with continuous learning"""
        logger.info("Starting paper trading mode...")
        
        # Initialize paper trader
        paper_config = {
            **self.config['trading'],
            'symbols': self.config['system']['symbols'],
            'use_live_data': self.ibkr_ingestor is not None
        }
        
        self.paper_trader = PaperTradingExecutor(paper_config)
        
        # Start paper trading
        self.paper_trader.start_trading()
        
        # Start continuous learning if enabled
        if self.config['system']['auto_model_updates'] and self.rl_trainer:
            self._start_continuous_learning()
        
        logger.info("Paper trading started")
    
    def _start_live_trading_mode(self):
        """Start live trading mode (placeholder - requires additional safety measures)"""
        logger.warning("Live trading mode not fully implemented - switching to paper trading")
        self._start_paper_trading_mode()
    
    def _start_continuous_learning(self):
        """Start continuous learning from trading experience"""
        def learning_loop():
            while self.is_running:
                try:
                    # Wait for training interval
                    sleep_hours = self.config['system']['training_frequency_hours']
                    time.sleep(sleep_hours * 3600)
                    
                    if not self.is_running:
                        break
                    
                    # Check if we have enough new trading data
                    if self.paper_trader:
                        status = self.paper_trader.get_current_status()
                        
                        if status['performance']['trades_executed'] < self.config['system']['min_training_samples']:
                            logger.info("Insufficient trading data for retraining")
                            continue
                    
                    # Retrain model with latest data
                    logger.info("Starting continuous learning cycle...")
                    
                    # Reload fresh data
                    if self._load_historical_data():
                        # Reinitialize trainer with new data
                        first_symbol = list(self.historical_data.keys())[0]
                        self.rl_trainer.load_data(self.historical_data[first_symbol], first_symbol)
                        
                        # Create new model
                        self.rl_trainer.create_model()
                        
                        # Train
                        training_stats = self.rl_trainer.train()
                        eval_results = self.rl_trainer.evaluate()
                        
                        # Register new model
                        training_data_hash = f"continuous_{self.training_cycles}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                        model_id = self.model_manager.register_model(
                            model=self.rl_trainer.model,
                            model_type='RL',
                            algorithm=self.config['rl_training']['algorithm'],
                            training_data_hash=training_data_hash,
                            performance_metrics=eval_results,
                            hyperparameters=self.config['rl_training'],
                            training_info=training_stats
                        )
                        
                        self.training_cycles += 1
                        self.last_model_update = datetime.now()
                        
                        logger.info(f"Continuous learning cycle {self.training_cycles} completed. Model: {model_id}")
                        
                        # Alert on new model
                        self.alerting.send_alert(self.alerting.create_alert(
                            severity=AlertSeverity.INFO,
                            alert_type=AlertType.UNUSUAL_ACTIVITY,
                            title="Model Retrained",
                            message=f"New model trained from live trading data: {model_id}",
                            details={
                                'model_id': model_id,
                                'training_return': eval_results.get('mean_profit', 0),
                                'validation_return': eval_results.get('mean_profit', 0),
                                'cycle': self.training_cycles
                            }
                        ))
                
                except Exception as e:
                    logger.error(f"Error in continuous learning: {e}")
                    time.sleep(1800)  # Sleep 30 minutes on error
        
        self.training_thread = threading.Thread(target=learning_loop, daemon=True)
        self.training_thread.start()
    
    def _start_monitoring(self):
        """Start system monitoring"""
        def monitoring_loop():
            while self.is_running:
                try:
                    # Save current state
                    self._save_current_state()
                    
                    # Monitor performance
                    self._monitor_performance()
                    
                    # Check system health
                    self._check_system_health()
                    
                    # Sleep until next monitoring cycle
                    time.sleep(self.config['monitoring']['save_results_frequency'])
                
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(60)  # Sleep 1 minute on error
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _save_current_state(self):
        """Save current system state"""
        results_dir = Path(self.config['monitoring']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
            'mode': self.config['system']['mode'],
            'training_cycles': self.training_cycles,
            'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None,
            'model_manager_stats': self.model_manager.get_registry_stats(),
            'current_deployed_model': self.model_manager.get_current_deployed_model()
        }
        
        # Add paper trading status if available
        if self.paper_trader:
            state['paper_trading'] = self.paper_trader.get_current_status()
        
        state_file = results_dir / f"system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _monitor_performance(self):
        """Monitor system performance"""
        if not self.paper_trader:
            return
        
        status = self.paper_trader.get_current_status()
        
        # Check for concerning performance
        portfolio = status['portfolio']
        
        if portfolio['total_return_pct'] < -self.config['trading']['max_drawdown_threshold'] * 100:
            self.alerting.send_alert(self.alerting.create_alert(
                severity=AlertSeverity.CRITICAL,
                alert_type=AlertType.RISK_LIMIT,
                title="Maximum Drawdown Exceeded",
                message=f"Portfolio return {portfolio['total_return_pct']:.2f}% exceeds maximum drawdown threshold",
                details={'current_return': portfolio['total_return_pct'], 'threshold': -self.config['trading']['max_drawdown_threshold'] * 100}
            ))
        
        # Record performance in model manager if model is deployed
        deployed_model = self.model_manager.get_current_deployed_model()
        if deployed_model:
            self.model_manager.record_deployment_performance(deployed_model, {
                'live_return': portfolio['total_return_pct'],
                'daily_pnl': portfolio['daily_pnl'],
                'trades_executed': status['performance']['trades_executed']
            })
    
    def _check_system_health(self):
        """Check system health metrics"""
        # Check if training is stuck
        if self.last_model_update:
            hours_since_update = (datetime.now() - self.last_model_update).total_seconds() / 3600
            max_hours = self.config['system']['training_frequency_hours'] * 2  # Allow 2x normal interval
            
            if hours_since_update > max_hours:
                self.alerting.send_alert(self.alerting.create_alert(
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.SYSTEM_ERROR,
                    title="Model Update Overdue",
                    message=f"No model updates for {hours_since_update:.1f} hours",
                    details={'hours_since_update': hours_since_update, 'expected_frequency': self.config['system']['training_frequency_hours']}
                ))
        
        # Check paper trading health
        if self.paper_trader and not self.paper_trader.is_trading:
            self.alerting.send_alert(self.alerting.create_alert(
                severity=AlertSeverity.ERROR,
                alert_type=AlertType.SYSTEM_ERROR,
                title="Paper Trading Stopped",
                message="Paper trading has stopped unexpectedly"
            ))
    
    def _save_final_results(self):
        """Save final results when stopping"""
        results_dir = Path(self.config['monitoring']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            'session_summary': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_runtime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
                'mode': self.config['system']['mode'],
                'training_cycles_completed': self.training_cycles,
            },
            'model_manager_final_stats': self.model_manager.get_registry_stats(),
            'models_created': self.model_manager.list_models(limit=10),
        }
        
        # Add paper trading results if available
        if self.paper_trader:
            final_results['paper_trading_final_status'] = self.paper_trader.get_current_status()
        
        results_file = results_dir / f"final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Final results saved to {results_file}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'is_running': self.is_running,
            'mode': self.config['system']['mode'],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600 if self.start_time else 0,
            'training_cycles': self.training_cycles,
            'last_model_update': self.last_model_update.isoformat() if self.last_model_update else None,
            'symbols': self.config['system']['symbols'],
            'model_manager': self.model_manager.get_registry_stats(),
            'deployed_model': self.model_manager.get_current_deployed_model()
        }
        
        if self.paper_trader:
            status['paper_trading'] = self.paper_trader.get_current_status()
        
        return status


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Adaptive Trading System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['training', 'paper_trading', 'live_trading'], 
                       help='Trading mode')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('adaptive_trading_system.log')
        ]
    )
    
    # Override mode if specified
    config_dict = {}
    if args.mode:
        config_dict = {'system': {'mode': args.mode}}
        
        # Save temporary config
        temp_config_file = 'temp_config.json'
        with open(temp_config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        args.config = temp_config_file
    
    # Create and start system
    system = AdaptiveTradingSystem(args.config)
    
    try:
        system.start()
        
        # Keep running until interrupted
        while system.is_running:
            time.sleep(10)
            
            # Print periodic status
            if datetime.now().second % 60 == 0:  # Every minute
                status = system.get_system_status()
                logger.info(f"System Status: Running for {status['uptime_hours']:.1f}h, "
                           f"Cycles: {status['training_cycles']}, "
                           f"Models: {status['model_manager']['total_models']}")
    
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        system.stop()
        
        # Cleanup temp config
        if args.mode and Path('temp_config.json').exists():
            os.remove('temp_config.json')


if __name__ == "__main__":
    main()