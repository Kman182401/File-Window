#!/usr/bin/env python3
"""
5-Loop Trading Pipeline Monitor

This script runs the trading pipeline for exactly 5 iterations while capturing
comprehensive telemetry on ML models, RL models, and the super-model decisions.
All trades are paper-only with extensive logging and monitoring.

Usage:
    python tools/pipeline_monitor_5loops.py

Environment Variables:
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID - IBKR connection settings
    ALLOW_ORDERS, DRY_RUN - Safety flags (forced to paper mode)
    PYTHONHASHSEED - Random seed for reproducibility
"""

import os
import sys
import json
import time
import signal
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Configurable timeout constants
PER_LOOP_TIMEOUT = int(os.getenv("PER_LOOP_TIMEOUT", "300"))  # Default 5 minutes per loop
OVERALL_TIMEOUT = int(os.getenv("OVERALL_TIMEOUT", "1800"))  # Default 30 minutes overall

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import project modules: prefer src/ package, fallback to legacy import path
try:
    from src.rl_trading_pipeline import RLTradingPipeline, default_config
except Exception:
    from rl_trading_pipeline import RLTradingPipeline, default_config
from tools.super_model import SuperModel, simulate_paper_trade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('pipeline_monitor')

# Try to import Historical Replay modules
REPLAY_AVAILABLE = False
try:
    from tools.historical_replay import HistoricalReplay, ReplayConfig
    from tools.simulation_broker import SimulationBroker
    REPLAY_AVAILABLE = True
    logger.info("Historical Replay Mode modules loaded successfully")
except ImportError as e:
    logger.warning(f"Historical Replay Mode not available: {e}")

class Pipeline5LoopMonitor:
    """
    Monitors and controls the trading pipeline for exactly 5 loops.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy()
        self.loop_count = 0
        self.target_loops = 5
        self.start_time = datetime.utcnow()
        self.loop_timeout = PER_LOOP_TIMEOUT  # Use configurable timeout
        
        # Telemetry storage
        self.loop_metrics = []
        self.ml_metrics = []
        self.rl_metrics = []
        self.super_decisions = []
        self.paper_trades = []
        self.errors = []
        
        # Initialize super model
        self.super_model = SuperModel(ml_weight=0.6, rl_weight=0.4)
        
        # Setup logging files
        self.timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.expanduser("~/logs/monitors")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.jsonl_log = os.path.join(self.log_dir, f"pipeline_5loops_{self.timestamp}.jsonl")
        self.human_log = os.path.join(self.log_dir, f"pipeline_5loops_{self.timestamp}.log")
        
        # Initialize Historical Replay Mode if enabled
        self.replay_mode = int(os.getenv("REPLAY_MODE", "0")) == 1
        self.replay = None
        self.sim_broker = None
        
        if self.replay_mode and REPLAY_AVAILABLE:
            logger.info("Initializing Historical Replay Mode...")
            try:
                # Setup replay configuration
                replay_config = ReplayConfig(
                    symbols=["ES", "NQ", "6B", "6E", "6A", "GC"],  # IBKR futures symbols
                    start_date="2024-01-01",
                    end_date="2024-12-31",
                    s3_bucket="omega-singularity-ml",
                    s3_prefix="market_data",
                    speed_multiplier=60.0,  # 60x speed for testing
                    cache_size_mb=512,
                    realism_mode="realistic"
                )
                
                # Initialize replay service
                self.replay = HistoricalReplay(replay_config)
                
                # Initialize simulation broker
                self.sim_broker = SimulationBroker(
                    initial_capital=100000.0,
                    commission_per_contract=2.25,
                    slippage_ticks=1
                )
                
                logger.info(f"Replay Mode initialized with {len(replay_config.symbols)} symbols")
                logger.info(f"Date range: {replay_config.start_date} to {replay_config.end_date}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Replay Mode: {e}")
                self.replay_mode = False
                self.replay = None
                self.sim_broker = None
        
        # Force paper-only mode
        self._enforce_safety_mode()
        
    def _enforce_safety_mode(self):
        """Force all safety settings for paper-only mode."""
        safety_env = {
            'ALLOW_ORDERS': '0',
            'DRY_RUN': '1', 
            'IBKR_PORT': '4002',  # Paper trading port
            'PYTHONHASHSEED': '0'  # Deterministic randomization
        }
        
        for key, value in safety_env.items():
            os.environ[key] = value
            logger.info(f"Safety mode: {key}={value}")
            
        # Update config for paper mode
        self.config.update({
            'paper_trading_only': True,
            'max_position_size': 1,  # Single contract only
            'disable_drift_detection': False,  # Re-enable for safety
            'drift_detection_threshold': 3.0
        })
        
    def run_5_loops(self) -> Dict[str, Any]:
        """
        Execute exactly 5 pipeline loops with comprehensive monitoring.
        
        Returns:
            Dict with summary metrics and status
        """
        logger.info("=" * 80)
        logger.info(f"Starting 5-Loop Trading Pipeline Monitor")
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Target loops: {self.target_loops}")
        logger.info(f"Loop timeout: {self.loop_timeout}s")
        logger.info("=" * 80)
        
        try:
            # Initialize pipeline
            pipeline = RLTradingPipeline(self.config)
            
            # Hook into pipeline for per-loop monitoring
            original_main_loop = pipeline._main_loop
            pipeline._main_loop = self._monitored_main_loop(pipeline, original_main_loop)
            
            # Run the loops
            start_time = time.time()
            while self.loop_count < self.target_loops:
                loop_start = time.time()
                logger.info(f"Starting loop {self.loop_count + 1}/{self.target_loops}")
                print(f"[monitor] Loop {self.loop_count + 1}/5: start (timeout={self.loop_timeout}s)", flush=True)
                
                try:
                    # Set timeout for this loop
                    signal.signal(signal.SIGALRM, self._loop_timeout_handler)
                    signal.alarm(self.loop_timeout)
                    
                    # Run one pipeline iteration
                    pipeline._main_loop()
                    
                    # Cancel timeout
                    signal.alarm(0)
                    
                    loop_duration = time.time() - loop_start
                    self._log_loop_completion(self.loop_count + 1, loop_duration)
                    print(f"[monitor] Loop {self.loop_count + 1}/5: done in {loop_duration:.1f}s", flush=True)
                    
                except Exception as e:
                    loop_duration = time.time() - loop_start
                    self._log_loop_error(self.loop_count + 1, e, loop_duration)
                    
                    # Continue to next loop on error
                    if self.loop_count < self.target_loops - 1:
                        logger.warning(f"Loop {self.loop_count + 1} failed, continuing to next loop")
                        time.sleep(5)  # Brief recovery pause
                    
                finally:
                    signal.alarm(0)  # Ensure timeout is cancelled
                    self.loop_count += 1
                    
            total_duration = time.time() - start_time
            
            # Generate final report
            final_report = self._generate_final_report(total_duration)
            
            # Add replay-specific metrics to final report
            if self.replay_mode and self.replay and REPLAY_AVAILABLE:
                replay_stats = self.replay.get_stats()
                final_report["replay_metrics"] = replay_stats
                
                if self.sim_broker:
                    broker_summary = self.sim_broker.get_portfolio_summary()
                    final_report["realistic_simulation_results"] = broker_summary
                    
                    # Export detailed trade log and equity curve
                    try:
                        trade_log_file = os.path.join(self.log_dir, f"trade_log_{self.timestamp}.csv")
                        trades_df = self.sim_broker.export_trade_log()
                        if not trades_df.empty:
                            trades_df.to_csv(trade_log_file, index=False)
                            final_report["log_files"]["trade_log"] = trade_log_file
                        
                        equity_curve_file = os.path.join(self.log_dir, f"equity_curve_{self.timestamp}.csv")
                        equity_df = self.sim_broker.export_equity_curve()
                        if not equity_df.empty:
                            equity_df.to_csv(equity_curve_file, index=False)
                            final_report["log_files"]["equity_curve"] = equity_curve_file
                            
                    except Exception as e:
                        logger.error(f"Error exporting broker data: {e}")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Fatal error in 5-loop monitor: {e}")
            logger.error(traceback.format_exc())
            return {
                "status": "FAILED",
                "error": str(e),
                "completed_loops": self.loop_count,
                "target_loops": self.target_loops
            }
            
    def _monitored_main_loop(self, pipeline, original_main_loop):
        """
        Wrapper around the original main loop to inject monitoring.
        """
        def monitored_loop():
            loop_start_time = datetime.utcnow()
            loop_metrics = {
                "loop_number": self.loop_count + 1,
                "start_time": loop_start_time.isoformat(),
                "ml_models": {},
                "rl_models": {},
                "super_decisions": [],
                "paper_trades": [],
                "errors": []
            }
            
            try:
                # Call original main loop
                original_main_loop()
                
                # Capture any ML metrics from pipeline
                if hasattr(pipeline, 'ml_models'):
                    loop_metrics["ml_models"] = self._extract_ml_metrics(pipeline.ml_models)
                
                # Capture any RL metrics from pipeline  
                if hasattr(pipeline, 'ppo_model'):
                    loop_metrics["rl_models"] = self._extract_rl_metrics(pipeline.ppo_model)
                
                # Generate super-model decisions
                super_decisions = self._generate_super_decisions(loop_metrics)
                loop_metrics["super_decisions"] = super_decisions
                
                # Simulate paper trades
                paper_trades = self._simulate_paper_trades(super_decisions)
                loop_metrics["paper_trades"] = paper_trades
                
                loop_metrics["status"] = "SUCCESS"
                
            except Exception as e:
                loop_metrics["status"] = "ERROR"
                loop_metrics["error"] = str(e)
                loop_metrics["traceback"] = traceback.format_exc()
                self.errors.append({
                    "loop": self.loop_count + 1,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            finally:
                loop_metrics["end_time"] = datetime.utcnow().isoformat()
                loop_metrics["duration_seconds"] = (
                    datetime.utcnow() - loop_start_time
                ).total_seconds()
                
                # Log to JSONL
                self._log_to_jsonl(loop_metrics)
                
                # Store for final report
                self.loop_metrics.append(loop_metrics)
                
        return monitored_loop
    
    def _extract_ml_metrics(self, ml_models: Dict) -> Dict[str, Any]:
        """Extract metrics from ML models."""
        ml_metrics = {}
        
        for ticker, models in ml_models.items():
            ticker_metrics = {}
            for model_name, model in models.items():
                try:
                    # Basic model info
                    model_info = {
                        "type": type(model).__name__,
                        "trained": True
                    }
                    
                    # Try to get feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        model_info["feature_importance_mean"] = float(np.mean(model.feature_importances_))
                        model_info["feature_importance_std"] = float(np.std(model.feature_importances_))
                    
                    # Try to get model score if available
                    if hasattr(model, 'score'):
                        model_info["has_score_method"] = True
                        
                    ticker_metrics[model_name] = model_info
                    
                except Exception as e:
                    ticker_metrics[model_name] = {
                        "type": type(model).__name__,
                        "error": str(e)
                    }
                    
            ml_metrics[ticker] = ticker_metrics
            
        return ml_metrics
    
    def _extract_rl_metrics(self, rl_model) -> Dict[str, Any]:
        """Extract metrics from RL model."""
        try:
            rl_metrics = {
                "model_type": type(rl_model).__name__,
                "trained": True
            }
            
            # Try to extract policy and value function info
            if hasattr(rl_model, 'policy'):
                rl_metrics["has_policy"] = True
                
            if hasattr(rl_model, 'learning_rate'):
                rl_metrics["learning_rate"] = rl_model.learning_rate
                
            return rl_metrics
            
        except Exception as e:
            return {
                "model_type": type(rl_model).__name__ if rl_model else "None",
                "error": str(e)
            }
    
    def _generate_super_decisions(self, loop_metrics: Dict) -> List[Dict]:
        """Generate super-model decisions for this loop."""
        decisions = []
        
        # Use replay data if available, otherwise use mock data
        if self.replay_mode and self.replay and REPLAY_AVAILABLE:
            # Use Historical Replay data for realistic decisions
            for ticker in self.replay.config.symbols:
                try:
                    # Get historical market data
                    bar = self.replay.get_next_bar(ticker)
                    if bar is None:
                        continue
                    
                    # Use actual ML predictions from loop_metrics if available
                    ml_pred = {
                        "prediction": np.random.randint(0, 3),  # Would come from actual model
                        "confidence": np.random.uniform(0.5, 0.95),
                        "model_ensemble": ["RandomForest", "XGBoost"]
                    }
                    
                    # Use actual RL actions from loop_metrics if available
                    rl_action = {
                        "action": np.random.randint(0, 3),  # Would come from PPO model
                        "value_estimate": np.random.uniform(-1.0, 1.0)
                    }
                    
                    # Real market context from historical data
                    market_ctx = {
                        "ticker": ticker,
                        "timestamp": bar.get("timestamp"),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume"),
                        "spread_bps": bar.get("spread_bps", 1.0),
                        "market_impact_bps": bar.get("market_impact_bps", 0.5),
                        "volatility_regime": self._classify_volatility(bar),
                        "price": bar.get("close", 100.0)
                    }
                    
                    # Get super-model decision with replay mode flag
                    decision = self.super_model.aggregate_signals(
                        ml_pred, rl_action, market_ctx, replay_mode=True
                    )
                    decision["ticker"] = ticker
                    decisions.append(decision)
                    
                except Exception as e:
                    logger.error(f"Error generating super decision for {ticker}: {e}")
        else:
            # Original mock data path
            mock_tickers = ["ES1!", "NQ1!", "EURUSD", "XAUUSD"]
            
            for ticker in mock_tickers:
                try:
                    # Mock ML prediction (would come from actual trained models)
                    ml_pred = {
                        "prediction": np.random.randint(0, 3),  # 0=SELL, 1=BUY, 2=HOLD
                        "confidence": np.random.uniform(0.5, 0.95),
                        "model_ensemble": ["RandomForest", "XGBoost"]
                    }
                    
                    # Mock RL action (would come from PPO model)
                    rl_action = {
                        "action": np.random.randint(0, 3),  # 0=HOLD, 1=BUY, 2=SELL  
                        "value_estimate": np.random.uniform(-1.0, 1.0)
                    }
                    
                    # Mock market context
                    market_ctx = {
                        "ticker": ticker,
                        "volatility_regime": np.random.choice(["low", "normal", "high"]),
                        "price": np.random.uniform(100, 5000)  # Mock price
                    }
                    
                    # Get super-model decision
                    decision = self.super_model.aggregate_signals(
                        ml_pred, rl_action, market_ctx
                    )
                    decision["ticker"] = ticker
                    decisions.append(decision)
                    
                except Exception as e:
                    logger.error(f"Error generating super decision for {ticker}: {e}")
                
        return decisions
    
    def _classify_volatility(self, bar: Dict) -> str:
        """Classify volatility regime based on price range."""
        if not bar or "high" not in bar or "low" not in bar or "close" not in bar:
            return "normal"
        
        try:
            price_range = (bar["high"] - bar["low"]) / bar["close"]
            if price_range < 0.002:  # < 0.2%
                return "low"
            elif price_range > 0.01:  # > 1%
                return "high"
            else:
                return "normal"
        except:
            return "normal"
    
    def _simulate_paper_trades(self, super_decisions: List[Dict]) -> List[Dict]:
        """Simulate paper trades based on super-model decisions."""
        trades = []
        
        for decision in super_decisions:
            try:
                if decision["decision"] != "HOLD":
                    # Get mock current price from market context
                    current_price = decision["rationale"]["market_context"].get("price", 100.0)
                    
                    trade = simulate_paper_trade(decision, current_price)
                    trade["ticker"] = decision.get("ticker", "UNKNOWN")
                    trades.append(trade)
                    
                    self.paper_trades.append(trade)
                    
            except Exception as e:
                logger.error(f"Error simulating paper trade: {e}")
                
        return trades
    
    def _log_to_jsonl(self, data: Dict):
        """Log structured data to JSONL file."""
        try:
            with open(self.jsonl_log, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log to JSONL: {e}")
    
    def _log_loop_completion(self, loop_num: int, duration: float):
        """Log successful loop completion."""
        logger.info(f"Loop {loop_num} COMPLETED in {duration:.1f}s")
        
    def _log_loop_error(self, loop_num: int, error: Exception, duration: float):
        """Log loop error."""
        logger.error(f"Loop {loop_num} FAILED after {duration:.1f}s: {error}")
        
    def _loop_timeout_handler(self, signum, frame):
        """Handle loop timeout."""
        print(f"[monitor] Loop {self.loop_count + 1}/5: TIMEOUT after {self.loop_timeout}s", flush=True)
        raise TimeoutError(f"Loop {self.loop_count + 1} exceeded {self.loop_timeout}s timeout")
    
    def _generate_final_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        # Calculate summary stats
        completed_loops = len([lm for lm in self.loop_metrics if lm.get("status") == "SUCCESS"])
        failed_loops = len([lm for lm in self.loop_metrics if lm.get("status") == "ERROR"])
        
        # Super-model stats
        super_stats = self.super_model.get_summary_stats()
        
        # Paper trade stats
        buy_trades = len([t for t in self.paper_trades if t["decision"] == "BUY"])
        sell_trades = len([t for t in self.paper_trades if t["decision"] == "SELL"])
        
        report = {
            "status": "SUCCESS" if completed_loops == self.target_loops else "PARTIAL",
            "timestamp": self.timestamp,
            "execution_summary": {
                "total_duration_seconds": total_duration,
                "target_loops": self.target_loops,
                "completed_loops": completed_loops,
                "failed_loops": failed_loops,
                "avg_loop_duration": total_duration / max(self.loop_count, 1)
            },
            "super_model_stats": super_stats,
            "paper_trading_summary": {
                "total_paper_trades": len(self.paper_trades),
                "buy_trades": buy_trades,
                "sell_trades": sell_trades,
                "trade_types": ["PAPER_SIMULATION"] * len(self.paper_trades)
            },
            "errors": self.errors,
            "log_files": {
                "jsonl": self.jsonl_log,
                "human": self.human_log
            },
            "loop_details": self.loop_metrics
        }
        
        # Save final report
        report_file = os.path.join(self.log_dir, f"pipeline_5loops_report_{self.timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("5-LOOP EXECUTION COMPLETE")
        logger.info(f"Status: {report['status']}")
        logger.info(f"Completed: {completed_loops}/{self.target_loops} loops")
        logger.info(f"Total Duration: {total_duration:.1f}s")
        logger.info(f"Paper Trades: {len(self.paper_trades)}")
        logger.info(f"Super-model Decisions: {super_stats.get('total_decisions', 0)}")
        logger.info(f"Errors: {len(self.errors)}")
        logger.info(f"Report saved: {report_file}")
        logger.info("=" * 80)
        
        return report


def main():
    """Main entry point for 5-loop monitoring."""
    
    # Ensure we're in paper mode
    logger.info("Initializing 5-loop pipeline monitor...")
    
    # Use default config but modify for monitoring
    config = default_config.copy()
    config.update({
        "use_mock_data": False,  # Use real IBKR data
        "disable_drift_detection": False,  # Enable for safety
        "drift_detection_threshold": 3.0,
    })
    
    # Initialize and run monitor
    monitor = Pipeline5LoopMonitor(config)
    report = monitor.run_5_loops()
    
    # Print final status
    print(f"\n{'='*60}")
    print(f"5-LOOP MONITOR FINAL STATUS: {report['status']}")
    print(f"Completed Loops: {report['execution_summary']['completed_loops']}/5")
    print(f"Total Duration: {report['execution_summary']['total_duration_seconds']:.1f}s")
    print(f"Paper Trades: {report['paper_trading_summary']['total_paper_trades']}")
    print(f"Errors: {len(report['errors'])}")
    
    if report['errors']:
        print(f"\nERRORS ENCOUNTERED:")
        for error in report['errors']:
            print(f"  Loop {error['loop']}: {error['error']}")
    
    print(f"\nLogs available at:")
    print(f"  JSONL: {report['log_files']['jsonl']}")
    print(f"  Human: {report['log_files']['human']}")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    if report['status'] == 'SUCCESS':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
