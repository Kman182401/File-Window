#!/usr/bin/env python3
"""
Algorithm Selector for Trading System

Smart algorithm selection based on system resources, market conditions,
and performance requirements. Ensures the system always has a working
trading brain that can make decisions and learn.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, Type, Iterable
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psutil

try:  # Optional dependency
    import torch
except Exception:  # pragma: no cover - fallback for CPU-only envs
    torch = None

from wfo.cpcv import CPCVConfig, CombinatorialPurgedCV
from wfo.metrics import deflated_sharpe_ratio, sharpe_ratio
from wfo.purging import PurgeConfig, apply_purge_embargo
from wfo.labeling import ensure_forward_label
from wfo_rl import RLAdapter, RLSpec, make_env_from_df, logistic_positions
from wfo.utils import enable_determinism, resolve_session_minutes

logger = logging.getLogger(__name__)

# Import system components
from system_optimization_config import get_available_memory_mb, get_optimal_batch_size
from memory_management_system import get_memory_manager
from error_handling_system import (
    robust_execution,
    ErrorCategory,
    ErrorSeverity,
    get_error_handler,
    FallbackChain
)

# Import RL agents
from sac_trading_agent import SACTradingAgent, create_sac_agent
from recurrent_ppo_agent import RecurrentPPOAgent, create_recurrent_ppo_agent
from enhanced_trading_environment import EnhancedTradingEnvironment

# Import standard components for fallback
try:
    from stable_baselines3 import PPO, A2C
    STANDARD_AGENTS_AVAILABLE = True
except ImportError:
    STANDARD_AGENTS_AVAILABLE = False
    logger.warning("Standard RL agents not available")


class AlgorithmType(Enum):
    """Available algorithm types."""
    SAC = "sac"                      # Soft Actor-Critic (best performance)
    RECURRENT_PPO = "recurrent_ppo"  # Temporal patterns
    PPO = "ppo"                      # Standard PPO
    A2C = "a2c"                      # Lightweight alternative
    RULE_BASED = "rule_based"        # Rule-based fallback


class PerformanceProfile(Enum):
    """Performance profiles for different scenarios."""
    HIGH_PERFORMANCE = "high"      # SAC with full features
    BALANCED = "balanced"          # RecurrentPPO
    MEMORY_EFFICIENT = "efficient" # Standard PPO
    MINIMAL = "minimal"            # A2C or rule-based


class SystemResourceMonitor:
    """Monitor system resources for algorithm selection."""
    
    def __init__(self):
        self.memory_threshold_high = 4000  # MB
        self.memory_threshold_medium = 2500  # MB
        self.cpu_threshold_high = 60  # percent
        
    def get_resource_profile(self) -> Dict[str, Any]:
        """Get current system resource profile."""
        memory_mb = get_available_memory_mb()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        profile = {
            'available_memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'memory_class': self._classify_memory(memory_mb),
            'cpu_class': self._classify_cpu(cpu_percent),
            'overall_class': self._classify_overall(memory_mb, cpu_percent)
        }
        
        return profile
    
    def _classify_memory(self, memory_mb: float) -> str:
        """Classify memory availability."""
        if memory_mb >= self.memory_threshold_high:
            return 'high'
        elif memory_mb >= self.memory_threshold_medium:
            return 'medium'
        else:
            return 'low'
    
    def _classify_cpu(self, cpu_percent: float) -> str:
        """Classify CPU utilization."""
        if cpu_percent <= self.cpu_threshold_high:
            return 'available'
        else:
            return 'busy'
    
    def _classify_overall(self, memory_mb: float, cpu_percent: float) -> str:
        """Overall system resource classification."""
        if memory_mb >= self.memory_threshold_high and cpu_percent <= self.cpu_threshold_high:
            return 'optimal'
        elif memory_mb >= self.memory_threshold_medium and cpu_percent <= 80:
            return 'good'
        elif memory_mb >= 1500:
            return 'constrained'
        else:
            return 'critical'


class RuleBasedTradingAgent:
    """
    Simple rule-based trading agent as ultimate fallback.
    
    This ensures the system ALWAYS has a trading brain that can:
    - Make trading decisions
    - Execute paper trades
    - Learn from results (basic parameter adaptation)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize rule-based agent."""
        self.config = config or {}
        
        # Strategy parameters (adaptive)
        self.ma_short = self.config.get('ma_short', 10)
        self.ma_long = self.config.get('ma_long', 30)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.bb_threshold = self.config.get('bb_threshold', 0.8)
        
        # Performance tracking
        self.trades_executed = 0
        self.profitable_trades = 0
        self.total_pnl = 0
        
        # Adaptive parameters
        self.last_adaptation = datetime.now()
        self.adaptation_frequency = timedelta(hours=24)
        
        logger.info("Initialized rule-based trading agent (fallback)")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        """
        Make prediction based on rules.
        
        Args:
            observation: Market observation
            deterministic: Ignored (always deterministic)
            
        Returns:
            Tuple of (action, state)
        """
        # Parse observation (assume it contains technical indicators)
        if len(observation) < 10:
            return np.array([0.0]), None  # Hold
        
        # Extract features (this is simplified - in practice, use proper feature mapping)
        close_price = observation[0] if len(observation) > 0 else 0
        rsi = observation[1] if len(observation) > 1 else 50
        ma_short = observation[2] if len(observation) > 2 else close_price
        ma_long = observation[3] if len(observation) > 3 else close_price
        bb_position = observation[4] if len(observation) > 4 else 0.5
        
        # Trading signals
        signals = []
        
        # 1. Moving Average Crossover
        if ma_short > ma_long:
            signals.append(1)  # Bullish
        elif ma_short < ma_long:
            signals.append(-1)  # Bearish
        else:
            signals.append(0)  # Neutral
        
        # 2. RSI Mean Reversion
        if rsi < self.rsi_oversold:
            signals.append(1)  # Oversold - buy
        elif rsi > self.rsi_overbought:
            signals.append(-1)  # Overbought - sell
        else:
            signals.append(0)
        
        # 3. Bollinger Bands
        if bb_position < 0.2:
            signals.append(1)  # Near lower band - buy
        elif bb_position > 0.8:
            signals.append(-1)  # Near upper band - sell
        else:
            signals.append(0)
        
        # Combine signals
        total_signal = np.mean(signals)
        
        # Convert to action
        if total_signal > 0.3:
            action = np.array([0.5])  # Buy with moderate size
        elif total_signal < -0.3:
            action = np.array([-0.5])  # Sell with moderate size
        else:
            action = np.array([0.0])  # Hold
        
        return action, None
    
    def learn_from_result(self, action: float, reward: float):
        """
        Learn from trading result (simple parameter adaptation).
        
        Args:
            action: Action taken
            reward: Reward received
        """
        self.trades_executed += 1
        self.total_pnl += reward
        
        if reward > 0:
            self.profitable_trades += 1
        
        # Adapt parameters based on performance
        if datetime.now() - self.last_adaptation > self.adaptation_frequency:
            self._adapt_parameters()
            self.last_adaptation = datetime.now()
    
    def _adapt_parameters(self):
        """Adapt strategy parameters based on performance."""
        if self.trades_executed == 0:
            return
        
        win_rate = self.profitable_trades / self.trades_executed
        avg_pnl = self.total_pnl / self.trades_executed
        
        logger.info(f"Rule-based agent adaptation: {win_rate:.2%} win rate, {avg_pnl:.4f} avg PnL")
        
        # Simple adaptation logic
        if win_rate < 0.4:
            # Poor performance - make strategy more conservative
            self.rsi_oversold = max(25, self.rsi_oversold - 2)
            self.rsi_overbought = min(75, self.rsi_overbought + 2)
            self.bb_threshold = min(0.9, self.bb_threshold + 0.05)
        elif win_rate > 0.6:
            # Good performance - make strategy more aggressive
            self.rsi_oversold = min(35, self.rsi_oversold + 1)
            self.rsi_overbought = max(65, self.rsi_overbought - 1)
            self.bb_threshold = max(0.7, self.bb_threshold - 0.02)
    
    def save(self, path: str):
        """Save agent parameters."""
        import json
        
        params = {
            'ma_short': self.ma_short,
            'ma_long': self.ma_long,
            'rsi_oversold': self.rsi_oversold,
            'rsi_overbought': self.rsi_overbought,
            'bb_threshold': self.bb_threshold,
            'trades_executed': self.trades_executed,
            'profitable_trades': self.profitable_trades,
            'total_pnl': self.total_pnl
        }
        
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Rule-based agent saved to {path}")
    
    def load(self, path: str):
        """Load agent parameters."""
        import json
        
        try:
            with open(path, 'r') as f:
                params = json.load(f)
            
            self.ma_short = params.get('ma_short', self.ma_short)
            self.ma_long = params.get('ma_long', self.ma_long)
            self.rsi_oversold = params.get('rsi_oversold', self.rsi_oversold)
            self.rsi_overbought = params.get('rsi_overbought', self.rsi_overbought)
            self.bb_threshold = params.get('bb_threshold', self.bb_threshold)
            self.trades_executed = params.get('trades_executed', 0)
            self.profitable_trades = params.get('profitable_trades', 0)
            self.total_pnl = params.get('total_pnl', 0)
            
            logger.info(f"Rule-based agent loaded from {path}")
            
        except Exception as e:
            logger.warning(f"Could not load rule-based agent: {e}")


class AlgorithmSelector:
    """
    Smart algorithm selection and management system.
    
    Ensures the trading system ALWAYS has a functioning trading brain:
    1. Selects optimal algorithm based on resources
    2. Handles algorithm failures with fallbacks
    3. Monitors performance and switches if needed
    4. Guarantees trading decisions are always possible
    """
    
    def __init__(self, env: Optional[EnhancedTradingEnvironment] = None):
        """
        Initialize algorithm selector.
        
        Args:
            env: Trading environment
        """
        self.env = env
        self.resource_monitor = SystemResourceMonitor()
        self.error_handler = get_error_handler()
        
        # Available agents
        self.agents = {}
        self.current_agent = None
        self.current_algorithm = None
        
        # Performance tracking
        self.performance_history = {}
        self.algorithm_failures = {}
        
        # Fallback chain
        self.fallback_chain = None
        
        logger.info("Initialized algorithm selector")
    
    @robust_execution(category=ErrorCategory.MODEL)
    def select_optimal_algorithm(self, 
                                 performance_profile: Optional[PerformanceProfile] = None) -> AlgorithmType:
        """
        Select optimal algorithm based on current system state.
        
        Args:
            performance_profile: Desired performance profile
            
        Returns:
            Selected algorithm type
        """
        # Get current system resources
        resources = self.resource_monitor.get_resource_profile()
        
        logger.info(f"Resource profile: {resources}")
        
        # Algorithm selection logic
        if performance_profile == PerformanceProfile.HIGH_PERFORMANCE:
            if resources['memory_class'] == 'high' and resources['cpu_class'] == 'available':
                return AlgorithmType.SAC
            else:
                logger.warning("High performance requested but resources insufficient, using RecurrentPPO")
                return AlgorithmType.RECURRENT_PPO
        
        elif performance_profile == PerformanceProfile.BALANCED:
            if resources['memory_class'] in ['high', 'medium']:
                return AlgorithmType.RECURRENT_PPO
            else:
                return AlgorithmType.PPO
        
        elif performance_profile == PerformanceProfile.MEMORY_EFFICIENT:
            return AlgorithmType.PPO if resources['memory_class'] != 'low' else AlgorithmType.A2C
        
        elif performance_profile == PerformanceProfile.MINIMAL:
            return AlgorithmType.A2C if STANDARD_AGENTS_AVAILABLE else AlgorithmType.RULE_BASED
        
        else:
            # Auto-select based on resources
            if resources['overall_class'] == 'optimal':
                return AlgorithmType.SAC
            elif resources['overall_class'] == 'good':
                return AlgorithmType.RECURRENT_PPO
            elif resources['overall_class'] == 'constrained':
                return AlgorithmType.PPO if STANDARD_AGENTS_AVAILABLE else AlgorithmType.A2C
            else:
                return AlgorithmType.RULE_BASED
    
    def create_agent(self, algorithm_type: AlgorithmType) -> Optional[Any]:
        """
        Create agent instance for specified algorithm.
        
        Args:
            algorithm_type: Algorithm type to create
            
        Returns:
            Created agent instance or None if failed
        """
        if self.env is None:
            logger.error("No environment provided for agent creation")
            return None
        
        try:
            if algorithm_type == AlgorithmType.SAC:
                return create_sac_agent(self.env)
            
            elif algorithm_type == AlgorithmType.RECURRENT_PPO:
                if hasattr(self.env, 'config') and hasattr(self.env.config, 'action_smoothness_weight'):
                    if self.env.config.action_smoothness_weight == 0:
                        self.env.config.action_smoothness_weight = 0.005
                        logger.info("Applied default action smoothness weight for RecurrentPPO")
                    if hasattr(self.env.config, 'transaction_entropy_gate_strength') and self.env.config.transaction_entropy_gate_strength == 0:
                        self.env.config.transaction_entropy_gate_strength = 0.5
                        logger.info("Enabled transaction entropy gate for RecurrentPPO")
                    if hasattr(self.env.config, 'use_domain_randomization') and not self.env.config.use_domain_randomization:
                        self.env.config.use_domain_randomization = True
                        logger.info("Activated domain randomization for RecurrentPPO environment")

                rp_config = {
                    'n_envs': 4,
                    'batch_size': 64,
                }
                return create_recurrent_ppo_agent(self.env, rp_config)
            
            elif algorithm_type == AlgorithmType.PPO and STANDARD_AGENTS_AVAILABLE:
                preferred_device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                config = {
                    'policy': 'MlpPolicy',
                    'learning_rate': 3e-4,
                    'n_steps': get_optimal_batch_size('rl_training') * 16,
                    'batch_size': get_optimal_batch_size('rl_training'),
                    'device': preferred_device,
                    'verbose': 1
                }
                return PPO(config['policy'], self.env, **{k:v for k,v in config.items() if k != 'policy'})
            
            elif algorithm_type == AlgorithmType.A2C and STANDARD_AGENTS_AVAILABLE:
                preferred_device = "cuda" if torch and torch.cuda.is_available() else "cpu"
                config = {
                    'policy': 'MlpPolicy',
                    'learning_rate': 3e-4,
                    'device': preferred_device,
                    'verbose': 1
                }
                return A2C(config['policy'], self.env, **{k:v for k,v in config.items() if k != 'policy'})
            
            elif algorithm_type == AlgorithmType.RULE_BASED:
                return RuleBasedTradingAgent()
            
            else:
                logger.error(f"Unsupported algorithm type: {algorithm_type}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create {algorithm_type.value} agent: {e}")
            self.algorithm_failures[algorithm_type] = self.algorithm_failures.get(algorithm_type, 0) + 1
            return None
    
    def initialize_agent(self, 
                        performance_profile: Optional[PerformanceProfile] = None) -> bool:
        """
        Initialize trading agent with fallback chain.
        
        Args:
            performance_profile: Desired performance profile
            
        Returns:
            True if agent successfully initialized
        """
        # Create fallback chain
        fallback_strategies = [
            lambda: self._try_algorithm(self.select_optimal_algorithm(performance_profile)),
            lambda: self._try_algorithm(AlgorithmType.RECURRENT_PPO),
            lambda: self._try_algorithm(AlgorithmType.PPO),
            lambda: self._try_algorithm(AlgorithmType.A2C),
            lambda: self._try_algorithm(AlgorithmType.RULE_BASED)
        ]
        
        self.fallback_chain = FallbackChain(fallback_strategies)
        
        try:
            agent = self.fallback_chain.execute()
            if agent is not None:
                self.current_agent = agent
                logger.info(f"Successfully initialized agent: {self.current_algorithm.value}")
                return True
            else:
                logger.critical("All agent initialization strategies failed!")
                return False
        except Exception as e:
            logger.critical(f"Agent initialization completely failed: {e}")
            return False
    
    def _try_algorithm(self, algorithm_type: AlgorithmType) -> Any:
        """
        Try to initialize specific algorithm.
        
        Args:
            algorithm_type: Algorithm to try
            
        Returns:
            Agent instance if successful
            
        Raises:
            Exception if algorithm fails
        """
        logger.info(f"Trying to initialize {algorithm_type.value} agent...")
        
        agent = self.create_agent(algorithm_type)
        if agent is None:
            raise Exception(f"{algorithm_type.value} agent creation failed")
        
        self.current_algorithm = algorithm_type
        return agent
    
    def predict(self, observation: np.ndarray, **kwargs) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Make prediction using current agent.
        
        Args:
            observation: Market observation
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (action, state)
        """
        if self.current_agent is None:
            logger.error("No active agent for prediction")
            return np.array([0.0]), None
        
        try:
            return self.current_agent.predict(observation, **kwargs)
        except Exception as e:
            logger.error(f"Prediction failed with {self.current_algorithm.value}: {e}")
            
            # Try to reinitialize agent
            if self.initialize_agent():
                return self.current_agent.predict(observation, **kwargs)
            else:
                # Ultimate fallback - return hold action
                return np.array([0.0]), None
    
    def train(self, total_timesteps: int = 10000, **kwargs):
        """
        Train current agent.
        
        Args:
            total_timesteps: Training timesteps
            **kwargs: Additional training arguments
        """
        if self.current_agent is None:
            logger.error("No active agent for training")
            return
        
        # Rule-based agent doesn't have traditional training
        if self.current_algorithm == AlgorithmType.RULE_BASED:
            logger.info("Rule-based agent doesn't require traditional training")
            return
        
        try:
            if hasattr(self.current_agent, 'train'):
                self.current_agent.train(total_timesteps, **kwargs)
            elif hasattr(self.current_agent, 'learn'):
                self.current_agent.learn(total_timesteps, **kwargs)
            else:
                logger.warning(f"Agent {self.current_algorithm.value} has no training method")
                
        except Exception as e:
            self.error_handler.handle_error(
                e,
                ErrorCategory.MODEL,
                ErrorSeverity.HIGH,
                {'component': 'agent_training', 'algorithm': self.current_algorithm.value}
            )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about current agent."""
        info = {
            'current_algorithm': self.current_algorithm.value if self.current_algorithm else None,
            'agent_available': self.current_agent is not None,
            'resource_profile': self.resource_monitor.get_resource_profile(),
            'algorithm_failures': dict(self.algorithm_failures)
        }
        
        # Add agent-specific info
        if self.current_agent:
            if hasattr(self.current_agent, 'get_memory_usage'):
                info['memory_usage'] = self.current_agent.get_memory_usage()
            
            if hasattr(self.current_agent, 'trades_executed'):
                info['trades_executed'] = self.current_agent.trades_executed
                info['profitable_trades'] = getattr(self.current_agent, 'profitable_trades', 0)
        
        return info
    
    def save_agent(self, path: str):
        """Save current agent."""
        if self.current_agent is None:
            logger.warning("No agent to save")
            return
        
        try:
            self.current_agent.save(path)
        except Exception as e:
            logger.error(f"Failed to save agent: {e}")
    
    def load_agent(self, path: str, algorithm_type: AlgorithmType):
        """Load agent from path."""
        try:
            agent = self.create_agent(algorithm_type)
            if agent:
                agent.load(path)
                self.current_agent = agent
                self.current_algorithm = algorithm_type
                logger.info(f"Loaded {algorithm_type.value} agent from {path}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")


def _coerce_strategy_entry(entry: Any) -> Dict[str, Any]:
    """Coerce either dict or dataclass strategy descriptors into a dict."""
    if isinstance(entry, dict):
        return dict(entry)
    if hasattr(entry, "__dict__"):
        return {k: v for k, v in entry.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Unsupported strategy entry type: {type(entry)!r}")


def _effective_trials(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 1.0
    if matrix.shape[1] == 1:
        return 1.0
    corr = np.nan_to_num(np.corrcoef(matrix.T), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    eigvals = np.linalg.eigvalsh(corr)
    denom = np.sum(eigvals ** 2) + 1e-12
    return float(max(1.0, (eigvals.sum() ** 2) / denom))


def select_strategies_with_cpcv(
    dataset: pd.DataFrame,
    strategies: Iterable[Any],
    cpcv_config: Dict[str, Any],
) -> list[Dict[str, Any]]:
    """Return a shortlist ranked by CPCV Sharpe and filtered by DSR."""

    if dataset.empty:
        logger.warning("CPCV selection skipped: empty dataset")
        return []

    base_df = dataset.reset_index(drop=True).copy()
    if "timestamp" not in base_df.columns:
        base_df["timestamp"] = pd.date_range(start="1970-01-01", periods=len(base_df), freq="T")
    if "returns" not in base_df.columns:
        if "close" in base_df.columns:
            base_df["returns"] = base_df["close"].pct_change().fillna(0.0)
        else:
            base_df["returns"] = 0.0

    label_lookahead = int(cpcv_config.get("label_lookahead_bars", 0))
    embargo_days = float(cpcv_config.get("embargo_days", 0))
    session_minutes = {k.upper(): v for k, v in (cpcv_config.get("session_minutes") or {}).items()}
    symbol = (cpcv_config.get("symbol") or "").upper()
    default_minutes = int(cpcv_config.get("minutes_per_trading_day", 390))
    minutes_per_day = session_minutes.get(symbol)
    if minutes_per_day is None:
        minutes_per_day = resolve_session_minutes(symbol, default_minutes)
    if minutes_per_day is None:
        minutes_per_day = default_minutes
    else:
        minutes_per_day = int(minutes_per_day)

    if cpcv_config.get("deterministic_debug"):
        logger.info("CPCV deterministic debug enabled (training may slow down)")
        enable_determinism(int(cpcv_config.get("deterministic_seed", 42)))

    ensure_forward_label(base_df, horizon=max(1, label_lookahead))
    embargo_bars = int(round(embargo_days * minutes_per_day))

    cpcv_cfg = CPCVConfig(
        n_groups=int(cpcv_config.get("groups", 12)),
        test_group_size=int(cpcv_config.get("test_group_size", 2)),
        embargo=embargo_bars,
        label_lookahead=label_lookahead,
        max_splits=cpcv_config.get("folds"),
        random_state=cpcv_config.get("random_state"),
    )
    splitter = CombinatorialPurgedCV(cpcv_cfg)

    log_root = Path(cpcv_config.get("log_dir", "artifacts/cpcv"))
    log_root.mkdir(parents=True, exist_ok=True)

    rl_fast_smoke = bool(cpcv_config.get("rl_fast_smoke", True))
    rl_overrides = dict(cpcv_config.get("rl_overrides", {}))
    dsr_threshold = float(cpcv_config.get("dsr_threshold", 0.05))
    costs_bps = float(cpcv_config.get("costs_bps", 0.0))

    idx = np.arange(len(base_df))
    evaluated: list[Dict[str, Any]] = []
    aggregated_matrix_inputs: Dict[str, np.ndarray] = {}

    for entry in strategies:
        strat = _coerce_strategy_entry(entry)
        strat_type = strat.get("type")
        name = strat.get("name", "unknown")
        logger.info("[CPCV] Evaluating strategy %s (%s)", name, strat_type)

        fold_returns: list[np.ndarray] = []
        fold_sharpes: list[float] = []

        for train_idx, test_idx in splitter.split(idx):
            train_idx = apply_purge_embargo(train_idx, test_idx, PurgeConfig(label_lookahead=label_lookahead, embargo=embargo_bars))
            if train_idx.size == 0 or len(test_idx) == 0:
                continue
            is_df = base_df.iloc[train_idx].reset_index(drop=True)
            oos_df = base_df.iloc[test_idx].reset_index(drop=True)
            ensure_forward_label(is_df, horizon=max(1, label_lookahead))
            ensure_forward_label(oos_df, horizon=max(1, label_lookahead))

            if strat_type == "rl_policy":
                rl_cfg = dict(strat.get("rl") or {})
                rl_cfg.update(rl_overrides)
                reward_kwargs = strat.get("reward") or {}
                spec = RLSpec(
                    algo=strat.get("algo", "RecurrentPPO"),
                    policy=strat.get("policy", "MlpPolicy"),
                    train_timesteps=int(rl_cfg.get("train_timesteps", 50_000)),
                    n_envs=int(rl_cfg.get("n_envs", 1)),
                    seed=int(rl_cfg.get("seed", 42)),
                    policy_kwargs=rl_cfg.get("policy_kwargs") or {},
                    algo_kwargs=rl_cfg.get("algo_kwargs") or {},
                    vecnormalize_obs=bool(rl_cfg.get("vecnormalize_obs", True)),
                    vecnormalize_reward=bool(rl_cfg.get("vecnormalize_reward", True)),
                    use_imitation_warmstart=bool(rl_cfg.get("use_imitation_warmstart", False)),
                    imitation_kwargs=rl_cfg.get("imitation_kwargs"),
                    warmstart_epochs=int(rl_cfg.get("warmstart_epochs", 5)),
                )
                adapter = RLAdapter(spec, fast_smoke=rl_fast_smoke)
                is_env_fn = make_env_from_df(is_df, costs_bps=costs_bps, reward_kwargs=reward_kwargs, eval_mode=False)
                oos_env_fn = make_env_from_df(oos_df, costs_bps=costs_bps, reward_kwargs=reward_kwargs, eval_mode=True)
                fold_idx = len(fold_returns) + 1
                fold_dir = log_root / name / f"fold_{fold_idx}"
                try:
                    model, vecnorm_path = adapter.fit_on_is(is_env_fn, str(fold_dir))
                    returns = adapter.score_on_oos(model, oos_env_fn, vecnorm_path)
                except RuntimeError as exc:
                    logger.warning(
                        "[CPCV] RL strategy %s failed: %s -- install gymnasium, stable-baselines3, sb3-contrib",
                        name,
                        exc,
                    )
                    fold_returns = []
                    break
            elif strat_type == "supervised" and strat.get("model") == "logistic":
                try:
                    positions = logistic_positions(is_df, oos_df, **(strat.get("params") or {}))
                except Exception as exc:  # pragma: no cover - dependency guard
                    logger.warning("[CPCV] Supervised strategy %s failed: %s", name, exc)
                    fold_returns = []
                    break
                returns_series = oos_df.get("returns")
                if returns_series is None:
                    returns_series = oos_df["close"].pct_change().fillna(0.0)
                returns = positions * returns_series.to_numpy(dtype=float)
            else:
                logger.debug("[CPCV] Strategy %s type %s not handled", name, strat_type)
                fold_returns = []
                break

            returns = np.asarray(returns, dtype=float)
            if returns.size == 0:
                continue
            fold_returns.append(returns)
            fold_sharpes.append(sharpe_ratio(returns))

        if not fold_returns or not fold_sharpes:
            continue

        aggregated = np.concatenate(fold_returns)
        aggregated_matrix_inputs[name] = aggregated
        evaluated.append(
            {
                "name": name,
                "strategy": entry,
                "median_sharpe": float(np.median(fold_sharpes)),
                "fold_sharpes": fold_sharpes,
                "aggregated": aggregated,
            }
        )

    if not evaluated:
        return []

    if aggregated_matrix_inputs:
        matrix = np.column_stack(list(aggregated_matrix_inputs.values()))
        trials_effective = _effective_trials(matrix)
    else:
        trials_effective = 1.0

    shortlist: list[Dict[str, Any]] = []
    for record in evaluated:
        dsr = deflated_sharpe_ratio(record["aggregated"], trials_effective=trials_effective)
        if dsr.p_value >= dsr_threshold:
            logger.info("[CPCV] Strategy %s filtered by DSR (p=%.4f)", record["name"], dsr.p_value)
            continue
        shortlist.append(
            {
                "name": record["name"],
                "strategy": record["strategy"],
                "median_sharpe": record["median_sharpe"],
                "sharpe_scores": [float(s) for s in record["fold_sharpes"]],
                "dsr": {"z_score": dsr.z_score, "p_value": dsr.p_value, "effective_trials": trials_effective},
            }
        )

    shortlist.sort(key=lambda item: item["median_sharpe"], reverse=True)
    top_k = cpcv_config.get("top_k")
    if isinstance(top_k, int) and top_k > 0:
        shortlist = shortlist[:top_k]
    return shortlist


def test_algorithm_selector():
    """Test algorithm selector functionality."""
    print("Testing Algorithm Selector")
    print("=" * 50)
    
    # Create dummy environment
    from enhanced_trading_environment import EnhancedTradingEnvironment, EnhancedTradingConfig
    
    config = EnhancedTradingConfig(
        use_dict_obs=False,
        normalize_observations=True
    )
    env = EnhancedTradingEnvironment(config=config)
    
    # Test 1: Algorithm selection
    print("\n1. Testing algorithm selection...")
    selector = AlgorithmSelector(env)
    
    # Test different profiles
    profiles = [
        PerformanceProfile.HIGH_PERFORMANCE,
        PerformanceProfile.BALANCED,
        PerformanceProfile.MEMORY_EFFICIENT,
        PerformanceProfile.MINIMAL
    ]
    
    for profile in profiles:
        algorithm = selector.select_optimal_algorithm(profile)
        print(f"   {profile.value}: {algorithm.value}")
    
    # Test 2: Agent initialization
    print("\n2. Testing agent initialization...")
    success = selector.initialize_agent(PerformanceProfile.BALANCED)
    if success:
        print(f"   ✓ Agent initialized: {selector.current_algorithm.value}")
    else:
        print("   ✗ Agent initialization failed")
    
    # Test 3: Prediction
    print("\n3. Testing prediction...")
    obs, _ = env.reset()
    
    if selector.current_agent:
        action, state = selector.predict(obs)
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action: {action}")
        print(f"   State: {state is not None}")
    
    # Test 4: Agent info
    print("\n4. Agent information:")
    info = selector.get_agent_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Test 5: Rule-based fallback
    print("\n5. Testing rule-based fallback...")
    rule_agent = RuleBasedTradingAgent()
    
    action, _ = rule_agent.predict(np.array([100, 45, 98, 102, 0.3]))
    print(f"   Rule-based action: {action}")
    
    # Test learning
    rule_agent.learn_from_result(action[0], 0.01)
    print(f"   Trades executed: {rule_agent.trades_executed}")
    
    print("\n✓ Algorithm selector test completed!")


if __name__ == "__main__":
    test_algorithm_selector()
