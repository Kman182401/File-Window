#!/usr/bin/env python3
"""
Algorithm Selector for Trading System

Smart algorithm selection based on system resources, market conditions,
and performance requirements. Ensures the system always has a working
trading brain that can make decisions and learn.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, Type
from enum import Enum
import psutil
from datetime import datetime, timedelta

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

logger = logging.getLogger(__name__)


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
                return create_recurrent_ppo_agent(self.env)
            
            elif algorithm_type == AlgorithmType.PPO and STANDARD_AGENTS_AVAILABLE:
                config = {
                    'policy': 'MlpPolicy',
                    'learning_rate': 3e-4,
                    'n_steps': get_optimal_batch_size('rl_training') * 16,
                    'batch_size': get_optimal_batch_size('rl_training'),
                    'device': 'cpu',
                    'verbose': 1
                }
                return PPO(config['policy'], self.env, **{k:v for k,v in config.items() if k != 'policy'})
            
            elif algorithm_type == AlgorithmType.A2C and STANDARD_AGENTS_AVAILABLE:
                config = {
                    'policy': 'MlpPolicy',
                    'learning_rate': 3e-4,
                    'device': 'cpu',
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