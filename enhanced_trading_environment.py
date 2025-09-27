#!/usr/bin/env python3
"""
Enhanced Trading Environment with Advanced Features

Production-ready Gymnasium environment with:
- Proper observation normalization (z-score, log scaling)
- Dict observation/action spaces for structured data
- Multi-dimensional actions (direction + position sizing)
- Dynamic episode management with market awareness
- Vectorized multi-asset support
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from dataclasses import dataclass, field
import logging
from collections import deque
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import system components
from system_optimization_config import get_available_memory_mb
from error_handling_system import robust_execution, ErrorCategory

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTradingConfig:
    """Enhanced configuration for trading environment."""
    
    # Observation parameters
    lookback_window: int = 50
    use_dict_obs: bool = True
    normalize_observations: bool = True
    normalization_window: int = 100
    
    # Action parameters
    use_continuous_actions: bool = True
    discrete_action_bins: int = 11  # For discrete position sizing
    max_position_size: float = 1.0
    
    # Market parameters
    symbols: List[str] = field(default_factory=lambda: ['ES1!'])
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    
    # Risk parameters
    max_drawdown: float = 0.20
    daily_loss_limit: float = 0.05
    position_timeout: int = 100
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.02
    use_take_profit: bool = True
    take_profit_pct: float = 0.05
    
    # Episode parameters
    episode_length: Optional[int] = None  # None for variable length
    use_market_sessions: bool = True
    random_start: bool = True
    min_episode_length: int = 100
    max_episode_length: int = 1000
    
    # Reward parameters (will be enhanced by reward engineering)
    use_multi_component_reward: bool = True
    sharpe_weight: float = 0.4
    pnl_weight: float = 0.3
    risk_weight: float = -0.2
    transaction_weight: float = -0.1
    action_smoothness_weight: float = 0.0
    decision_interval: int = 1
    transaction_entropy_window: int = 120
    transaction_entropy_threshold: float = 0.9
    transaction_entropy_gate_strength: float = 0.5
    use_domain_randomization: bool = False
    slippage_range: Tuple[float, float] = (0.0003, 0.001)
    transaction_cost_range: Tuple[float, float] = (0.0005, 0.002)
    latency_ms_range: Tuple[int, int] = (0, 250)
    fill_rate_beta: Tuple[float, float] = (5.0, 2.0)
    decision_interval_range: Tuple[int, int] = (1, 3)


class ObservationNormalizer:
    """
    Handles observation normalization with multiple strategies.
    """
    
    def __init__(self, config: EnhancedTradingConfig):
        self.config = config
        self.price_history = deque(maxlen=config.normalization_window)
        self.volume_history = deque(maxlen=config.normalization_window)
        self.indicator_stats = {}
        
    def normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """
        Normalize price data using percentage returns with clipping.
        """
        if len(prices) < 2:
            return np.zeros_like(prices)
        
        # Calculate returns
        returns = np.diff(prices) / (prices[:-1] + 1e-8)
        
        # Clip extreme values
        returns = np.clip(returns, -0.1, 0.1)
        
        # Scale
        returns = returns * 10
        
        # Pad to original length
        returns = np.concatenate([[0], returns])
        
        return returns
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize volume using log transformation.
        """
        # Store history
        self.volume_history.extend(volume.tolist())
        
        if len(self.volume_history) < 10:
            return np.zeros_like(volume)
        
        # Calculate rolling mean
        rolling_mean = np.mean(list(self.volume_history))
        
        # Log normalization
        normalized = np.log1p(volume / (rolling_mean + 1e-8))
        
        return normalized
    
    def normalize_indicators(self, indicators: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize technical indicators using rolling z-score.
        """
        normalized = {}
        
        for name, values in indicators.items():
            if name not in self.indicator_stats:
                self.indicator_stats[name] = {
                    'history': deque(maxlen=self.config.normalization_window),
                    'mean': 0,
                    'std': 1
                }
            
            # Update history
            self.indicator_stats[name]['history'].extend(values.tolist())
            
            if len(self.indicator_stats[name]['history']) >= 10:
                # Calculate rolling statistics
                history = np.array(list(self.indicator_stats[name]['history']))
                mean = np.mean(history)
                std = np.std(history) + 1e-8
                
                # Z-score normalization
                normalized[name] = (values - mean) / std
                
                # Clip to reasonable range
                normalized[name] = np.clip(normalized[name], -3, 3)
            else:
                normalized[name] = np.zeros_like(values)
        
        return normalized
    
    def normalize_portfolio_state(self, portfolio: Dict[str, float]) -> np.ndarray:
        """
        Normalize portfolio state information.
        """
        return np.array([
            portfolio.get('position', 0) / self.config.max_position_size,
            portfolio.get('cash_ratio', 1.0),
            np.clip(portfolio.get('unrealized_pnl', 0) / 1000, -1, 1),
            np.clip(portfolio.get('drawdown', 0) / self.config.max_drawdown, 0, 1)
        ], dtype=np.float32)


class EnhancedTradingEnvironment(gym.Env):
    """
    Enhanced Gymnasium-compatible trading environment with advanced features.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, 
                 data: Optional[pd.DataFrame] = None,
                 config: Optional[EnhancedTradingConfig] = None,
                 costs_bps: float = 0.0,
                 lambda_var: float = 0.0,
                 lambda_dd: float = 0.0,
                 h_var: int = 60,
                 hindsight_H: int = 0,
                 hindsight_weight: float = 0.0,
                 use_hindsight_in_training: bool = False,
                 eval_mode: bool = False):
        """
        Initialize enhanced trading environment.
        
        Args:
            data: Market data DataFrame
            config: Environment configuration
        """
        super().__init__()
        
        self.config = config or EnhancedTradingConfig()
        self.data = data
        self.normalizer = ObservationNormalizer(self.config)

        # Risk-aware reward parameters
        self.costs_bps = float(costs_bps)
        self.lambda_var = float(lambda_var)
        self.lambda_dd = float(lambda_dd)
        self.h_var = max(1, int(h_var))
        self.hindsight_H = max(0, int(hindsight_H))
        self.hindsight_weight = float(hindsight_weight)
        self.use_hindsight_in_training = bool(use_hindsight_in_training)
        self.eval_mode = bool(eval_mode)
        self._return_window: deque[float] = deque(maxlen=self.h_var)
        self._portfolio_peak: float = self.config.initial_balance
        self._last_drawdown: float = 0.0
        self._last_position: float = 0.0
        self._last_price: Optional[float] = None
        self._position_change: float = 0.0

        # Initialize spaces
        self._setup_spaces()

        # Trading state
        self.current_step = 0
        self.episode_start_step = 0
        self.balance = self.config.initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value_history = [self.config.initial_balance]

        # Base configuration for domain randomization
        self._base_transaction_cost = self.config.transaction_cost
        self._base_slippage = self.config.slippage
        self._base_decision_interval = self.config.decision_interval
        self.current_latency_ms: float = 0.0
        self.current_fill_rate: float = 1.0
        
        # Metrics
        self.metrics = {
            'total_reward': 0,
            'total_profit': 0,
            'n_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
            'action_delta_mean': 0,
            'turnover': 0,
            'transaction_entropy': 0,
            'latency_ms': 0,
            'fill_rate': 1.0,
            'decision_interval': self.config.decision_interval,
            'volatility_proxy': 0,
            'drawdown_increment': 0,
            'last_return': 0,
        }

        # Action smoothness / turnover / entropy tracking
        self.prev_action: Optional[np.ndarray] = None
        self.action_delta_history: List[float] = []
        self.turnover_history: List[float] = []
        self.transaction_history: deque = deque(maxlen=self.config.transaction_entropy_window)
        self.decision_interval_counter = 0
        self.last_action_vector: Optional[np.ndarray] = None

        logger.info(f"Initialized enhanced trading environment with {len(self.config.symbols)} symbols")
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        
        if self.config.use_dict_obs:
            # Dict observation space
            self.observation_space = spaces.Dict({
                'market_data': spaces.Box(
                    low=-5.0, high=5.0,
                    shape=(self.config.lookback_window, 10),  # OHLCV + indicators
                    dtype=np.float32
                ),
                'portfolio_state': spaces.Box(
                    low=-1.0, high=1.0,
                    shape=(4,),  # position, cash_ratio, pnl, drawdown
                    dtype=np.float32
                ),
                'market_regime': spaces.Discrete(4)  # bull, bear, sideways, volatile
            })
        else:
            # Flat observation space
            obs_dim = self.config.lookback_window * 10 + 4 + 1
            self.observation_space = spaces.Box(
                low=-5.0, high=5.0,
                shape=(obs_dim,),
                dtype=np.float32
            )
        
        if self.config.use_continuous_actions:
            # Continuous action space: [direction, size]
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )
        else:
            # Discrete action space
            self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[Union[np.ndarray, Dict], Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if options is not None and "eval_mode" in options:
            self.eval_mode = bool(options["eval_mode"])
        
        # Reset trading state
        self.balance = self.config.initial_balance
        self.position = 0
        self.entry_price = 0
        self.portfolio_value_history = [self.config.initial_balance]
        self._return_window.clear()
        self._portfolio_peak = self.config.initial_balance
        self._last_drawdown = 0.0
        self._last_position = 0.0
        self._position_change = 0.0
        self._last_price = None
        
        # Reset episode position
        if self.config.random_start and self.data is not None:
            max_start = len(self.data) - self.config.max_episode_length
            self.episode_start_step = self.np_random.integers(
                self.config.lookback_window,
                max(self.config.lookback_window + 1, max_start)
            )
        else:
            self.episode_start_step = self.config.lookback_window
        
        self.current_step = self.episode_start_step
        
        # Reset metrics
        self.metrics = {
            'total_reward': 0,
            'total_profit': 0,
            'n_trades': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
            'action_delta_mean': 0,
            'turnover': 0,
            'transaction_entropy': 0,
            'latency_ms': 0,
            'fill_rate': 1.0,
            'decision_interval': self.config.decision_interval,
            'volatility_proxy': 0,
            'drawdown_increment': 0,
            'last_return': 0,
        }

        # Reset smoothness tracking
        if self.config.use_continuous_actions:
            self.prev_action = np.zeros(2, dtype=np.float32)
        else:
            self.prev_action = np.zeros(1, dtype=np.float32)
        self.action_delta_history = []
        self.turnover_history = []
        self.transaction_history: deque = deque(maxlen=self.config.transaction_entropy_window)
        self.decision_interval_counter = 0
        self.last_action_vector: Optional[np.ndarray] = None

        self._apply_domain_randomization()
        self._update_dynamic_metric_snapshots()

        return self._get_observation(), self._get_info()
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[Union[np.ndarray, Dict], float, bool, bool, Dict]:
        """
        Execute one trading step.
        
        Args:
            action: Trading action
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        prev_position = float(self.position)
        prev_price = self._get_current_price()

        # Parse action
        raw_action = action

        if self.config.decision_interval > 1:
            if self.decision_interval_counter % self.config.decision_interval != 0:
                raw_action = self.last_action_vector if self.last_action_vector is not None else action
            self.decision_interval_counter += 1
        else:
            self.decision_interval_counter = 1

        if self.config.use_continuous_actions:
            direction = float(np.asarray(raw_action)[0])  # -1 to 1
            size = float(np.asarray(raw_action)[1])  # 0 to 1
        else:
            if isinstance(raw_action, np.ndarray):
                raw_action = float(np.asarray(raw_action).reshape(-1)[0])
            direction = float(raw_action - 1)
            size = 1.0

        action_vector = np.array([direction, size], dtype=np.float32) if self.config.use_continuous_actions else np.array([direction], dtype=np.float32)
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action_vector)

        action_delta = float(np.linalg.norm(action_vector - self.prev_action))
        self.action_delta_history.append(action_delta)

        # Execute trade and advance state
        gated_direction, gated_size = self._apply_transaction_entropy_gate(direction, size)
        self._execute_trade(gated_direction, gated_size)
        self.current_step += 1

        # Fetch next price for reward computation
        next_price = prev_price
        if self.data is not None and 'close' in self.data.columns and self.current_step < len(self.data):
            next_price = float(self.data.iloc[self.current_step]['close'])
        elif self.data is None:
            next_price = prev_price

        denom = prev_price if abs(prev_price) > 1e-8 else 1.0
        ret_tp1 = (next_price - prev_price) / denom
        delta_position = float(self.position - prev_position)

        rolling_var = self._update_volatility_proxy(ret_tp1)
        drawdown_increment = self._record_portfolio_value(next_price)
        pnl_component = float(self.position * ret_tp1)
        transaction_penalty = (self.costs_bps * 1e-4) * abs(delta_position)
        reward = pnl_component - transaction_penalty

        if not self.eval_mode:
            reward -= self.lambda_var * rolling_var
            reward -= self.lambda_dd * drawdown_increment
            reward += self._compute_hindsight_bonus(prev_price)

        reward = float(reward)

        smooth_weight = getattr(self.config, 'action_smoothness_weight', 0.0)
        if smooth_weight > 0:
            reward -= smooth_weight * (action_delta ** 2)

        self._last_position = float(self.position)
        self._last_price = next_price

        self.metrics['volatility_proxy'] = rolling_var
        self.metrics['drawdown_increment'] = drawdown_increment
        self.metrics['last_return'] = ret_tp1
        self.metrics['position'] = float(self.position)

        # Check termination
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        if self.action_delta_history:
            self.metrics['action_delta_mean'] = float(np.mean(self.action_delta_history[-50:]))
        if self.turnover_history:
            self.metrics['turnover'] = float(np.mean(self.turnover_history[-50:]))
        transaction_entropy = self._compute_transaction_entropy()
        self.metrics['transaction_entropy'] = transaction_entropy
        self._update_dynamic_metric_snapshots()

        self.prev_action = action_vector
        self.last_action_vector = np.array([gated_direction, gated_size], dtype=np.float32) if self.config.use_continuous_actions else np.array([gated_direction], dtype=np.float32)

        # Update metrics
        self.metrics['total_reward'] += reward

        # Get observation and info (after metric updates)
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def _execute_trade(self, direction: float, size: float):
        """Execute trade based on action."""
        if self.data is None:
            return
        
        current_price = self._get_current_price()
        
        # Calculate target position
        target_position = direction * size * self.config.max_position_size
        
        # Calculate position change
        position_change = target_position - self.position
        
        self._position_change = position_change

        if abs(position_change) > 0.01:  # Minimum position change
            # Calculate transaction cost
            transaction_cost = abs(position_change) * current_price * self.config.transaction_cost
            
            # Calculate slippage
            if position_change > 0:  # Buying
                execution_price = current_price * (1 + self.config.slippage)
            else:  # Selling
                execution_price = current_price * (1 - self.config.slippage)
            
            # Update balance
            self.balance -= position_change * execution_price + transaction_cost
            
            # Update position
            if self.position == 0 and position_change != 0:
                # Opening position
                self.entry_price = execution_price
                self.metrics['n_trades'] += 1
            elif self.position != 0 and target_position == 0:
                # Closing position
                profit = self.position * (current_price - self.entry_price)
                self.metrics['total_profit'] += profit
            
            self.position = target_position
            self.turnover_history.append(abs(position_change))
        else:
            self.turnover_history.append(0.0)

        # Record action sign for transaction entropy
        self._record_transaction(direction, size)

        # Update latest seen price for reward computations
        self._last_price = current_price
    
    def _update_volatility_proxy(self, new_return: float) -> float:
        """Maintain rolling variance proxy for returns."""
        self._return_window.append(float(new_return))
        if len(self._return_window) < 2:
            return 0.0
        return float(np.var(np.asarray(self._return_window, dtype=np.float64)))

    def _record_portfolio_value(self, price: float) -> float:
        """Track portfolio value, update drawdown stats, and return incremental penalty."""
        portfolio_value = self.balance + self.position * price
        if not self.portfolio_value_history:
            self.portfolio_value_history = [portfolio_value]
        else:
            self.portfolio_value_history.append(portfolio_value)
        self._portfolio_peak = max(self._portfolio_peak, portfolio_value)
        if self._portfolio_peak <= 0:
            drawdown = 0.0
        else:
            drawdown = (self._portfolio_peak - portfolio_value) / (self._portfolio_peak + 1e-12)
        incremental = max(0.0, drawdown - self._last_drawdown)
        self._last_drawdown = drawdown
        self.metrics['max_drawdown'] = max(self.metrics.get('max_drawdown', 0.0), drawdown)
        self.metrics['current_drawdown'] = drawdown
        return incremental

    def _compute_hindsight_bonus(self, price_t: float) -> float:
        """Optional hindsight reward component available only during training."""
        if self.eval_mode or not self.use_hindsight_in_training:
            return 0.0
        if self.hindsight_weight == 0.0 or self.hindsight_H <= 0:
            return 0.0
        if self.data is None or 'close' not in self.data.columns:
            return 0.0
        base_idx = max(0, self.current_step - 1)
        future_idx = min(len(self.data) - 1, base_idx + self.hindsight_H)
        if future_idx <= base_idx:
            return 0.0
        future_price = float(self.data.iloc[future_idx]['close'])
        return self.hindsight_weight * float(self.position) * (future_price - price_t)
    
    def _get_observation(self) -> Union[np.ndarray, Dict]:
        """Get current observation with normalization."""
        if self.data is None:
            # Return dummy observation
            if self.config.use_dict_obs:
                return {
                    'market_data': np.zeros((self.config.lookback_window, 10), dtype=np.float32),
                    'portfolio_state': np.zeros(4, dtype=np.float32),
                    'market_regime': 0
                }
            else:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get market data window
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Normalize market data
        market_features = []
        
        if self.config.normalize_observations and 'close' in window_data.columns:
            # Price features
            prices = window_data['close'].values
            price_features = self.normalizer.normalize_prices(prices)
            market_features.append(price_features)
            
            # Volume features
            if 'volume' in window_data.columns:
                volume = window_data['volume'].values
                volume_features = self.normalizer.normalize_volume(volume)
                market_features.append(volume_features)
            
            # Technical indicators
            indicators = {}
            for col in ['rsi', 'macd', 'bb_position']:
                if col in window_data.columns:
                    indicators[col] = window_data[col].values
            
            if indicators:
                normalized_indicators = self.normalizer.normalize_indicators(indicators)
                for values in normalized_indicators.values():
                    market_features.append(values)
        
        # Pad if necessary
        if market_features:
            market_data = np.column_stack(market_features)
        else:
            market_data = np.zeros((self.config.lookback_window, 10))
        
        # Ensure correct shape
        if market_data.shape[0] < self.config.lookback_window:
            padding = np.zeros((self.config.lookback_window - market_data.shape[0], market_data.shape[1]))
            market_data = np.vstack([padding, market_data])

        if market_data.shape[1] < 10:
            col_padding = np.zeros((market_data.shape[0], 10 - market_data.shape[1]))
            market_data = np.hstack([market_data, col_padding])
        elif market_data.shape[1] > 10:
            market_data = market_data[:, :10]
        
        # Portfolio state
        portfolio_state = self.normalizer.normalize_portfolio_state({
            'position': self.position,
            'cash_ratio': self.balance / self.config.initial_balance,
            'unrealized_pnl': self.position * (self._get_current_price() - self.entry_price) if self.position != 0 else 0,
            'drawdown': self._calculate_drawdown()
        })
        
        # Market regime (simplified)
        market_regime = self._detect_market_regime()
        
        if self.config.use_dict_obs:
            return {
                'market_data': market_data.astype(np.float32),
                'portfolio_state': portfolio_state.astype(np.float32),
                'market_regime': market_regime
            }
        else:
            # Flatten everything
            flat_obs = np.concatenate([
                market_data.flatten(),
                portfolio_state,
                [market_regime]
            ])
            return flat_obs.astype(np.float32)

    def _get_current_price(self) -> float:
        """Get current market price."""
        if self.data is None or self.current_step >= len(self.data):
            return 100.0
        
        if 'close' in self.data.columns:
            return float(self.data.iloc[self.current_step]['close'])
        return 100.0
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown."""
        if len(self.portfolio_value_history) == 0:
            return 0.0
        
        peak = np.max(self.portfolio_value_history)
        current = self.portfolio_value_history[-1]
        
        if peak > 0:
            return (peak - current) / peak
        return 0.0
    
    def _detect_market_regime(self) -> int:
        """Detect current market regime (simplified)."""
        if self.data is None or self.current_step < 20:
            return 0  # Unknown
        
        # Get recent prices
        recent_prices = self.data.iloc[self.current_step-20:self.current_step]['close'].values
        
        # Calculate trend
        returns = np.diff(recent_prices) / recent_prices[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Classify regime
        if mean_return > 0.001 and std_return < 0.02:
            return 0  # Bull
        elif mean_return < -0.001 and std_return < 0.02:
            return 1  # Bear
        elif abs(mean_return) < 0.0005 and std_return < 0.01:
            return 2  # Sideways
        else:
            return 3  # Volatile

    def _apply_domain_randomization(self) -> None:
        """Sample randomized environment parameters per episode."""
        if not self.config.use_domain_randomization:
            self.config.transaction_cost = self._base_transaction_cost
            self.config.slippage = self._base_slippage
            self.config.decision_interval = self._base_decision_interval
            self.current_latency_ms = 0.0
            self.current_fill_rate = 1.0
            self.decision_interval_counter = 0
            return

        tc_low, tc_high = self.config.transaction_cost_range
        sl_low, sl_high = self.config.slippage_range
        lat_low, lat_high = self.config.latency_ms_range
        di_low, di_high = self.config.decision_interval_range
        alpha, beta = self.config.fill_rate_beta

        self.config.transaction_cost = float(np.random.uniform(tc_low, tc_high))
        self.config.slippage = float(np.random.uniform(sl_low, sl_high))
        self.config.decision_interval = int(np.random.randint(di_low, di_high + 1))
        self.config.decision_interval = max(1, self.config.decision_interval)

        self.current_latency_ms = float(np.random.uniform(lat_low, lat_high))
        self.current_fill_rate = float(np.random.beta(alpha, beta))
        self.decision_interval_counter = 0
        self.last_action_vector = None

    def _update_dynamic_metric_snapshots(self) -> None:
        """Refresh metrics that reflect randomized parameters."""
        self.metrics['latency_ms'] = self.current_latency_ms
        self.metrics['fill_rate'] = self.current_fill_rate
        self.metrics['decision_interval'] = self.config.decision_interval

    def _record_transaction(self, direction: float, size: float) -> None:
        """Track transactions for entropy calculation."""
        if self.transaction_history.maxlen == 0:
            return
        signed_size = float(direction * size)
        self.transaction_history.append(signed_size)

    def _apply_transaction_entropy_gate(self, direction: float, size: float) -> Tuple[float, float]:
        """Scale trade size in high-entropy regimes."""
        gate_strength = self.config.transaction_entropy_gate_strength
        if gate_strength <= 0:
            return direction, size

        entropy = self._compute_transaction_entropy()
        threshold = self.config.transaction_entropy_threshold
        if entropy <= threshold:
            return direction, size

        excess = entropy - threshold
        scale = max(0.0, 1.0 - gate_strength * excess)
        scaled_size = np.clip(size * scale, 0.0, 1.0)
        return direction, scaled_size

    def _compute_transaction_entropy(self) -> float:
        """Rolling entropy over sell/hold/buy bins (normalized to [0,1])."""
        if not self.transaction_history:
            return 0.0

        data = np.array(self.transaction_history, dtype=np.float32)
        buy_prob = np.mean(data > 1e-6)
        sell_prob = np.mean(data < -1e-6)
        hold_prob = max(0.0, 1.0 - buy_prob - sell_prob)
        probs = np.array([sell_prob, hold_prob, buy_prob], dtype=np.float32)
        probs = probs[probs > 0]
        if probs.size == 0:
            return 0.0
        entropy = -float(np.sum(probs * np.log(probs + 1e-8)) / np.log(3.0))
        return entropy
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Check drawdown limit
        if self._calculate_drawdown() > self.config.max_drawdown:
            return True
        
        # Check data end
        if self.data is not None and self.current_step >= len(self.data) - 1:
            return True
        
        # Check bankruptcy
        portfolio_value = self.balance + self.position * self._get_current_price()
        if portfolio_value <= 0:
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated."""
        # Check episode length
        episode_length = self.current_step - self.episode_start_step
        
        if self.config.episode_length and episode_length >= self.config.episode_length:
            return True
        
        if episode_length >= self.config.max_episode_length:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.balance + self.position * self._get_current_price(),
            'metrics': self.metrics.copy()
        }
        
        # Calculate additional metrics
        if len(self.portfolio_value_history) > 1:
            returns = np.diff(self.portfolio_value_history) / self.portfolio_value_history[:-1]
            
            if len(returns) > 0:
                info['metrics']['sharpe_ratio'] = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
                info['metrics']['max_drawdown'] = self._calculate_drawdown()
                
                if self.metrics['n_trades'] > 0:
                    # Estimate win rate (simplified)
                    info['metrics']['win_rate'] = max(0, min(1, 0.5 + np.mean(returns) * 100))
        
        return info
    
    def render(self, mode: str = 'human'):
        """Render environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Portfolio Value: ${self.balance + self.position * self._get_current_price():.2f}")
            print(f"Metrics: {self.metrics}")
        elif mode == 'ansi':
            return str(self._get_info())

    def set_eval_mode(self, flag: bool) -> None:
        """Toggle evaluation mode at runtime."""
        self.eval_mode = bool(flag)


def test_enhanced_environment():
    """Test enhanced trading environment."""
    print("Testing Enhanced Trading Environment")
    print("=" * 50)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
        'volume': np.random.randint(1000, 10000, 1000),
        'rsi': 50 + np.random.randn(1000) * 20,
        'macd': np.random.randn(1000) * 0.1,
        'bb_position': np.random.uniform(0, 1, 1000)
    })
    
    # Test 1: Dict observation space
    print("\n1. Testing Dict observation space...")
    config = EnhancedTradingConfig(use_dict_obs=True, normalize_observations=True)
    env = EnhancedTradingEnvironment(data, config)
    
    obs, info = env.reset()
    print(f"   Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"   Market data shape: {obs['market_data'].shape}")
        print(f"   Portfolio state shape: {obs['portfolio_state'].shape}")
        print(f"   Market regime: {obs['market_regime']}")
    
    # Test 2: Continuous action space
    print("\n2. Testing continuous action space...")
    action = env.action_space.sample()
    print(f"   Action: {action}")
    print(f"   Direction: {action[0]:.3f}, Size: {action[1]:.3f}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Reward: {reward:.6f}")
    print(f"   Portfolio value: ${info['portfolio_value']:.2f}")
    
    # Test 3: Normalization
    print("\n3. Testing observation normalization...")
    print(f"   Market data range: [{obs['market_data'].min():.2f}, {obs['market_data'].max():.2f}]")
    print(f"   Portfolio state range: [{obs['portfolio_state'].min():.2f}, {obs['portfolio_state'].max():.2f}]")
    
    # Test 4: Episode management
    print("\n4. Testing episode management...")
    config2 = EnhancedTradingConfig(
        random_start=True,
        episode_length=50,
        use_market_sessions=True
    )
    env2 = EnhancedTradingEnvironment(data, config2)
    
    for i in range(100):
        action = env2.action_space.sample()
        obs, reward, terminated, truncated, info = env2.step(action)
        
        if terminated or truncated:
            print(f"   Episode ended at step {i+1}")
            print(f"   Reason: {'terminated' if terminated else 'truncated'}")
            print(f"   Final metrics: {info['metrics']}")
            break
    
    # Test 5: Multi-component reward
    print("\n5. Testing multi-component reward...")
    total_reward = 0
    obs, _ = env.reset()
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, _, _, info = env.step(action)
        total_reward += reward
    
    print(f"   Total reward (10 steps): {total_reward:.6f}")
    print(f"   Sharpe ratio: {info['metrics'].get('sharpe_ratio', 0):.3f}")
    print(f"   Max drawdown: {info['metrics'].get('max_drawdown', 0):.3f}")
    
    print("\n✓ Enhanced environment test completed!")


if __name__ == "__main__":
    test_enhanced_environment()
