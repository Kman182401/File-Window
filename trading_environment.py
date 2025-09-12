"""
Trading Environment for Reinforcement Learning

Gym-compatible environment that allows RL agents to learn trading strategies
from market data with realistic execution simulation.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
from utils.market_data_validator import MarketDataValidator
from monitoring.alerting_system import get_alerting_system, AlertSeverity, AlertType

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Configuration for trading environment"""
    # Market data
    lookback_window: int = 50  # Number of past bars to include in observation
    
    # Trading parameters
    initial_balance: float = 100000.0
    max_position: float = 1.0  # Maximum position size (1.0 = 100% of balance)
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    
    # Risk management
    max_drawdown: float = 0.20  # 20% max drawdown
    daily_loss_limit: float = 0.05  # 5% daily loss limit
    position_timeout: int = 100  # Max bars to hold position
    
    # Reward parameters
    profit_reward_scale: float = 1.0
    risk_penalty_scale: float = 2.0
    transaction_penalty: float = 0.1
    drawdown_penalty_scale: float = 5.0
    
    # Features to include in observation
    price_features: List[str] = None
    technical_features: List[str] = None
    volume_features: List[str] = None
    
    def __post_init__(self):
        if self.price_features is None:
            self.price_features = ['open', 'high', 'low', 'close']
        if self.technical_features is None:
            self.technical_features = ['sma_20', 'sma_50', 'rsi', 'bb_upper', 'bb_lower']
        if self.volume_features is None:
            self.volume_features = ['volume', 'vol_sma_20']


class TradingEnvironment(gym.Env):
    """
    Trading Environment for RL agents
    
    Observation Space: Box with normalized market features over lookback window
    Action Space: Discrete(3) - 0: Hold/Close, 1: Buy/Long, 2: Sell/Short
    """
    
    def __init__(self, config: TradingConfig = None):
        super().__init__()
        
        self.config = config or TradingConfig()
        self.validator = MarketDataValidator()
        self.alerting = get_alerting_system()
        
        # Environment state
        self.data: Optional[pd.DataFrame] = None
        self.current_step = 0
        self.max_steps = 0
        
        # Trading state
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.position = 0.0  # Current position size (-1 to 1)
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.daily_returns = []
        self.max_equity = self.config.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # Hold, Buy, Sell
        
        # Calculate observation dimension
        self._setup_observation_space()
        
        # Episode tracking
        self.episode_count = 0
        self.episode_profit = 0.0
        
        logger.info(f"Trading Environment initialized with config: {self.config}")
    
    def _setup_observation_space(self):
        """Setup observation space based on config"""
        # Calculate total features
        n_price_features = len(self.config.price_features)
        n_tech_features = len(self.config.technical_features) 
        n_vol_features = len(self.config.volume_features)
        
        # Additional state features
        n_state_features = 4  # position, drawdown, time_in_position, balance_ratio
        
        total_features = (n_price_features + n_tech_features + n_vol_features) * self.config.lookback_window + n_state_features
        
        # Normalized observation space
        self.observation_space = gym.spaces.Box(
            low=-5.0,  # Allow for extreme normalized values
            high=5.0,
            shape=(total_features,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space: {total_features} features ({self.config.lookback_window} window)")
    
    def set_data(self, data: pd.DataFrame):
        """Set market data for the environment"""
        # Validate data
        clean_data, report = self.validator.validate_ohlcv(data, "TRADING_ENV")
        
        if report['removed_rows'] > 0:
            logger.warning(f"Data validation removed {report['removed_rows']} rows")
        
        self.data = clean_data.copy()
        self.max_steps = len(self.data) - self.config.lookback_window - 1
        
        # Normalize features for better RL training
        self._normalize_data()
        
        logger.info(f"Environment loaded {len(self.data)} bars of data")
    
    def _normalize_data(self):
        """Normalize features for stable RL training"""
        # Normalize price features (returns)
        for col in self.config.price_features:
            if col in self.data.columns:
                self.data[f'{col}_norm'] = self.data[col].pct_change().fillna(0).clip(-0.1, 0.1) * 10
        
        # Normalize technical indicators
        for col in self.config.technical_features:
            if col in self.data.columns:
                # Z-score normalization
                mean_val = self.data[col].rolling(window=100, min_periods=10).mean()
                std_val = self.data[col].rolling(window=100, min_periods=10).std()
                self.data[f'{col}_norm'] = ((self.data[col] - mean_val) / (std_val + 1e-8)).fillna(0).clip(-3, 3)
        
        # Normalize volume features
        for col in self.config.volume_features:
            if col in self.data.columns:
                # Log normalize volume
                self.data[f'{col}_norm'] = np.log1p(self.data[col] / self.data[col].rolling(window=50, min_periods=10).mean()).fillna(0).clip(-2, 2)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        if self.data is None:
            raise ValueError("No market data loaded. Call set_data() first.")
        
        # Reset trading state
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.position = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Reset performance tracking
        self.equity_curve = [self.balance]
        self.trade_history = []
        self.daily_returns = []
        self.max_equity = self.balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.episode_profit = 0.0
        
        # Random start position (avoid overfitting to start of data)
        if options and options.get('deterministic', False):
            self.current_step = self.config.lookback_window
        else:
            max_start = max(self.config.lookback_window, self.max_steps - 1000)  # Keep at least 1000 steps
            self.current_step = self.np_random.integers(self.config.lookback_window, max_start)
        
        self.episode_count += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one trading step"""
        if self.current_step >= self.max_steps:
            raise ValueError("Episode is done. Call reset().")
        
        # Get current market data
        current_price = self.data.iloc[self.current_step]['close']
        prev_balance = self.balance
        
        # Execute trading action
        reward = self._execute_action(action, current_price)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps - 1
        
        # Move to next step
        self.current_step += 1
        
        # Get next observation
        observation = self._get_observation() if not (terminated or truncated) else self._get_observation()
        
        info = self._get_info()
        
        # Log significant events
        if abs(reward) > 10:  # Significant reward/penalty
            logger.debug(f"Step {self.current_step}: Action={action}, Reward={reward:.3f}, Balance={self.balance:.2f}")
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and return reward"""
        reward = 0.0
        
        # Action mapping: 0=Hold/Close, 1=Buy/Long, 2=Sell/Short
        target_position = 0.0 if action == 0 else (1.0 if action == 1 else -1.0)
        
        # Calculate position change
        position_change = target_position - self.position
        
        if abs(position_change) > 0.01:  # Significant position change
            # Calculate transaction costs
            trade_size = abs(position_change)
            transaction_cost = trade_size * self.config.transaction_cost
            slippage_cost = trade_size * self.config.slippage
            total_cost = (transaction_cost + slippage_cost) * self.balance
            
            # Apply costs
            self.balance -= total_cost
            reward -= total_cost / self.initial_balance * 100  # Penalty for transaction costs
            
            # Close existing position first
            if abs(self.position) > 0.01:
                # Calculate P&L from closing position
                position_pnl = self.position * (current_price - self.position_entry_price) / self.position_entry_price
                position_pnl *= self.balance  # Convert to dollar terms
                
                self.balance += position_pnl
                reward += position_pnl / self.initial_balance * self.config.profit_reward_scale * 100
                
                # Track trade result
                is_winning = position_pnl > 0
                self.trade_history.append({
                    'entry_step': self.position_entry_step,
                    'exit_step': self.current_step,
                    'entry_price': self.position_entry_price,
                    'exit_price': current_price,
                    'position_size': self.position,
                    'pnl': position_pnl,
                    'holding_period': self.current_step - self.position_entry_step,
                    'winning': is_winning
                })
                
                self.total_trades += 1
                if is_winning:
                    self.winning_trades += 1
            
            # Enter new position
            if abs(target_position) > 0.01:
                self.position = target_position
                self.position_entry_price = current_price
                self.position_entry_step = self.current_step
            else:
                self.position = 0.0
                self.position_entry_price = 0.0
                self.position_entry_step = 0
        
        # Add unrealized P&L to balance for current position
        if abs(self.position) > 0.01:
            unrealized_pnl = self.position * (current_price - self.position_entry_price) / self.position_entry_price
            total_balance = self.balance + unrealized_pnl * self.balance
            
            # Reward for unrealized gains, penalty for losses
            pnl_reward = unrealized_pnl * self.config.profit_reward_scale * 10
            reward += pnl_reward
            
            # Penalty for holding position too long
            holding_period = self.current_step - self.position_entry_step
            if holding_period > self.config.position_timeout:
                reward -= 1.0  # Penalty for overly long positions
        else:
            total_balance = self.balance
        
        # Update balance for next step
        self.balance = total_balance
        
        return reward
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        current_balance = self.balance
        
        # Update equity curve
        self.equity_curve.append(current_balance)
        
        # Calculate drawdown
        if current_balance > self.max_equity:
            self.max_equity = current_balance
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.max_equity - current_balance) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Calculate daily return (approximate)
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
    
    def _check_termination(self) -> bool:
        """Check if episode should be terminated"""
        # Max drawdown exceeded
        if self.current_drawdown > self.config.max_drawdown:
            logger.warning(f"Episode terminated: Max drawdown {self.current_drawdown:.3f} exceeded")
            return True
        
        # Daily loss limit exceeded (approximate)
        if len(self.daily_returns) > 0:
            recent_return = sum(self.daily_returns[-20:]) if len(self.daily_returns) >= 20 else sum(self.daily_returns)
            if recent_return < -self.config.daily_loss_limit:
                logger.warning(f"Episode terminated: Daily loss limit exceeded {recent_return:.3f}")
                return True
        
        # Balance too low
        if self.balance < self.initial_balance * 0.1:  # Lost 90% of capital
            logger.warning(f"Episode terminated: Balance too low {self.balance:.2f}")
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.current_step < self.config.lookback_window:
            raise ValueError(f"Current step {self.current_step} < lookback window {self.config.lookback_window}")
        
        # Get historical data window
        start_idx = self.current_step - self.config.lookback_window
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Collect features
        features = []
        
        # Price features (normalized)
        for col in self.config.price_features:
            norm_col = f'{col}_norm'
            if norm_col in window_data.columns:
                features.extend(window_data[norm_col].values)
            else:
                # Fallback: use raw normalized data
                values = window_data[col].pct_change().fillna(0).clip(-0.1, 0.1).values * 10
                features.extend(values)
        
        # Technical features (normalized)
        for col in self.config.technical_features:
            norm_col = f'{col}_norm'
            if norm_col in window_data.columns:
                features.extend(window_data[norm_col].values)
            else:
                # Fallback: z-score normalize
                values = window_data[col].fillna(window_data[col].mean()).values
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-8
                features.extend(((values - mean_val) / std_val).clip(-3, 3))
        
        # Volume features (normalized)
        for col in self.config.volume_features:
            norm_col = f'{col}_norm'
            if norm_col in window_data.columns:
                features.extend(window_data[norm_col].values)
            else:
                # Fallback: log normalize
                values = window_data[col].fillna(window_data[col].mean()).values
                mean_val = np.mean(values) + 1e-8
                features.extend(np.log1p(values / mean_val).clip(-2, 2))
        
        # State features
        state_features = [
            self.position,  # Current position
            self.current_drawdown,  # Current drawdown
            (self.current_step - self.position_entry_step) / 100.0 if self.position != 0 else 0,  # Time in position
            (self.balance / self.initial_balance - 1.0)  # Balance ratio change
        ]
        
        features.extend(state_features)
        
        # Convert to numpy array and ensure correct dtype
        observation = np.array(features, dtype=np.float32)
        
        # Ensure observation is finite
        observation = np.nan_to_num(observation, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about current state"""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        total_return = (self.balance / self.initial_balance - 1.0) * 100
        
        return {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'episode': self.episode_count,
            'step': self.current_step
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if len(self.equity_curve) < 2:
            return {}
        
        total_return = (self.balance / self.initial_balance - 1.0) * 100
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Calculate Sharpe ratio (approximation)
        if len(self.daily_returns) > 1:
            returns_array = np.array(self.daily_returns)
            sharpe = np.sqrt(252) * np.mean(returns_array) / (np.std(returns_array) + 1e-8)
        else:
            sharpe = 0.0
        
        # Calculate maximum consecutive losses
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in self.trade_history:
            if not trade['winning']:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return {
            'total_return': total_return,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe,
            'max_consecutive_losses': max_consecutive_losses,
            'final_balance': self.balance,
            'episode_length': len(self.equity_curve),
            'avg_trade_pnl': np.mean([t['pnl'] for t in self.trade_history]) if self.trade_history else 0,
            'profit_factor': self._calculate_profit_factor()
        }
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trade_history:
            return 0.0
        
        gross_profit = sum(t['pnl'] for t in self.trade_history if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trade_history if t['pnl'] < 0))
        
        return gross_profit / max(gross_loss, 1e-8)
    
    def render(self, mode='human'):
        """Render environment (for debugging)"""
        if mode == 'human':
            metrics = self.get_performance_metrics()
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
                  f"Position: {self.position:.2f}, Return: {metrics.get('total_return', 0):.2f}%")


def create_trading_environment(config: TradingConfig = None) -> TradingEnvironment:
    """Factory function to create trading environment"""
    return TradingEnvironment(config)


def test_environment():
    """Test the trading environment"""
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2021-01-01', freq='1H')
    n_points = len(dates)
    
    # Generate synthetic price data
    price = 100
    prices = []
    for _ in range(n_points):
        price *= (1 + np.random.normal(0, 0.001))
        prices.append(price)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Add technical indicators (simplified)
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = np.random.uniform(20, 80, n_points)  # Simplified RSI
    data['bb_upper'] = data['sma_20'] * 1.02
    data['bb_lower'] = data['sma_20'] * 0.98
    data['vol_sma_20'] = data['volume'].rolling(20).mean()
    
    # Create and test environment
    config = TradingConfig(lookback_window=20)
    env = create_trading_environment(config)
    env.set_data(data)
    
    # Test episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: Action={action}, Reward={reward:.3f}, Balance=${info['balance']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"Final metrics: {env.get_performance_metrics()}")
    print("Environment test completed!")


if __name__ == "__main__":
    test_environment()