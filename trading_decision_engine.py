"""
Trading Decision Engine

Central decision-making system that combines:
- Rule-based strategies
- RL-trained models
- Risk management
- Real-time feature processing

This is the "brain" of the trading system.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import threading
import time
from pathlib import Path

# Local imports
import sys
sys.path.append('/home/ubuntu')

from trading_strategies import (
    TradingStrategy, TradeSignal, create_default_strategy, 
    MovingAverageCrossoverStrategy, RSIMeanReversionStrategy,
    BollingerBandStrategy, EnsembleStrategy
)
from rl_trainer import RLTrainer
from trading_environment import TradingConfig
from utils.market_data_validator import MarketDataValidator
from monitoring.alerting_system import get_alerting_system, AlertSeverity, AlertType
from configs.market_data_config import (
    MAX_POSITION_EXPOSURE, MAX_ORDER_SIZE, 
    MAX_DAILY_LOSS_PCT, MAX_TRADES_PER_DAY
)

logger = logging.getLogger(__name__)


class DecisionMode(Enum):
    """Decision making modes"""
    RULE_BASED = "rule_based"      # Use only rule-based strategies
    RL_ONLY = "rl_only"           # Use only RL model
    HYBRID = "hybrid"             # Combine rule-based and RL
    ENSEMBLE = "ensemble"         # Weighted ensemble of all methods


@dataclass
class TradingDecision:
    """Represents a trading decision"""
    timestamp: datetime
    symbol: str
    action: int  # 0=Hold/Close, 1=Buy/Long, 2=Sell/Short
    confidence: float  # 0.0 to 1.0
    position_size: float  # Recommended position size (0.0 to 1.0)
    price: float
    reasoning: str
    
    # Risk metrics
    expected_return: float = 0.0
    expected_risk: float = 0.0
    max_loss: float = 0.0
    
    # Source information
    strategy_votes: Dict[str, float] = None
    rl_prediction: Optional[Tuple[int, float]] = None
    features: Dict[str, float] = None
    
    def __post_init__(self):
        if self.strategy_votes is None:
            self.strategy_votes = {}
        if self.features is None:
            self.features = {}


@dataclass
class PortfolioState:
    """Current portfolio state"""
    total_balance: float = 100000.0
    available_balance: float = 100000.0
    positions: Dict[str, float] = None  # symbol -> position size
    unrealized_pnl: Dict[str, float] = None  # symbol -> unrealized PnL
    daily_pnl: float = 0.0
    total_trades_today: int = 0
    max_drawdown: float = 0.0
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.unrealized_pnl is None:
            self.unrealized_pnl = {}


class TradingDecisionEngine:
    """
    Main trading decision engine that combines all strategies and models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.validator = MarketDataValidator()
        self.alerting = get_alerting_system()
        
        # Decision components
        self.rule_strategies: List[TradingStrategy] = []
        self.rl_trainer: Optional[RLTrainer] = None
        self.ensemble_strategy: Optional[EnsembleStrategy] = None
        
        # State management
        self.portfolio = PortfolioState()
        self.decision_mode = DecisionMode(self.config.get('decision_mode', 'hybrid'))
        self.feature_buffer = {}  # Symbol -> recent features
        self.decision_history = []
        
        # Risk management
        self.risk_limits = {
            'max_position': MAX_POSITION_EXPOSURE,
            'max_order_size': MAX_ORDER_SIZE,
            'max_daily_loss_pct': MAX_DAILY_LOSS_PCT,
            'max_trades_per_day': MAX_TRADES_PER_DAY
        }
        
        # Performance tracking
        self.decisions_made = 0
        self.successful_decisions = 0
        self.last_decision_time = None
        
        # Initialize components
        self._initialize_strategies()
        self._initialize_rl_model()
        
        logger.info(f"Trading Decision Engine initialized in {self.decision_mode.value} mode")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'decision_mode': 'hybrid',
            'rl_model_path': None,
            'feature_buffer_size': 100,
            'min_confidence_threshold': 0.3,
            'max_position_concentration': 0.5,  # Max 50% in any one position
            'risk_multiplier': 1.0,
            'enable_ensemble': True,
            'ensemble_weights': {
                'ma_crossover': 0.3,
                'rsi_mean_reversion': 0.2,
                'bollinger_bands': 0.2,
                'rl_model': 0.3
            },
            'lookback_window': 50,
            'models_dir': 'models'
        }
    
    def _initialize_strategies(self):
        """Initialize rule-based strategies"""
        # Create individual strategies
        ma_fast = MovingAverageCrossoverStrategy(fast_period=10, slow_period=30)
        ma_slow = MovingAverageCrossoverStrategy(fast_period=20, slow_period=50)
        rsi_strategy = RSIMeanReversionStrategy(rsi_period=14, oversold=25, overbought=75)
        bb_strategy = BollingerBandStrategy(period=20, std_multiplier=2.0)
        
        self.rule_strategies = [ma_fast, ma_slow, rsi_strategy, bb_strategy]
        
        # Create ensemble if enabled
        if self.config.get('enable_ensemble', True):
            weights = [
                self.config['ensemble_weights'].get('ma_crossover', 0.3),
                0.2,  # Second MA strategy
                self.config['ensemble_weights'].get('rsi_mean_reversion', 0.2),
                self.config['ensemble_weights'].get('bollinger_bands', 0.2)
            ]
            self.ensemble_strategy = EnsembleStrategy(self.rule_strategies, weights)
        
        logger.info(f"Initialized {len(self.rule_strategies)} rule-based strategies")
    
    def _initialize_rl_model(self):
        """Initialize RL model if available"""
        model_path = self.config.get('rl_model_path')
        
        if model_path and Path(model_path).exists():
            try:
                self.rl_trainer = RLTrainer(self.config)
                
                # Create minimal environment for prediction
                trading_config = TradingConfig(lookback_window=self.config['lookback_window'])
                
                # Load model
                success = self.rl_trainer.create_model(load_existing=True, model_path=model_path)
                
                if success:
                    logger.info(f"RL model loaded from {model_path}")
                else:
                    self.rl_trainer = None
                    logger.warning("Failed to load RL model")
            
            except Exception as e:
                logger.error(f"Failed to initialize RL model: {e}")
                self.rl_trainer = None
        else:
            logger.info("No RL model path provided or model not found")
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """Update market data for a symbol"""
        # Validate data
        clean_data, report = self.validator.validate_ohlcv(data, symbol)
        
        if report['removed_rows'] > 0:
            logger.warning(f"Cleaned {report['removed_rows']} invalid rows for {symbol}")
        
        # Store recent features
        if len(clean_data) > 0:
            features = self._extract_features(clean_data.tail(1))
            
            if symbol not in self.feature_buffer:
                self.feature_buffer[symbol] = []
            
            self.feature_buffer[symbol].append({
                'timestamp': clean_data.index[-1],
                'features': features,
                'price': clean_data.iloc[-1]['close']
            })
            
            # Keep only recent features
            max_buffer = self.config['feature_buffer_size']
            if len(self.feature_buffer[symbol]) > max_buffer:
                self.feature_buffer[symbol] = self.feature_buffer[symbol][-max_buffer:]
    
    def _extract_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Extract features from market data"""
        if len(data) == 0:
            return {}
        
        row = data.iloc[-1]
        features = {}
        
        # Price features
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in row:
                features[col] = float(row[col])
        
        # Technical indicators (if available)
        tech_indicators = ['sma_20', 'sma_50', 'rsi', 'bb_upper', 'bb_lower', 'bb_middle']
        for indicator in tech_indicators:
            if indicator in row:
                features[indicator] = float(row[indicator])
        
        # Derived features
        if 'close' in features and 'open' in features:
            features['price_change_pct'] = (features['close'] - features['open']) / features['open']
        
        if 'sma_20' in features and 'sma_50' in features:
            features['ma_cross_signal'] = 1.0 if features['sma_20'] > features['sma_50'] else -1.0
        
        return features
    
    def make_decision(self, symbol: str, current_data: pd.DataFrame) -> TradingDecision:
        """
        Make trading decision for a symbol
        
        Args:
            symbol: Trading symbol
            current_data: Recent market data including current bar
            
        Returns:
            TradingDecision with recommendation
        """
        if len(current_data) < 50:
            return self._create_hold_decision(symbol, current_data, "Insufficient data")
        
        current_price = current_data.iloc[-1]['close']
        timestamp = current_data.index[-1] if hasattr(current_data.index, '__getitem__') else datetime.now()
        
        # Check risk limits first
        risk_check = self._check_risk_limits(symbol)
        if risk_check['blocked']:
            return self._create_hold_decision(symbol, current_data, f"Risk limit: {risk_check['reason']}")
        
        # Get features
        features = self._extract_features(current_data.tail(1))
        
        strategy_votes = {}
        rl_prediction = None
        confidence_scores = []
        
        # 1. Rule-based strategies
        if self.decision_mode in [DecisionMode.RULE_BASED, DecisionMode.HYBRID, DecisionMode.ENSEMBLE]:
            for strategy in self.rule_strategies:
                try:
                    # Update strategy state
                    current_position = self.portfolio.positions.get(symbol, 0.0)
                    strategy.update_position(current_position)
                    
                    # Get signal
                    signal = strategy.generate_signal(current_data, len(current_data) - 1)
                    strategy_votes[strategy.name] = {
                        'action': signal.action,
                        'confidence': signal.confidence,
                        'reasoning': signal.reasoning
                    }
                    
                    if signal.confidence > 0.1:
                        confidence_scores.append(signal.confidence)
                
                except Exception as e:
                    logger.error(f"Error in strategy {strategy.name}: {e}")
        
        # 2. RL model prediction
        if self.decision_mode in [DecisionMode.RL_ONLY, DecisionMode.HYBRID] and self.rl_trainer:
            try:
                # Prepare observation (simplified)
                observation = self._prepare_rl_observation(current_data, symbol)
                if observation is not None:
                    action, rl_confidence = self.rl_trainer.predict(observation)
                    rl_prediction = (action, rl_confidence)
                    
                    strategy_votes['rl_model'] = {
                        'action': action,
                        'confidence': rl_confidence,
                        'reasoning': 'RL model prediction'
                    }
                    confidence_scores.append(rl_confidence)
            
            except Exception as e:
                logger.error(f"Error in RL prediction: {e}")
        
        # 3. Ensemble decision
        if self.decision_mode == DecisionMode.ENSEMBLE and self.ensemble_strategy:
            try:
                current_position = self.portfolio.positions.get(symbol, 0.0)
                self.ensemble_strategy.update_position(current_position)
                
                ensemble_signal = self.ensemble_strategy.generate_signal(current_data, len(current_data) - 1)
                strategy_votes['ensemble'] = {
                    'action': ensemble_signal.action,
                    'confidence': ensemble_signal.confidence,
                    'reasoning': ensemble_signal.reasoning
                }
                confidence_scores.append(ensemble_signal.confidence)
            
            except Exception as e:
                logger.error(f"Error in ensemble strategy: {e}")
        
        # Combine all signals
        final_decision = self._combine_signals(
            strategy_votes, 
            rl_prediction, 
            features, 
            symbol, 
            current_price, 
            timestamp
        )
        
        # Apply position sizing and final risk checks
        final_decision = self._apply_position_sizing(final_decision, symbol)
        final_decision = self._final_risk_check(final_decision, symbol)
        
        # Record decision
        self.decision_history.append(final_decision)
        self.decisions_made += 1
        self.last_decision_time = timestamp
        
        # Alert on significant decisions
        if final_decision.confidence > 0.7 and final_decision.action != 0:
            self.alerting.send_alert(self.alerting.create_alert(
                severity=AlertSeverity.INFO,
                alert_type=AlertType.UNUSUAL_ACTIVITY,
                title=f"High Confidence Trading Signal",
                message=f"{symbol}: {final_decision.reasoning}",
                details={
                    'symbol': symbol,
                    'action': final_decision.action,
                    'confidence': final_decision.confidence,
                    'position_size': final_decision.position_size
                }
            ))
        
        return final_decision
    
    def _combine_signals(self, strategy_votes: Dict[str, Dict], 
                        rl_prediction: Optional[Tuple[int, float]],
                        features: Dict[str, float],
                        symbol: str, price: float, timestamp: datetime) -> TradingDecision:
        """Combine all trading signals into final decision"""
        
        if not strategy_votes:
            return self._create_hold_decision(symbol, pd.DataFrame(), "No valid signals")
        
        # Weighted voting based on confidence
        action_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # Hold, Buy, Sell
        total_weight = 0.0
        reasoning_parts = []
        
        # Weight strategies by their configured importance
        weights = self.config['ensemble_weights']
        
        for strategy_name, vote in strategy_votes.items():
            action = vote['action']
            confidence = vote['confidence']
            
            # Get strategy weight
            weight = weights.get(strategy_name.lower().replace('_', '_'), 1.0)
            if 'ma_crossover' in strategy_name.lower():
                weight = weights.get('ma_crossover', 1.0)
            elif 'rsi' in strategy_name.lower():
                weight = weights.get('rsi_mean_reversion', 1.0)
            elif 'bollinger' in strategy_name.lower():
                weight = weights.get('bollinger_bands', 1.0)
            elif 'rl' in strategy_name.lower():
                weight = weights.get('rl_model', 1.0)
            
            weighted_confidence = confidence * weight
            action_scores[action] += weighted_confidence
            total_weight += weight
            
            if confidence > 0.1:
                reasoning_parts.append(f"{strategy_name}: {vote['reasoning']} (conf: {confidence:.2f})")
        
        # Determine final action
        if total_weight == 0:
            final_action = 0
            final_confidence = 0.0
        else:
            final_action = max(action_scores, key=action_scores.get)
            final_confidence = action_scores[final_action] / total_weight
        
        # Apply minimum confidence threshold
        min_confidence = self.config['min_confidence_threshold']
        if final_confidence < min_confidence:
            final_action = 0
            final_confidence = 0.0
            reasoning = f"Confidence {final_confidence:.2f} below threshold {min_confidence}"
        else:
            reasoning = " | ".join(reasoning_parts[:3])  # Keep top 3 reasons
        
        # Calculate position size based on confidence
        base_position_size = 0.1  # 10% base position
        confidence_multiplier = min(final_confidence * 2, 1.0)  # Max 100% at confidence 0.5+
        position_size = base_position_size * confidence_multiplier * self.config.get('risk_multiplier', 1.0)
        
        return TradingDecision(
            timestamp=timestamp,
            symbol=symbol,
            action=final_action,
            confidence=final_confidence,
            position_size=position_size,
            price=price,
            reasoning=reasoning,
            strategy_votes=strategy_votes,
            rl_prediction=rl_prediction,
            features=features
        )
    
    def _prepare_rl_observation(self, data: pd.DataFrame, symbol: str) -> Optional[np.ndarray]:
        """Prepare observation for RL model"""
        try:
            lookback = self.config['lookback_window']
            if len(data) < lookback:
                return None
            
            # Get recent data window
            window_data = data.tail(lookback)
            
            # Extract features (simplified version of environment observation)
            features = []
            
            # Price returns
            close_returns = window_data['close'].pct_change().fillna(0).clip(-0.1, 0.1) * 10
            features.extend(close_returns.values)
            
            # Technical indicators (normalized)
            if 'rsi' in window_data.columns:
                rsi_norm = (window_data['rsi'] - 50) / 50
                features.extend(rsi_norm.fillna(0).values)
            else:
                features.extend([0] * lookback)
            
            # Volume (log normalized)
            if 'volume' in window_data.columns:
                vol_norm = np.log1p(window_data['volume'] / window_data['volume'].rolling(20).mean().fillna(1))
                features.extend(vol_norm.fillna(0).clip(-2, 2).values)
            else:
                features.extend([0] * lookback)
            
            # Add state features
            current_position = self.portfolio.positions.get(symbol, 0.0)
            state_features = [
                current_position,  # Current position
                self.portfolio.daily_pnl / self.portfolio.total_balance,  # Daily P&L ratio
                0,  # Time in position (simplified)
                (self.portfolio.total_balance / 100000 - 1.0)  # Balance change
            ]
            features.extend(state_features)
            
            observation = np.array(features, dtype=np.float32)
            observation = np.nan_to_num(observation, nan=0.0, posinf=5.0, neginf=-5.0)
            
            return observation
        
        except Exception as e:
            logger.error(f"Error preparing RL observation: {e}")
            return None
    
    def _check_risk_limits(self, symbol: str) -> Dict[str, Union[bool, str]]:
        """Check if trading is allowed based on risk limits"""
        # Check daily loss limit
        daily_loss_pct = abs(min(0, self.portfolio.daily_pnl)) / self.portfolio.total_balance * 100
        if daily_loss_pct >= self.risk_limits['max_daily_loss_pct']:
            return {'blocked': True, 'reason': f'Daily loss limit exceeded: {daily_loss_pct:.2f}%'}
        
        # Check daily trade limit
        if self.portfolio.total_trades_today >= self.risk_limits['max_trades_per_day']:
            return {'blocked': True, 'reason': f'Daily trade limit exceeded: {self.portfolio.total_trades_today}'}
        
        # Check position concentration
        current_position = abs(self.portfolio.positions.get(symbol, 0.0))
        max_concentration = self.config['max_position_concentration']
        position_value = current_position * self.portfolio.total_balance
        
        if position_value / self.portfolio.total_balance > max_concentration:
            return {'blocked': True, 'reason': f'Position concentration too high: {position_value/self.portfolio.total_balance*100:.1f}%'}
        
        return {'blocked': False, 'reason': 'OK'}
    
    def _apply_position_sizing(self, decision: TradingDecision, symbol: str) -> TradingDecision:
        """Apply position sizing rules"""
        if decision.action == 0:  # Hold
            return decision
        
        # Get current position
        current_position = self.portfolio.positions.get(symbol, 0.0)
        
        # Calculate maximum allowed position change
        max_order_size = self.risk_limits['max_order_size']
        max_position = self.risk_limits['max_position']
        
        # Adjust position size based on current portfolio state
        available_balance_ratio = self.portfolio.available_balance / self.portfolio.total_balance
        decision.position_size *= available_balance_ratio  # Reduce size if low available balance
        
        # Ensure we don't exceed position limits
        if decision.action == 1:  # Buy
            max_additional = max_position - current_position
            decision.position_size = min(decision.position_size, max_additional, max_order_size)
        elif decision.action == 2:  # Sell
            max_reduction = current_position + max_position  # Can go from +max to -max
            decision.position_size = min(decision.position_size, max_reduction, max_order_size)
        
        # Minimum position size check
        if decision.position_size < 0.01:  # Less than 1%
            decision.action = 0
            decision.confidence = 0.0
            decision.reasoning = "Position size too small after risk adjustments"
        
        return decision
    
    def _final_risk_check(self, decision: TradingDecision, symbol: str) -> TradingDecision:
        """Final risk check before execution"""
        if decision.action == 0:
            return decision
        
        # Check if market conditions are suitable for trading
        if symbol in self.feature_buffer and len(self.feature_buffer[symbol]) > 0:
            recent_features = self.feature_buffer[symbol][-1]['features']
            
            # Don't trade during extreme volatility (simplified check)
            if 'price_change_pct' in recent_features:
                price_change = abs(recent_features['price_change_pct'])
                if price_change > 0.05:  # 5% price change
                    decision.action = 0
                    decision.confidence = 0.0
                    decision.reasoning = f"Extreme volatility detected: {price_change*100:.1f}%"
        
        return decision
    
    def _create_hold_decision(self, symbol: str, data: pd.DataFrame, reason: str) -> TradingDecision:
        """Create a hold decision"""
        price = data.iloc[-1]['close'] if len(data) > 0 else 100.0
        timestamp = data.index[-1] if len(data) > 0 and hasattr(data.index, '__getitem__') else datetime.now()
        
        return TradingDecision(
            timestamp=timestamp,
            symbol=symbol,
            action=0,
            confidence=0.0,
            position_size=0.0,
            price=price,
            reasoning=reason
        )
    
    def update_portfolio_state(self, symbol: str, position: float, pnl: float):
        """Update portfolio state after trade execution"""
        self.portfolio.positions[symbol] = position
        self.portfolio.daily_pnl += pnl
        
        if pnl != 0:  # Realized P&L indicates a trade was made
            self.portfolio.total_trades_today += 1
        
        # Update balance
        self.portfolio.total_balance += pnl
        
        # Recalculate available balance (simplified)
        position_value = sum(abs(pos) for pos in self.portfolio.positions.values()) * self.portfolio.total_balance
        self.portfolio.available_balance = self.portfolio.total_balance - position_value
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics"""
        recent_decisions = self.decision_history[-100:] if len(self.decision_history) >= 100 else self.decision_history
        
        if not recent_decisions:
            return {'no_decisions': True}
        
        # Calculate stats
        total_decisions = len(recent_decisions)
        hold_decisions = sum(1 for d in recent_decisions if d.action == 0)
        buy_decisions = sum(1 for d in recent_decisions if d.action == 1)
        sell_decisions = sum(1 for d in recent_decisions if d.action == 2)
        
        avg_confidence = np.mean([d.confidence for d in recent_decisions])
        high_confidence_decisions = sum(1 for d in recent_decisions if d.confidence > 0.7)
        
        return {
            'total_decisions': total_decisions,
            'hold_decisions': hold_decisions,
            'buy_decisions': buy_decisions,
            'sell_decisions': sell_decisions,
            'avg_confidence': avg_confidence,
            'high_confidence_decisions': high_confidence_decisions,
            'decision_mode': self.decision_mode.value,
            'portfolio_state': {
                'total_balance': self.portfolio.total_balance,
                'daily_pnl': self.portfolio.daily_pnl,
                'positions': dict(self.portfolio.positions),
                'trades_today': self.portfolio.total_trades_today
            }
        }


def test_decision_engine():
    """Test the trading decision engine"""
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-02-01', freq='1H')
    n_points = len(dates)
    
    # Generate price data
    price = 100
    prices = []
    for _ in range(n_points):
        price *= (1 + np.random.normal(0, 0.01))
        prices.append(price)
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 5000, n_points)
    }, index=dates)
    
    # Add technical indicators
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    data['rsi'] = np.random.uniform(30, 70, n_points)
    data['bb_upper'] = data['close'] * 1.02
    data['bb_lower'] = data['close'] * 0.98
    
    # Create decision engine
    config = {
        'decision_mode': 'hybrid',
        'min_confidence_threshold': 0.2,
        'enable_ensemble': True
    }
    
    engine = TradingDecisionEngine(config)
    
    # Test decisions
    print("Testing Trading Decision Engine...")
    
    for i in range(60, len(data), 10):  # Test every 10 bars after warmup
        current_data = data.iloc[:i+1]
        decision = engine.make_decision("TEST", current_data)
        
        if decision.confidence > 0.3:
            print(f"Step {i}: {decision.action} (conf: {decision.confidence:.2f}) - {decision.reasoning}")
        
        # Simulate portfolio update
        if decision.action != 0:
            simulated_pnl = np.random.normal(0, 100)  # Random P&L for testing
            engine.update_portfolio_state("TEST", decision.position_size if decision.action == 1 else -decision.position_size, simulated_pnl)
    
    # Print stats
    stats = engine.get_decision_stats()
    print(f"\nDecision Stats: {stats}")
    
    print("Decision engine test completed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_decision_engine()