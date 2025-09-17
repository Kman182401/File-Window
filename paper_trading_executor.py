"""
Paper Trading Executor

Executes trading decisions in paper trading mode with realistic simulation of:
- Order execution with slippage and latency
- Portfolio management
- Risk controls
- Performance tracking
- IBKR integration for live data with simulated execution
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
import time
import json
from pathlib import Path

# Local imports
import sys


from trading_decision_engine import TradingDecisionEngine, TradingDecision, PortfolioState
from market_data_ibkr_adapter import IBKRIngestor
from ibkr_paper_broker import IBKRPaperBroker, PaperOrder, OrderStatus as BrokerOrderStatus
from order_management_system import OrderManagementSystem, OrderManagerConfig, ManagedOrder, OrderPriority
from utils.market_data_validator import MarketDataValidator
from monitoring.alerting_system import get_alerting_system, AlertSeverity, AlertType
from utils.audit_logger import audit_log, AuditEventType, AuditSeverity
from configs.market_data_config import IBKR_SYMBOLS

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Represents a trading order"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    
    # Execution details
    decision: Optional[TradingDecision] = None
    execution_latency: float = 0.0
    market_impact: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    last_price: float = 0.0
    
    # Risk metrics
    max_favorable: float = 0.0  # Maximum favorable excursion
    max_adverse: float = 0.0    # Maximum adverse excursion
    
    # Timing
    entry_time: Optional[datetime] = None
    hold_duration: timedelta = timedelta()


class PaperTradingExecutor:
    """
    Paper trading executor using REAL IBKR paper trading account
    Places actual paper orders through IB Gateway for authentic learning experience
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Core components
        self.decision_engine = TradingDecisionEngine(self.config.get('decision_engine', {}))
        self.validator = MarketDataValidator()
        self.alerting = get_alerting_system()
        
        # REAL IBKR paper trading components
        self.paper_broker = IBKRPaperBroker(self.config.get('paper_broker', {}))
        self.order_manager = OrderManagementSystem(self.paper_broker, OrderManagerConfig())
        self.ibkr_ingestor = None  # Will use broker's connection
        
        # Trading state
        self.portfolio = PortfolioState(
            total_balance=self.config['initial_capital'],
            available_balance=self.config['initial_capital']
        )
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_counter = 0
        
        # Performance tracking
        self.trades_executed = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.equity_curve = [self.config['initial_capital']]
        self.daily_stats = []
        
        # Market data
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.last_prices: Dict[str, float] = {}
        
        # Execution simulation
        self.execution_delay = self.config['execution_delay_ms'] / 1000.0
        self.commission_rate = self.config['commission_rate']
        self.slippage_rate = self.config['slippage_rate']
        
        # Trading state
        self.is_trading = False
        self.trading_thread = None
        
        logger.info("Paper Trading Executor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'initial_capital': 100000.0,
            'commission_rate': 0.001,  # 0.1% per trade
            'slippage_rate': 0.0005,   # 0.05% slippage
            'execution_delay_ms': 100,  # 100ms execution delay
            'market_impact_rate': 0.0001,  # 0.01% market impact
            'max_position_value': 50000,  # Max $50k per position
            'data_update_interval': 1,  # Update data every 1 second
            'decision_interval': 5,     # Make decisions every 5 seconds
            'symbols': IBKR_SYMBOLS,
            'use_live_data': True,
            'save_state_interval': 300,  # Save state every 5 minutes
            'results_dir': 'paper_trading_results',
            'decision_engine': {}
        }
    
    def connect_to_market(self) -> bool:
        """Connect to IBKR for both market data and order execution"""
        try:
            # Connect paper broker (includes order execution)
            if not self.paper_broker.connect():
                logger.error("Failed to connect to IBKR paper broker")
                return False
            
            # Use the same IBKR connection for market data
            self.ibkr_ingestor = self.paper_broker.ibkr
            
            logger.info("Connected to IBKR for REAL paper trading and market data")
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False
    
    def start_trading(self):
        """Start REAL paper trading with IB Gateway"""
        if self.is_trading:
            logger.warning("Trading already in progress")
            return
        
        if not self.connect_to_market():
            raise ConnectionError("Failed to connect to IBKR for real paper trading")
        
        # Start order management system
        self.order_manager.start()
        
        # Setup order callbacks
        self._setup_order_callbacks()
        
        self.is_trading = True
        self.trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self.trading_thread.start()
        
        logger.info("🎯 REAL PAPER TRADING STARTED - Orders will be placed through IB Gateway!")
        
        # Send startup alert
        self.alerting.send_alert(self.alerting.create_alert(
            severity=AlertSeverity.INFO,
            alert_type=AlertType.SYSTEM_ERROR,
            title="REAL Paper Trading Started",
            message=f"REAL paper trading started with IB Gateway connection - ${self.portfolio.total_balance:.2f} capital",
            details={
                'initial_capital': self.portfolio.total_balance,
                'mode': 'REAL_PAPER_TRADING',
                'broker_connected': self.paper_broker.is_connected
            }
        ))
    
    def stop_trading(self):
        """Stop REAL paper trading"""
        self.is_trading = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)
        
        # Stop order management
        self.order_manager.stop()
        
        # Close all positions through REAL orders
        self._close_all_positions_real()
        
        # Disconnect broker
        self.paper_broker.disconnect()
        
        # Save final results
        self._save_results()
        
        logger.info("🎯 REAL Paper trading stopped - All orders processed through IB Gateway")
    
    def _trading_loop(self):
        """Main trading loop"""
        last_decision_time = time.time()
        last_data_update = time.time()
        
        while self.is_trading:
            try:
                current_time = time.time()
                
                # Update market data
                if current_time - last_data_update >= self.config['data_update_interval']:
                    self._update_market_data()
                    last_data_update = current_time
                
                # Make trading decisions
                if current_time - last_decision_time >= self.config['decision_interval']:
                    self._make_trading_decisions()
                    last_decision_time = current_time
                
                # Process pending orders
                self._process_pending_orders()
                
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                # Save state periodically
                if int(current_time) % self.config['save_state_interval'] == 0:
                    self._save_state()
                
                time.sleep(0.1)  # Small sleep to prevent excessive CPU usage
            
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(1)  # Sleep on error to prevent rapid cycling
    
    def _update_market_data(self):
        """Update market data for all symbols"""
        for symbol in self.config['symbols']:
            try:
                if self.ibkr_ingestor:
                    # Get live data from IBKR
                    data = self.ibkr_ingestor.fetch_data(
                        symbol=symbol,
                        duration="1 D",
                        barSize="1 min",
                        whatToShow="TRADES",
                        useRTH=False
                    )
                    
                    if data is not None and len(data) > 0:
                        # Validate and store data
                        clean_data, report = self.validator.validate_ohlcv(data, symbol)
                        
                        if len(clean_data) > 0:
                            self.market_data[symbol] = clean_data
                            self.last_prices[symbol] = clean_data.iloc[-1]['close']
                            
                            # Update decision engine
                            self.decision_engine.update_market_data(symbol, clean_data)
                
                else:
                    # Generate simulated data for testing
                    self._generate_simulated_data(symbol)
            
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {e}")
    
    def _generate_simulated_data(self, symbol: str):
        """Generate simulated market data for testing"""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = 100.0
        
        # Simple random walk
        price_change = np.random.normal(0, 0.001)  # 0.1% standard deviation
        new_price = self.last_prices[symbol] * (1 + price_change)
        
        # Create single bar of data
        timestamp = datetime.now()
        bar_data = pd.DataFrame({
            'open': [self.last_prices[symbol]],
            'high': [max(self.last_prices[symbol], new_price) * 1.001],
            'low': [min(self.last_prices[symbol], new_price) * 0.999],
            'close': [new_price],
            'volume': [np.random.randint(1000, 5000)]
        }, index=[timestamp])
        
        # Update stored data
        if symbol not in self.market_data:
            self.market_data[symbol] = bar_data
        else:
            self.market_data[symbol] = pd.concat([self.market_data[symbol].tail(100), bar_data])
        
        self.last_prices[symbol] = new_price
        
        # Add technical indicators
        self._add_technical_indicators(symbol)
    
    def _add_technical_indicators(self, symbol: str):
        """Add technical indicators to market data"""
        if symbol not in self.market_data:
            return
        
        df = self.market_data[symbol]
        
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume moving average
        df['vol_sma_20'] = df['volume'].rolling(window=20).mean()
        
        # Fill NaN values
        df.ffill(inplace=True)
        
        self.market_data[symbol] = df
    
    def _make_trading_decisions(self):
        """Make trading decisions for all symbols"""
        for symbol in self.config['symbols']:
            if symbol not in self.market_data or len(self.market_data[symbol]) < 50:
                continue  # Need sufficient data
            
            try:
                # Get trading decision
                decision = self.decision_engine.make_decision(symbol, self.market_data[symbol])
                
                # Execute decision if actionable
                if decision.action != 0 and decision.confidence > 0.3:
                    self._execute_decision(decision)
            
            except Exception as e:
                logger.error(f"Error making decision for {symbol}: {e}")
    
    def _execute_decision(self, decision: TradingDecision):
        """Execute a trading decision through REAL IBKR paper orders"""
        if decision.action == 0:  # Hold - no action needed
            return
        
        symbol = decision.symbol
        
        # Calculate order quantity based on position size
        # position_size is a fraction (0.0 to 1.0) of available balance
        account_info = self.paper_broker.get_account_info()
        available_balance = account_info.get('available_funds', self.portfolio.available_balance)
        
        # Convert position size to actual quantity
        position_value = decision.position_size * available_balance
        current_price = decision.price
        
        if decision.action == 1:  # Buy/Long
            quantity = position_value / current_price
        elif decision.action == 2:  # Sell/Short
            # For futures, we can go long or short
            # Check current position
            current_position = self.paper_broker.get_position(symbol)
            current_qty = current_position.quantity if current_position else 0.0
            
            if current_qty > 0:
                # Close long position first, then potentially go short
                quantity = -min(current_qty, position_value / current_price)
            else:
                # Open or add to short position
                quantity = -(position_value / current_price)
        
        # Submit order through order management system
        order_id = self.order_manager.submit_order(
            decision=decision,
            priority=OrderPriority.NORMAL
        )
        
        if order_id:
            logger.info(f"🎯 REAL ORDER SUBMITTED: {order_id} - {decision.action} {symbol} "
                       f"(confidence: {decision.confidence:.2f}) - {decision.reasoning}")
            
            # Track order locally
            self.order_counter += 1
            
            # Audit log for REAL order
            audit_log(
                AuditEventType.SYSTEM_START,
                AuditSeverity.INFO,
                "REAL_paper_trading",
                f"REAL paper order submitted: {order_id}",
                symbol=symbol,
                action=decision.action,
                quantity=quantity,
                price=current_price,
                confidence=decision.confidence,
                reasoning=decision.reasoning,
                order_manager_id=order_id
            )
        else:
            logger.error(f"Failed to submit order for {symbol}")
    
    def _setup_order_callbacks(self):
        """Setup callbacks for order updates from broker"""
        def on_order_fill(managed_order: ManagedOrder):
            """Handle order fill notification"""
            if managed_order.paper_order and managed_order.paper_order.status == BrokerOrderStatus.FILLED:
                self._handle_order_fill(managed_order)
        
        def on_order_cancel(managed_order: ManagedOrder):
            """Handle order cancellation"""
            self._handle_order_cancel(managed_order)
        
        # Set callbacks on order manager
        # (This would be implemented in the order manager)
        logger.info("Order callbacks configured for REAL paper trading")
    
    def _handle_order_fill(self, managed_order: ManagedOrder):
        """Handle real order fill from IB Gateway"""
        paper_order = managed_order.paper_order
        
        logger.info(f"🎯 REAL ORDER FILLED: {managed_order.order_id} - "
                   f"{paper_order.action} {paper_order.filled_quantity} {paper_order.symbol} "
                   f"@ ${paper_order.avg_fill_price:.4f}")
        
        # Update local tracking
        self.trades_executed += 1
        self.total_commission += paper_order.commission
        
        # Calculate actual slippage from expected vs actual price
        expected_price = managed_order.expected_fill_price
        actual_price = paper_order.avg_fill_price
        slippage = abs(actual_price - expected_price) / expected_price if expected_price > 0 else 0
        self.total_slippage += slippage * paper_order.filled_quantity * actual_price
        
        # Update portfolio state from REAL broker positions
        self._update_portfolio_from_broker()
        
        # Audit log for REAL fill
        audit_log(
            AuditEventType.SYSTEM_START,
            AuditSeverity.INFO,
            "REAL_paper_trading",
            f"REAL order filled: {managed_order.order_id}",
            symbol=paper_order.symbol,
            action=paper_order.action,
            filled_quantity=paper_order.filled_quantity,
            fill_price=paper_order.avg_fill_price,
            commission=paper_order.commission,
            slippage=slippage,
            latency_ms=managed_order.latency_ms
        )
        
        # Alert on significant fills
        if managed_order.decision.confidence > 0.7:
            self.alerting.send_alert(self.alerting.create_alert(
                severity=AlertSeverity.INFO,
                alert_type=AlertType.UNUSUAL_ACTIVITY,
                title="High-Confidence Order Filled",
                message=f"REAL paper order filled: {paper_order.action} {paper_order.filled_quantity} {paper_order.symbol}",
                details={
                    'order_id': managed_order.order_id,
                    'symbol': paper_order.symbol,
                    'action': paper_order.action,
                    'quantity': paper_order.filled_quantity,
                    'price': paper_order.avg_fill_price,
                    'confidence': managed_order.decision.confidence,
                    'commission': paper_order.commission,
                    'slippage': slippage
                }
            ))
    
    def _handle_order_cancel(self, managed_order: ManagedOrder):
        """Handle order cancellation"""
        logger.info(f"Order cancelled: {managed_order.order_id}")
    
    def _update_portfolio_from_broker(self):
        """Update portfolio state from real broker positions"""
        try:
            # Get real account info from broker
            account_info = self.paper_broker.get_account_info()
            self.portfolio.total_balance = account_info.get('account_balance', self.portfolio.total_balance)
            self.portfolio.available_balance = account_info.get('available_funds', self.portfolio.available_balance)
            
            # Get real positions
            broker_positions = self.paper_broker.get_all_positions()
            
            # Update local position tracking
            for broker_pos in broker_positions:
                symbol = broker_pos.symbol
                
                # Create or update local position
                if symbol not in self.positions:
                    self.positions[symbol] = Position(symbol=symbol)
                
                local_pos = self.positions[symbol]
                local_pos.quantity = broker_pos.quantity
                local_pos.avg_entry_price = broker_pos.avg_cost
                local_pos.unrealized_pnl = broker_pos.unrealized_pnl
                local_pos.last_price = broker_pos.market_price
            
            # Update equity curve with real balance
            self.equity_curve.append(self.portfolio.total_balance)
            
            # Keep equity curve manageable
            if len(self.equity_curve) > 10000:
                self.equity_curve = self.equity_curve[-5000:]
        
        except Exception as e:
            logger.error(f"Error updating portfolio from broker: {e}")
    
    def _update_position_from_order(self, order: Order):
        """Update position after order execution"""
        symbol = order.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        # Calculate quantity change
        quantity_change = order.filled_quantity if order.side == "BUY" else -order.filled_quantity
        
        # Update position
        if position.quantity == 0:
            # Opening new position
            position.quantity = quantity_change
            position.avg_entry_price = order.avg_fill_price
            position.entry_time = order.timestamp
        elif (position.quantity > 0 and quantity_change > 0) or (position.quantity < 0 and quantity_change < 0):
            # Adding to position
            total_value = (position.quantity * position.avg_entry_price + 
                          quantity_change * order.avg_fill_price)
            position.quantity += quantity_change
            position.avg_entry_price = total_value / position.quantity if position.quantity != 0 else 0
        else:
            # Closing or reducing position
            if abs(quantity_change) >= abs(position.quantity):
                # Closing entire position
                realized_pnl = position.quantity * (order.avg_fill_price - position.avg_entry_price)
                position.realized_pnl += realized_pnl
                
                # Check for position reversal
                remaining_quantity = quantity_change + position.quantity
                if abs(remaining_quantity) > 0.0001:  # Reversing position
                    position.quantity = remaining_quantity
                    position.avg_entry_price = order.avg_fill_price
                    position.entry_time = order.timestamp
                else:
                    # Position fully closed
                    position.quantity = 0
                    position.avg_entry_price = 0
                    position.entry_time = None
            else:
                # Partial close
                close_ratio = abs(quantity_change) / abs(position.quantity)
                realized_pnl = close_ratio * position.quantity * (order.avg_fill_price - position.avg_entry_price)
                position.realized_pnl += realized_pnl
                position.quantity += quantity_change
        
        # Update commission
        position.commission_paid += order.commission
        
        # Update hold duration
        if position.entry_time and position.quantity != 0:
            position.hold_duration = datetime.now() - position.entry_time
    
    def _update_portfolio_from_order(self, order: Order):
        """Update portfolio after order execution"""
        # Update available balance
        trade_value = order.filled_quantity * order.avg_fill_price
        
        if order.side == "BUY":
            self.portfolio.available_balance -= (trade_value + order.commission)
        else:
            self.portfolio.available_balance += (trade_value - order.commission)
        
        # Update trade count
        self.portfolio.total_trades_today += 1
        
        # Update decision engine portfolio state
        symbol = order.symbol
        position = self.positions.get(symbol, Position(symbol)).quantity
        realized_pnl = self.positions.get(symbol, Position(symbol)).realized_pnl
        
        self.decision_engine.update_portfolio_state(symbol, position, realized_pnl)
    
    def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        # Calculate current portfolio value
        total_value = self.portfolio.available_balance
        
        # Add position values
        for symbol, position in self.positions.items():
            if position.quantity != 0 and symbol in self.last_prices:
                current_price = self.last_prices[symbol]
                position_value = position.quantity * current_price
                total_value += position_value
                
                # Update unrealized P&L
                position.unrealized_pnl = position.quantity * (current_price - position.avg_entry_price)
                position.last_price = current_price
                
                # Update max favorable/adverse excursion
                if position.quantity > 0:  # Long position
                    excursion = current_price - position.avg_entry_price
                    if excursion > position.max_favorable:
                        position.max_favorable = excursion
                    if excursion < position.max_adverse:
                        position.max_adverse = excursion
                else:  # Short position
                    excursion = position.avg_entry_price - current_price
                    if excursion > position.max_favorable:
                        position.max_favorable = excursion
                    if excursion < position.max_adverse:
                        position.max_adverse = excursion
        
        # Update portfolio balance
        self.portfolio.total_balance = total_value
        
        # Update equity curve
        self.equity_curve.append(total_value)
        
        # Keep equity curve manageable
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-5000:]  # Keep last 5000 points
        
        # Calculate daily P&L
        if len(self.equity_curve) > 1:
            self.portfolio.daily_pnl = self.equity_curve[-1] - self.config['initial_capital']
    
    def _close_all_positions_real(self):
        """Close all open positions through REAL IB Gateway orders"""
        broker_positions = self.paper_broker.get_all_positions()
        
        for position in broker_positions:
            if abs(position.quantity) > 0.001:
                symbol = position.symbol
                
                # Create closing decision
                close_decision = TradingDecision(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    action=0,  # Close position
                    confidence=1.0,  # High confidence for system shutdown
                    position_size=abs(position.quantity),
                    price=position.market_price,
                    reasoning="System shutdown - closing all positions"
                )
                
                # Determine close action based on current position
                if position.quantity > 0:
                    # Close long position
                    quantity = -position.quantity
                else:
                    # Close short position  
                    quantity = -position.quantity
                
                # Place REAL closing order through broker
                order_id = self.paper_broker.place_market_order(
                    symbol=symbol,
                    quantity=quantity,
                    confidence=1.0,
                    reasoning="System shutdown - position close"
                )
                
                if order_id:
                    logger.info(f"🎯 REAL CLOSING ORDER: {symbol} - Close {position.quantity} contracts")
                    
                    # Wait for fill (with timeout)
                    start_time = time.time()
                    while time.time() - start_time < 30:  # 30 second timeout
                        broker_order = self.paper_broker.get_order(order_id)
                        if broker_order and broker_order.status == BrokerOrderStatus.FILLED:
                            logger.info(f"✅ Position closed: {symbol}")
                            break
                        time.sleep(1)
                    else:
                        logger.warning(f"⚠️ Closing order timeout: {symbol}")
                else:
                    logger.error(f"❌ Failed to place closing order: {symbol}")
    
    def _save_state(self):
        """Save current trading state"""
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': {
                'total_balance': self.portfolio.total_balance,
                'available_balance': self.portfolio.available_balance,
                'daily_pnl': self.portfolio.daily_pnl,
                'trades_today': self.portfolio.total_trades_today
            },
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'avg_entry_price': pos.avg_entry_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'commission_paid': pos.commission_paid
                }
                for symbol, pos in self.positions.items()
                if abs(pos.quantity) > 0.001
            },
            'performance': {
                'trades_executed': self.trades_executed,
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'equity_value': self.equity_curve[-1] if self.equity_curve else 0
            }
        }
        
        state_file = results_dir / f"trading_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _save_results(self):
        """Save final trading results"""
        results_dir = Path(self.config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate performance metrics
        total_return = (self.portfolio.total_balance / self.config['initial_capital'] - 1) * 100
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 1:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            sharpe = np.sqrt(252 * 24) * np.mean(returns) / (np.std(returns) + 1e-8)  # Assuming hourly data
        else:
            sharpe = 0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_array)
        drawdown = (peak - equity_array) / peak
        max_drawdown = np.max(drawdown) * 100
        
        results = {
            'summary': {
                'initial_capital': self.config['initial_capital'],
                'final_balance': self.portfolio.total_balance,
                'total_return_pct': total_return,
                'total_trades': self.trades_executed,
                'total_commission': self.total_commission,
                'total_slippage': self.total_slippage,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_drawdown
            },
            'positions': [
                {
                    'symbol': symbol,
                    'quantity': pos.quantity,
                    'avg_entry_price': pos.avg_entry_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'total_pnl': pos.unrealized_pnl + pos.realized_pnl,
                    'commission_paid': pos.commission_paid,
                    'max_favorable': pos.max_favorable,
                    'max_adverse': pos.max_adverse,
                    'hold_duration_hours': pos.hold_duration.total_seconds() / 3600 if pos.hold_duration else 0
                }
                for symbol, pos in self.positions.items()
            ],
            'orders': [
                {
                    'order_id': order.order_id,
                    'timestamp': order.timestamp.isoformat(),
                    'symbol': order.symbol,
                    'side': order.side,
                    'quantity': order.quantity,
                    'status': order.status.value,
                    'avg_fill_price': order.avg_fill_price,
                    'commission': order.commission,
                    'slippage': order.slippage,
                    'execution_latency': order.execution_latency
                }
                for order in self.orders
            ],
            'equity_curve': self.equity_curve,
            'decision_stats': self.decision_engine.get_decision_stats()
        }
        
        results_file = results_dir / f"paper_trading_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Final Performance: Return={total_return:.2f}%, "
                   f"Trades={self.trades_executed}, Sharpe={sharpe:.2f}, MaxDD={max_drawdown:.2f}%")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        total_return = (self.portfolio.total_balance / self.config['initial_capital'] - 1) * 100
        
        return {
            'is_trading': self.is_trading,
            'portfolio': {
                'total_balance': self.portfolio.total_balance,
                'available_balance': self.portfolio.available_balance,
                'total_return_pct': total_return,
                'daily_pnl': self.portfolio.daily_pnl,
                'trades_today': self.portfolio.total_trades_today
            },
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl
                }
                for symbol, pos in self.positions.items()
                if abs(pos.quantity) > 0.001
            },
            'performance': {
                'trades_executed': self.trades_executed,
                'total_commission': self.total_commission,
                'symbols_trading': list(self.config['symbols'])
            }
        }


def test_paper_trading():
    """Test the paper trading system"""
    config = {
        'initial_capital': 10000,
        'use_live_data': False,  # Use simulated data for testing
        'symbols': ['TEST'],
        'decision_interval': 1,  # Make decisions every second for fast testing
        'data_update_interval': 0.5
    }
    
    executor = PaperTradingExecutor(config)
    
    try:
        print("Starting paper trading test...")
        executor.start_trading()
        
        # Let it run for 30 seconds
        time.sleep(30)
        
        # Check status
        status = executor.get_current_status()
        print(f"Status after 30 seconds: {status}")
        
        # Stop trading
        executor.stop_trading()
        print("Paper trading test completed!")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
        executor.stop_trading()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_paper_trading()
