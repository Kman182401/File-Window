"""
Simulation Broker - Ultra-lightweight paper trading engine
Optimized for minimal memory and CPU usage on m5.large
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class SimulatedOrder:
    """Lightweight order representation."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    order_id: str = field(default_factory=lambda: str(time.time_ns()))


@dataclass
class Position:
    """Efficient position tracking."""
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_position(self, quantity: int, price: float):
        """Update position with new trade."""
        if self.quantity == 0:
            # New position
            self.quantity = quantity
            self.avg_price = price
        elif (self.quantity > 0 and quantity > 0) or (self.quantity < 0 and quantity < 0):
            # Adding to position
            total_cost = (self.quantity * self.avg_price) + (quantity * price)
            self.quantity += quantity
            self.avg_price = total_cost / self.quantity if self.quantity != 0 else 0
        else:
            # Reducing or closing position
            if abs(quantity) >= abs(self.quantity):
                # Position closed or reversed
                self.realized_pnl += (price - self.avg_price) * self.quantity
                remaining = quantity + self.quantity
                self.quantity = remaining
                self.avg_price = price if remaining != 0 else 0
            else:
                # Partial close
                self.realized_pnl += (price - self.avg_price) * (-quantity)
                self.quantity += quantity


class SimulationBroker:
    """
    Ultra-efficient paper trading broker optimized for m5.large.
    
    Features:
    - Minimal memory footprint (<100MB)
    - O(1) position lookups
    - Realistic slippage and fees
    - No database required
    """
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission_per_contract: float = 2.25,
                 slippage_ticks: int = 1,
                 tick_size: float = 0.25,
                 max_positions: int = 10):
        """
        Initialize broker with trading parameters.
        
        Args:
            initial_capital: Starting capital
            commission_per_contract: Commission per futures contract
            slippage_ticks: Number of ticks slippage for market orders
            tick_size: Minimum price movement
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.max_positions = max_positions
        
        # Position tracking (memory-efficient)
        self.positions: Dict[str, Position] = {}
        
        # Order tracking (keep only recent 100)
        self.orders = deque(maxlen=100)
        self.filled_orders = deque(maxlen=100)
        
        # Trade log (keep only recent 100)
        self.trades = deque(maxlen=100)
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.peak_equity = initial_capital
        
        # Equity curve (keep only recent 1000 points)
        self.equity_curve = deque(maxlen=1000)
        self.equity_curve.append(initial_capital)
        
    def _calculate_slippage(self, side: OrderSide, price: float) -> float:
        """Calculate slippage for market order."""
        slippage = self.slippage_ticks * self.tick_size
        if side == OrderSide.BUY:
            return price + slippage  # Pay more when buying
        else:
            return price - slippage  # Receive less when selling
    
    def _calculate_commission(self, quantity: int) -> float:
        """Calculate commission for trade."""
        return abs(quantity) * self.commission
    
    def place_order(self, order: SimulatedOrder, current_price: float) -> Dict[str, Any]:
        """
        Place and immediately execute order (market orders only for now).
        
        Args:
            order: Order to execute
            current_price: Current market price
            
        Returns:
            Execution result
        """
        # Check position limits
        if len(self.positions) >= self.max_positions and order.symbol not in self.positions:
            return {
                'status': 'REJECTED',
                'reason': 'Max positions reached',
                'order_id': order.order_id
            }
        
        # Calculate execution price with slippage
        if order.order_type == OrderType.MARKET:
            exec_price = self._calculate_slippage(order.side, current_price)
        else:
            exec_price = current_price  # Simplified for now
        
        # Calculate costs
        commission = self._calculate_commission(order.quantity)
        trade_value = abs(order.quantity) * exec_price * 50  # ES/NQ multiplier
        
        # Check if we have enough capital
        required_margin = trade_value * 0.1  # 10% margin requirement
        if self.cash < required_margin + commission:
            return {
                'status': 'REJECTED',
                'reason': 'Insufficient funds',
                'order_id': order.order_id,
                'required': required_margin + commission,
                'available': self.cash
            }
        
        # Execute trade
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(symbol=order.symbol)
        
        position = self.positions[order.symbol]
        
        # Update position
        trade_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity
        position.update_position(trade_qty, exec_price)
        
        # Update cash
        self.cash -= commission
        self.total_commission += commission
        
        # Calculate slippage cost
        slippage_cost = abs(exec_price - current_price) * abs(order.quantity) * 50
        self.total_slippage += slippage_cost
        
        # Record trade
        trade = {
            'timestamp': time.time(),
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': exec_price,
            'commission': commission,
            'slippage': exec_price - current_price
        }
        self.trades.append(trade)
        self.total_trades += 1
        
        # Update filled orders
        self.filled_orders.append(order)
        
        # Update equity curve
        current_equity = self.get_total_equity(current_price)
        self.equity_curve.append(current_equity)
        
        # Update drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        return {
            'status': 'FILLED',
            'order_id': order.order_id,
            'fill_price': exec_price,
            'commission': commission,
            'timestamp': time.time()
        }
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        return self.positions.copy()
    
    def update_unrealized_pnl(self, current_prices: Dict[str, float]):
        """Update unrealized P&L for all positions."""
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.quantity != 0:
                current_price = current_prices[symbol]
                position.unrealized_pnl = (current_price - position.avg_price) * position.quantity * 50
    
    def get_total_equity(self, current_price: float = None) -> float:
        """Calculate total account equity."""
        total = self.cash
        
        for position in self.positions.values():
            if position.quantity != 0:
                total += position.realized_pnl + position.unrealized_pnl
                
        return total
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio performance metrics."""
        current_equity = self.get_total_equity()
        total_return = (current_equity - self.initial_capital) / self.initial_capital
        
        # Calculate win rate
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        # Calculate Sharpe ratio (simplified)
        if len(self.equity_curve) > 2:
            returns = np.diff(list(self.equity_curve)) / list(self.equity_curve)[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return {
            'total_equity': current_equity,
            'cash': self.cash,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'open_positions': len([p for p in self.positions.values() if p.quantity != 0])
        }
    
    def reset(self):
        """Reset broker to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.filled_orders.clear()
        self.trades.clear()
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_capital
        self.equity_curve.clear()
        self.equity_curve.append(self.initial_capital)
    
    def get_trade_log(self) -> List[Dict]:
        """Get recent trade log."""
        return list(self.trades)
    
    def close_all_positions(self, current_prices: Dict[str, float]) -> List[Dict]:
        """Close all open positions at current prices."""
        results = []
        
        for symbol, position in list(self.positions.items()):
            if position.quantity != 0 and symbol in current_prices:
                # Create closing order
                side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                order = SimulatedOrder(
                    symbol=symbol,
                    side=side,
                    quantity=abs(position.quantity)
                )
                
                # Execute closing order
                result = self.place_order(order, current_prices[symbol])
                results.append(result)
                
        return results


class FastBroker(SimulationBroker):
    """
    Even faster broker with minimal features for rapid testing.
    No slippage, no commission, simplified calculations.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(
            initial_capital=initial_capital,
            commission_per_contract=0,
            slippage_ticks=0
        )
    
    def _calculate_slippage(self, side: OrderSide, price: float) -> float:
        """No slippage in fast mode."""
        return price
    
    def _calculate_commission(self, quantity: int) -> float:
        """No commission in fast mode."""
        return 0.0