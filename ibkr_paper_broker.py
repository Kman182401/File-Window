"""
IBKR Paper Broker Interface

Real paper trading execution through IB Gateway
- Places actual paper trades through IBKR's paper trading account
- Handles real order execution, fills, and rejections
- Maintains real portfolio positions and balances
- Provides authentic trading experience for RL learning
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# IB imports
from ib_insync import Trade, Position, PortfolioItem

# Local imports
import sys


from market_data_ibkr_adapter import IBKRIngestor
from monitoring.alerting_system import get_alerting_system, AlertSeverity, AlertType
from utils.audit_logger import audit_log, AuditEventType, AuditSeverity
from configs.market_data_config import MAX_POSITION_EXPOSURE, MAX_ORDER_SIZE, MAX_DAILY_LOSS_PCT

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration matching IB statuses"""
    PENDING = "Pending"
    SUBMITTED = "Submitted"
    PRESUBMITTED = "PreSubmitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"
    INACTIVE = "Inactive"


@dataclass
class PaperOrder:
    """Represents a paper trading order"""
    order_id: str
    symbol: str
    action: str  # "BUY" or "SELL"
    quantity: float
    order_type: str  # "MARKET" or "LIMIT"
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    
    # Execution details
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    remaining_quantity: float = 0.0
    commission: float = 0.0
    
    # Timestamps
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    
    # IB Trade object
    ib_trade: Optional[Trade] = None
    
    # Metadata
    confidence: float = 0.0
    reasoning: str = ""
    
    def __post_init__(self):
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity


@dataclass
class PaperPosition:
    """Represents a paper trading position"""
    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    market_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # IB Position object
    ib_position: Optional[Position] = None


class IBKRPaperBroker:
    """
    Paper trading broker using real IBKR paper trading account
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        
        # Core components
        self.ibkr = IBKRIngestor()
        self.alerting = get_alerting_system()
        
        # Trading state
        self.orders: Dict[str, PaperOrder] = {}  # order_id -> PaperOrder
        self.positions: Dict[str, PaperPosition] = {}  # symbol -> PaperPosition
        self.account_balance: float = 0.0
        self.available_funds: float = 0.0
        
        # Order tracking
        self.order_counter = 0
        self.is_connected = False
        
        # Callbacks
        self.order_callbacks: List[Callable] = []
        self.position_callbacks: List[Callable] = []
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread = None
        
        logger.info("IBKR Paper Broker initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'max_position_size': MAX_POSITION_EXPOSURE,
            'max_order_size': MAX_ORDER_SIZE,
            'max_daily_loss_pct': MAX_DAILY_LOSS_PCT,
            'order_timeout_minutes': 60,  # Cancel orders after 1 hour
            'position_update_interval': 5,  # Update positions every 5 seconds
            'account_update_interval': 30,  # Update account every 30 seconds
            'enable_risk_checks': True,
            'paper_account_only': True  # Safety: only allow paper accounts
        }
    
    def connect(self) -> bool:
        """Connect to IBKR Gateway"""
        try:
            # Connection is handled by IBKRIngestor
            # Just verify we can get account info (indicates paper account)
            if self.config['paper_account_only']:
                account_summary = self.ibkr.get_account_summary()
                if not account_summary:
                    logger.error("Failed to get account summary - cannot verify paper account")
                    return False
                
                # Check if this is a paper account (usually has "DU" prefix)
                account_id = None
                for tag, info in account_summary.items():
                    if tag == "AccountCode":
                        account_id = info.get('value', '')
                        break
                
                if account_id and not account_id.startswith('DU'):
                    logger.error(f"SAFETY CHECK: Account {account_id} does not appear to be a paper account!")
                    self.alerting.send_alert(self.alerting.create_alert(
                        severity=AlertSeverity.CRITICAL,
                        alert_type=AlertType.SECURITY,
                        title="NON-PAPER ACCOUNT DETECTED",
                        message=f"Account {account_id} may be a live account - trading blocked for safety",
                        details={'account_id': account_id}
                    ))
                    return False
                
                logger.info(f"Connected to paper account: {account_id}")
            
            self.is_connected = True
            
            # Update account info
            self._update_account_info()
            
            # Update positions
            self._update_positions()
            
            # Start monitoring
            self._start_monitoring()
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from IBKR Gateway"""
        self.is_connected = False
        
        # Stop monitoring
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=10)
        
        logger.info("Disconnected from IBKR Paper Broker")
    
    def place_market_order(self, symbol: str, quantity: float, 
                          confidence: float = 0.0, reasoning: str = "") -> Optional[str]:
        """
        Place a market order
        
        Args:
            symbol: Trading symbol (e.g., "ES1!")
            quantity: Order quantity (positive for BUY, negative for SELL)
            confidence: Decision confidence (0.0-1.0)
            reasoning: Decision reasoning text
            
        Returns:
            Order ID if successful, None if failed
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        # Determine action and quantity
        action = "BUY" if quantity > 0 else "SELL"
        abs_quantity = abs(quantity)
        
        # Risk checks
        if self.config['enable_risk_checks']:
            if not self._check_order_risk(symbol, abs_quantity, action):
                return None
        
        # Create paper order
        order_id = f"PAPER_{self.order_counter:06d}"
        self.order_counter += 1
        
        paper_order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=abs_quantity,
            order_type="MARKET",
            submitted_time=datetime.now(),
            confidence=confidence,
            reasoning=reasoning
        )
        
        try:
            # Place order with IBKR
            ib_trade = self.ibkr.place_market_order(symbol, abs_quantity, action)
            
            if ib_trade:
                paper_order.ib_trade = ib_trade
                paper_order.status = OrderStatus.SUBMITTED
                
                # Store order
                self.orders[order_id] = paper_order
                
                logger.info(f"Market order placed: {order_id} - {action} {abs_quantity} {symbol}")
                
                # Audit log
                audit_log(
                    AuditEventType.SYSTEM_START,
                    AuditSeverity.INFO,
                    "IBKRPaperBroker",
                    f"Market order placed: {order_id}",
                    symbol=symbol,
                    action=action,
                    quantity=abs_quantity,
                    order_type="MARKET",
                    confidence=confidence
                )
                
                # Trigger callbacks
                self._notify_order_callbacks(paper_order)
                
                return order_id
            
            else:
                logger.error(f"Failed to place market order: {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def place_limit_order(self, symbol: str, quantity: float, price: float,
                         confidence: float = 0.0, reasoning: str = "") -> Optional[str]:
        """
        Place a limit order
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity (positive for BUY, negative for SELL)
            price: Limit price
            confidence: Decision confidence
            reasoning: Decision reasoning
            
        Returns:
            Order ID if successful, None if failed
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None
        
        # Determine action and quantity
        action = "BUY" if quantity > 0 else "SELL"
        abs_quantity = abs(quantity)
        
        # Risk checks
        if self.config['enable_risk_checks']:
            if not self._check_order_risk(symbol, abs_quantity, action):
                return None
        
        # Create paper order
        order_id = f"PAPER_{self.order_counter:06d}"
        self.order_counter += 1
        
        paper_order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            action=action,
            quantity=abs_quantity,
            order_type="LIMIT",
            limit_price=price,
            submitted_time=datetime.now(),
            confidence=confidence,
            reasoning=reasoning
        )
        
        try:
            # Place order with IBKR
            ib_trade = self.ibkr.place_limit_order(symbol, abs_quantity, price, action)
            
            if ib_trade:
                paper_order.ib_trade = ib_trade
                paper_order.status = OrderStatus.SUBMITTED
                
                # Store order
                self.orders[order_id] = paper_order
                
                logger.info(f"Limit order placed: {order_id} - {action} {abs_quantity} {symbol} @ ${price}")
                
                # Audit log
                audit_log(
                    AuditEventType.SYSTEM_START,
                    AuditSeverity.INFO,
                    "IBKRPaperBroker",
                    f"Limit order placed: {order_id}",
                    symbol=symbol,
                    action=action,
                    quantity=abs_quantity,
                    price=price,
                    order_type="LIMIT",
                    confidence=confidence
                )
                
                # Trigger callbacks
                self._notify_order_callbacks(paper_order)
                
                return order_id
            
            else:
                logger.error(f"Failed to place limit order: {symbol}")
                return None
        
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        paper_order = self.orders[order_id]
        
        if not paper_order.ib_trade:
            logger.error(f"No IB trade found for order: {order_id}")
            return False
        
        try:
            success = self.ibkr.cancel_order(paper_order.ib_trade)
            
            if success:
                paper_order.status = OrderStatus.CANCELLED
                logger.info(f"Order cancelled: {order_id}")
                
                # Trigger callbacks
                self._notify_order_callbacks(paper_order)
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        if order_id not in self.orders:
            return None
        
        paper_order = self.orders[order_id]
        
        if paper_order.ib_trade:
            # Update from IB
            ib_status = self.ibkr.get_order_status(paper_order.ib_trade)
            paper_order.status = OrderStatus(ib_status)
        
        return paper_order.status
    
    def get_order(self, order_id: str) -> Optional[PaperOrder]:
        """Get order details"""
        return self.orders.get(order_id)
    
    def get_all_orders(self) -> List[PaperOrder]:
        """Get all orders"""
        return list(self.orders.values())
    
    def get_open_orders(self) -> List[PaperOrder]:
        """Get open orders"""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PRESUBMITTED]]
    
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[PaperPosition]:
        """Get all positions"""
        return [pos for pos in self.positions.values() if abs(pos.quantity) > 0.001]
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        return {
            'account_balance': self.account_balance,
            'available_funds': self.available_funds,
            'positions': len(self.get_all_positions()),
            'open_orders': len(self.get_open_orders()),
            'total_orders': len(self.orders),
            'is_connected': self.is_connected
        }
    
    def add_order_callback(self, callback: Callable[[PaperOrder], None]):
        """Add callback for order updates"""
        self.order_callbacks.append(callback)
    
    def add_position_callback(self, callback: Callable[[PaperPosition], None]):
        """Add callback for position updates"""
        self.position_callbacks.append(callback)
    
    def _check_order_risk(self, symbol: str, quantity: float, action: str) -> bool:
        """Check order against risk limits"""
        # Check order size limit
        if quantity > self.config['max_order_size']:
            logger.warning(f"Order size {quantity} exceeds limit {self.config['max_order_size']}")
            return False
        
        # Check position limit
        current_position = self.positions.get(symbol, PaperPosition(symbol)).quantity
        
        if action == "BUY":
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity
        
        if abs(new_position) > self.config['max_position_size']:
            logger.warning(f"New position {new_position} would exceed limit {self.config['max_position_size']}")
            return False
        
        # Check available funds (simplified)
        if self.available_funds < 1000:  # Need minimum funds
            logger.warning("Insufficient available funds for trading")
            return False
        
        return True
    
    def _update_account_info(self):
        """Update account balance and available funds"""
        try:
            account_summary = self.ibkr.get_account_summary()
            
            for tag, info in account_summary.items():
                if tag == "NetLiquidationByCurrency":
                    self.account_balance = float(info.get('value', 0))
                elif tag == "AvailableFunds":
                    self.available_funds = float(info.get('value', 0))
        
        except Exception as e:
            logger.error(f"Failed to update account info: {e}")
    
    def _update_positions(self):
        """Update positions from IBKR"""
        try:
            ib_positions = self.ibkr.get_positions()
            
            # Update existing positions
            for ib_pos in ib_positions:
                # Map IB contract to our symbol
                symbol = self._map_contract_to_symbol(ib_pos.contract)
                if symbol:
                    paper_pos = self.positions.get(symbol, PaperPosition(symbol))
                    
                    paper_pos.quantity = ib_pos.position
                    paper_pos.avg_cost = ib_pos.avgCost
                    paper_pos.market_price = ib_pos.marketPrice
                    paper_pos.market_value = ib_pos.marketValue
                    paper_pos.unrealized_pnl = ib_pos.unrealizedPNL
                    paper_pos.ib_position = ib_pos
                    
                    self.positions[symbol] = paper_pos
                    
                    # Trigger callbacks
                    self._notify_position_callbacks(paper_pos)
        
        except Exception as e:
            logger.error(f"Failed to update positions: {e}")
    
    def _update_orders(self):
        """Update order statuses from IBKR"""
        try:
            # Check all pending orders
            for order in self.get_open_orders():
                if order.ib_trade:
                    ib_status = self.ibkr.get_order_status(order.ib_trade)
                    old_status = order.status
                    order.status = OrderStatus(ib_status)
                    
                    # Update fill information if filled
                    if order.status == OrderStatus.FILLED and old_status != OrderStatus.FILLED:
                        order.filled_time = datetime.now()
                        order.filled_quantity = order.quantity
                        order.remaining_quantity = 0.0
                        
                        # Get fill details from trade
                        if hasattr(order.ib_trade, 'fills') and order.ib_trade.fills:
                            fill = order.ib_trade.fills[-1]  # Get latest fill
                            order.avg_fill_price = fill.execution.price
                            order.commission = fill.commissionReport.commission if fill.commissionReport else 0.0
                        
                        logger.info(f"Order filled: {order.order_id} - {order.action} {order.filled_quantity} {order.symbol} @ ${order.avg_fill_price}")
                        
                        # Trigger callbacks
                        self._notify_order_callbacks(order)
                    
                    # Cancel old orders
                    if order.submitted_time and datetime.now() - order.submitted_time > timedelta(minutes=self.config['order_timeout_minutes']):
                        logger.info(f"Cancelling old order: {order.order_id}")
                        self.cancel_order(order.order_id)
        
        except Exception as e:
            logger.error(f"Failed to update orders: {e}")
    
    def _start_monitoring(self):
        """Start background monitoring of orders and positions"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Started IBKR paper broker monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        last_position_update = time.time()
        last_account_update = time.time()
        
        while self.monitoring_active and self.is_connected:
            try:
                current_time = time.time()
                
                # Update orders frequently
                self._update_orders()
                
                # Update positions periodically
                if current_time - last_position_update >= self.config['position_update_interval']:
                    self._update_positions()
                    last_position_update = current_time
                
                # Update account periodically
                if current_time - last_account_update >= self.config['account_update_interval']:
                    self._update_account_info()
                    last_account_update = current_time
                
                time.sleep(1)  # Check every second
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _map_contract_to_symbol(self, contract) -> Optional[str]:
        """Map IB contract to our symbol format"""
        try:
            if contract.secType == "FUT":
                if contract.symbol == "ES" and contract.exchange == "CME":
                    return "ES1!"
                elif contract.symbol == "NQ" and contract.exchange == "CME":
                    return "NQ1!"
                elif contract.symbol == "6B" and contract.exchange == "CME":
                    return "GBPUSD"
                elif contract.symbol == "6E" and contract.exchange == "CME":
                    return "EURUSD"
                elif contract.symbol == "6A" and contract.exchange == "CME":
                    return "AUDUSD"
                elif contract.symbol == "GC" and contract.exchange == "COMEX":
                    return "XAUUSD"
            
            return None
        
        except Exception as e:
            logger.error(f"Error mapping contract to symbol: {e}")
            return None
    
    def _notify_order_callbacks(self, order: PaperOrder):
        """Notify order callbacks"""
        for callback in self.order_callbacks:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")
    
    def _notify_position_callbacks(self, position: PaperPosition):
        """Notify position callbacks"""
        for callback in self.position_callbacks:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in position callback: {e}")


def test_paper_broker():
    """Test the paper broker"""
    broker = IBKRPaperBroker()
    
    # Connect
    if not broker.connect():
        print("Failed to connect to IBKR")
        return
    
    print("Connected to IBKR Paper Broker")
    
    # Get account info
    account_info = broker.get_account_info()
    print(f"Account info: {account_info}")
    
    # Test small market order
    order_id = broker.place_market_order("ES1!", 1, confidence=0.8, reasoning="Test order")
    if order_id:
        print(f"Order placed: {order_id}")
        
        # Wait for fill
        time.sleep(10)
        
        order = broker.get_order(order_id)
        if order:
            print(f"Order status: {order.status}")
            print(f"Filled quantity: {order.filled_quantity}")
            print(f"Fill price: {order.avg_fill_price}")
    
    # Get positions
    positions = broker.get_all_positions()
    print(f"Positions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.quantity} @ {pos.avg_cost}")
    
    # Disconnect
    broker.disconnect()
    print("Test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_paper_broker()