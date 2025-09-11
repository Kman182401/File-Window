"""
Order Management System

Handles the complete order lifecycle for paper trading:
- Order validation and risk checks
- Order routing to IBKR Paper Broker
- Order status tracking and updates
- Order timeout and cancellation management
- Performance analytics per order
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue

# Local imports
import sys
sys.path.append('/home/ubuntu')

from ibkr_paper_broker import IBKRPaperBroker, PaperOrder, OrderStatus
from trading_decision_engine import TradingDecision
from monitoring.alerting_system import get_alerting_system, AlertSeverity, AlertType
from utils.audit_logger import audit_log, AuditEventType, AuditSeverity

logger = logging.getLogger(__name__)


class OrderPriority(Enum):
    """Order priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ManagedOrder:
    """Wrapper for orders with additional management data"""
    order_id: str
    symbol: str
    decision: TradingDecision
    priority: OrderPriority = OrderPriority.NORMAL
    
    # Order details
    paper_order: Optional[PaperOrder] = None
    broker_order_id: Optional[str] = None
    
    # Lifecycle tracking
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Performance tracking
    expected_fill_price: float = 0.0
    actual_fill_price: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    error_message: str = ""
    retry_count: int = 0
    
    # Callbacks
    fill_callback: Optional[Callable] = None
    cancel_callback: Optional[Callable] = None


class OrderQueue:
    """Priority queue for orders"""
    
    def __init__(self, max_size: int = 1000):
        self.queue = queue.PriorityQueue(maxsize=max_size)
        self.size = 0
    
    def put(self, order: ManagedOrder):
        """Add order to queue"""
        # Priority queue uses lower numbers for higher priority
        priority_value = 5 - order.priority.value
        self.queue.put((priority_value, self.size, order))
        self.size += 1
    
    def get(self, timeout: float = 1.0) -> Optional[ManagedOrder]:
        """Get next order from queue"""
        try:
            _, _, order = self.queue.get(timeout=timeout)
            return order
        except queue.Empty:
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty"""
        return self.queue.empty()


class OrderManagerStats:
    """Order management statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_orders = 0
        self.filled_orders = 0
        self.cancelled_orders = 0
        self.rejected_orders = 0
        self.avg_fill_time_ms = 0.0
        self.avg_slippage = 0.0
        self.total_commission = 0.0
        self.fill_rate = 0.0
        
        # Performance by symbol
        self.symbol_stats: Dict[str, Dict[str, Any]] = {}


class OrderManagerConfig:
    """Configuration for order manager"""
    
    def __init__(self):
        self.max_order_queue_size = 500
        self.order_timeout_minutes = 60
        self.max_retries = 3
        self.retry_delay_seconds = 5
        self.enable_risk_checks = True
        self.enable_order_throttling = True
        self.max_orders_per_minute = 10
        self.order_processing_threads = 2
        self.status_update_interval = 1  # seconds
        self.cleanup_interval = 300  # 5 minutes


class OrderManagementSystem:
    """
    Complete order management system for paper trading
    """
    
    def __init__(self, broker: IBKRPaperBroker, config: OrderManagerConfig = None):
        self.broker = broker
        self.config = config or OrderManagerConfig()
        self.alerting = get_alerting_system()
        
        # Order tracking
        self.orders: Dict[str, ManagedOrder] = {}  # order_id -> ManagedOrder
        self.order_queue = OrderQueue(self.config.max_order_queue_size)
        self.order_counter = 0
        
        # Statistics
        self.stats = OrderManagerStats()
        
        # Threading
        self.is_running = False
        self.processor_threads: List[threading.Thread] = []
        self.status_thread: Optional[threading.Thread] = None
        
        # Rate limiting
        self.order_timestamps: List[datetime] = []
        
        # Setup broker callbacks
        self.broker.add_order_callback(self._on_order_update)
        
        logger.info("Order Management System initialized")
    
    def start(self):
        """Start the order management system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start order processor threads
        for i in range(self.config.order_processing_threads):
            thread = threading.Thread(
                target=self._order_processor_loop, 
                name=f"OrderProcessor-{i}",
                daemon=True
            )
            thread.start()
            self.processor_threads.append(thread)
        
        # Start status update thread
        self.status_thread = threading.Thread(
            target=self._status_update_loop,
            name="OrderStatusUpdater", 
            daemon=True
        )
        self.status_thread.start()
        
        logger.info("Order Management System started")
    
    def stop(self):
        """Stop the order management system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.processor_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        
        if self.status_thread and self.status_thread.is_alive():
            self.status_thread.join(timeout=5)
        
        logger.info("Order Management System stopped")
    
    def submit_order(self, decision: TradingDecision, 
                    priority: OrderPriority = OrderPriority.NORMAL) -> str:
        """
        Submit a trading decision for order execution
        
        Args:
            decision: TradingDecision from decision engine
            priority: Order priority level
            
        Returns:
            Order ID for tracking
        """
        # Generate order ID
        order_id = f"OMS_{self.order_counter:06d}"
        self.order_counter += 1
        
        # Create managed order
        managed_order = ManagedOrder(
            order_id=order_id,
            symbol=decision.symbol,
            decision=decision,
            priority=priority,
            expected_fill_price=decision.price
        )
        
        # Store order
        self.orders[order_id] = managed_order
        
        # Add to processing queue
        self.order_queue.put(managed_order)
        
        logger.info(f"Order submitted: {order_id} - {decision.action} {decision.symbol} (priority: {priority.name})")
        
        # Update stats
        self.stats.total_orders += 1
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.error(f"Order not found: {order_id}")
            return False
        
        managed_order = self.orders[order_id]
        
        # Can only cancel if broker order exists and not already filled
        if (managed_order.broker_order_id and 
            managed_order.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED]):
            
            success = self.broker.cancel_order(managed_order.broker_order_id)
            
            if success:
                managed_order.status = OrderStatus.CANCELLED
                managed_order.cancelled_at = datetime.now()
                
                logger.info(f"Order cancelled: {order_id}")
                
                # Update stats
                self.stats.cancelled_orders += 1
                
                # Trigger callback
                if managed_order.cancel_callback:
                    try:
                        managed_order.cancel_callback(managed_order)
                    except Exception as e:
                        logger.error(f"Error in cancel callback: {e}")
                
                return True
        
        return False
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        if order_id not in self.orders:
            return None
        
        return self.orders[order_id].status
    
    def get_order(self, order_id: str) -> Optional[ManagedOrder]:
        """Get order details"""
        return self.orders.get(order_id)
    
    def get_open_orders(self) -> List[ManagedOrder]:
        """Get all open orders"""
        return [order for order in self.orders.values() 
                if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PRESUBMITTED]]
    
    def get_filled_orders(self) -> List[ManagedOrder]:
        """Get all filled orders"""
        return [order for order in self.orders.values() if order.status == OrderStatus.FILLED]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get order management statistics"""
        # Calculate current stats
        filled_orders = self.get_filled_orders()
        
        if filled_orders:
            self.stats.filled_orders = len(filled_orders)
            self.stats.fill_rate = len(filled_orders) / max(self.stats.total_orders, 1) * 100
            
            # Calculate averages
            fill_times = [
                (order.filled_at - order.submitted_at).total_seconds() * 1000
                for order in filled_orders 
                if order.filled_at and order.submitted_at
            ]
            
            if fill_times:
                self.stats.avg_fill_time_ms = sum(fill_times) / len(fill_times)
            
            slippages = [order.slippage for order in filled_orders if order.slippage > 0]
            if slippages:
                self.stats.avg_slippage = sum(slippages) / len(slippages)
            
            # Total commission
            commissions = [order.paper_order.commission for order in filled_orders 
                          if order.paper_order and order.paper_order.commission > 0]
            if commissions:
                self.stats.total_commission = sum(commissions)
        
        return {
            'total_orders': self.stats.total_orders,
            'filled_orders': self.stats.filled_orders,
            'cancelled_orders': self.stats.cancelled_orders,
            'rejected_orders': self.stats.rejected_orders,
            'fill_rate_pct': self.stats.fill_rate,
            'avg_fill_time_ms': self.stats.avg_fill_time_ms,
            'avg_slippage_pct': self.stats.avg_slippage * 100,
            'total_commission': self.stats.total_commission,
            'open_orders': len(self.get_open_orders()),
            'queue_size': not self.order_queue.empty()
        }
    
    def _order_processor_loop(self):
        """Main order processing loop"""
        while self.is_running:
            try:
                # Get next order from queue
                order = self.order_queue.get(timeout=1.0)
                if not order:
                    continue
                
                # Process the order
                self._process_order(order)
            
            except Exception as e:
                logger.error(f"Error in order processor: {e}")
                time.sleep(1)
    
    def _process_order(self, order: ManagedOrder):
        """Process a single order"""
        try:
            # Rate limiting check
            if self.config.enable_order_throttling:
                if not self._check_rate_limit():
                    logger.warning(f"Rate limit exceeded, requeueing order: {order.order_id}")
                    time.sleep(1)
                    self.order_queue.put(order)
                    return
            
            # Risk checks
            if self.config.enable_risk_checks:
                if not self._validate_order(order):
                    order.status = OrderStatus.REJECTED
                    order.error_message = "Failed risk validation"
                    self.stats.rejected_orders += 1
                    return
            
            # Submit to broker
            decision = order.decision
            
            if decision.action == 0:  # Hold - no order needed
                order.status = OrderStatus.CANCELLED
                order.error_message = "Hold decision - no order placed"
                return
            
            # Determine order quantity and action
            quantity = decision.position_size if decision.action == 1 else -decision.position_size
            
            # Place market order (default for now)
            broker_order_id = self.broker.place_market_order(
                symbol=decision.symbol,
                quantity=quantity,
                confidence=decision.confidence,
                reasoning=decision.reasoning
            )
            
            if broker_order_id:
                order.broker_order_id = broker_order_id
                order.submitted_at = datetime.now()
                order.status = OrderStatus.SUBMITTED
                
                # Record rate limiting
                self.order_timestamps.append(datetime.now())
                
                logger.info(f"Order submitted to broker: {order.order_id} -> {broker_order_id}")
            
            else:
                order.status = OrderStatus.REJECTED
                order.error_message = "Broker rejected order"
                self.stats.rejected_orders += 1
                
                # Retry logic
                if order.retry_count < self.config.max_retries:
                    order.retry_count += 1
                    logger.info(f"Retrying order {order.order_id} (attempt {order.retry_count})")
                    time.sleep(self.config.retry_delay_seconds)
                    self.order_queue.put(order)
        
        except Exception as e:
            logger.error(f"Error processing order {order.order_id}: {e}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            self.stats.rejected_orders += 1
    
    def _status_update_loop(self):
        """Status update and cleanup loop"""
        while self.is_running:
            try:
                # Check for order timeouts
                self._check_order_timeouts()
                
                # Cleanup old rate limiting records
                self._cleanup_rate_limiting()
                
                # Cleanup old orders (optional)
                self._cleanup_old_orders()
                
                time.sleep(self.config.status_update_interval)
            
            except Exception as e:
                logger.error(f"Error in status update loop: {e}")
                time.sleep(5)
    
    def _check_rate_limit(self) -> bool:
        """Check if order submission is within rate limits"""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Count recent orders
        recent_orders = sum(1 for ts in self.order_timestamps if ts > one_minute_ago)
        
        return recent_orders < self.config.max_orders_per_minute
    
    def _cleanup_rate_limiting(self):
        """Clean up old rate limiting records"""
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        self.order_timestamps = [ts for ts in self.order_timestamps if ts > one_minute_ago]
    
    def _validate_order(self, order: ManagedOrder) -> bool:
        """Validate order before submission"""
        decision = order.decision
        
        # Check decision confidence
        if decision.confidence < 0.1:
            logger.warning(f"Order {order.order_id} has very low confidence: {decision.confidence}")
            return False
        
        # Check position size
        if decision.position_size <= 0:
            logger.warning(f"Order {order.order_id} has invalid position size: {decision.position_size}")
            return False
        
        # Check for valid action
        if decision.action not in [0, 1, 2]:
            logger.warning(f"Order {order.order_id} has invalid action: {decision.action}")
            return False
        
        return True
    
    def _check_order_timeouts(self):
        """Check for and cancel timed out orders"""
        timeout_cutoff = datetime.now() - timedelta(minutes=self.config.order_timeout_minutes)
        
        for order in self.get_open_orders():
            if order.submitted_at and order.submitted_at < timeout_cutoff:
                logger.info(f"Cancelling timed out order: {order.order_id}")
                self.cancel_order(order.order_id)
    
    def _cleanup_old_orders(self):
        """Clean up very old completed orders to save memory"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours
        
        old_order_ids = []
        for order_id, order in self.orders.items():
            if (order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED] and
                order.created_at < cutoff_time):
                old_order_ids.append(order_id)
        
        # Remove old orders (keep some for statistics)
        if len(old_order_ids) > 1000:  # Only cleanup if we have many orders
            for order_id in old_order_ids[:-100]:  # Keep last 100
                del self.orders[order_id]
            
            logger.info(f"Cleaned up {len(old_order_ids) - 100} old orders")
    
    def _on_order_update(self, paper_order: PaperOrder):
        """Handle order updates from broker"""
        # Find our managed order
        managed_order = None
        for order in self.orders.values():
            if order.broker_order_id == paper_order.order_id:
                managed_order = order
                break
        
        if not managed_order:
            logger.warning(f"Received update for unknown order: {paper_order.order_id}")
            return
        
        # Update managed order
        managed_order.paper_order = paper_order
        managed_order.status = paper_order.status
        
        # Handle fill
        if paper_order.status == OrderStatus.FILLED and not managed_order.filled_at:
            managed_order.filled_at = datetime.now()
            managed_order.actual_fill_price = paper_order.avg_fill_price
            
            # Calculate slippage
            if managed_order.expected_fill_price > 0:
                managed_order.slippage = abs(
                    managed_order.actual_fill_price - managed_order.expected_fill_price
                ) / managed_order.expected_fill_price
            
            # Calculate latency
            if managed_order.submitted_at:
                latency = (managed_order.filled_at - managed_order.submitted_at).total_seconds() * 1000
                managed_order.latency_ms = latency
            
            logger.info(f"Order filled: {managed_order.order_id} - "
                       f"Price: ${paper_order.avg_fill_price}, "
                       f"Slippage: {managed_order.slippage:.4f}, "
                       f"Latency: {managed_order.latency_ms:.1f}ms")
            
            # Trigger fill callback
            if managed_order.fill_callback:
                try:
                    managed_order.fill_callback(managed_order)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")
            
            # Send alert for significant trades
            if managed_order.decision.confidence > 0.7:
                self.alerting.send_alert(self.alerting.create_alert(
                    severity=AlertSeverity.INFO,
                    alert_type=AlertType.UNUSUAL_ACTIVITY,
                    title="High-Confidence Order Filled",
                    message=f"Order {managed_order.order_id} filled: {paper_order.action} {paper_order.quantity} {paper_order.symbol}",
                    details={
                        'order_id': managed_order.order_id,
                        'symbol': paper_order.symbol,
                        'quantity': paper_order.quantity,
                        'fill_price': paper_order.avg_fill_price,
                        'confidence': managed_order.decision.confidence,
                        'slippage': managed_order.slippage
                    }
                ))


def test_order_management():
    """Test the order management system"""
    from ibkr_paper_broker import IBKRPaperBroker
    from trading_decision_engine import TradingDecision
    
    # Create components
    broker = IBKRPaperBroker()
    oms = OrderManagementSystem(broker)
    
    # Start systems
    if not broker.connect():
        print("Failed to connect broker")
        return
    
    oms.start()
    
    # Create test decision
    decision = TradingDecision(
        timestamp=datetime.now(),
        symbol="ES1!",
        action=1,  # Buy
        confidence=0.8,
        position_size=0.1,
        price=4500.0,
        reasoning="Test order"
    )
    
    # Submit order
    order_id = oms.submit_order(decision, OrderPriority.HIGH)
    print(f"Order submitted: {order_id}")
    
    # Wait and check status
    time.sleep(15)
    
    order = oms.get_order(order_id)
    if order:
        print(f"Order status: {order.status}")
        print(f"Broker order ID: {order.broker_order_id}")
    
    # Get statistics
    stats = oms.get_statistics()
    print(f"Statistics: {stats}")
    
    # Stop systems
    oms.stop()
    broker.disconnect()
    
    print("Test completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_order_management()