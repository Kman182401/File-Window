#!/usr/bin/env python3
"""
Order Safety Wrapper
Centralized safety checks for all order placements in the trading system.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from collections import deque
import threading

logger = logging.getLogger(__name__)

# Configuration from environment
MAX_DAILY_LOSS_PCT = float(os.getenv('MAX_DAILY_LOSS_PCT', '0.02'))
MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '20'))
MAX_POSITION_EXPOSURE = int(os.getenv('MAX_POSITION_EXPOSURE', '3'))
MAX_ORDER_SIZE = int(os.getenv('MAX_ORDER_SIZE', '2'))
ERROR_THRESHOLD = int(os.getenv('ERROR_THRESHOLD', '3'))
ERROR_WINDOW_MINUTES = int(os.getenv('ERROR_WINDOW_MINUTES', '5'))
DEDUP_WINDOW_SECONDS = int(os.getenv('DEDUP_WINDOW_SECONDS', '5'))

# Allowed symbols
SYMBOL_ALLOWLIST = {'ES', 'NQ', 'MES', 'MNQ', '6E', '6B', '6A', 'GC', 'MGC'}

class OrderSafetyManager:
    """Manages safety checks and circuit breakers for order placement"""
    
    def __init__(self):
        self.daily_pnl = 0.0
        self.starting_balance = 100000.0  # Should be set from account
        self.trade_count_today = 0
        self.current_positions = {}  # symbol -> quantity
        self.error_timestamps = deque(maxlen=ERROR_THRESHOLD)
        self.recent_orders = {}  # (symbol, side) -> timestamp
        self.circuit_breaker_active = False
        self.lock = threading.RLock()
        self.session_start = datetime.now()
        
    def reset_daily_counters(self):
        """Reset counters at start of trading day"""
        with self.lock:
            self.daily_pnl = 0.0
            self.trade_count_today = 0
            self.error_timestamps.clear()
            self.recent_orders.clear()
            self.circuit_breaker_active = False
            logger.info("Daily safety counters reset")
    
    def update_pnl(self, pnl_change: float):
        """Update daily P&L"""
        with self.lock:
            self.daily_pnl += pnl_change
            logger.info(f"Daily P&L updated: ${self.daily_pnl:.2f}")
    
    def record_error(self):
        """Record an order error for circuit breaker"""
        with self.lock:
            self.error_timestamps.append(time.time())
            
            # Check if circuit breaker should activate
            if len(self.error_timestamps) >= ERROR_THRESHOLD:
                window_start = time.time() - (ERROR_WINDOW_MINUTES * 60)
                recent_errors = sum(1 for ts in self.error_timestamps if ts > window_start)
                
                if recent_errors >= ERROR_THRESHOLD:
                    self.circuit_breaker_active = True
                    logger.error(f"Circuit breaker ACTIVATED: {recent_errors} errors in {ERROR_WINDOW_MINUTES} minutes")
                    return True
            return False
    
    def check_symbol_allowlist(self, symbol: str) -> bool:
        """Check if symbol is allowed for trading"""
        # Extract root symbol (e.g., ES from ES1!)
        root = symbol.rstrip('1!').upper()
        allowed = root in SYMBOL_ALLOWLIST
        if not allowed:
            logger.warning(f"Symbol {symbol} (root: {root}) not in allowlist")
        return allowed
    
    def check_position_limits(self, symbol: str, quantity: int, side: str) -> bool:
        """Check if order would exceed position limits"""
        with self.lock:
            current_pos = self.current_positions.get(symbol, 0)
            
            # Calculate new position after order
            if side == 'BUY':
                new_pos = current_pos + quantity
            else:  # SELL
                new_pos = current_pos - quantity
            
            # Check absolute position limit
            if abs(new_pos) > MAX_POSITION_EXPOSURE:
                logger.warning(f"Position limit exceeded: {symbol} current={current_pos} new={new_pos} limit={MAX_POSITION_EXPOSURE}")
                return False
            
            # Check order size limit
            if quantity > MAX_ORDER_SIZE:
                logger.warning(f"Order size limit exceeded: {quantity} > {MAX_ORDER_SIZE}")
                return False
                
            return True
    
    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        with self.lock:
            loss_pct = abs(self.daily_pnl / self.starting_balance) if self.starting_balance > 0 else 0
            
            if self.daily_pnl < 0 and loss_pct >= MAX_DAILY_LOSS_PCT:
                logger.error(f"Daily loss limit reached: {loss_pct:.2%} >= {MAX_DAILY_LOSS_PCT:.2%}")
                return False
            return True
    
    def check_trade_count_limit(self) -> bool:
        """Check if daily trade count limit has been reached"""
        with self.lock:
            if self.trade_count_today >= MAX_TRADES_PER_DAY:
                logger.warning(f"Daily trade count limit reached: {self.trade_count_today} >= {MAX_TRADES_PER_DAY}")
                return False
            return True
    
    def check_market_hours(self, symbol: str) -> bool:
        """Check if current time is within trading hours for symbol
        
        Note: Uses server local time. Ensure server is in US/Eastern or adjust accordingly.
        Set ALLOW_OUTSIDE_RTH=1 to trade outside regular hours.
        """
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        
        # Basic RTH check for futures (can be enhanced with symbol-specific hours)
        if symbol.startswith(('ES', 'NQ', 'MES', 'MNQ')):
            # E-mini futures RTH: 9:30 AM - 4:00 PM ET (cash session)
            # Note: Globex trades nearly 24h - this is just for RTH enforcement
            in_rth = (hour == 9 and minute >= 30) or (10 <= hour < 16)
            
            if not in_rth:
                if not os.getenv('ALLOW_OUTSIDE_RTH', '0') == '1':
                    logger.warning(f"Outside RTH for {symbol}: {hour:02d}:{minute:02d} (RTH: 9:30-16:00 ET)")
                    return False
        elif symbol.startswith(('6E', '6B', '6A')):  # FX futures
            # FX futures trade Sunday 5pm - Friday 4pm CT with brief daily break
            # For simplicity, block only during weekend
            weekday = now.weekday()
            if weekday >= 5:  # Saturday or Sunday
                if not os.getenv('ALLOW_OUTSIDE_RTH', '0') == '1':
                    logger.warning(f"Weekend - markets closed for {symbol}")
                    return False
        
        return True
    
    def check_duplicate_order(self, symbol: str, side: str) -> bool:
        """Check for duplicate orders within dedup window"""
        with self.lock:
            key = (symbol, side)
            now = time.time()
            
            if key in self.recent_orders:
                last_time = self.recent_orders[key]
                if now - last_time < DEDUP_WINDOW_SECONDS:
                    logger.warning(f"Duplicate order detected: {symbol} {side} within {DEDUP_WINDOW_SECONDS}s")
                    return False
            
            # Record this order
            self.recent_orders[key] = now
            return True
    
    def check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is active"""
        with self.lock:
            if self.circuit_breaker_active:
                logger.error("Circuit breaker is ACTIVE - orders blocked")
                return False
            return True
    
    def pre_order_checks(self, symbol: str, side: str, quantity: int) -> Tuple[bool, str]:
        """
        Run all pre-order safety checks
        
        Returns:
            (passed, reason) - True if all checks pass, False and reason if any fail
        """
        checks = [
            (self.check_circuit_breaker(), "circuit_breaker_active"),
            (self.check_symbol_allowlist(symbol), "symbol_not_allowed"),
            (self.check_daily_loss_limit(), "daily_loss_limit_exceeded"),
            (self.check_trade_count_limit(), "trade_count_limit_exceeded"),
            (self.check_position_limits(symbol, quantity, side), "position_limit_exceeded"),
            (self.check_market_hours(symbol), "outside_market_hours"),
            (self.check_duplicate_order(symbol, side), "duplicate_order"),
        ]
        
        for passed, reason in checks:
            if not passed:
                logger.error(f"Pre-order check failed: {reason} for {symbol} {side} qty={quantity}")
                return False, reason
        
        logger.info(f"Pre-order checks PASSED for {symbol} {side} qty={quantity}")
        return True, "checks_passed"
    
    def record_order_placed(self, symbol: str, side: str, quantity: int):
        """Record that an order was successfully placed"""
        with self.lock:
            self.trade_count_today += 1
            
            # Update position tracking
            if side == 'BUY':
                self.current_positions[symbol] = self.current_positions.get(symbol, 0) + quantity
            else:  # SELL
                self.current_positions[symbol] = self.current_positions.get(symbol, 0) - quantity
            
            logger.info(f"Order recorded: {symbol} {side} {quantity} | Daily trades: {self.trade_count_today}")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        with self.lock:
            self.circuit_breaker_active = False
            self.error_timestamps.clear()
            logger.info("Circuit breaker manually reset")


# Global instance
safety_manager = OrderSafetyManager()


def safe_place_order(ib, contract, order, symbol: str, side: str, quantity: int, context: str = None):
    """
    Safe wrapper for order placement with comprehensive safety checks
    
    Args:
        ib: IB connection instance
        contract: IB contract object
        order: IB order object
        symbol: Symbol being traded
        side: 'BUY' or 'SELL'
        quantity: Order quantity
        context: Optional context ('parent', 'child_tp', 'child_sl') for bracket orders
        
    Returns:
        Trade object if successful, None if blocked by safety checks
    """
    # Short-circuit if gate disabled
    if os.getenv("ORDER_GATE_ENABLED", "1") != "1":
        return ib.placeOrder(contract, order)  # nosem: safe_place_order internal bypass
    
    # Skip duplicate/RTH checks for bracket children (they don't add exposure)
    skip_dup = context in {"child_tp", "child_sl"}
    skip_rth = context in {"child_tp", "child_sl"}
    
    # Run pre-order checks (skip some checks for children)
    if skip_dup or skip_rth:
        # For children, only check basic safety (not duplicate/RTH)
        passed = True
        reason = "child_order_allowed"
    else:
        passed, reason = safety_manager.pre_order_checks(symbol, side, quantity)
    
    if not passed:
        logger.warning(f"[order_blocked] symbol={symbol} side={side} qty={quantity} reason={reason}")
        return None
    
    try:
        # Place the order
        trade = ib.placeOrder(contract, order)  # nosem: safe_place_order internal placement
        
        # Record successful placement (only for parent orders that add exposure)
        if context != "child_tp" and context != "child_sl":
            safety_manager.record_order_placed(symbol, side, quantity)
        
        logger.info(f"[order_ok] symbol={symbol} side={side} qty={quantity} context={context or 'parent'}")
        return trade
        
    except Exception as e:
        logger.error(f"[SAFETY] Order placement failed: {e}")
        safety_manager.record_error()
        raise


def initialize_safety_manager(starting_balance: float = None):
    """Initialize safety manager with account details"""
    if starting_balance:
        safety_manager.starting_balance = starting_balance
    safety_manager.reset_daily_counters()
    logger.info(f"Safety manager initialized with balance: ${safety_manager.starting_balance:,.2f}")


# Export main components
__all__ = ['safe_place_order', 'safety_manager', 'initialize_safety_manager']