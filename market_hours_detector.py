#!/usr/bin/env python3
"""
Market Hours Detection and Historical Training Safety Module

Provides safe market hour detection for enabling historical training only when
markets are closed, preventing conflicts with live trading operations.

Key Safety Features:
- Multiple market hour detection methods (US, European, Asian markets)
- Timezone-aware calculations with UTC normalization
- Holiday calendar integration
- Safety buffers around market open/close
- Live/historical mode conflict prevention
- Comprehensive logging and monitoring

Author: AI Trading System
Version: 1.0.0
"""

import os
import logging
from datetime import datetime, time, timedelta, UTC
from typing import Dict, List, Tuple, Optional, Union
import pytz
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class MarketHours:
    """Market hours configuration for different exchanges"""
    name: str
    timezone: str
    open_time: time
    close_time: time
    days_open: List[int]  # 0=Monday, 6=Sunday
    holidays: List[str] = None  # YYYY-MM-DD format
    
class MarketHoursDetector:
    """
    Detects market hours across multiple exchanges to determine safe historical training windows
    """
    
    def __init__(self):
        """Initialize market hours for major trading venues"""
        self.markets = {
            'US_EQUITY': MarketHours(
                name='US Stock Market',
                timezone='America/New_York',
                open_time=time(9, 30),
                close_time=time(16, 0),
                days_open=[0, 1, 2, 3, 4],  # Mon-Fri
                holidays=[  # Major 2024 holidays
                    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29',
                    '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02',
                    '2024-10-14', '2024-11-28', '2024-12-25'
                ]
            ),
            'US_FUTURES': MarketHours(
                name='CME Futures',
                timezone='America/Chicago',
                open_time=time(17, 0),  # Sunday 5 PM CT
                close_time=time(16, 0),  # Friday 4 PM CT (session end)
                days_open=[6, 0, 1, 2, 3, 4],  # Sun 5PM CT through Fri 4PM CT
                holidays=[
                    # Representative CME holiday closures/early closes (2025)
                    # Note: Keep this list data-driven via config in future iterations
                    '2025-01-01',  # New Year’s Day
                    '2025-01-20',  # MLK Day
                    '2025-02-17',  # Presidents’ Day
                    '2025-04-18',  # Good Friday (CME closed for equities)
                    '2025-05-26',  # Memorial Day
                    '2025-06-19',  # Juneteenth (modified hours)
                    '2025-07-04',  # Independence Day
                    '2025-09-01',  # Labor Day
                    '2025-11-27',  # Thanksgiving Day
                    '2025-12-25',  # Christmas Day
                ],
            ),
            'FOREX': MarketHours(
                name='Forex Market',
                timezone='UTC',
                open_time=time(21, 0),  # Sunday 9 PM UTC (Sydney open)
                close_time=time(22, 0),  # Friday 10 PM UTC (NY close)
                days_open=[0, 1, 2, 3, 4, 5, 6],  # Nearly 24/7
            ),
            'LONDON': MarketHours(
                name='London Stock Exchange',
                timezone='Europe/London',
                open_time=time(8, 0),
                close_time=time(16, 30),
                days_open=[0, 1, 2, 3, 4],  # Mon-Fri
            ),
            'TOKYO': MarketHours(
                name='Tokyo Stock Exchange',
                timezone='Asia/Tokyo',
                open_time=time(9, 0),
                close_time=time(15, 0),
                days_open=[0, 1, 2, 3, 4],  # Mon-Fri
            )
        }
        
        # Safety buffer: Don't train within N minutes of market open/close
        self.safety_buffer_minutes = 30

        # Daily CME maintenance window (CT) e.g., 16:15–16:30
        # Some products vary; keep configurable at detector level
        self.cme_maintenance_start = time(16, 15)
        self.cme_maintenance_end = time(16, 30)
        
        # Primary markets for your futures trading
        self.primary_markets = ['US_FUTURES', 'US_EQUITY']
        
    def is_market_open(self, market_name: str, dt: datetime = None) -> bool:
        """Check if a specific market is currently open"""
        if dt is None:
            dt = datetime.now(UTC).astimezone(pytz.UTC)
            
        if market_name not in self.markets:
            logger.warning(f"Unknown market: {market_name}")
            return False
            
        market = self.markets[market_name]
        
        # Convert to market timezone
        market_tz = pytz.timezone(market.timezone)
        market_time = dt.astimezone(market_tz)
        
        # Check if it's a trading day
        if market_time.weekday() not in market.days_open:
            return False
            
        # Check holidays
        if market.holidays:
            date_str = market_time.strftime('%Y-%m-%d')
            if date_str in market.holidays:
                return False
                
        # Check if within trading hours
        current_time = market_time.time()
        
        # Handle special daily maintenance for CME futures
        if market_name == 'US_FUTURES':
            # In exchange local time (America/Chicago)
            if self.cme_maintenance_start <= current_time < self.cme_maintenance_end:
                return False

        # Handle overnight markets (like futures)
        if market.open_time > market.close_time:  # Crosses midnight
            return current_time >= market.open_time or current_time <= market.close_time
        else:  # Normal day session
            return market.open_time <= current_time <= market.close_time

    def get_market_state(self, market_name: str, dt: Optional[datetime] = None) -> Tuple[str, Optional[datetime]]:
        """
        Return (state, next_transition_at_utc) where state in {'open','maintenance','closed'}
        Times are computed in the market's local timezone and returned as UTC.
        """
        if dt is None:
            dt = datetime.now(UTC).astimezone(pytz.UTC)

        if market_name not in self.markets:
            return 'closed', None

        market = self.markets[market_name]
        tz = pytz.timezone(market.timezone)
        now_local = dt.astimezone(tz)
        t = now_local.time()

        # Maintenance detection (CME)
        if market_name == 'US_FUTURES' and self.cme_maintenance_start <= t < self.cme_maintenance_end:
            # Next transition is maintenance end today (or next valid day)
            end_local = now_local.replace(hour=self.cme_maintenance_end.hour,
                                          minute=self.cme_maintenance_end.minute,
                                          second=0, microsecond=0)
            return 'maintenance', end_local.astimezone(pytz.UTC)

        # Open/closed
        if self.is_market_open(market_name, dt):
            # Find next close (for futures this may be same-day 16:00 CT or via days_open)
            # Compute close today in local time; if already past close (overnight session), pick next day’s close
            base = now_local
            close_local = base.replace(hour=market.close_time.hour,
                                       minute=market.close_time.minute,
                                       second=0, microsecond=0)
            if market.open_time > market.close_time and t <= market.close_time:
                # Overnight branch where close is later same local day
                next_transition = close_local
            else:
                # If current time already beyond today's close, move to next valid day
                if t > market.close_time:
                    next_day = base + timedelta(days=1)
                else:
                    next_day = base
                next_transition = next_day.replace(hour=market.close_time.hour,
                                                    minute=market.close_time.minute,
                                                    second=0, microsecond=0)
            return 'open', next_transition.astimezone(pytz.UTC)

        # Closed state; compute next open
        next_open = self.get_next_market_open(market_name, dt)
        return 'closed', next_open
            
    def are_primary_markets_closed(self, dt: datetime = None) -> bool:
        """Check if all primary markets are closed (safe for historical training)"""
        if dt is None:
            dt = datetime.now(UTC).astimezone(pytz.UTC)
            
        for market_name in self.primary_markets:
            if self.is_market_open(market_name, dt):
                logger.info(f"Market {market_name} is open, historical training not safe")
                return False
                
        return True
        
    def get_next_market_open(self, market_name: str, dt: datetime = None) -> Optional[datetime]:
        """Get the next market open time"""
        if dt is None:
            dt = datetime.now(UTC).astimezone(pytz.UTC)
            
        if market_name not in self.markets:
            return None
            
        market = self.markets[market_name]
        market_tz = pytz.timezone(market.timezone)
        
        # Start searching from current time
        search_dt = dt.astimezone(market_tz)
        
        # Search up to 7 days ahead
        for days_ahead in range(8):
            test_date = search_dt + timedelta(days=days_ahead)
            
            # Check if it's a trading day
            if test_date.weekday() in market.days_open:
                # Check holidays
                if not market.holidays or test_date.strftime('%Y-%m-%d') not in market.holidays:
                    # Create market open datetime
                    open_dt = test_date.replace(
                        hour=market.open_time.hour,
                        minute=market.open_time.minute,
                        second=0,
                        microsecond=0
                    )
                    
                    # Only return if it's in the future
                    if open_dt > search_dt:
                        return open_dt.astimezone(pytz.UTC)
                        
        return None
        
    def get_safe_training_window(self, hours_ahead: int = 8) -> Tuple[bool, Optional[datetime], str]:
        """
        Get safe training window information
        
        Returns:
            (is_safe_now, next_conflict_time, reason)
        """
        now = datetime.now(UTC).astimezone(pytz.UTC)
        
        # Check if markets are currently closed
        if not self.are_primary_markets_closed(now):
            reason = "Primary markets are currently open"
            return False, None, reason
            
        # Check safety buffer around market opens
        for market_name in self.primary_markets:
            next_open = self.get_next_market_open(market_name, now)
            if next_open:
                time_to_open = (next_open - now).total_seconds() / 60  # minutes
                if time_to_open <= self.safety_buffer_minutes:
                    reason = f"{market_name} opens in {time_to_open:.0f} minutes (within safety buffer)"
                    return False, next_open, reason
                    
        # Check if we have enough time for training
        earliest_open = min([
            self.get_next_market_open(market, now) 
            for market in self.primary_markets
        ], default=None)
        
        if earliest_open:
            available_hours = (earliest_open - now).total_seconds() / 3600
            if available_hours < 1:  # Need at least 1 hour for training
                reason = f"Only {available_hours:.1f} hours until market open, insufficient for training"
                return False, earliest_open, reason
                
        return True, earliest_open, "Safe training window available"
        
    def create_training_schedule(self, training_duration_hours: int = 2) -> Dict:
        """Create a safe training schedule"""
        is_safe, next_conflict, reason = self.get_safe_training_window()
        
        schedule = {
            'is_safe_now': is_safe,
            'can_start_training': is_safe,
            'reason': reason,
            'next_market_open': next_conflict,
            'recommended_duration_hours': training_duration_hours,
            'max_safe_duration_hours': 0,
            'market_status': {}
        }
        
        # Get status of all markets
        now = datetime.now(UTC).astimezone(pytz.UTC)
        for market_name, market in self.markets.items():
            schedule['market_status'][market_name] = {
                'is_open': self.is_market_open(market_name, now),
                'next_open': self.get_next_market_open(market_name, now)
            }
            
        # Calculate max safe duration
        if next_conflict:
            max_hours = (next_conflict - now).total_seconds() / 3600 - (self.safety_buffer_minutes / 60)
            schedule['max_safe_duration_hours'] = max(0, max_hours)
            
        return schedule

def is_historical_training_safe() -> bool:
    """
    Simple function to check if historical training is safe right now
    Returns True if markets are closed and safe for historical training
    """
    detector = MarketHoursDetector()
    is_safe, _, reason = detector.get_safe_training_window()
    
    if is_safe:
        logger.info("Historical training is SAFE - markets closed with sufficient buffer")
    else:
        logger.info(f"Historical training is NOT SAFE - {reason}")
        
    return is_safe

if __name__ == "__main__":
    # Test the market hours detector
    detector = MarketHoursDetector()
    
    print("=== Market Hours Detection Test ===")
    
    # Check current status
    schedule = detector.create_training_schedule()
    print(f"Safe for training: {schedule['is_safe_now']}")
    print(f"Reason: {schedule['reason']}")
    
    if schedule['next_market_open']:
        print(f"Next market opens: {schedule['next_market_open']}")
        
    print("\n=== Market Status ===")
    for market, status in schedule['market_status'].items():
        open_status = "OPEN" if status['is_open'] else "CLOSED"
        print(f"{market}: {open_status}")
        
    print(f"\n=== Training Window ===")
    print(f"Max safe duration: {schedule['max_safe_duration_hours']:.1f} hours")
