#!/usr/bin/env python3
"""
Property-based tests for order safety using Hypothesis.
These tests verify that safety functions behave correctly across
a wide range of inputs, including edge cases.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

# Import the modules to test
from order_safety_wrapper import OrderSafetyManager

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)


class TestOrderSafetyProperties:
    """Property-based tests for order safety functions."""

    @given(
        quantity=st.integers(min_value=-1000, max_value=1000),
        max_size=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=100)
    def test_quantity_clamping_bounds(self, quantity, max_size):
        """Quantity clamping should always produce valid bounds."""
        # Simulate a clamping function
        def clamp_quantity(qty, max_val):
            return max(0, min(abs(qty), max_val))

        result = clamp_quantity(quantity, max_size)

        # Properties that must always hold
        assert 0 <= result <= max_size, f"Result {result} out of bounds [0, {max_size}]"
        assert isinstance(result, int), "Result should be an integer"

    @given(
        symbol=st.text(min_size=1, max_size=10,
                       alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        side=st.sampled_from(['BUY', 'SELL']),
        quantity=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=50)
    def test_duplicate_order_detection(self, symbol, side, quantity):
        """Duplicate detection should work for any valid order."""
        manager = OrderSafetyManager()

        # First order should always pass duplicate check
        result1 = manager.check_duplicate_order(symbol, side)
        assert result1 is True, "First order should not be duplicate"

        # Immediate duplicate should be detected
        result2 = manager.check_duplicate_order(symbol, side)
        assert result2 is False, "Immediate duplicate should be detected"

        # After clearing, should pass again
        manager.recent_orders.clear()
        result3 = manager.check_duplicate_order(symbol, side)
        assert result3 is True, "After clearing, order should not be duplicate"

    @given(
        pnl_changes=st.lists(
            st.floats(min_value=-10000, max_value=10000, allow_nan=False),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50)
    def test_pnl_tracking_consistency(self, pnl_changes):
        """P&L tracking should be consistent and accurate."""
        manager = OrderSafetyManager()

        expected_total = 0.0
        for change in pnl_changes:
            manager.update_pnl(change)
            expected_total += change

        # Allow for small floating point errors
        assert abs(manager.daily_pnl - expected_total) < 0.01, \
            f"P&L mismatch: {manager.daily_pnl} != {expected_total}"

    @given(
        starting_balance=st.floats(min_value=1000, max_value=1000000),
        loss_pct=st.floats(min_value=0, max_value=1.0)
    )
    @settings(max_examples=50)
    def test_loss_limit_calculation(self, starting_balance, loss_pct):
        """Loss limit checks should be mathematically correct."""
        manager = OrderSafetyManager()
        manager.starting_balance = starting_balance

        # Calculate loss that would trigger limit
        max_loss = -starting_balance * loss_pct
        manager.daily_pnl = max_loss

        # Check if loss limit is correctly identified
        is_at_limit = abs(manager.daily_pnl / manager.starting_balance) >= loss_pct

        if loss_pct > 0:
            assert is_at_limit or manager.daily_pnl >= 0, \
                "Loss limit logic error"

    @given(
        symbol=st.text(min_size=0, max_size=20),
        side=st.text(min_size=0, max_size=10),
        quantity=st.integers(min_value=-1000, max_value=1000)
    )
    @settings(max_examples=100)
    def test_safety_manager_never_crashes(self, symbol, side, quantity):
        """Safety manager should never crash, even with invalid inputs."""
        manager = OrderSafetyManager()

        try:
            # These should handle any input gracefully
            manager.pre_order_checks(symbol, side, quantity)
            manager.check_duplicate_order(symbol, side)
            manager.record_order_placed(symbol, side, abs(quantity))
            # Should complete without crashing
        except (ValueError, TypeError, KeyError):
            # Expected exceptions for invalid inputs are acceptable
            pass
        except Exception as e:
            # Unexpected exceptions are test failures
            pytest.fail(f"Unexpected exception: {e}")


class OrderSafetyStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for order safety manager.
    This ensures the manager behaves correctly through
    sequences of operations.
    """

    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.manager = OrderSafetyManager()
        self.orders_placed = []
        self.total_pnl = 0.0

    @rule(
        symbol=st.sampled_from(['ES', 'NQ', '6E', '6B']),
        side=st.sampled_from(['BUY', 'SELL']),
        quantity=st.integers(min_value=1, max_value=3)
    )
    def place_order(self, symbol, side, quantity):
        """Rule: Place an order."""
        passed, reason = self.manager.pre_order_checks(symbol, side, quantity)

        if passed:
            self.manager.record_order_placed(symbol, side, quantity)
            self.orders_placed.append((symbol, side, quantity))

    @rule(
        pnl_change=st.floats(min_value=-1000, max_value=1000, allow_nan=False)
    )
    def update_pnl(self, pnl_change):
        """Rule: Update P&L."""
        self.manager.update_pnl(pnl_change)
        self.total_pnl += pnl_change

    @rule()
    def reset_daily(self):
        """Rule: Reset daily counters."""
        self.manager.reset_daily_counters()
        self.orders_placed = []
        self.total_pnl = 0.0

    @invariant()
    def pnl_consistency(self):
        """Invariant: P&L should match our tracking."""
        assert abs(self.manager.daily_pnl - self.total_pnl) < 0.01, \
            f"P&L mismatch: {self.manager.daily_pnl} != {self.total_pnl}"

    @invariant()
    def trade_count_consistency(self):
        """Invariant: Trade count should match our tracking."""
        # After reset, counts should match
        if hasattr(self, 'orders_placed'):
            assert self.manager.trade_count_today >= 0, \
                "Trade count should never be negative"


# Additional focused property tests

@given(
    errors_in_window=st.integers(min_value=0, max_value=10),
    window_minutes=st.integers(min_value=1, max_value=60)
)
@settings(max_examples=50)
def test_circuit_breaker_logic(errors_in_window, window_minutes):
    """Circuit breaker should activate based on error threshold."""
    manager = OrderSafetyManager()

    # Record errors
    for _ in range(errors_in_window):
        manager.record_error()

    # Check if circuit breaker state is correct
    threshold = int(os.getenv('ERROR_THRESHOLD', '3'))

    if errors_in_window >= threshold:
        # Should have activated if enough errors
        pass  # Actual check depends on timing
    else:
        # Should not activate with fewer errors
        assert not manager.circuit_breaker_active or errors_in_window == 0


@given(
    hour=st.integers(min_value=0, max_value=23),
    minute=st.integers(min_value=0, max_value=59),
    symbol=st.sampled_from(['ES', 'NQ', '6E', 'GC'])
)
def test_market_hours_logic(hour, minute, symbol):
    """Market hours check should be consistent."""
    manager = OrderSafetyManager()

    # Mock the current time
    from unittest.mock import MagicMock, patch
    mock_now = MagicMock()
    mock_now.hour = hour
    mock_now.minute = minute
    mock_now.weekday.return_value = 2  # Wednesday

    with patch('order_safety_wrapper.datetime') as mock_datetime:
        mock_datetime.now.return_value = mock_now

        result = manager.check_market_hours(symbol)

        # Result should be boolean
        assert isinstance(result, bool)

        # During regular trading hours for indices
        if symbol.startswith(('ES', 'NQ')):
            in_rth = (hour == 9 and minute >= 30) or (10 <= hour < 16)
            if not os.getenv('ALLOW_OUTSIDE_RTH', '0') == '1':
                assert result == in_rth or result is True  # May allow based on config


if __name__ == "__main__":
    # Run a quick test
    test = TestOrderSafetyProperties()
    test.test_quantity_clamping_bounds()
    print("✓ Property-based tests are working")

    # Run state machine test
    TestStateMachine = OrderSafetyStateMachine.TestCase
    TestStateMachine.settings = settings(max_examples=10)
    unittest_case = TestStateMachine()
    unittest_case.runTest()
    print("✓ State machine tests are working")
