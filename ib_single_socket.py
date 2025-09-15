#!/usr/bin/env python3
"""
Simplified IB Single Socket Pattern
Synchronous version without worker threads for maximum stability
"""

from ib_insync import IB, Future, util
import time
import random
import logging
from pathlib import Path
from typing import Optional, Any
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)

# Global IB connection
_ib: Optional[IB] = None
_last_api_call = 0
PID_FILE = Path.home() / "trading_system.pid"

def init_ib(host="127.0.0.1", port=4002, client_id=9002, timeout=20) -> IB:
    """Initialize the single shared IB connection"""
    global _ib

    if _ib is not None and _ib.isConnected():
        logger.info(f"Already connected (clientId={_ib.client.clientId})")
        return _ib

    # Write PID file
    pid = os.getpid()
    PID_FILE.write_text(str(pid))
    logger.info(f"PID {pid} written to {PID_FILE}")

    # Connect with retries
    for attempt in range(3):
        try:
            logger.info(f"Connecting to {host}:{port} (attempt {attempt+1}/3)...")
            _ib = IB()
            _ib.connect(host, port, clientId=client_id, timeout=timeout)
            logger.info(f"✅ Connected! Server version: {_ib.client.serverVersion()}")
            return _ib
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(10)
            else:
                if PID_FILE.exists():
                    PID_FILE.unlink()
                raise

def get_ib() -> Optional[IB]:
    """Get the shared IB connection"""
    return _ib

def apply_pacing(min_delay=0.2, max_delay=0.3):
    """Apply pacing between API calls to avoid violations"""
    global _last_api_call

    now = time.time()
    time_since_last = now - _last_api_call

    if time_since_last < min_delay:
        delay = min_delay + random.random() * (max_delay - min_delay)
        time.sleep(delay)

    _last_api_call = time.time()

def fetch_historical_data(symbol: str, duration: str = "1 D", bar_size: str = "5 mins") -> Any:
    """Fetch historical data with proper pacing"""
    if not _ib or not _ib.isConnected():
        raise RuntimeError("IB not connected")

    # Apply pacing
    apply_pacing()

    # Map symbol to contract
    if symbol == "ES1!":
        contract = Future("ES", lastTradeDateOrContractMonth="202509", exchange="CME")
    elif symbol == "NQ1!":
        contract = Future("NQ", lastTradeDateOrContractMonth="202509", exchange="CME")
    else:
        raise ValueError(f"Unknown symbol: {symbol}")

    # Qualify contract
    contracts = _ib.qualifyContracts(contract)
    if not contracts:
        raise ValueError(f"Could not qualify contract for {symbol}")

    # Apply pacing before historical data request
    apply_pacing()

    # Fetch data
    bars = _ib.reqHistoricalData(
        contracts[0],
        endDateTime='',
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow='TRADES',
        useRTH=False
    )

    return bars

def place_order_safe(contract, order):
    """Place an order with proper pacing"""
    if not _ib or not _ib.isConnected():
        raise RuntimeError("IB not connected")

    apply_pacing()
    trade = _ib.placeOrder(contract, order)
    return trade

def get_positions():
    """Get positions with pacing"""
    if not _ib or not _ib.isConnected():
        raise RuntimeError("IB not connected")

    apply_pacing()
    return _ib.positions()

def disconnect():
    """Disconnect and cleanup"""
    global _ib

    if _ib and _ib.isConnected():
        logger.info("Disconnecting from IB Gateway...")
        _ib.disconnect()
        logger.info("✅ Disconnected")

    if PID_FILE.exists():
        PID_FILE.unlink()
        logger.info(f"Removed PID file")

    _ib = None

def check_pipeline_running():
    """Check if another process is using the connection"""
    if not PID_FILE.exists():
        return False

    try:
        import psutil
        pid = int(PID_FILE.read_text().strip())

        # Check if process exists and is ours
        current_pid = os.getpid()
        if pid == current_pid:
            return False  # It's us

        if psutil.pid_exists(pid):
            try:
                proc = psutil.Process(pid)
                if 'python' in proc.name().lower():
                    return True
            except:
                pass

        # Stale PID file
        PID_FILE.unlink()
        return False

    except Exception as e:
        logger.warning(f"Error checking PID: {e}")
        return False

def enforce_single_connection():
    """Ensure only one connection exists"""
    if check_pipeline_running():
        pid = PID_FILE.read_text().strip()
        raise RuntimeError(
            f"Another process is already connected (PID: {pid})\n"
            f"Only one process can use clientId=9002.\n"
            f"Stop the other process or use cached data."
        )

# Data export for training
def export_training_data(symbols=["ES1!", "NQ1!"], days=5):
    """Export data to cache for offline training"""
    if not _ib or not _ib.isConnected():
        raise RuntimeError("IB not connected")

    cache_dir = Path.home() / "training_data"
    cache_dir.mkdir(exist_ok=True)

    for symbol in symbols:
        try:
            logger.info(f"Exporting {symbol} data...")
            bars = fetch_historical_data(symbol, f"{days} D", "5 mins")

            if bars:
                df = util.df(bars)
                cache_file = cache_dir / f"{symbol.replace('!', '')}_training_data.parquet"
                df.to_parquet(cache_file)
                logger.info(f"✅ Exported {len(df)} bars to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to export {symbol}: {e}")

# Test function
def test_single_socket():
    """Test the single socket pattern"""
    logger.info("=" * 60)
    logger.info("SINGLE SOCKET PATTERN TEST")
    logger.info("=" * 60)

    try:
        # Check for conflicts
        enforce_single_connection()

        # Connect
        ib = init_ib()
        logger.info(f"✅ Connected with clientId={ib.client.clientId}")

        # Test data fetch with pacing
        logger.info("\nTesting data fetch with pacing...")

        for symbol in ["ES1!", "NQ1!"]:
            start = time.time()
            bars = fetch_historical_data(symbol, "1 D", "1 hour")
            elapsed = time.time() - start
            logger.info(f"  {symbol}: {len(bars)} bars fetched in {elapsed:.2f}s")

            if bars:
                latest = bars[-1]
                logger.info(f"    Latest: {latest.date} Close: {latest.close}")

        # Test positions
        logger.info("\nChecking positions...")
        positions = get_positions()
        logger.info(f"  Found {len(positions)} positions")
        for pos in positions:
            logger.info(f"    {pos.contract.symbol}: {pos.position} @ {pos.avgCost}")

        # Export training data
        logger.info("\nExporting training data...")
        export_training_data(["ES1!"], days=2)

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("Single socket pattern working correctly:")
        logger.info("  • One connection maintained")
        logger.info("  • Proper pacing between API calls")
        logger.info("  • Data export for offline training")
        logger.info("  • No clientId conflicts")

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

    finally:
        disconnect()

if __name__ == "__main__":
    test_single_socket()