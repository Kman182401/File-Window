#!/usr/bin/env python3
"""
Minimal Pipeline Test - Establishing Stable Baseline
Target: <100ms decision latency, zero errors for 10 minutes
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ib_insync import IB, Future, util
import psutil

# Apply baseline settings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["DRIFT_ENABLED"] = "0"
os.environ["DRY_RUN"] = "1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler('/home/ubuntu/logs/minimal_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MinimalTradingPipeline:
    """Minimal, stable trading pipeline for baseline establishment"""

    def __init__(self):
        self.ib = None
        self.symbols = ["ES1!", "NQ1!"]  # Start with 2 only
        self.features = [
            "ret_1", "ret_5", "atr", "rsi",
            "volume", "hour", "position_held"
        ]  # 7 core features only
        self.positions = {}
        self.last_decision_time = {}
        self.error_count = 0
        self.start_time = time.time()
        self.decision_count = 0
        self.total_latency = 0

    def connect(self):
        """Establish IB Gateway connection"""
        try:
            self.ib = IB()
            self.ib.connect('127.0.0.1', 4002, clientId=9002, timeout=20)
            logger.info(f"✅ Connected to IB Gateway (server v{self.ib.client.serverVersion()})")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False

    def fetch_data(self, symbol):
        """Fetch minimal market data"""
        try:
            # Map symbol to futures contract with front month
            if symbol == "ES1!":
                contract = Future('ES', lastTradeDateOrContractMonth='202509', exchange='CME')
            elif symbol == "NQ1!":
                contract = Future('NQ', lastTradeDateOrContractMonth='202509', exchange='CME')
            else:
                return None

            contracts = self.ib.qualifyContracts(contract)
            if not contracts:
                logger.warning(f"Could not qualify contract for {symbol}")
                return None
            contract = contracts[0]

            # Fetch last 100 bars (minimal)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='5 mins',
                whatToShow='TRADES',
                useRTH=False
            )

            if bars:
                df = util.df(bars)
                return df
            return None

        except Exception as e:
            logger.warning(f"Data fetch error for {symbol}: {e}")
            self.error_count += 1
            return None

    def calculate_features(self, df, symbol):
        """Calculate minimal feature set"""
        try:
            features = {}

            # Returns
            features['ret_1'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) if len(df) > 1 else 0
            features['ret_5'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) > 5 else 0

            # ATR (simplified)
            if len(df) > 14:
                high_low = df['high'] - df['low']
                features['atr'] = high_low.rolling(14).mean().iloc[-1]
            else:
                features['atr'] = 0

            # RSI (simplified)
            if len(df) > 14:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
            else:
                features['rsi'] = 50

            # Volume
            features['volume'] = df['volume'].iloc[-1] if 'volume' in df else 0

            # Time
            features['hour'] = datetime.now().hour

            # Position
            features['position_held'] = self.positions.get(symbol, 0)

            return features

        except Exception as e:
            logger.warning(f"Feature calculation error for {symbol}: {e}")
            return None

    def make_decision(self, features, symbol):
        """Simple rule-based decision for baseline"""
        try:
            # Simple momentum strategy
            if features['ret_1'] > 0.001 and features['rsi'] < 70:
                return 1  # Buy signal
            elif features['ret_1'] < -0.001 and features['rsi'] > 30:
                return -1  # Sell signal
            else:
                return 0  # Hold

        except Exception as e:
            logger.warning(f"Decision error for {symbol}: {e}")
            return 0

    def run_iteration(self):
        """Run one pipeline iteration"""
        iteration_start = time.time()

        for symbol in self.symbols:
            try:
                symbol_start = time.time()

                # Fetch data
                df = self.fetch_data(symbol)
                if df is None or df.empty:
                    continue

                # Calculate features
                features = self.calculate_features(df, symbol)
                if features is None:
                    continue

                # Make decision
                decision_start = time.time()
                action = self.make_decision(features, symbol)
                decision_latency = (time.time() - decision_start) * 1000  # ms

                self.decision_count += 1
                self.total_latency += decision_latency

                # Log decision
                symbol_time = time.time() - symbol_start
                logger.info(f"  {symbol}: action={action:+d}, latency={decision_latency:.1f}ms, total={symbol_time:.2f}s")

            except Exception as e:
                logger.error(f"Iteration error for {symbol}: {e}")
                self.error_count += 1

        iteration_time = time.time() - iteration_start
        logger.info(f"Iteration complete: {iteration_time:.2f}s")

    def get_stats(self):
        """Get pipeline statistics"""
        runtime = time.time() - self.start_time
        avg_latency = self.total_latency / max(1, self.decision_count)

        # Memory stats
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        # Socket stats
        import subprocess
        result = subprocess.run(
            ["ss", "-tanp", "|", "grep", ":4002"],
            capture_output=True,
            text=True,
            shell=True
        )
        close_wait = result.stdout.count("CLOSE-WAIT")

        return {
            'runtime_minutes': runtime / 60,
            'error_count': self.error_count,
            'error_rate_per_hour': (self.error_count / runtime) * 3600,
            'decision_count': self.decision_count,
            'avg_decision_latency_ms': avg_latency,
            'memory_mb': memory_mb,
            'close_wait_sockets': close_wait
        }

    def run(self, duration_minutes=10):
        """Run pipeline for specified duration"""
        logger.info("=" * 60)
        logger.info("MINIMAL PIPELINE TEST - STABLE BASELINE")
        logger.info("=" * 60)

        if not self.connect():
            return False

        end_time = time.time() + (duration_minutes * 60)
        iteration = 0

        try:
            while time.time() < end_time:
                iteration += 1
                logger.info(f"\n--- Iteration {iteration} ---")

                self.run_iteration()

                # Show stats every 5 iterations
                if iteration % 5 == 0:
                    stats = self.get_stats()
                    logger.info("📊 Current Stats:")
                    logger.info(f"  Runtime: {stats['runtime_minutes']:.1f} minutes")
                    logger.info(f"  Errors: {stats['error_count']} ({stats['error_rate_per_hour']:.1f}/hour)")
                    logger.info(f"  Avg Decision Latency: {stats['avg_decision_latency_ms']:.1f}ms")
                    logger.info(f"  Memory: {stats['memory_mb']:.1f}MB")
                    logger.info(f"  CLOSE-WAIT sockets: {stats['close_wait_sockets']}")

                # Sleep between iterations
                time.sleep(10)  # 10 second intervals

        except KeyboardInterrupt:
            logger.info("\n⚠️ Test interrupted by user")

        finally:
            # Final stats
            stats = self.get_stats()
            logger.info("\n" + "=" * 60)
            logger.info("FINAL RESULTS")
            logger.info("=" * 60)

            # Success criteria
            success = (
                stats['error_rate_per_hour'] < 1 and
                stats['avg_decision_latency_ms'] < 100 and
                stats['close_wait_sockets'] < 10
            )

            if success:
                logger.info("✅ SUCCESS - ALL CRITERIA MET")
            else:
                logger.info("❌ FAILED - CRITERIA NOT MET")

            logger.info(f"  Runtime: {stats['runtime_minutes']:.1f} minutes")
            logger.info(f"  Total Errors: {stats['error_count']}")
            logger.info(f"  Error Rate: {stats['error_rate_per_hour']:.1f}/hour")
            logger.info(f"  Decisions Made: {stats['decision_count']}")
            logger.info(f"  Avg Decision Latency: {stats['avg_decision_latency_ms']:.1f}ms")
            logger.info(f"  Final Memory: {stats['memory_mb']:.1f}MB")
            logger.info(f"  CLOSE-WAIT Sockets: {stats['close_wait_sockets']}")

            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                logger.info("✅ Disconnected from IB Gateway")

            return success

if __name__ == "__main__":
    # Run 10-minute test
    pipeline = MinimalTradingPipeline()
    success = pipeline.run(duration_minutes=1)  # Start with 1 minute test

    sys.exit(0 if success else 1)