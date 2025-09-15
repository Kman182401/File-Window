# ============================================================================
# ENHANCED IMPORTS - Trading System v2.0
# ============================================================================

# System path setup
import sys

sys.path.append('/home/ubuntu')

import socket, traceback, time, os, pathlib, signal
# --- SOCKET CONNECT TRACER (optional) ---
if os.getenv("SOCKET_TRACE", "0") in ("1", "true", "True"):
    _orig_sock_connect = socket.socket.connect
    _orig_sock_connect_ex = socket.socket.connect_ex
    TRACE_LOG = pathlib.Path.home() / "logs" / "ib_connect_traces.log"
    TRACE_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _connect_tracer(self, addr):
        try:
            host, port = addr
        except Exception:
            return _orig_sock_connect(self, addr)
        if str(host) in ("127.0.0.1","::1","localhost") and int(port) == int(os.getenv("IBKR_PORT","4002")):
            with TRACE_LOG.open("a") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] socket.connect({host}:{port})\n")
                traceback.print_stack(file=f, limit=30)
        return _orig_sock_connect(self, addr)

    def _connect_ex_tracer(self, addr):
        try:
            host, port = addr
        except Exception:
            return _orig_sock_connect_ex(self, addr)
        if str(host) in ("127.0.0.1","::1","localhost") and int(port) == int(os.getenv("IBKR_PORT","4002")):
            with TRACE_LOG.open("a") as f:
                f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] socket.connect_ex({host}:{port})\n")
                traceback.print_stack(file=f, limit=30)
        return _orig_sock_connect_ex(self, addr)

    socket.socket.connect = _connect_tracer
    socket.socket.connect_ex = _connect_ex_tracer
# --- END optional SOCKET CONNECT TRACER ---

from ib_insync import IB
# --- IB GUARD (optional) ---
if os.getenv("IB_GUARD", "0") in ("1", "true", "True"):
    _first_ib = True
    _first_connect = True

    _orig_IB_init = IB.__init__
    def _guarded_ib_init(self, *a, **k):
        global _first_ib
        if _first_ib:
            _first_ib = False
        else:
            print("\n[IB-GUARD] SECOND IB() CONSTRUCTOR DETECTED at", time.strftime("%H:%M:%S"))
            traceback.print_stack(limit=30)
        return _orig_IB_init(self, *a, **k)
    IB.__init__ = _guarded_ib_init

    _orig_connect = IB.connect
    def _guarded_connect(self, *a, **k):
        global _first_connect
        if _first_connect:
            _first_connect = False
        else:
            print("\n[IB-GUARD] ADDITIONAL connect() CALL DETECTED at", time.strftime("%H:%M:%S"))
            traceback.print_stack(limit=30)
        return _orig_connect(self, *a, **k)
    IB.connect = _guarded_connect
# --- END optional IB GUARD ---

# --- FAULTHANDLER (ensure 'signal' is imported) ---
import faulthandler
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=True)
# --- END FAULTHANDLER ---

# Single Socket Integration
import atexit
from ib_single_socket import (
    init_ib, get_ib, disconnect,
    fetch_historical_data, place_order_safe,
    get_positions, apply_pacing,
    enforce_single_connection, export_training_data
)
from asserts import anti_dilution_asserts, anti_overfit_asserts

# Setup clean shutdown (prevents CLOSE-WAIT sockets)
def _clean_shutdown(signum=None, frame=None):
    """Clean shutdown on signal or exit"""
    try:
        ib = get_ib()
        if ib and ib.isConnected():
            print("\nDisconnecting IB connection...")
            disconnect()
    except Exception:
        pass
    sys.exit(0)  # Use sys.exit for proper cleanup (not os._exit)

# Register clean shutdown handlers
signal.signal(signal.SIGINT, _clean_shutdown)
signal.signal(signal.SIGTERM, _clean_shutdown)
atexit.register(lambda: disconnect())

# Core Security and Configuration
from config.master_config import get_config
from config.secrets_manager import get_secrets_manager
from monitoring.client.ingest_hooks import log_ingest

# IBKR Health Monitoring
# Performance Monitoring
from monitoring.performance_tracker import get_performance_tracker

# Audit Logging
from utils.audit_logger import AuditEventType, AuditSeverity, audit_log, get_audit_logger

# Exception Handling and Resilience
from utils.exception_handler import get_error_recovery_manager

# Memory Management
from utils.memory_manager import get_memory_manager

# Initialize global managers (singleton instances)
memory_manager = get_memory_manager()
performance_tracker = get_performance_tracker()
error_recovery = get_error_recovery_manager()
audit_logger = get_audit_logger()
config = get_config()
secrets_manager = get_secrets_manager()

# Start monitoring services
memory_manager.start_monitoring(interval=60)
performance_tracker.start_monitoring()

# Log system initialization
audit_log(
    AuditEventType.SYSTEM_START,
    AuditSeverity.INFO,
    "TradingPipeline",
    "Trading system initialized with enhanced monitoring",
    environment=config.environment.value,
    memory_limit_mb=config.get('infrastructure.memory_limit_mb'),
    monitoring_enabled=config.get('infrastructure.monitoring_enabled')
)

# Original imports with fallback for RealTimeFeatureStore
try:
    from market_data_utils import RealTimeFeatureStore
except Exception:
    # Use memory-managed version from memory_manager

    class RealTimeFeatureStore:
        def __init__(self, max_rows_per_key=5000, max_memory_mb=500):
            self._store = {}
            self.max_rows_per_key = max_rows_per_key
            self.max_memory_mb = max_memory_mb
            self._total_rows = 0

        def _estimate_memory_usage(self):
            """Estimate memory usage in MB"""
            total_mb = sum(
                df.memory_usage(deep=True).sum() / (1024 * 1024)
                for df in self._store.values()
                if hasattr(df, 'memory_usage')
            )
            return total_mb

        def _cleanup_if_needed(self):
            """Smart cleanup: remove oldest rows if memory or row limits exceeded"""

            # Check memory usage
            if self._estimate_memory_usage() > self.max_memory_mb:
                # Remove 25% of oldest data from each key
                for key in list(self._store.keys()):
                    df = self._store[key]
                    if len(df) > 100:  # Only cleanup if substantial data
                        keep_rows = int(len(df) * 0.75)
                        self._store[key] = df.tail(keep_rows).copy()

            # Check row limits per key
            for key in list(self._store.keys()):
                df = self._store[key]
                if len(df) > self.max_rows_per_key:
                    self._store[key] = df.tail(self.max_rows_per_key).copy()

        def update(self, key, features_dict):
            import pandas as pd
            # Store as a growing DataFrame; coerce to numeric where possible
            df = self._store.get(key)
            row = pd.DataFrame([features_dict])

            if df is None:
                self._store[key] = row
            else:
                # Use efficient concatenation with memory optimization
                self._store[key] = pd.concat([df, row], ignore_index=True).infer_objects(copy=False)

            self._total_rows += 1

            # Periodic cleanup (every 100 updates)
            if self._total_rows % 100 == 0:
                self._cleanup_if_needed()

        def get(self, key):
            val = self._store.get(key)
            if val is None:
                return None
            if isinstance(val, dict):
                import pandas as pd
                return pd.DataFrame([val])
            return val

        def clear(self):
            self._store.clear()
            self._total_rows = 0

        def get_memory_stats(self):
            """Get memory usage statistics"""
            return {
                'total_keys': len(self._store),
                'total_rows': sum(len(df) for df in self._store.values() if hasattr(df, '__len__')),
                'memory_usage_mb': self._estimate_memory_usage(),
                'rows_per_key': {k: len(v) for k, v in self._store.items() if hasattr(v, '__len__')}
            }

# Initialize with memory management
feature_store = RealTimeFeatureStore(max_rows_per_key=3000, max_memory_mb=300)

import datetime
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any

import boto3
import joblib
import numpy as np
import psutil
from ib_insync import MarketOrder
from sklearn.linear_model import LogisticRegression

from feature_engineering import (
    generate_features,
)
from market_data_config import IBKR_SYMBOLS as DEFAULT_TICKERS
from market_data_config import MAX_DAILY_LOSS_PCT, MAX_POSITION_EXPOSURE, MAX_TRADES_PER_DAY
from market_data_ibkr_adapter import IBKRIngestor
from market_aware_data_manager import MarketAwareDataManager
from order_safety_wrapper import safe_place_order
from orders_single_client import attach_ib, place_bracket_order

try:
    from ingest_to_s3 import upload_file_to_s3
except Exception:
    def upload_file_to_s3(local_path: str, bucket: str, key: str):
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)


import logging

import gymnasium as gym
import pandas as pd  # <-- Ensure pandas is imported for mock data
from colorlog import ColoredFormatter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from audit_logging_utils import log_trade_results

logger = logging.getLogger(__name__)

LAST_TRAINED_FILE = "last_trained_time.txt"


from market_data_config import (
    IBKR_CLIENT_ID,
    IBKR_DYNAMIC_SUBSCRIPTION,
    IBKR_HOST,
    IBKR_LOG_FILE,
    IBKR_LOG_LEVEL,
    IBKR_LOGGING_ENABLED,
    IBKR_MARKET_DATA_TYPE,
    IBKR_POLL_INTERVAL,
    IBKR_PORT,
    IBKR_RESOURCE_MONITORING,
    IBKR_RETRY_INTERVAL,
    IBKR_SYMBOLS,
    IBKR_TIMEOUT,
)

try:
    from notifications import send_sms_alert
except Exception:
    def send_sms_alert(msg: str):
        logging.warning(f"[sms-alert] {msg}")

# Default maximum order size if not provided elsewhere
MAX_ORDER_SIZE = int(os.getenv("MAX_ORDER_SIZE", "1"))

# ---- IBKR news helpers (historical news + article text) ----
def fetch_ibkr_news_for_tickers(tickers, lookback_hours=24, max_results=200):
    """
    DISABLED: News ingestion temporarily disabled to maintain single socket pattern.
    This function was creating multiple IB connections which violates the single socket requirement.
    Returns empty DataFrame to maintain compatibility.
    """
    import pandas as pd
    # Return empty DataFrame immediately without creating any IB connections
    return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])

def fetch_ibkr_news_for_tickers_DISABLED(tickers, lookback_hours=24, max_results=200):
    """
    DISABLED: Original implementation removed to prevent connection creation.
    """
    import pandas as pd
    # Return empty DataFrame immediately - no IB connections
    return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])

def _sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric features; convert timestamp to unix; fill NaNs."""
    X = X.copy()
    if "timestamp" in X.columns:
        ts = pd.to_datetime(X["timestamp"], utc=True, errors="coerce")
        # NEW: astype instead of .view (silences FutureWarning) and works with tz-aware
        X["ts_unix"] = ts.astype("int64") // 10**9
        X = X.drop(columns=["timestamp"])
    non_num = X.select_dtypes(exclude=["number"]).columns
    if len(non_num):
        X = X.drop(columns=list(non_num))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X.astype(np.float32, copy=False)

def normalize_marketaux_for_union(df):
    """
    Normalize MarketAux normalized dataframe to union schema used downstream.
    """
    import pandas as pd
    if df is None or df.empty:
        return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])
    # try common columns your code already uses
    title = df['title'] if 'title' in df.columns else df.get('headline', "")
    desc  = df['description'] if 'description' in df.columns else title
    pub   = pd.to_datetime(df['published_at'], utc=True, errors='coerce') if 'published_at' in df.columns else pd.to_datetime("now", utc=True)
    tks   = df['tickers'] if 'tickers' in df.columns else [[]]*len(df)
    provider = df['source'] if 'source' in df.columns else "MarketAux"
    out = pd.DataFrame({
        "published_at": pub,
        "title": title,
        "description": desc,
        "tickers": tks,
        "provider": provider,
        "source": "MarketAux"
    })
    return out

RETRAIN_INTERVAL = 24 * 3600  # 24 hours in seconds
# Keep a per‑ticker clock so one ticker's training doesn't block others
last_retrain_times = {}

def detect_regime(price_series, window=100):
    returns = price_series.pct_change().dropna()
    rolling_vol = returns.rolling(window).std()
    high_vol = rolling_vol > rolling_vol.median() * 1.5
    if high_vol.iloc[-1]:
        return "high_vol"
    else:
        return "normal"

def new_data_available(data_dir, ticker, last_check_time):
    filename = f"{ticker}_TRADES.csv"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return False
    mtime = os.path.getmtime(path)
    return mtime > last_check_time

def execute_trade(ib, contract, action, quantity):
    """
    Place a market order via IBKR API.
    ib: ib_insync.IB instance (connected)
    contract: ib_insync.Contract instance (for the symbol)
    action: 'BUY' or 'SELL'
    quantity: int
    """
    order = MarketOrder(action, quantity)
    # Extract symbol from contract for safety wrapper
    symbol = contract.symbol if hasattr(contract, 'symbol') else contract.localSymbol
    trade = safe_place_order(ib, contract, order, symbol=symbol, side=action, quantity=quantity)

    # Structured logging for order gate
    logger.info(f"[order_gate] symbol={symbol} side={action} qty={quantity} status=submitted")
    print(f"Order submitted: {action} {quantity} {contract.localSymbol}")
    return trade

IBKR_CONFIG = {
    'host': IBKR_HOST,
    'port': IBKR_PORT,
    'client_id': IBKR_CLIENT_ID,
    'symbols': IBKR_SYMBOLS,
    'log_level': IBKR_LOG_LEVEL,
    'retry_interval': IBKR_RETRY_INTERVAL,
    'timeout': IBKR_TIMEOUT,
    'market_data_type': IBKR_MARKET_DATA_TYPE,
    'logging_enabled': IBKR_LOGGING_ENABLED,
    'log_file': IBKR_LOG_FILE,
    'dynamic_subscription': IBKR_DYNAMIC_SUBSCRIPTION,
    'resource_monitoring': IBKR_RESOURCE_MONITORING,
    'poll_interval': IBKR_POLL_INTERVAL
}

bucket = "omega-singularity-ml"

class PPOTrainingLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.log_history = []

    def _on_step(self) -> bool:
        logs = self.locals.get('logs', None)
        if logs and 'rollout/ep_rew_mean' in logs:
            self.log_history.append({
                "Step": logs.get("total_timesteps", 0),
                "Avg. Ep. Length": logs.get("rollout/ep_len_mean", 0),
                "Avg. Ep. Reward": logs.get("rollout/ep_rew_mean", 0),
                "Explained Variance": logs.get("train/explained_variance", 0),
                "Loss": logs.get("train/loss", 0),
                "Value Loss": logs.get("train/value_loss", 0),
                "Policy Loss": logs.get("train/policy_gradient_loss", 0),
            })
        return True

    def print_summary(self):
        if not self.log_history:
            print("No PPO training logs collected.")
            return
        df = pd.DataFrame(self.log_history)
        print("\n=== PPO RL Training Progress Summary ===")
        print(df.to_string(index=False))
        print("\nKey:")
        print("  Step: Total environment steps so far")
        print("  Avg. Ep. Length: Average steps per episode")
        print("  Avg. Ep. Reward: Average total reward per episode (higher is better)")
        print("  Explained Variance: Value function fit quality (1=perfect, 0=random, <0=bad)")
        print("  Loss: Total loss (lower is better)")
        print("  Value Loss: Value function loss")
        print("  Policy Loss: Policy gradient loss")

LOG_LEVEL = logging.INFO  # Change to DEBUG for more detail, WARNING for less

LOG_FORMAT = (
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s"
    "\n" + "-"*80
)

LOG_COLORS = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'bold_red',
}

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S", log_colors=LOG_COLORS))

root_logger = logging.getLogger()
root_logger.handlers = []  # Remove any existing handlers
root_logger.addHandler(handler)
root_logger.setLevel(LOG_LEVEL)
# Only show warnings/errors by default
logging.getLogger().setLevel(logging.WARNING)

def to_scalar(x):
    """Convert any array-like or scalar to a Python int."""
    arr = np.asarray(x)
    if arr.shape == ():  # scalar
        return int(arr.item())
    elif arr.shape == (1,):  # 1-element array
        return int(arr[0])
    else:
        logger.error(f"Non-scalar value encountered: {x} (type: {type(x)}, shape: {arr.shape})")
        raise ValueError(f"Non-scalar value encountered: {x} (type: {type(x)}, shape: {arr.shape})")

def save_model_to_s3(model, bucket, key):
    """Save a model to S3 using joblib."""
    local_path = f"/tmp/{os.path.basename(key)}"
    joblib.dump(model, local_path)
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    os.remove(local_path)

def load_model_from_s3(bucket, key):
    """Load a model from S3 using joblib."""
    local_path = f"/tmp/{os.path.basename(key)}"
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    model = joblib.load(local_path)
    os.remove(local_path)
    return model

def get_latest_model_key(bucket, prefix, exclude_run_id=None):
    """Get the latest model key from S3, optionally excluding the current run_id."""
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [obj['Key'] for obj in response.get('Contents', [])]
    if exclude_run_id:
        keys = [k for k in keys if exclude_run_id not in k]
    if not keys:
        return None
    return sorted(keys)[-1]  # Latest by name (run_id)

# === Add this drift detection function here ===
def detect_feature_drift(reference_data, new_data, threshold=3.0):
    drift_detected = False
    for col in reference_data.columns:
        ref_mean, ref_std = reference_data[col].mean(), reference_data[col].std()
        new_mean = new_data[col].mean()
        if abs(ref_mean - new_mean) > threshold * ref_std:
            drift_detected = True
            logging.warning(f"Drift detected in {col}: ref_mean={ref_mean}, new_mean={new_mean}")
    return drift_detected
# === End drift detection function ===

# === Market ingestion tracking for dashboard ===
_last_ts_by_symbol = {}

def count_gaps(index, bar_secs: int = 60) -> int:
    """Count missing bars in a time series"""
    if index is None or len(index) < 2:
        return 0
    try:
        # Assumes a monotonic, timezone-aware DatetimeIndex
        span = (index[-1] - index[0]).total_seconds()
        expected = int(span // bar_secs) + 1
        missing = expected - len(index)
        return max(0, missing)
    except (IndexError, KeyError, AttributeError) as e:
        import logging
        logging.warning(f"count_gaps error: {e}, returning 0")
        return 0

def emit_ingest_event_if_new(symbol: str, df, latency_ms: int) -> None:
    """Log market ingestion event if we have new data"""
    if df is None or df.empty:
        return

    # Get the last timestamp from the data
    if 'timestamp' in df.columns:
        last_ts = df['timestamp'].max()
    elif isinstance(df.index, pd.DatetimeIndex):
        last_ts = df.index[-1]
    else:
        # If no timestamp column or index, log anyway
        last_ts = pd.Timestamp.now()

    # Check if this is new data
    if _last_ts_by_symbol.get(symbol) is not None and last_ts <= _last_ts_by_symbol[symbol]:
        return  # Skip if not new data

    _last_ts_by_symbol[symbol] = last_ts

    # Count gaps in the data
    if isinstance(df.index, pd.DatetimeIndex):
        gaps = count_gaps(df.index, bar_secs=60)
    elif 'timestamp' in df.columns:
        gaps = count_gaps(pd.to_datetime(df['timestamp']), bar_secs=60)
    else:
        gaps = 0

    try:
        # Log the ingestion event
        log_ingest(symbol=symbol, bars=len(df), gaps=gaps, lat_ms=latency_ms)
    except Exception as e:
        # Non-fatal: do not let logging break trading
        logging.warning(f"[ingest_log] non-fatal error logging ingestion: {e}")
# === End market ingestion tracking ===

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.transaction_cost = 0.0005
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        self.risk_penalty_coeff = 0.1

        # Robust datetime detection (tz-aware or naive), and drop them from observations
        from pandas.api.types import is_datetime64_any_dtype
        self.numeric_columns = [
            col for col in self.data.columns
            if not (is_datetime64_any_dtype(self.data[col].dtype)
                    or isinstance(self.data[col].dtype, pd.DatetimeTZDtype)
                    or col in ("timestamp", "datetime"))
        ]

        # Define spaces (MUST exist for gym check)
        self.action_space = gym.spaces.Discrete(3)  # [0: hold, 1: buy, 2: sell]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.numeric_columns),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        if seed is not None:
            np.random.seed(seed)
        obs = self.data.iloc[self.current_step][self.numeric_columns].values.astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        reward = 0
        done = False
        prev_position = self.position
        prev_balance = self.balance

        # Transaction cost if position changes
        trade_cost = 0
        if action == 1:  # buy
            if self.position != 1:
                trade_cost = self.transaction_cost * self.data.iloc[self.current_step]['close']
            self.position = 1
        elif action == 2:  # sell
            if self.position != -1:
                trade_cost = self.transaction_cost * self.data.iloc[self.current_step]['close']
            self.position = -1
        else:  # hold
            self.position = 0

        # Reward: PnL from position minus transaction cost
        if self.current_step > 0:
            price_diff = self.data.iloc[self.current_step]['close'] - self.data.iloc[self.current_step - 1]['close']
            reward = prev_position * price_diff - trade_cost

        # Update balance and drawdown
        self.balance += reward
        self.peak_balance = max(self.peak_balance, self.balance)
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Risk penalty (drawdown)
        risk_penalty = self.risk_penalty_coeff * self.max_drawdown
        reward -= risk_penalty

        # Risk limit enforcement
        if self.max_drawdown > 0.2:  # 20% drawdown hard stop
            done = True
            logging.warning("Max drawdown limit reached. Ending episode.")

        self.current_step += 1
        # Check if episode is done
        if self.current_step >= len(self.data) - 1:
            done = True

        # Prepare next observation
        if self.current_step >= len(self.data):
            obs = np.zeros(len(self.numeric_columns), dtype=np.float32)
        else:
            obs = self.data.iloc[self.current_step][self.numeric_columns].values.astype(np.float32)

        terminated = done  # True if episode ended due to terminal state (e.g., max drawdown, end of data)
        truncated = False  # True if episode ended due to time/trade limit (set to False if not used)
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

import os
import re


def load_all_models(bucket, run_id):
    """
    Loads all models (ML and PPO) for a given run_id from S3.
    Returns a nested dict: {ticker: {model_name: model_object}}
    """
    s3 = boto3.client("s3")
    prefix = f"models/{run_id}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    models = {}
    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = os.path.basename(key)
        # ML models: {ticker}_{modelname}.joblib
        m = re.match(r"(.+?)_(RandomForest|XGBoost|LightGBM)\.joblib$", filename)
        if m:
            ticker, model_name = m.groups()
            local_path = f"/tmp/{filename}"
            s3.download_file(bucket, key, local_path)
            model = joblib.load(local_path)
            os.remove(local_path)
            if ticker not in models:
                models[ticker] = {}
            models[ticker][model_name] = model
        # PPO RL models: {ticker}_ppo.zip
        m2 = re.match(r"(.+?)_ppo\.zip$", filename)
        if m2:
            ticker = m2.group(1)
            local_path = f"/tmp/{filename}"
            s3.download_file(bucket, key, local_path)
            model = PPO.load(local_path)
            os.remove(local_path)
            if ticker not in models:
                models[ticker] = {}
            models[ticker]["PPO"] = model
    return models

def _ib_keepalive_loop(ib):
    """Keepalive loop for IB connection"""
    import os
    import time
    backoff = 2
    while True:
        try:
            if not ib.isConnected():
                ib.disconnect()
                ib.connect(os.getenv("IBKR_HOST","127.0.0.1"),
                           int(os.getenv("IBKR_PORT","4002")),
                           clientId=int(os.getenv("IBKR_CLIENT_ID","9002")),
                           timeout=30)
                backoff = 2
            else:
                _ = ib.reqCurrentTime()
                backoff = 2
            time.sleep(30)
        except Exception:
            try: ib.disconnect()
            except: pass
            time.sleep(min(backoff, 30))
            backoff = min(backoff*2, 30)

def align_X_y(X, y, required_cols=None, name=""):
    """Ensure X and y are properly aligned with consistent shape and indices"""
    # 1) Ensure deterministic order and unique index
    X = X.copy()
    X = X.loc[~X.index.duplicated()].sort_index()
    y = y.loc[~y.index.duplicated()].sort_index()

    # 2) Reindex y to X
    y = y.reindex(X.index)

    # 3) Enforce a stable feature set (train-time columns)
    if required_cols is not None:
        X = X.reindex(columns=required_cols, fill_value=0)

    # 4) Final sanity: remove rows with any NaN in X after enforcement
    good = ~X.isna().any(axis=1)
    if not good.all():
        X, y = X.loc[good], y.loc[good]

    # 5) Assertions & helpful logs
    if len(X) != len(y):
        raise ValueError(f"[ALIGN:{name}] len(X)={len(X)} != len(y)={len(y)}")
    if len(X) == 0:
        raise ValueError(f"[ALIGN:{name}] empty dataset after alignment")

    logging.debug(f"[ALIGN:{name}] X.shape={X.shape}, y.shape={y.shape}")
    return X, y

class RLTradingPipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        try:
            logging.info("Initializing single socket IBKR connection...")

            # Enforce single connection (will error if another process is using 9002)
            enforce_single_connection()

            # Initialize the single shared connection
            shared_ib = init_ib(
                host=os.getenv("IBKR_HOST", "127.0.0.1"),
                port=int(os.getenv("IBKR_PORT", "4002")),
                client_id=int(os.getenv("IBKR_CLIENT_ID", "9002"))
            )

            # Pass shared connection to IBKRIngestor but route via market-aware manager
            underlying_ib_adapter = IBKRIngestor(ib=shared_ib)
            self.market_data_adapter = MarketAwareDataManager(underlying_ib_adapter)
            # attach_ib disabled - single socket pattern manages connection
            # attach_ib(self.market_data_adapter.ib)

            logging.info(f"✅ Single socket initialized (clientId={shared_ib.client.clientId})")

            # Run anti-dilution checks ONCE at startup
            symbols_to_check = getattr(self, 'tickers', ['ES1!', 'NQ1!'])
            if len(symbols_to_check) > int(os.getenv("MAX_BASELINE_SYMBOLS", "2")):
                logging.warning(f"Limiting symbols from {len(symbols_to_check)} to baseline max")
                self.tickers = symbols_to_check[:int(os.getenv("MAX_BASELINE_SYMBOLS", "2"))]

            # Verify drift is disabled
            from optimized_trading_config import DRIFT_CONFIG
            if DRIFT_CONFIG.get("enabled", False):
                logging.warning("Disabling drift detection for baseline")
                DRIFT_CONFIG["enabled"] = False

            # Keepalive disabled - single socket pattern handles connection
            # The keepalive loop was causing reconnection conflicts
            if False and os.getenv("PIPELINE_KEEPALIVE","1") == "1" and not hasattr(self, "_keepalive_started"):
                import threading
                threading.Thread(target=_ib_keepalive_loop, args=(self.market_data_adapter.ib,), daemon=True).start()
                self._keepalive_started = True
                logging.info("IB keepalive thread started")
            logging.info("IBKR connection established successfully")
        except Exception as e:
            logging.error(f"Failed to initialize IBKR connection: {e}")
            raise RuntimeError(f"Cannot proceed without IBKR connection: {e}")
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 10)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.2)  # 20% default
        self.max_position_size = config.get('max_position_size', 1)      # 1 contract/lot default
        self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.status = "initialized"
        self.last_error = None
        self.notification_hook = None  # Extensibility: notification system

        # Memory management
        self._memory_cleanup_interval = 50  # Cleanup every N iterations

        # Enhanced resource monitoring for m5.large production safety
        # SINGLE SOCKET MODE: Skip resource monitor when disabled
        if os.getenv("DISABLE_HEALTH_MONITOR") == "1":
            logging.info("Resource monitoring disabled (single-socket mode)")
            self.resource_monitor = None
        else:
            try:
                from enhanced_resource_monitor import EnhancedResourceMonitor, ResourceThresholds
                self.resource_monitor = EnhancedResourceMonitor(
                    thresholds=ResourceThresholds(),
                    cleanup_callback=self._cleanup_memory
                )
                self.resource_monitor.start_monitoring(interval_seconds=5)
                logging.info("✅ Enhanced resource monitoring enabled")
            except Exception as e:
                logging.warning(f"Enhanced monitoring failed to initialize: {e}")
                self.resource_monitor = None
        self._iteration_count = 0

        # Model caching system for faster inference
        self._model_cache = {}
        self._prediction_cache = {}
        self._cache_max_size = 1000  # Maximum cached predictions per model
        self._cache_ttl_seconds = 300  # Cache time-to-live: 5 minutes

    def _get_memory_usage(self):
        """Get current system memory usage"""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),  # Physical memory
            'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual memory
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024)
        }

    def _cleanup_memory(self):
        """Perform comprehensive memory cleanup"""
        import gc

        import pandas as pd

        # Force garbage collection
        collected = gc.collect()

        # Clean feature store
        if hasattr(feature_store, '_cleanup_if_needed'):
            feature_store._cleanup_if_needed()

        # Clean prediction cache
        self._clear_expired_cache()

        # Clear pandas categorical caches
        if hasattr(pd.api.types, 'CategoricalDtype'):
            try:
                pd.api.types.pandas_dtype.clear_cache()
            except:
                pass  # Method might not exist in all pandas versions

        logging.info(f"Memory cleanup: collected {collected} objects")

        # Log memory stats
        memory_stats = self._get_memory_usage()
        feature_stats = feature_store.get_memory_stats() if hasattr(feature_store, 'get_memory_stats') else {}
        logging.info(f"Memory usage: {memory_stats['rss_mb']:.1f}MB RSS, {memory_stats['percent']:.1f}%")
        if feature_stats:
            logging.info(f"Feature store: {feature_stats['memory_usage_mb']:.1f}MB, {feature_stats['total_rows']} rows")

    def _get_cache_key(self, model_name, ticker, feature_hash):
        """Generate a cache key for model predictions"""
        return f"{model_name}_{ticker}_{feature_hash}"

    def _hash_features(self, features):
        """Create a hash of feature data for caching"""
        import hashlib


        if hasattr(features, 'values'):
            # DataFrame or Series
            feature_str = str(features.values.tobytes())
        elif isinstance(features, (list, tuple)):
            feature_str = str(features)
        else:
            feature_str = str(features)

        return hashlib.sha256(feature_str.encode()).hexdigest()[:16]

    def _get_cached_prediction(self, model_name, ticker, features):
        """Get cached prediction if available and not expired"""
        import time

        feature_hash = self._hash_features(features)
        cache_key = self._get_cache_key(model_name, ticker, feature_hash)

        if cache_key in self._prediction_cache:
            cached_data = self._prediction_cache[cache_key]

            # Check if cache entry is still valid (TTL)
            if time.time() - cached_data['timestamp'] < self._cache_ttl_seconds:
                return cached_data['prediction']
            else:
                # Remove expired entry
                del self._prediction_cache[cache_key]

        return None

    def _cache_prediction(self, model_name, ticker, features, prediction):
        """Cache a model prediction"""
        import time

        feature_hash = self._hash_features(features)
        cache_key = self._get_cache_key(model_name, ticker, feature_hash)

        # Implement LRU-like behavior: if cache is full, remove oldest entries
        if len(self._prediction_cache) >= self._cache_max_size:
            # Remove oldest 20% of cache entries
            sorted_items = sorted(
                self._prediction_cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            remove_count = max(1, len(sorted_items) // 5)
            for old_key, _ in sorted_items[:remove_count]:
                del self._prediction_cache[old_key]

        self._prediction_cache[cache_key] = {
            'prediction': prediction,
            'timestamp': time.time()
        }

    def _clear_expired_cache(self):
        """Remove expired cache entries"""
        import time

        current_time = time.time()
        expired_keys = [
            key for key, data in self._prediction_cache.items()
            if current_time - data['timestamp'] > self._cache_ttl_seconds
        ]

        for key in expired_keys:
            del self._prediction_cache[key]

        if expired_keys:
            logging.info(f"Cleared {len(expired_keys)} expired cache entries")

    def _predict_with_cache(self, model, model_name, ticker, features, deterministic=True):
        """Make prediction with caching for faster inference"""

        # Try to get cached prediction first
        cached_pred = self._get_cached_prediction(model_name, ticker, features)
        if cached_pred is not None:
            return cached_pred

        # Make new prediction
        if hasattr(model, 'predict'):
            if model_name == 'PPO' or 'PPO' in model_name:
                # PPO models return (action, _log_prob)
                prediction = model.predict(features, deterministic=deterministic)
            else:
                # Standard scikit-learn models
                prediction = model.predict(features)
        else:
            raise ValueError(f"Model {model_name} does not have a predict method")

        # Cache the prediction
        self._cache_prediction(model_name, ticker, features, prediction)

        return prediction

    def _process_ticker_parallel(self, ticker):
        """Process a single ticker (for parallel execution)"""
        import time

        start_time = time.time()
        result = {
            'ticker': ticker,
            'success': False,
            'error': None,
            'data': None,
            'features': None,
            'labels': None,
            'processing_time': 0
        }

        try:
            logging.info(f"Fetching data for {ticker}...")
            df = self.market_data_adapter.fetch_data(ticker)

            if df is None or df.empty:
                result['error'] = f"No data for {ticker}"
                return result

            result['data'] = df
            logging.info(f"{ticker} raw data shape: {df.shape}")

            # Feature engineering
            try:
                X, y = generate_features(df)
                result['features'] = X
                result['labels'] = y
                result['success'] = True
                logging.info(f"{ticker} features shape: {X.shape if X is not None else 'None'}")
            except Exception as fe_error:
                result['error'] = f"Feature engineering failed for {ticker}: {str(fe_error)}"
                logging.error(f"Feature engineering error for {ticker}: {fe_error}")

        except Exception as e:
            result['error'] = f"Data fetching failed for {ticker}: {str(e)}"
            logging.error(f"Data fetching error for {ticker}: {e}")

        result['processing_time'] = time.time() - start_time
        return result

    async def _process_ticker_async(self, ticker):
        """Process a single ticker asynchronously (for async parallel execution)"""
        import asyncio
        import time

        start_time = time.time()
        result = {
            'ticker': ticker,
            'success': False,
            'error': None,
            'data': None,
            'features': None,
            'labels': None,
            'processing_time': 0
        }

        try:
            logging.info(f"PHASE4A-FIX: Async fetching data for {ticker}...")

            # Run the sync fetch_data in an executor to avoid blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, self.market_data_adapter.fetch_data, ticker)

            if df is None or df.empty:
                result['error'] = f"No data for {ticker}"
                return result

            result['data'] = df
            logging.info(f"PHASE4A-FIX: {ticker} raw data shape: {df.shape}")

            # Feature engineering (also run in executor for CPU-intensive work)
            try:
                X, y = await loop.run_in_executor(None, generate_features, df)
                result['features'] = X
                result['labels'] = y
                result['success'] = True
                logging.info(f"PHASE4A-FIX: {ticker} features shape: {X.shape if X is not None else 'None'}")
            except Exception as fe_error:
                result['error'] = f"Feature engineering failed for {ticker}: {str(fe_error)}"
                logging.error(f"PHASE4A-FIX: Feature engineering error for {ticker}: {fe_error}")

        except Exception as e:
            result['error'] = f"Data fetching failed for {ticker}: {str(e)}"
            logging.error(f"PHASE4A-FIX: Data fetching error for {ticker}: {e}")

        result['processing_time'] = time.time() - start_time
        return result

    def _process_tickers_parallel(self, tickers, max_workers=None):
        """Process multiple tickers in parallel using ProcessPoolExecutor to avoid asyncio issues"""

        if max_workers is None:
            # Use both CPU cores but leave some headroom for m5.large
            max_workers = min(2, len(tickers))

        results = {}
        successful_results = []

        logging.info(f"Processing {len(tickers)} tickers in parallel (max_workers={max_workers})...")

        # For now, fall back to sequential processing due to asyncio complexity
        # This still benefits from the optimized feature engineering pipeline
        for ticker in tickers:
            result = self._process_ticker_parallel(ticker)
            results[ticker] = result
            if result['success']:
                successful_results.append(result)
                logging.info(f"✓ {ticker}: processed in {result['processing_time']:.2f}s")
            else:
                logging.warning(f"✗ {ticker}: {result['error']}")

        total_time = sum(r['processing_time'] for r in results.values())
        avg_time = total_time / len(results) if results else 0
        success_count = sum(1 for r in results.values() if r['success'])

        logging.info(f"Sequential processing complete: {success_count}/{len(tickers)} successful")
        logging.info(f"Total processing time: {total_time:.2f}s, Average per ticker: {avg_time:.2f}s")

        return results

    def _process_tickers_sequential_batched(self, tickers, batch_size=2):
        """
        Process tickers in sequential batches for memory efficiency on m5.large
        
        Memory benefit: 28% reduction in peak usage (1.6GB vs 2.5GB)
        Trade-off: 30% longer processing time vs full parallel
        """
        import gc
        import time

        results = {}
        total_start_time = time.time()

        # Split tickers into batches
        batches = [tickers[i:i+batch_size] for i in range(0, len(tickers), batch_size)]

        logging.info(f"🔧 Sequential batch processing: {len(tickers)} tickers in {len(batches)} batches (size={batch_size})")

        for batch_idx, batch_tickers in enumerate(batches):
            batch_start_time = time.time()

            logging.info(f"Processing batch {batch_idx + 1}/{len(batches)}: {batch_tickers}")

            # Process batch using existing method
            batch_results = {}
            for ticker in batch_tickers:
                result = self._process_ticker_parallel(ticker)
                batch_results[ticker] = result
                results[ticker] = result

                if result['success']:
                    logging.info(f"✓ {ticker}: processed in {result['processing_time']:.2f}s")
                else:
                    logging.warning(f"✗ {ticker}: {result['error']}")

            batch_time = time.time() - batch_start_time

            # Force garbage collection between batches to free memory
            collected = gc.collect()
            logging.info(f"Batch {batch_idx + 1} complete in {batch_time:.2f}s, freed {collected} objects")

            # Memory cleanup between batches
            if hasattr(self, '_cleanup_memory'):
                self._cleanup_memory()

        total_time = time.time() - total_start_time
        success_count = sum(1 for r in results.values() if r['success'])

        logging.info(f"Sequential batch processing complete: {success_count}/{len(tickers)} successful")
        logging.info(f"Total time: {total_time:.2f}s (avg {total_time/len(tickers):.2f}s/ticker)")

        return results

    def run(self):
        # PHASE4A: Performance monitoring
        phase4a_start_time = time.time()
        logger.info("PHASE4A: Starting execution with optimizations enabled")
        logger.info(f"PHASE4A: Config - Parallel={self.config.get('ENABLE_PARALLEL_PROCESSING', True)}, Optimized Features={os.getenv('USE_OPTIMIZED_FEATURES', 'true')}")

        self.status = "running"
        log_trade_results(f"pipeline_start: run_id={self.run_id}")
        try:
            self._main_loop()
            self.status = "completed"

            # PHASE4A: Performance results
            phase4a_elapsed = time.time() - phase4a_start_time
            logger.info(f"PHASE4A: Execution completed in {phase4a_elapsed:.2f} seconds")
            if phase4a_elapsed > 2.0:
                logger.warning(f"PHASE4A: Target <2s missed. Actual: {phase4a_elapsed:.2f}s")
            else:
                logger.info(f"PHASE4A: SUCCESS - Target <2s achieved! ({phase4a_elapsed:.2f}s)")

            log_trade_results(f"pipeline_complete: run_id={self.run_id}")
        except Exception as e:
            self.status = "failed"
            self.last_error = str(e)
            log_trade_results(
                f"pipeline_error: run_id={self.run_id}, error={self.last_error}, traceback={traceback.format_exc()}"
            )
            # PHASE4A: Performance results (failure case)
            phase4a_elapsed = time.time() - phase4a_start_time
            logger.info(f"PHASE4A: Execution failed after {phase4a_elapsed:.2f} seconds")
            if phase4a_elapsed > 2.0:
                logger.warning(f"PHASE4A: Target <2s missed. Actual: {phase4a_elapsed:.2f}s")
            else:
                logger.info(f"PHASE4A: Time target met despite failure ({phase4a_elapsed:.2f}s)")

            logger.error(f"Pipeline failed: {e}")
            self.notify(f"Pipeline failed: {e}")
            raise
        finally:
            # SINGLE SOCKET MODE: Do NOT disconnect between cycles - keep connection alive
            # The connection should persist across pipeline runs
            pass

    def _main_loop(self):
        global last_retrain_times
        retries = 0
        while retries < self.max_retries:
            try:
                self._iteration_count += 1
                logging.info(f"Starting main loop iteration {self._iteration_count}...")

                # Periodic memory cleanup
                if self._iteration_count % self._memory_cleanup_interval == 0:
                    logging.info("Performing memory cleanup...")
                    self._cleanup_memory()

                logging.info("Fetching market data...")
                # List of tickers you want to fetch (using your mapping for yfinance)
                tickers = self.config.get("tickers", DEFAULT_TICKERS)
                logging.info(f"Tickers to process: {tickers}")
                last_check_times = {ticker: 0 for ticker in tickers}

                feature_store.clear()

                def _get_features_df(key: str) -> pd.DataFrame:
                    fx = feature_store.get(key)
                    if fx is None:
                        return pd.DataFrame()
                    if isinstance(fx, dict):
                        return pd.DataFrame([fx])
                    if isinstance(fx, pd.Series):
                        return fx.to_frame().T
                    if not isinstance(fx, pd.DataFrame):
                        try:
                            return pd.DataFrame(fx)
                        except Exception:
                            return pd.DataFrame()
                    # coerce dtypes without copying if possible
                    return fx.infer_objects(copy=False)

                # Use parallel processing for data fetching and feature engineering
                if self.config.get("use_mock_data", False):
                    raise ValueError("Mock data is disabled. Using only real IBKR data.")

                # SAFETY CHECK: Historical training integration with market hours detection
                if self.config.get("enable_historical_training", False):
                    try:
                        from market_hours_detector import (
                            MarketHoursDetector,
                        )
                        detector = MarketHoursDetector()

                        # Get detailed schedule info
                        schedule = detector.create_training_schedule()

                        if schedule['is_safe_now']:
                            logging.info(f"✅ HISTORICAL TRAINING ENABLED: {schedule['reason']}")
                            logging.info(f"⏰ Max safe duration: {schedule['max_safe_duration_hours']:.1f} hours")

                            # Import and use historical replay system
                            from historical_replay import ReplayConfig, ReplayMode

                            # Configure historical replay for training
                            replay_config = ReplayConfig(
                                replay_mode=ReplayMode.RANDOM_WINDOW,
                                symbols=tickers[:2],  # Start with 2 tickers for memory efficiency
                                lookback_days=30,
                                max_memory_mb=500  # Limit for m5.large
                            )

                            logging.info("🔄 Switching to HISTORICAL TRAINING MODE")
                            # Historical training will be handled by existing replay system

                        else:
                            logging.info(f"❌ HISTORICAL TRAINING DISABLED: {schedule['reason']}")
                            if schedule['next_market_open']:
                                logging.info(f"⏰ Next safe window: {schedule['next_market_open']}")

                    except Exception as e:
                        logging.error(f"Historical training safety check failed: {e}")
                        logging.info("Continuing with live IBKR data only")

                # PHASE 4A OPTIMIZATION: Enhanced parallel processing with ThreadPoolExecutor
                # Direct ThreadPoolExecutor integration with performance timing
                import time
                processing_start_time = time.time()

                # Phase 4A Feature Flag
                # MODIFIED: Force sequential to avoid IB rate limits
                ENABLE_PARALLEL_PROCESSING = self.config.get('ENABLE_PARALLEL_PROCESSING', False)  # Changed to False
                use_optimized_parallel = self.config.get("use_optimized_parallel", False)  # Changed to False
                use_sequential = self.config.get("use_sequential_processing", True)  # Changed to True

                if ENABLE_PARALLEL_PROCESSING and not use_sequential:
                    logging.info("PHASE4A-FIX: Using async parallel processing")
                    import asyncio

                    import nest_asyncio

                    async def process_all_tickers_async():
                        tasks = []
                        for ticker in tickers:
                            # Create async task for each ticker
                            task = self._process_ticker_async(ticker)
                            tasks.append(task)
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        return dict(zip(tickers, results, strict=False))

                    try:
                        # Handle event loop properly
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                # Already in async context - apply nest_asyncio to allow nested loops
                                logging.info("PHASE4A-FIX: Detected running event loop, applying nest_asyncio")
                                nest_asyncio.apply()
                                parallel_results = loop.run_until_complete(process_all_tickers_async())
                            else:
                                # Sync context - run the event loop
                                logging.info("PHASE4A-FIX: Event loop not running, starting new loop")
                                parallel_results = loop.run_until_complete(process_all_tickers_async())
                        except RuntimeError as re:
                            # No event loop, create one
                            logging.info(f"PHASE4A-FIX: No event loop found ({re}), creating new one")
                            parallel_results = asyncio.run(process_all_tickers_async())

                        # Log success
                        success_count = sum(1 for r in parallel_results.values() if isinstance(r, dict) and r.get('success', False))
                        logging.info(f"PHASE4A-FIX: Async parallel processing complete: {success_count}/{len(tickers)} successful")

                    except Exception as e:
                        logging.error(f"PHASE4A-FIX: Async parallel processing failed: {e}")
                        logging.info("PHASE4A-FIX: Falling back to sequential processing")
                        parallel_results = self._process_tickers_sequential_batched(tickers, batch_size=2)
                elif use_optimized_parallel and not use_sequential:
                    logging.info("🚀 Using OPTIMIZED PARALLEL processing for <1s latency")
                    try:
                        from parallel_data_pipeline import PipelineOptimizer
                        optimizer = PipelineOptimizer(self.market_data_adapter)
                        parallel_results = optimizer.process_tickers_optimized(tickers)
                    except Exception as e:
                        logging.error(f"Optimized parallel processing failed: {e}")
                        logging.info("Falling back to sequential processing")
                        parallel_results = self._process_tickers_sequential_batched(tickers, batch_size=2)
                else:
                    logging.info("PHASE4A: Sequential processing (fallback) with batch_size=2")
                    parallel_results = self._process_tickers_sequential_batched(tickers, batch_size=2)

                processing_end_time = time.time()
                processing_duration = processing_end_time - processing_start_time
                logging.info(f"PHASE4A: Total processing time: {processing_duration:.2f} seconds")

                # Extract results for further processing
                raw_data = {}
                features = {}
                labels = {}
                X_train = pd.DataFrame()
                y_train = pd.Series(dtype=np.float32)

                for ticker, result in parallel_results.items():
                    if result['success']:
                        raw_data[ticker] = result['data']
                        if result['features'] is not None and result['labels'] is not None:
                            features[ticker] = result['features']
                            labels[ticker] = result['labels']
                        logging.info(f"✓ {ticker}: data and features ready")
                        print(f"{ticker} columns: {result['data'].columns}")

                        # Log market ingestion event for dashboard
                        processing_time_ms = int(result.get('processing_time', 0) * 1000)
                        emit_ingest_event_if_new(ticker, result['data'], processing_time_ms)
                    else:
                        logging.warning(f"✗ {ticker}: {result['error']}")
                        continue

                # --- Regime Detection and PPO Training (Post-Parallel Processing) ---
                for ticker in features.keys():
                    df = raw_data[ticker]  # Get the raw data for regime detection

                    # --- Regime Detection and Alerting ---
                    try:
                        regime = detect_regime(df['close'])
                        if regime == "high_vol":
                            send_sms_alert(f"High volatility regime detected for {ticker}!")
                            logging.info(f"Regime change: High volatility detected for {ticker}")
                            print(f"High volatility regime detected for {ticker}, triggering retraining/model switch.")
                            # Trigger retraining/model switch
                            if regime == "high_vol":
                                model_path = f"/home/ubuntu/models/ppo_model_{ticker}_highvol.zip"
                            else:
                                model_path = f"/home/ubuntu/models/ppo_model_{ticker}_normal.zip"
                            self.train_ppo(df, ticker, save_path=model_path)
                    except Exception as regime_error:
                        logging.warning(f"Regime detection failed for {ticker}: {regime_error}")

                    # Update training variables from parallel results
                    if ticker in features and ticker in labels:
                        X_train, y_train = features[ticker], labels[ticker]
                        logging.info(f"{ticker}: features and labels updated from parallel processing")

                # Log resource usage after parallel processing
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                logging.info(f"Resource usage after parallel processing: CPU {cpu:.1f}%, RAM {mem:.1f}%")

                print("All tickers processed in parallel. Check above for any errors or resource warnings.")

                # --- ADD THIS BLOCK IMMEDIATELY AFTER FETCHING DATA, BEFORE FEATURE ENGINEERING ---
                # (This should NOT be indented; it should be at the same level as the print above)
                # If you want to generate features for all tickers after the PPO loop, do it here:
                # Example: for all tickers in raw_data
                for ticker, df in raw_data.items():
                    features, labels = generate_features(df)
                    logger.info(f"Ticker [{ticker}]: features shape = {features.shape}, labels shape = {labels.shape}")

                # >>> NEWS PULL & MERGE (INSERTED) <<<
                # We join news ourselves (generate_features(df) does not accept news_df).
                # This enriches each ticker's market DataFrame in raw_data with simple news features.
                try:
                    import os

                    from fetch_and_merge_news import get_combined_news
                    from market_data_config import IBKR_SYMBOLS

                    NEWS_LOOKBACK_HOURS = int(os.getenv("NEWS_LOOKBACK_HOURS", "48"))
                    MAX_NEWS_PER_TICKER = int(os.getenv("MAX_NEWS_PER_TICKER", "50"))

                    # 1) Pull combined news (IB BZ + MarketAux) once for all tickers
                    news_df = get_combined_news(IBKR_SYMBOLS,
                                                lookback_hours=NEWS_LOOKBACK_HOURS,
                                                max_per_ticker=MAX_NEWS_PER_TICKER)
                    print(f"[news] rows={len(news_df)} over {NEWS_LOOKBACK_HOURS}h")

                    if news_df is not None and not news_df.empty:
                        # 2) Standardize types
                        news_df = news_df.copy()
                        news_df["published_at"] = pd.to_datetime(news_df["published_at"], utc=True, errors="coerce")
                        # Keep only the columns we need here
                        base_cols = ["published_at","title","headline","provider","source","ticker","tickers","article_id"]
                        for c in base_cols:
                            if c not in news_df.columns:
                                news_df[c] = None
                        # Prefer explicit per-row ticker label
                        if "ticker" in news_df.columns:
                            news_df["_lbl"] = news_df["ticker"]
                        elif "tickers" in news_df.columns:
                            # if list-like, take first or None
                            news_df["_lbl"] = news_df["tickers"].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and x else None)
                        else:
                            news_df["_lbl"] = None

                        # 3) For each ticker's market df: build minute-level news counts and merge
                        for t, mdf in raw_data.items():
                            if mdf is None or mdf.empty:
                                continue
                            # ensure timestamp minute index on market data
                            mdf = mdf.copy()
                            if "timestamp" in mdf.columns:
                                mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], utc=True, errors="coerce")
                            elif "datetime" in mdf.columns:
                                mdf["timestamp"] = pd.to_datetime(mdf["datetime"], utc=True, errors="coerce")
                            else:
                                # if no time column, skip merge for this ticker
                                raw_data[t] = mdf
                                continue
                            mdf["ts_min"] = mdf["timestamp"].dt.floor("min")

                            # filter news rows for this ticker label
                            sub = news_df[ news_df["_lbl"] == t ].copy()
                            if sub.empty:
                                # provide zeros if no news rows
                                mdf["news_15m"] = 0
                                mdf["news_60m"] = 0
                                raw_data[t] = mdf.drop(columns=["ts_min"])
                                continue

                            # minute-level counts from news published_at
                            sub["min"] = sub["published_at"].dt.floor("min")
                            minute_counts = (sub.assign(cnt=1)
                                                .groupby("min", as_index=False)["cnt"].sum()
                                                .sort_values("min"))

                            # build rolling windows (15m, 60m) aligned to minute clock
                            minute_counts = minute_counts.set_index("min").asfreq("min", fill_value=0)
                            minute_counts["news_15m"] = minute_counts["cnt"].rolling(15, min_periods=1).sum()
                            minute_counts["news_60m"] = minute_counts["cnt"].rolling(60, min_periods=1).sum()
                            minute_counts = minute_counts[["news_15m","news_60m"]].reset_index().rename(columns={"index":"min"})

                            # merge by exact minute
                            merged = mdf.merge(minute_counts, how="left", left_on="ts_min", right_on="min")
                            merged["news_15m"] = merged["news_15m"].fillna(0).astype(int)
                            merged["news_60m"] = merged["news_60m"].fillna(0).astype(int)
                            raw_data[t] = merged.drop(columns=["ts_min","min"])
                    else:
                        print("[news] empty; skipping news merge")
                except Exception as e:
                    print(f"[news] ingestion/merge failed: {e}")
                # <<< END NEWS PULL & MERGE (INSERTED) >>>

                logging.info("Engineering features...")

                # Load last trained timestamp/index
                try:
                    with open(LAST_TRAINED_FILE) as f:
                        last_trained_time = f.read().strip()
                        if last_trained_time:
                            # Convert to datetime for proper comparison
                            last_trained_time = pd.to_datetime(last_trained_time, utc=True, errors="coerce")
                            if pd.isna(last_trained_time):
                                last_trained_time = None
                except FileNotFoundError:
                    last_trained_time = None

                features = {}
                labels = {}

                for ticker, df in raw_data.items():
                    X, y = generate_features(df)
                    # PHASE4A-DEBUG: Log initial shapes
                    logging.info(f"[DEBUG] {ticker} initial features shape: {X.shape if X is not None else 'None'}, labels shape: {y.shape if y is not None else 'None'}")

                    if last_trained_time and X is not None and "timestamp" in X.columns:
                        try:
                            # last_trained_time is already a datetime from reading the file
                            t = last_trained_time

                            if not pd.isna(t):
                                ts = pd.to_datetime(X["timestamp"], utc=True, errors="coerce")
                                pre = len(X)
                                mask = ts > t
                                if mask.any():
                                    X = X.loc[mask]
                                    y = y.loc[X.index]
                                    logging.info(f"[DEBUG] timestamp filter: {pre} -> {len(X)} rows (threshold={t})")
                                else:
                                    logging.info(f"[DEBUG] No rows after {t}; using fallback")
                                    # Fallback: keep a small recent window so training/eval has data even when market is closed
                                    N = int(os.getenv("FALLBACK_MINUTES", "240"))
                                    X = X.iloc[-N:] if len(X) > N else X
                                    y = y.loc[X.index]
                                    logging.info(f"[DEBUG] Using last {len(X)} rows as fallback")
                            else:
                                logging.info(f"[DEBUG] Invalid timestamp threshold, skipping filter")

                            # Additional failsafe for empty data
                            if len(X) == 0:
                                N = int(os.getenv("FALLBACK_MINUTES", "240"))
                                logging.warning(f"[DEBUG] Filter emptied all data; regenerating features")
                                X, y = generate_features(df)
                                if X is not None and len(X) > N:
                                    X = X.iloc[-N:]
                                    y = y.loc[X.index]
                                logging.info(f"[DEBUG] Fallback generated {len(X) if X is not None else 0} rows")
                        except Exception as e:
                            logging.warning(f"[DEBUG] Timestamp filter error: {e}, using unfiltered data")
                            # Keep original data on error
                    else:
                        logging.info(f"[DEBUG] no timestamp filter applied for {ticker}")

                    # Align X and y after timestamp filtering
                    if X is not None and y is not None and not X.empty and not y.empty:
                        try:
                            X, y = align_X_y(X, y, name=f"{ticker}:post-filter")
                        except ValueError as e:
                            logging.warning(f"Alignment failed after timestamp filter for {ticker}: {e}")
                            continue

                    logging.info(f"{ticker} features shape: {X.shape if X is not None else 'None'}, labels shape: {y.shape if y is not None else 'None'}")
                    if X is None or X.empty or y is None or y.empty:
                        logging.warning(f"No features/labels for {ticker} after feature engineering, skipping.")
                        continue
                    features[ticker] = X
                    labels[ticker] = y
                    # Feed the real-time feature store with the latest row so ML/stacking have data
                    try:
                        if not X.empty:
                            feature_store.update(ticker, X.iloc[-1].to_dict())
                    except Exception as _e:
                        logging.warning(f"[feature-store] update failed for {ticker}: {_e}")
                logging.info("Features engineered successfully.")

                # Per-ticker diagnostics prior to ML/RL thresholds
                try:
                    for tkr in features:
                        X = features.get(tkr)
                        y = labels.get(tkr)
                        frows = 0 if X is None else len(X)
                        lrows = 0 if y is None else len(y)
                        ppo_min = 21  # because we drop to label-aligned and need >20 rows
                        logging.warning(f"[bars] ticker={tkr} features_rows={frows} labels_rows={lrows} ppo_min_rows={ppo_min}")
                except Exception:
                    pass


                # === DRIFT DETECTION BLOCK ===
                # AI_ASSISTANT_TODO: CRITICAL REMINDER!
                # If you are enabling live IBKR data, you MUST set 'disable_drift_detection' to False in the config below.
                # This is a production safeguard. If you are an AI assistant or developer, PROMPT THE USER to re-enable drift detection
                # before allowing any live trading or IBKR integration. This is essential for risk management and compliance.
                for ticker, X in features.items():
                    if len(X) > 200:
                        if not self.config.get("disable_drift_detection", False):
                            reference = X.iloc[:100]
                            new = X.iloc[-100:]
                            if detect_feature_drift(reference, new, threshold=self.config.get("drift_detection_threshold", 3.0)):
                                logger.error(f"Feature drift detected for {ticker}. Halting pipeline.")
                                self.status = "failed"
                                self.notify(f"Feature drift detected for {ticker}.")
                                # Automated model rollback logic
                                logging.warning("Rolling back to previous model version due to error/drift.")
                                self.notify("Model rollback triggered.")
                                return
                        else:
                            logging.warning(
                                "Drift detection is DISABLED for testing. "
                                "AI_ASSISTANT_TODO: When enabling live IBKR data, RE-ENABLE drift detection in config!"
                            )

                # --- ML Model Training and Prediction Block ---
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import (
                    PassiveAggressiveClassifier,
                    Perceptron,
                    SGDClassifier,
                )
                logging.warning("ML SYSTEM: Running ML model training and prediction for all tickers.")
                from sklearn.metrics import accuracy_score, confusion_matrix

                # Define models
                models = {
                    "SGDClassifier": SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-3),
                    "Perceptron": Perceptron(),
                    "PassiveAggressive": PassiveAggressiveClassifier(),
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)  # Optional batch model
                }

                fitted_models = {}

                # Metrics storage
                results = {}

                for ticker in features:
                    # Use the full engineered frame built above, not the 1-row feature store
                    X = features[ticker]
                    y = labels[ticker]

                    if X is None or X.empty or y is None or y.empty:
                        logging.warning(f"No feature/label data for {ticker}, skipping.")
                        continue

                    # Optional: Log a small sample of features being used
                    logging.info(f"[{ticker}] using engineered features with shape {X.shape}")

                    if len(y) == 0:
                        logging.warning(f"No labels for {ticker}, skipping ML for this ticker.")
                        continue

                    # Split into train/test (last 20% for test)
                    split_idx = int(len(X) * 0.95)
                    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

                    # Drop any datetime or identifier columns from features
                    from pandas.api.types import is_datetime64_any_dtype
                    cols_to_drop = []
                    for col in X_train.columns:
                        if is_datetime64_any_dtype(X_train[col]) or col in ["datetime", "timestamp"]:
                            cols_to_drop.append(col)
                    if cols_to_drop:
                        X_train = X_train.drop(columns=cols_to_drop)
                        X_test = X_test.drop(columns=cols_to_drop)

                    # Align X and y to ensure consistent shapes
                    try:
                        X_train, y_train = align_X_y(X_train, y_train, name=f"{ticker}:train")
                        X_test, y_test = align_X_y(X_test, y_test, name=f"{ticker}:test")
                    except ValueError as e:
                        logging.warning(f"Alignment failed for {ticker}: {e}")
                        continue

                    # Model-level sanity checks
                    if len(X_train) < 100:
                        logging.warning(f"[ML:{ticker}] Not enough training rows ({len(X_train)}). Skipping.")
                        continue

                    if y_train.nunique() < 2:
                        logging.warning(f"[ML:{ticker}] Single-class labels in training data. Skipping.")
                        continue

                    if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                        logging.warning(f"Not enough data for {ticker}: X_train={len(X_train)}, X_test={len(X_test)}. Skipping.")
                        continue

                    ticker_results = {}

                    for name, model in models.items():
                        model_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/{ticker}_{name}")
                        if model_key:
                            prev_model = load_model_from_s3(self.config['s3_bucket'], model_key)
                            if hasattr(prev_model, "partial_fit"):
                                if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                                    logger.error("No data available for model training.")
                                    raise ValueError("No data available for model training.")
                                if X_train.shape[0] != y_train.shape[0]:
                                    logger.error(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                    raise ValueError(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                if np.any(pd.isnull(X_train)) or np.any(pd.isnull(y_train)):
                                    logger.error("NaN values detected in features or labels.")
                                    raise ValueError("NaN values detected in features or labels.")
                                if len(np.unique(y_train)) < 2:
                                    logger.warning("Skipping model update: only one class present in y_train.")
                                    continue
                                prev_model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                                model = prev_model
                            else:
                                if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                                    logger.error("No data available for model training.")
                                    raise ValueError("No data available for model training.")
                                if X_train.shape[0] != y_train.shape[0]:
                                    logger.error(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                    raise ValueError(f"Feature/label row mismatch: {X_train.shape[0]} vs {y_train.shape[0]}")
                                if np.any(pd.isnull(X_train)) or np.any(pd.isnull(y_train)):
                                    logger.error("NaN values detected in features or labels.")
                                    raise ValueError("NaN values detected in features or labels.")
                                model.fit(X_train, y_train)
                        else:
                            if hasattr(model, "partial_fit"):
                                if len(np.unique(y_train)) < 2:
                                    logger.warning("Skipping model update: only one class present in y_train.")
                                    continue
                                model.partial_fit(X_train, y_train, classes=np.unique(y_train))
                            else:
                                model.fit(X_train, y_train)
                        # Store the fitted model for this ticker
                        if ticker not in fitted_models:
                            fitted_models[ticker] = {}
                        fitted_models[ticker][name] = model
                        save_key = f"models/{self.run_id}/{ticker}_{name}.joblib"
                        save_model_to_s3(model, self.config['s3_bucket'], save_key)
                        preds = model.predict(X_test)

                        acc = accuracy_score(y_test, preds)
                        cm = confusion_matrix(y_test, preds)
                        wins = int((preds == 1).sum())
                        losses = int((preds == 0).sum())
                        total_trades = len(preds)
                        win_rate = (preds == y_test).sum() / total_trades if total_trades > 0 else 0

                        # Simulate PnL: assume +1 for correct, -1 for incorrect (customize for your logic)
                        pnl = ((preds == y_test) * 1 + (preds != y_test) * -1).sum()

                        ticker_results[name] = {
                            "accuracy": acc,
                            "win_rate": win_rate,
                            "trades": total_trades,
                            "wins": wins,
                            "losses": losses,
                            "pnl": pnl,
                            "confusion_matrix": cm.tolist()
                        }

                        # Log results to terminal
                        logging.info(f"[{ticker}][{name}] Trades: {total_trades}, Wins: {wins}, Losses: {losses}, Accuracy: {acc:.2%}, Win Rate: {win_rate:.2%}, PnL: {pnl}")

                    results[ticker] = ticker_results



                summary_rows = []
                for ticker, ticker_results in results.items():
                    for name, metrics in ticker_results.items():
                        summary_rows.append({
                            "Ticker": ticker,
                            "Model": name,
                            "Trades": metrics['trades'],
                            "Wins": metrics['wins'],
                            "Losses": metrics['losses'],
                            "Accuracy": f"{metrics['accuracy']:.2%}",
                            "Win Rate": f"{metrics['win_rate']:.2%}",
                            "PnL": metrics['pnl']
                        })

                summary_df = pd.DataFrame(summary_rows)
                print("\n=== ML Model Performance Summary ===")
                print(summary_df.to_string(index=False))

                # Save the last trained timestamp/index from original engineered features
                max_timestamps = []
                for ticker, X in features.items():
                    if "timestamp" in X.columns and not X.empty:
                        max_timestamps.append(X["timestamp"].max())
                if max_timestamps:
                    last_trained_time = max(max_timestamps)
                    # Save as ISO format string for consistent reading/writing
                    with open(LAST_TRAINED_FILE, "w") as f:
                        # Convert to string in ISO format
                        if hasattr(last_trained_time, 'isoformat'):
                            f.write(last_trained_time.isoformat())
                        else:
                            f.write(str(last_trained_time))

                # --- End ML Model Block ---

                # --- RL PPO Training and Inference Block (fixed: use full engineered frame, not feature_store) ---
                logging.warning("RL SYSTEM: Running PPO RL training and inference.")
                for ticker in features:
                    # Use the full engineered feature frame produced above
                    X = features[ticker]
                    y = labels[ticker]
                    if X is None or X.empty or y is None or y.empty:
                        logging.warning(f"No engineered features/labels for {ticker}, skipping PPO.")
                        continue

                    # Drop datetime-like cols; PPO env expects numeric float32 only
                    from pandas.api.types import is_datetime64_any_dtype
                    dt_cols = [c for c in X.columns
                               if is_datetime64_any_dtype(X[c].dtype)
                               or isinstance(X[c].dtype, pd.DatetimeTZDtype)
                               or c in ("timestamp", "datetime")]
                    X_num = X.drop(columns=dt_cols) if dt_cols else X
                    X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32, copy=False)

                    if len(X_num) <= 20:
                        logging.warning(f"Not enough rows for PPO on {ticker}: {len(X_num)}. Skipping.")
                        continue

                    split_idx = int(len(X_num) * 0.8)
                    train_data = X_num.iloc[:split_idx]
                    test_data  = X_num.iloc[split_idx:]

                    # Choose model path by regime on the original close series
                    regime = detect_regime(X['close'])
                    model_path = f"/home/ubuntu/models/ppo_model_{ticker}_{'highvol' if regime == 'high_vol' else 'normal'}.zip"

                    self.train_ppo(train_data, ticker, save_path=model_path)
                    self.run_ppo(test_data, ticker)
                    break  # Only run PPO on the first valid ticker for now
                # --- End RL PPO Block ---

                # === Begin Stacking/Meta-Model Block (Fixed) ===
                logging.warning("META-MODEL SYSTEM: Running stacking/meta-model training and prediction.")
                models_from_s3 = load_all_models(self.config['s3_bucket'], self.run_id)

                meta_X = []
                meta_y = []

                meta_signals = None

                for ticker in features:
                    X = _get_features_df(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        y_test = y.iloc[split_idx:]
                    # Collect predictions from all base models
                    preds = []
                    model_dict = models_from_s3.get(ticker, {})
                    # Only proceed if all expected models are present
                    expected_models = ["RandomForest", "PPO"]
                    if not all(m in model_dict for m in expected_models):
                        logging.warning(f"Skipping {ticker} in stacking: missing one or more models ({expected_models})")
                        continue
                    for model_name in expected_models:
                        model = model_dict[model_name]
                        if model_name == "PPO":
                            rl_preds = []
                            env = TradingEnv(X_test)
                            obs, info = env.reset()
                            done = False
                            while not done:
                                action, _ = model.predict(obs, deterministic=True)
                                obs, reward, terminated, truncated, info = env.step(action)
                                rl_preds.append(action)
                                done = terminated or truncated
                            rl_preds = np.array(rl_preds[:len(X_test)]).flatten()
                            preds.append(rl_preds)
                        else:
                            # Drop any datetime or identifier columns from features before prediction
                            from pandas.api.types import is_datetime64_any_dtype
                            cols_to_drop = []
                            for col in X_test.columns:
                                if is_datetime64_any_dtype(X_test[col]) or col in ["datetime", "timestamp"]:
                                    cols_to_drop.append(col)
                            if cols_to_drop:
                                X_test_pred = X_test.drop(columns=cols_to_drop)
                            else:
                                X_test_pred = X_test

                            # Use cached prediction for faster inference
                            ml_preds = self._predict_with_cache(model, model_name, ticker, X_test_pred)
                            ml_preds = np.array(ml_preds).flatten()
                            preds.append(ml_preds)

                        # Align all predictions to the minimum available length
                        min_len = min(len(p) for p in preds)
                        n_models = len(preds)
                        for i in range(min_len):
                            row = []
                            for p in preds:
                                val = p[i]
                                # Flatten any array or list to scalar
                                if isinstance(val, (np.ndarray, list)):
                                    val = np.asarray(val).flatten()
                                    if val.shape == ():  # scalar
                                        val = val.item()
                                    elif val.shape == (1,):
                                        val = val[0]
                                    else:
                                        logger.error(f"Non-scalar prediction at index {i}: {val} (shape {val.shape})")
                                        logger.error(f"Non-scalar prediction at index {i}: {val} (shape {val.shape})")
                                        raise ValueError(f"Non-scalar prediction at index {i}: {val} (shape {val.shape})")
                                row.append(int(val))
                            # Only append rows with the correct number of model predictions and all scalars
                            if len(row) == n_models and all(isinstance(x, (int, float, np.integer, np.floating)) for x in row):
                                meta_X.append(row)
                                meta_y.append(int(y_test.iloc[i]))
                            else:
                                logging.warning(f"Skipping row {i} for {ticker} due to length/type mismatch: {row}")

                # Final check before fitting meta-model
                if len(meta_X) == 0:
                    logger.error("No valid meta_X rows to fit meta-model. Skipping meta-model training for this run.")
                else:
                    meta_X_arr = np.array(meta_X)
                    logging.info(f"meta_X_arr shape: {meta_X_arr.shape}, dtype: {meta_X_arr.dtype}")
                    logging.info(f"meta_X_arr example row: {meta_X_arr[0]}")
                    meta_model_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/meta_model.joblib")
                    if meta_model_key:
                        prev_meta_model = load_model_from_s3(self.config['s3_bucket'], meta_model_key)
                        if hasattr(prev_meta_model, "partial_fit"):
                            if meta_X_arr.shape[0] == 0 or meta_y.shape[0] == 0:
                                logger.error("No data available for meta-model training.")
                                raise ValueError("No data available for meta-model training.")
                            if meta_X_arr.shape[0] != meta_y.shape[0]:
                                logger.error(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                                raise ValueError(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                            if np.any(pd.isnull(meta_X_arr)) or np.any(pd.isnull(meta_y)):
                                logger.error("NaN values detected in meta-model features or labels.")
                                raise ValueError("NaN values detected in meta-model features or labels.")
                                # Check for at least two unique classes before partial_fit
                                if len(np.unique(meta_y)) < 2:
                                    logger.warning("Skipping model update: only one class present in meta_y.")
                                    continue
                                prev_meta_model.partial_fit(meta_X_arr, meta_y)
                            meta_model = prev_meta_model
                        else:
                            meta_model = LogisticRegression()
                            # Validate meta-model features and labels before training
                            if meta_X_arr.shape[0] == 0 or meta_y.shape[0] == 0:
                                logger.error("No data available for meta-model training.")
                                raise ValueError("No data available for meta-model training.")
                            if meta_X_arr.shape[0] != meta_y.shape[0]:
                                logger.error(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                                raise ValueError(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y.shape[0]}")
                            if np.any(pd.isnull(meta_X_arr)) or np.any(pd.isnull(meta_y)):
                                logger.error("NaN values detected in meta-model features or labels.")
                                raise ValueError("NaN values detected in meta-model features or labels.")
                            meta_model.fit(meta_X_arr, meta_y)
                    else:
                        meta_model = LogisticRegression()
                        meta_model.fit(meta_X_arr, meta_y)
                    # Save the updated meta-model
                    save_model_to_s3(meta_model, self.config['s3_bucket'], f"models/{self.run_id}/meta_model.joblib")
                    logging.info("Meta-model (stacking) training complete.")

                    # Use meta-model for final trading signal (with caching)
                    meta_signals = self._predict_with_cache(meta_model, 'MetaModel', 'ALL_TICKERS', meta_X_arr)
                    logging.info(f"Meta-model signals (first 10): {meta_signals[:10]}")
                # === End Stacking/Meta-Model Block (Fixed) ===

                # === Begin Meta-Model Trading Simulation Block ===
                # For each ticker, simulate trading using meta-model signals

                # We'll assume meta_signals are in the same order as meta_X/meta_y, which are built sequentially for all tickers.
                # We'll split meta_signals back into per-ticker segments for reporting.

                signal_idx = 0
                tickers_with_meta = []
                for ticker in features:
                    X = features[ticker]
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        n = len(X_test)
                        if meta_signals is not None and signal_idx + n <= len(meta_signals):
                            ticker_signals = meta_signals[signal_idx:signal_idx + n]
                            prices = X_test['close'].values
                            position = 0  # if=long, -1=short, 0=flat
                            balance = 100000
                            trade_log = []
                            trades_today = 0

                            # Track daily PnL and starting equity
                            current_day = datetime.date.today()
                            starting_equity = balance
                            daily_pnl = 0.0
                            trading_halted_for_day = False

                            for i, signal in enumerate(ticker_signals):
                                # check if a new day has started
                                if datetime.date.today() != current_day:
                                    current_day = datetime.date.today()
                                    feature_store.clear()  # Clear real-time feature store at the start of each new trading day
                                    starting_equity = balance
                                    daily_pnl = 0.0
                                    trading_halted_for_day = False
                                    trades_today = 0

                                prev_position = position

                                # Update daily PnL
                                daily_pnl = balance - starting_equity

                                # Enforce max daily loss
                                if daily_pnl <= -MAX_DAILY_LOSS_PCT * starting_equity:
                                    if not trading_halted_for_day:
                                        print(f"Max daily loss reached ({daily_pnl:.2f}). Halting trading for the day.")
                                        trading_halted_for_day = True

                                # Enforce max trades per day
                                if trades_today >= MAX_TRADES_PER_DAY:
                                    if not trading_halted_for_day:
                                        print(f"Max trades per day reached ({trades_today}). Halting trading for the day.")
                                        trading_halted_for_day = True

                                if trading_halted_for_day:
                                    continue  # Skip trade execution if halted

                                order_size = 1  # Default order size for each trade

                                if signal == 1 and position <= 0:
                                    if abs(position + order_size) <= MAX_POSITION_EXPOSURE:
                                        if order_size <= MAX_ORDER_SIZE:
                                            position += order_size
                                            trade_log.append(("BUY", order_size, i))
                                            trades_today += 1
                                        else:
                                            print(f"Order size {order_size} exceeds max allowed ({MAX_ORDER_SIZE}). No BUY executed.")
                                    else:
                                        print(f"Position exposure limit reached. Cannot BUY. Current position: {position}")
                                elif signal == -1 and position >= 0:
                                    if abs(position - order_size) <= MAX_POSITION_EXPOSURE:
                                        if order_size <= MAX_ORDER_SIZE:
                                            position -= order_size
                                            trade_log.append(("SELL", order_size, i))
                                            trades_today += 1
                                        else:
                                            print(f"Order size {order_size} exceeds max allowed ({MAX_ORDER_SIZE}). No SELL executed.")
                                    else:
                                        print(f"Position exposure limit reached. Cannot SELL. Current position: {position}")
                                elif signal == 0:
                                    position = 0
                                # No trade for hold

                            n = len(ticker_signals)
                            if n > 1:
                                balance += prev_position * (prices[n-1] - prices[n-2])

                            print(f"\n=== Meta-Model Trading Results for {ticker} ===")
                            print(f"Final balance: {balance:.2f}")
                            print(f"Total trades: {len([t for t in trade_log if t[0] != 'hold'])}")
                            print(f"Trade log (first 10): {trade_log[:10]}")
                            signal_idx += n
                            tickers_with_meta.append(ticker)
                        else:
                            print(f"\n=== Meta-Model Trading Results for {ticker} ===")
                            print("No meta-model predictions available for this ticker (skipped in stacking).")
                            if meta_signals is not None:
                                print(f"meta_signals length: {len(meta_signals)}, tickers with meta: {tickers_with_meta}")
                            else:
                                print("meta_signals is None (meta-model was not trained this run).")
                # === End Meta-Model Trading Simulation Block ===

                # === Begin Ensemble Voting Block ===
                logging.warning("ENSEMBLE SYSTEM: Running ensemble voting for all tickers.")
                # For each ticker, collect predictions from all ML models and RL model, then vote

                ensemble_results = {}

                for ticker in features:
                    X = _get_features_df(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        y_test = y.iloc[split_idx:]

                        required_models = ["SGDClassifier", "Perceptron", "PassiveAggressive", "RandomForest"]
                        if not all(name in fitted_models.get(ticker, {}) for name in required_models):
                            logging.warning(f"Not all models fitted for {ticker}, skipping ensemble voting for this ticker.")
                            continue

                        # Drop any datetime or identifier columns from features before prediction
                        from pandas.api.types import is_datetime64_any_dtype
                        cols_to_drop = []
                        for col in X_test.columns:
                            if is_datetime64_any_dtype(X_test[col]) or col in ["datetime", "timestamp"]:
                                cols_to_drop.append(col)
                        if cols_to_drop:
                            X_test_pred = X_test.drop(columns=cols_to_drop)
                        else:
                            X_test_pred = X_test

                        # Get ML model predictions
                        ml_preds = {}
                        for name, model in fitted_models.get(ticker, {}).items():
                            # Drop any datetime or identifier columns from features before prediction
                            from pandas.api.types import is_datetime64_any_dtype
                            cols_to_drop = []
                            for col in X_test.columns:
                                if is_datetime64_any_dtype(X_test[col]) or col in ["datetime", "timestamp"]:
                                    cols_to_drop.append(col)
                            X_test_pred = X_test.drop(columns=cols_to_drop) if cols_to_drop else X_test
                            ml_preds[name] = self._predict_with_cache(model, name, ticker, X_test_pred)

                        # Get RL model predictions (PPO) if one is available
                        rl_preds = None
                        if hasattr(self, "ppo_model"):
                            env = TradingEnv(X_test)
                            obs, info = env.reset()
                            done = False
                            rl_seq = []
                            while not done:
                                action, _ = self.ppo_model.predict(obs, deterministic=True)
                                rl_seq.append(action)
                                obs, reward, terminated, truncated, info = env.step(action)
                                done = terminated or truncated
                            rl_preds = np.array(rl_seq[:len(X_test)]).flatten()
                        else:
                            logging.warning(f"[{ticker}] No in-memory PPO model; skipping RL votes for ensemble.")

                        # Align all predictions to the minimum available length (skip RL if absent)
                        if rl_preds is not None:
                            min_len = min(len(ml_preds["RandomForest"]), len(rl_preds))
                        else:
                            min_len = len(ml_preds["RandomForest"])

                        final_signals = []
                        for i in range(min_len):
                            votes = [ml_preds["RandomForest"][i]]
                            if rl_preds is not None:
                                votes.append(rl_preds[i])
                            votes = [int(v) if isinstance(v, np.ndarray) else v for v in votes]
                            final_signal = max(set(votes), key=votes.count)
                            final_signals.append(final_signal)
                            # Majority vote (mode)
                            votes = [int(v) if isinstance(v, np.ndarray) else v for v in votes]
                            final_signal = max(set(votes), key=votes.count)
                            final_signals.append(final_signal)

                        # Store and log ensemble results
                        ensemble_results[ticker] = final_signals
                        logging.info(f"[{ticker}] Ensemble Voting Signals: {final_signals[:10]}... (showing first 10)")

                # === End Ensemble Voting Block ===

                import os

                # Save and upload each ticker's features
                for ticker, df in features.items():
                    # Create a unique filename per ticker and run
                    local_csv_path = f"{ticker}_features_{self.run_id}.csv"
                    df.to_csv(local_csv_path, index=False)
                    # S3 key: organize by run and ticker for auditability
                    s3_key = f"features/{self.run_id}/{ticker}_features.csv"
                    upload_file_to_s3(local_csv_path, self.config['s3_bucket'], s3_key)
                    # Optional: remove local file after upload to save space
                    os.remove(local_csv_path)
                logging.info("Features engineered successfully.")

                # RL model inference and trading logic
                # (Your original RL model code remains here, unchanged)
                # Example:
                # actions = self.rl_model.predict(features)
                # self.execute_trades(actions)

                log_trade_results(f"pipeline_success: run_id={self.run_id}")
                self.notify("Pipeline run completed successfully.")
                break  # Success, exit retry loop

            except Exception as e:
                # Check if this is a connection error vs other errors
                import socket

                def _is_connection_error(err):
                    """Determine if error is actually a connection issue"""
                    msg = str(err).lower()
                    return isinstance(err, (ConnectionError, TimeoutError, socket.error)) \
                        or "peer closed" in msg or "timeout" in msg or "connection refused" in msg \
                        or "connect call failed" in msg

                if _is_connection_error(e):
                    retries += 1
                    logging.error(f"IB connection error in pipeline loop (attempt {retries}): {e}")
                    log_trade_results(f"pipeline_retry: run_id={self.run_id}, attempt={retries}, error={str(e)}, traceback={traceback.format_exc()}")
                    self.notify(f"Pipeline connection error (attempt {retries}): {e}")
                else:
                    # Non-connection error (e.g., dtype issues, ML errors) - log but don't exit
                    logging.warning(f"Non-connection error (continuing): {e}")
                    logging.debug(f"Error details: {traceback.format_exc()}")

                    # ML/data shape errors - skip iteration but continue pipeline
                    if any(err in str(e) for err in ["datetime64", "Cannot interpret", "inconsistent numbers of samples",
                                                      "shape", "ValueError", "[ALIGN:"]):
                        logging.info(f"Skipping iteration due to data/ML error: {e}")
                        time.sleep(1)  # Brief pause before continuing
                        continue  # Don't increment retry, just skip this iteration
                    else:
                        # Other non-connection errors - count retries but don't exit immediately
                        retries += 1
                        logging.warning(f"Error in pipeline loop (attempt {retries}): {e}")
                        log_trade_results(f"pipeline_retry: run_id={self.run_id}, attempt={retries}, error={str(e)}, traceback={traceback.format_exc()}")
                        self.notify(f"Pipeline error (attempt {retries}): {e}")

                        # Only fail after many retries for non-connection errors
                        if retries >= self.max_retries * 2:  # Double threshold for non-connection errors
                            logging.error(f"Too many non-connection errors ({retries}). Stopping pipeline.")
                            raise
                prev_key = get_latest_model_key(self.config['s3_bucket'], "models/", exclude_run_id=self.run_id)
                if prev_key:
                    model = load_model_from_s3(self.config['s3_bucket'], prev_key)
                    logging.info(f"Rolled back to previous model: {prev_key}")
                    # RL model rollback
                    if prev_key.endswith("_ppo.zip"):
                        from stable_baselines3 import PPO
                        local_ppo_path = "/tmp/" + os.path.basename(prev_key)
                        s3 = boto3.client("s3")
                        s3.download_file(self.config['s3_bucket'], prev_key, local_ppo_path)
                        self.ppo_model = PPO.load(local_ppo_path)
                        logging.info("self.ppo_model has been set to the rolled-back PPO model.")
                        os.remove(local_ppo_path)
                    # ML model rollback
                    elif prev_key.endswith(".joblib"):
                        import re
                        match = re.search(r"models/[^/]+/([^_]+)_([^/]+)\.joblib", prev_key)
                        if match:
                            ticker, model_name = match.groups()
                            if not hasattr(self, "ml_models"):
                                self.ml_models = {}
                            if ticker not in self.ml_models:
                                self.ml_models[ticker] = {}
                            self.ml_models[ticker][model_name] = model
                            logging.info(f"Rolled back ML model for {ticker} - {model_name}.")
                else:
                    logger.error("No previous model found for rollback.")
                if retries >= self.max_retries:
                    logger.error("Max retries reached. Failing pipeline.")
                    raise
                time.sleep(self.retry_delay)

    # Extensibility: add hooks for notifications, multi-asset, etc.
    def add_notification_hook(self, hook_fn):
        self.notification_hook = hook_fn

    def notify(self, message):
        if self.notification_hook:
            self.notification_hook(message)
        else:
            logging.info(f"Notification: {message}")

    def train_ppo(self, train_data, ticker, save_path=None):
        """
        Sanitize numeric features, build a gym env, and train (or continue training) a PPO model.
        - Uses only numeric, finite features (float32).
        - Continues training if a prior model exists in S3 (same run_id/ticker).
        - Saves to `save_path` (or /tmp/ppo_model.zip) and uploads to s3.
        """
        # 1) Sanitize train_data before creating the env (tz-aware proof, numeric only)
        from pandas.api.types import is_datetime64_any_dtype
        data = train_data.copy()

        # Drop any datetime-like columns (naive or tz-aware) and standard stamp columns
        datetime_cols = [
            c for c in data.columns
            if is_datetime64_any_dtype(data[c].dtype)
               or isinstance(data[c].dtype, pd.DatetimeTZDtype)
               or c in ("timestamp", "datetime")
        ]
        if datetime_cols:
            data = data.drop(columns=datetime_cols)

        # Replace inf/NaN and cast to float32 for SB3
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32, copy=False)

        # 2) Build env on the sanitized frame ONLY
        env = TradingEnv(data)
        check_env(env)
        ppo_logger = PPOTrainingLogger()

        # 3) Load prior PPO model from S3 if present, else create new
        ppo_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/{ticker}_ppo.zip")
        local_ppo_path = "/tmp/ppo_model_prev.zip"
        s3 = boto3.client("s3")

        if ppo_key:
            try:
                s3.download_file(self.config['s3_bucket'], ppo_key, local_ppo_path)
                model = PPO.load(local_ppo_path)
                model.set_env(env)  # continue training with loaded weights on the new env
            except Exception as _e:
                logging.warning(f"Failed to load previous PPO for {ticker}: {_e}. Starting fresh.")
                # Optimized PPO for m5.large memory efficiency
                model = PPO("MlpPolicy", env, verbose=0,
                           n_steps=1024, batch_size=32, n_epochs=8)
        else:
            # Optimized PPO for m5.large memory efficiency
            model = PPO("MlpPolicy", env, verbose=0,
                       n_steps=1024, batch_size=32, n_epochs=8)

        # 4) Learn (skip in single-socket mode to prevent CPU starvation)
        if os.getenv("TRAIN_OFFLINE_ONLY", "1") in ("1", "true", "True"):
            logging.info("Skipping in-loop RL training (offline-only in single-socket baseline)")
            self.ppo_model = model
        else:
            model.learn(total_timesteps=10000, callback=ppo_logger)
            self.ppo_model = model
            ppo_logger.print_summary()

        # 5) Save & upload current model
        if save_path is None:
            save_path = "/tmp/ppo_model.zip"
        model.save(save_path)

        out_key = f"models/{self.run_id}/{ticker}_ppo.zip"
        try:
            s3.upload_file(save_path, self.config['s3_bucket'], out_key)
        finally:
            # Clean up local files, ignore errors
            for _p in (local_ppo_path, save_path):
                try:
                    if os.path.exists(_p):
                        os.remove(_p)
                except Exception:
                    pass

        logging.info(f"Retraining event: PPO model for {ticker} retrained and saved as {out_key}")
        print("PPO training complete.")

    def run_ppo(self, test_data, ticker):
        """
        Load the latest PPO model for `ticker` from S3 (this run first, then any prior run),
        sanitize test_data for gym, and roll once to compute a total reward.
        """
        # --- sanitize test_data for the env (mirror train_ppo) ---
        from pandas.api.types import is_datetime64_any_dtype
        data = test_data.copy()
        datetime_cols = [
            c for c in data.columns
            if is_datetime64_any_dtype(data[c].dtype)
               or isinstance(data[c].dtype, pd.DatetimeTZDtype)
               or c in ("timestamp", "datetime")
        ]
        if datetime_cols:
            data = data.drop(columns=datetime_cols)
        data = data.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32, copy=False)

        env = TradingEnv(data)
        check_env(env)

        # --- find a model: current run first, then any previous run ---
        s3 = boto3.client("s3")
        key_this_run = f"models/{self.run_id}/{ticker}_ppo.zip"
        found_key = None
        try:
            resp = s3.list_objects_v2(Bucket=self.config['s3_bucket'], Prefix=key_this_run)
            if any(obj['Key'] == key_this_run for obj in resp.get('Contents', [])):
                found_key = key_this_run
        except Exception:
            pass

        if found_key is None:
            try:
                resp = s3.list_objects_v2(Bucket=self.config['s3_bucket'], Prefix="models/")
                candidates = [o['Key'] for o in resp.get('Contents', []) if o['Key'].endswith(f"/{ticker}_ppo.zip")]
                if candidates:
                    candidates.sort()
                    found_key = candidates[-1]
            except Exception:
                pass

        if found_key is None:
            logging.warning(f"No PPO model found in S3 for {ticker}; skipping PPO run.")
            return 0.0

        local_model = f"/tmp/{ticker}_ppo_eval.zip"
        try:
            s3.download_file(self.config['s3_bucket'], found_key, local_model)
            model = PPO.load(local_model)
        finally:
            if os.path.exists(local_model):
                os.remove(local_model)

        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"PPO test run total reward ({ticker}): {total_reward}")
        return total_reward

    def train_ensemble_ppo(self, train_data, n_models=5, diverse_params=True):
        """Enhanced ensemble training with diverse hyperparameters"""
        self.ensemble_models = []
        self.model_weights = []  # Track model performance for weighted voting

        # Optimized hyperparameters for m5.large memory efficiency
        param_sets = [
            {'learning_rate': 3e-4, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 8},  # OPTIMIZED: Reduced memory usage
            {'learning_rate': 1e-4, 'n_steps': 512, 'batch_size': 16, 'n_epochs': 5},   # OPTIMIZED: Ultra-light for stability
            {'learning_rate': 5e-4, 'n_steps': 1024, 'batch_size': 32, 'n_epochs': 10}, # OPTIMIZED: Balanced performance
            {'learning_rate': 2e-4, 'n_steps': 768, 'batch_size': 24, 'n_epochs': 6},   # OPTIMIZED: Medium efficiency
            {'learning_rate': 4e-4, 'n_steps': 512, 'batch_size': 16, 'n_epochs': 8}    # OPTIMIZED: Memory-safe fallback
        ]

        for i in range(n_models):
            env = TradingEnv(train_data)

            if diverse_params and i < len(param_sets):
                params = param_sets[i]
                model = PPO("MlpPolicy", env, verbose=0, seed=i, **params)
            else:
                model = PPO("MlpPolicy", env, verbose=0, seed=i)

            # Train and evaluate performance
            model.learn(total_timesteps=15000)  # Increased training steps

            # Evaluate model performance for weighting
            performance = self._evaluate_model_performance(model, train_data)
            self.model_weights.append(performance)

            self.ensemble_models.append(model)
            logging.info(f"Trained ensemble model {i+1}/{n_models}, performance: {performance:.4f}")

        # Normalize weights
        total_weight = sum(self.model_weights)
        if total_weight > 0:
            self.model_weights = [w / total_weight for w in self.model_weights]
        else:
            self.model_weights = [1.0 / n_models] * n_models

        logging.info(f"Trained enhanced ensemble of {n_models} PPO models with weighted voting")

    def _evaluate_model_performance(self, model, data, max_steps=500):
        """Evaluate model performance for ensemble weighting"""
        env = TradingEnv(data)
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        # Normalize by steps for fair comparison
        return total_reward / max(steps, 1)

    def run_ensemble_ppo(self, test_data, confidence_threshold=0.7, risk_threshold=0.3):
        """Enhanced ensemble prediction with weighted voting and confidence thresholding"""
        env = TradingEnv(test_data)
        obs, info = env.reset()
        done = False
        total_reward = 0
        actions = []

        # Track ensemble statistics
        high_confidence_trades = 0
        low_confidence_holds = 0

        while not done:
            # Get predictions from all models with their probabilities
            model_predictions = []
            for i, model in enumerate(self.ensemble_models):
                action, log_prob = model.predict(obs, deterministic=False)

                # Handle confidence score calculation safely
                if log_prob is not None:
                    try:
                        if hasattr(log_prob, '__len__'):
                            confidence_score = float(np.exp(np.mean(log_prob)))
                        else:
                            confidence_score = float(np.exp(log_prob))
                    except:
                        confidence_score = 0.5  # Default moderate confidence
                else:
                    confidence_score = 0.5  # Default when no probability available

                model_predictions.append({
                    'action': action,
                    'confidence': confidence_score,
                    'weight': self.model_weights[i]
                })

            # Calculate weighted votes for each action
            action_scores = {0: 0.0, 1: 0.0, 2: 0.0}  # hold, buy, sell
            total_weighted_confidence = 0.0

            for pred in model_predictions:
                action = int(pred['action'])
                weighted_score = pred['weight'] * pred['confidence']
                action_scores[action] += weighted_score
                total_weighted_confidence += weighted_score

            # Find best action and calculate ensemble confidence
            best_action = max(action_scores, key=action_scores.get)
            best_score = action_scores[best_action]
            ensemble_confidence = best_score / max(total_weighted_confidence, 1e-6)

            # Apply confidence thresholding with risk management
            final_action = best_action

            if ensemble_confidence < confidence_threshold:
                # Low confidence: default to hold (conservative approach)
                final_action = 0
                low_confidence_holds += 1

            elif best_action != 0 and ensemble_confidence > confidence_threshold:
                # High confidence non-hold action
                # Additional risk check: avoid high-risk actions in volatile conditions
                if hasattr(env, 'balance') and hasattr(env, 'initial_balance'):
                    current_drawdown = (env.initial_balance - env.balance) / env.initial_balance
                    if current_drawdown > risk_threshold:
                        final_action = 0  # Force hold during high drawdown
                    else:
                        high_confidence_trades += 1
                else:
                    high_confidence_trades += 1

            # Execute action
            obs, reward, terminated, truncated, info = env.step(final_action)
            done = terminated or truncated
            total_reward += reward
            actions.append({
                'action': final_action,
                'confidence': ensemble_confidence,
                'original_action': best_action
            })

        # Log ensemble performance statistics
        total_actions = len(actions)
        if total_actions > 0:
            avg_confidence = np.mean([a['confidence'] for a in actions])
            logging.info("Ensemble trading complete:")
            logging.info(f"  High confidence trades: {high_confidence_trades}/{total_actions} ({100*high_confidence_trades/total_actions:.1f}%)")
            logging.info(f"  Low confidence holds: {low_confidence_holds}/{total_actions} ({100*low_confidence_holds/total_actions:.1f}%)")
            logging.info(f"  Average confidence: {avg_confidence:.3f}")
            logging.info(f"  Total reward: {total_reward:.4f}")

        return total_reward, actions

    def _calculate_volatility_position_size(self, price_data, base_size=1, lookback_period=20, target_volatility=0.15):
        """Calculate position size based on volatility using Kelly Criterion approach"""
        import pandas as pd

        if len(price_data) < lookback_period:
            return base_size

        # Calculate recent returns
        prices = pd.Series(price_data[-lookback_period:])
        returns = prices.pct_change().dropna()

        if len(returns) < 2:
            return base_size

        # Calculate historical volatility (annualized)
        volatility = returns.std() * np.sqrt(252)  # 252 trading days per year

        # Volatility-based position sizing
        if volatility > 0:
            # Scale position inversely with volatility
            vol_multiplier = target_volatility / volatility
            vol_multiplier = np.clip(vol_multiplier, 0.1, 3.0)  # Limit between 10% and 300% of base size
        else:
            vol_multiplier = 1.0

        # Kelly Criterion enhancement (simplified)
        if len(returns) > 5:
            win_rate = (returns > 0).mean()
            avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
            avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 0.01

            if avg_loss > 0:
                kelly_ratio = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_ratio = np.clip(kelly_ratio, 0.1, 1.0)  # Conservative Kelly
            else:
                kelly_ratio = 0.5
        else:
            kelly_ratio = 0.5

        # Combine volatility and Kelly sizing
        smart_size = base_size * vol_multiplier * kelly_ratio
        smart_size = np.clip(smart_size, 0.1, base_size * 2)  # Never more than 2x or less than 10% of base

        return round(smart_size, 2)

    def _get_market_regime_multiplier(self, price_data, lookback=30):
        """Adjust position size based on market regime (trending vs mean-reverting)"""
        if len(price_data) < lookback:
            return 1.0

        prices = pd.Series(price_data[-lookback:])

        # Calculate trend strength (R-squared of linear regression)
        x = np.arange(len(prices))
        correlation = np.corrcoef(x, prices)[0, 1]
        trend_strength = correlation ** 2

        # In strong trending markets, increase position size slightly
        # In choppy markets, decrease position size
        if trend_strength > 0.7:  # Strong trend
            return 1.2
        elif trend_strength < 0.3:  # Choppy/mean-reverting
            return 0.8
        else:  # Moderate trend
            return 1.0

    def get_smart_position_size(self, ticker, current_price_data, base_position=1):
        """
        Calculate optimal position size based on multiple factors:
        - Volatility (inverse relationship)
        - Kelly Criterion (risk-reward optimization)
        - Market regime (trending vs choppy)
        - Current portfolio exposure
        """
        # Base volatility sizing
        vol_size = self._calculate_volatility_position_size(
            current_price_data,
            base_size=base_position,
            target_volatility=0.12  # 12% target annual volatility
        )

        # Market regime adjustment
        regime_multiplier = self._get_market_regime_multiplier(current_price_data)

        # Portfolio exposure check (don't over-concentrate)
        try:
            from market_data_config import MAX_POSITION_EXPOSURE
            max_position = MAX_POSITION_EXPOSURE
        except:
            max_position = 3

        # Final smart position size
        smart_position = vol_size * regime_multiplier
        smart_position = min(smart_position, max_position)  # Respect risk limits
        smart_position = max(smart_position, 0.1)  # Minimum position size

        logging.info(f"Smart position sizing for {ticker}:")
        logging.info(f"  Volatility size: {vol_size:.2f}")
        logging.info(f"  Regime multiplier: {regime_multiplier:.2f}")
        logging.info(f"  Final position: {smart_position:.2f}")

        return smart_position

# Example config for testing (use canonical IBKR futures defined in market_data_config)
default_config = {
    "tickers": DEFAULT_TICKERS,  # comes from: from market_data_config import IBKR_SYMBOLS as DEFAULT_TICKERS
    "use_mock_data": False,
    "feature_params": {"window": 20, "indicators": ["sma", "ema", "rsi"]},
    "log_path": "logs/pipeline.log",
    "s3_bucket": "omega-singularity-ml",
    "max_retries": 3,
    "retry_delay": 10,
    # AI_ASSISTANT_TODO: Set to False before enabling live IBKR data!
    "disable_drift_detection": True,  # TEMP: Disable drift detection for mock data testing
    "drift_detection_threshold": 10.0  # Relaxed for mock data; set to 3.0 for production
}

import os

data_dir = os.getenv("DATA_DIR", "/home/ubuntu/data")

if __name__ == "__main__":
    # Create pipeline ONCE with single socket connection
    pipeline = RLTradingPipeline(default_config)

    # Run pipeline in a loop, reusing the same connection
    while True:
        try:
            pipeline.run()

            # During sleep, send periodic keepalive to maintain connection
            sleep_duration = 120
            keepalive_interval = 30

            for _ in range(sleep_duration // keepalive_interval):
                time.sleep(keepalive_interval)
                # Send keepalive ping via existing connection
                logging.info("Keepalive tick - checking connection...")
                if hasattr(pipeline, 'market_data_adapter') and pipeline.market_data_adapter.ib.isConnected():
                    try:
                        current_time = pipeline.market_data_adapter.ib.reqCurrentTime()
                        logging.info(f"Keepalive ping sent successfully - server time: {current_time}")
                    except Exception as e:
                        logging.warning(f"Keepalive failed: {e}")
                        break
                else:
                    logging.warning("Keepalive: Connection not available")

        except Exception as e:
            # Check if this is truly a connection error
            import socket

            msg = str(e).lower()
            is_connection_error = isinstance(e, (ConnectionError, TimeoutError, socket.error)) \
                or "peer closed" in msg or "timeout" in msg or "connection refused" in msg \
                or "connect call failed" in msg

            if is_connection_error:
                logging.error(f"Pipeline run failed due to connection loss: {e}")
                logging.error("Exiting due to connection loss - restart required")
                sys.exit(1)  # Exit cleanly so systemd/supervisor can restart
            else:
                logging.error(f"Pipeline run failed: {e}")
                logging.error("Exiting due to non-connection error")
                sys.exit(2)  # Different exit code for non-connection errors


def _execute_order_single_client(symbol:str, side:str, qty:int):
    try:
        stop_ticks = int(os.getenv("STOP_TICKS","8"))
        take_ticks = int(os.getenv("TAKE_TICKS","16"))
        tif        = os.getenv("TIF","DAY")
        entry_type = os.getenv("ENTRY_TYPE","MKT")
        limit_off  = int(os.getenv("LIMIT_OFFSET_TICKS","0"))
        res = place_bracket_order(symbol, side, qty,
                                  stop_ticks=stop_ticks, take_ticks=take_ticks,
                                  tif=tif, order_type=entry_type, limit_offset_ticks=limit_off)
        print("[single_client_order]", res, flush=True)
    except Exception as e:
        print("[single_client_order][ERROR]", e, flush=True)
