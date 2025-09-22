import logging
import os
import uuid
from pathlib import Path

from datetime import datetime, timezone
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

from monitoring.omni_monitor import emit_event

try:
    import torch  # type: ignore
    from ml.sequence_forecaster import (
        SequenceForecasterArtifact,
        load_artifact as load_sequence_artifact,
    )
    from ml.trainers.fit_sequence_forecaster import fit_seq_forecaster_for_symbol
except Exception as _seq_import_exc:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    load_sequence_artifact = None  # type: ignore
    SequenceForecasterArtifact = None  # type: ignore
    fit_seq_forecaster_for_symbol = None  # type: ignore
    logging.getLogger(__name__).info("Sequence forecaster unavailable: %s", _seq_import_exc)

try:
    from utils.alerts import send_sms_alert
except Exception:
    def send_sms_alert(msg: str):
        logging.info(f"[ALERT] {msg}")

try:
    from market_data_utils import RealTimeFeatureStore
except Exception:
    class RealTimeFeatureStore:
        def __init__(self):
            self._store = {}
        def update(self, key, features_dict):
            import pandas as pd
            # Store as a growing DataFrame; coerce to numeric where possible
            df = self._store.get(key)
            if isinstance(features_dict, pd.DataFrame):
                row = features_dict.reset_index(drop=True)
            else:
                row = pd.DataFrame([features_dict])
            self._store[key] = row if df is None else pd.concat([df, row], ignore_index=True).infer_objects(copy=False)
        def get(self, key):
            return self._store.get(key)
        def clear(self):
            self._store.clear()
feature_store = RealTimeFeatureStore()

from market_data_config import MAX_POSITION_EXPOSURE
from ib_single_socket import init_ib
import datetime
from market_data_config import MAX_DAILY_LOSS_PCT, MAX_TRADES_PER_DAY
from ib_insync import MarketOrder, util
from account_summary_lookahead import AccountSummaryLookahead
from market_data_ibkr_adapter import IBKRIngestor
from utils.persist_market_data import persist_bars
from market_data_config import IBKR_SYMBOLS
TICKERS = IBKR_SYMBOLS
import psutil
from feature_engineering import classify_news_type
from sklearn.linear_model import LogisticRegression
import joblib
import os as os_mod
import numpy as np
import random
import math
import json
import hashlib
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from collections import deque, defaultdict
from typing import List, Dict, Any, Tuple, Optional
import boto3
import io
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict
from feature_engineering import generate_features, add_technical_indicators, process_ticker, list_s3_files, read_s3_file
try:
    from ingest_to_s3 import upload_file_to_s3
except Exception:
    def upload_file_to_s3(local_path: str, bucket: str, key: str):
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, bucket, key)

from market_data_ibkr_adapter import IBKRIngestor

from feature_engineering import generate_features
from audit_logging_utils import log_trade_results

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from colorlog import ColoredFormatter

import warnings
from stable_baselines3.common.callbacks import BaseCallback

import logging
logger = logging.getLogger(__name__)

LAST_TRAINED_FILE = "last_trained_time.txt"


def _coerce_epoch(value: Any) -> Optional[int]:
    """Best-effort conversion of persisted timestamps into epoch seconds."""
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(value)

    if isinstance(value, datetime):
        return int(value.timestamp())

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return int(float(candidate))
        except ValueError:
            try:
                dt = date_parser.parse(candidate)
            except Exception:
                logger.warning("Unable to parse last trained timestamp '%s'", value)
                return None
            return int(dt.timestamp())

    logger.warning("Unsupported last trained timestamp type: %r", type(value))
    return None


def _ppo_observation_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return OHLCV-only frame suitable for PPO models."""

    base_cols = [col for col in ("open", "high", "low", "close", "volume") if col in df.columns]
    if len(base_cols) < 5:
        return None

    frame = df[base_cols].copy()
    for col in base_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna()
    return frame

try:
    from news_ingestion_marketaux import fetch_marketaux_news, normalize_marketaux_news
except Exception:
    def fetch_marketaux_news(*a, **k):
        import pandas as pd
        return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])
    def normalize_marketaux_news(df):
        return df
try:
    from news_data_utils import engineer_news_features
except Exception:
    def engineer_news_features(df):
        return df

# Ensure the health monitor stays passive in single-socket mode; any
# additional socket probes can cause the gateway to reset the API port.
os.environ.setdefault("DISABLE_HEALTH_MONITOR", "1")

from market_data_config import (
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID, IBKR_SYMBOLS, IBKR_LOG_LEVEL,
    IBKR_RETRY_INTERVAL, IBKR_TIMEOUT, IBKR_MARKET_DATA_TYPE,
    IBKR_LOGGING_ENABLED, IBKR_LOG_FILE, IBKR_DYNAMIC_SUBSCRIPTION,
    IBKR_RESOURCE_MONITORING, IBKR_POLL_INTERVAL
)

# Allocate dedicated client IDs for auxiliary sessions so that we never
# collide with the primary market-data channel (IBKR_CLIENT_ID). IBKR closes
# duplicate client IDs immediately ("Connection reset"), which is exactly what
# we observed when the news fetch and the main ingestor both used the same ID.
# Offsetting keeps the original ID for the trading/market-data session while
# giving deterministic IDs to short-lived helpers.
IBKR_NEWS_CLIENT_ID = IBKR_CLIENT_ID + 100
IBKR_AUX_CLIENT_ID = IBKR_CLIENT_ID + 101
IBKR_CONNECT_TIMEOUT = int(os.getenv('IBKR_CONNECT_TIMEOUT', '30'))

# ---- IBKR news helpers (historical news + article text) ----
def fetch_ibkr_news_for_tickers(tickers, lookback_hours=24, max_results=200):
    """
    Best-effort IBKR Historical News fetch for the given tickers.
    Works with whatever news providers your account is entitled to.
    Returns a pandas.DataFrame with columns:
    ['published_at','title','description','tickers','provider','source']
    """
    import re
    import pandas as pd
    from datetime import datetime, timedelta
    from ib_insync import IB
    from market_data_ibkr_adapter import IBKRIngestor

    rows = []

    # Connect (separate lightweight session for news)
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_NEWS_CLIENT_ID, timeout=IBKR_CONNECT_TIMEOUT)
    except Exception:
        # If IB Gateway isn’t ready, return empty DF; pipeline will continue with MarketAux
        return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])

    # Discover provider codes you have access to
    try:
        providers = ib.reqNewsProviders() or []
    except Exception:
        providers = []
    provider_codes = ",".join(p.code for p in providers) if providers else ""
    if not provider_codes:
        ib.disconnect()
        return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])

    # Use your existing adapter to resolve exact (qualified) contracts quickly
    # Reuse the lightweight IB connection for contract discovery to avoid
    # spinning up another persistent socket. We still assign a dedicated client
    # ID so that, if the adapter ever needs to originate API requests, it does
    # not collide with the primary trading session.
    ingestor = IBKRIngestor(
        host=IBKR_HOST,
        port=IBKR_PORT,
        clientId=IBKR_AUX_CLIENT_ID,
        ib=ib
    )

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(hours=lookback_hours)

    for tvsym in tickers:
        try:
            contract = ingestor._get_contract(tvsym)  # nearest non‑expired, qualified
            conId = contract.conId
            items = ib.reqHistoricalNews(
                conId=conId,
                providerCodes=provider_codes,
                startDateTime=start_dt,
                endDateTime=end_dt,
                totalResults=max_results,
                options=[]
            ) or []
            for it in items:
                # Pull full article text when available
                try:
                    article = ib.reqNewsArticle(it.providerCode, it.articleId)
                    text = getattr(article, "articleText", "") or ""
                except Exception:
                    text = ""
                # strip simple html if present
                text_clean = re.sub(r"<[^>]+>", " ", text).strip()
                # build row
                rows.append({
                    "published_at": pd.to_datetime(getattr(it, "time", end_dt), utc=True),
                    "title": getattr(it, "headline", "") or "",
                    "description": text_clean if text_clean else getattr(it, "headline", ""),
                    "tickers": [tvsym],
                    "provider": it.providerCode,
                    "source": "IBKR"
                })
        except Exception:
            # keep going on ticker errors; we’re in learning mode
            continue

    ib.disconnect()
    if not rows:
        return pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])
    df = pd.DataFrame(rows).sort_values("published_at")
    return df

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
last_retrain_time = 0

def detect_regime(price_series, window=100):
    returns = price_series.pct_change().dropna()
    rolling_vol = returns.rolling(window).std()
    high_vol = rolling_vol > rolling_vol.median() * 1.5
    if high_vol.iloc[-1]:
        return "high_vol"
    else:
        return "normal"

def new_data_available(data_dir, ticker, last_check_time):
    """Return True if a newer Parquet shard exists for this ticker.

    Parquet layout (env-first):
      $DATA_DIR/symbol=<ticker>/date=YYYY-MM-DD/bars.parquet
    """
    base = Path(data_dir) / f"symbol={ticker}"
    if not base.exists():
        return False
    parts = sorted(base.rglob("bars.parquet"))
    if not parts:
        return False
    latest = parts[-1]
    try:
        mtime = os_mod.path.getmtime(latest)
    except Exception:
        return False
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
    trade = ib.placeOrder(contract, order)
    print(f"Order submitted: {action} {quantity} {contract.localSymbol}")
    return trade

# Default data dir for new_data_available checks (env-first)
data_dir = os_mod.getenv("DATA_DIR", str(Path.home() / ".local/share/m5_trader/data"))

# === Combined News Ingestion (MarketAux + IBKR) ===
_news_env_flag = os_mod.getenv("ENABLE_IBKR_NEWS", os_mod.getenv("ENABLE_NEWS_ANALYSIS", "0"))
_news_enabled = _news_env_flag.lower() in ("1", "true", "yes")


def fetch_union_news(
    symbols: Optional[List[str]] = None,
    *,
    lookback_hours: int = 24,
    max_results: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbols = symbols or IBKR_SYMBOLS
    entity_types = "indices,commodities,currencies,futures"

    try:
        ma_list = fetch_marketaux_news(
            symbols=symbols,
            limit=20,
            entity_types=entity_types,
            countries="us",
            language="en",
            filter_entities="false",
            must_have_entities="false",
            min_match_score=0.05,
        )
        ma_raw = normalize_marketaux_news(ma_list)
    except Exception as exc:
        logging.warning(f"MarketAux fetch failed: {exc}")
        ma_raw = None

    ma_df = normalize_marketaux_for_union(ma_raw)

    try:
        ibkr_df = fetch_ibkr_news_for_tickers(symbols, lookback_hours=lookback_hours, max_results=max_results)
    except Exception as exc:
        logging.warning(f"IBKR news fetch failed: {exc}")
        ibkr_df = pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])

    try:
        news_df = pd.concat([ma_df, ibkr_df], ignore_index=True)
        if not news_df.empty:
            news_df['published_min'] = news_df['published_at'].dt.floor('min')
            news_df = news_df.drop_duplicates(subset=['published_min', 'title']).drop(columns=['published_min'])
    except Exception as exc:
        logging.warning(f"News analysis disabled due to error: {exc}")
        news_df = pd.DataFrame()

    return ma_df, ibkr_df, news_df


# News placeholders (updated inside the main loop when enabled)
ma_df = pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])
ibkr_df = pd.DataFrame(columns=['published_at','title','description','tickers','provider','source'])
news_df = pd.DataFrame()

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

def _s3_enabled() -> bool:
    try:
        return str(os_mod.getenv("S3_ENABLE", "0")).lower() in ("1", "true", "yes")
    except Exception:
        return False


MODELS_ROOT_DIR = Path(os_mod.getenv("MODELS_DIR", str(Path.home() / "models"))).expanduser()
LOCAL_STACKING_DIR = MODELS_ROOT_DIR / "stacking"
LOCAL_PPO_DIR = MODELS_ROOT_DIR / "ppo"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_feature_signature(columns: List[str]) -> str:
    normalized = [c.strip().lower() for c in columns]
    joined = "|".join(normalized)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]


def ppo_filename(ticker: str, regime_suffix: str, signature: str) -> str:
    return f"{ticker}_ppo_{regime_suffix}__{signature}.zip"


def legacy_ppo_filename(ticker: str, regime_suffix: str) -> str:
    return f"{ticker}_ppo_{regime_suffix}.zip"


def save_model_local(model, run_id: str, filename: str) -> Path:
    run_dir = _ensure_dir(LOCAL_STACKING_DIR / str(run_id))
    dest = run_dir / filename
    joblib.dump(model, dest)
    logging.info("Saved local model artifact: %s", dest)
    return dest


def load_local_models(run_id: str) -> Dict[str, Dict[str, Any]]:
    models: Dict[str, Dict[str, Any]] = {}
    run_dir = LOCAL_STACKING_DIR / str(run_id)

    if not run_dir.exists():
        candidates = sorted([p for p in LOCAL_STACKING_DIR.glob("*") if p.is_dir()], reverse=True)
        for candidate in candidates:
            if candidate.name != str(run_id):
                logging.info("No local models for run %s; falling back to %s", run_id, candidate.name)
                run_dir = candidate
                break
        else:
            logging.info("No local model directory available under %s", LOCAL_STACKING_DIR)
            return models

    for path in run_dir.glob("*.joblib"):
        match = re.match(r"(.+?)_([A-Za-z0-9]+)\.joblib$", path.name)
        if not match:
            continue
        ticker, model_name = match.groups()
        try:
            model = joblib.load(path)
        except Exception as exc:
            logging.warning("Failed to load local model %s: %s", path, exc)
            continue
        models.setdefault(ticker, {})[model_name] = model

    for path in run_dir.glob("*_ppo*.zip"):
        match = re.match(r"(.+?)_ppo__(?P<sig>[a-f0-9]+)\.zip$", path.name)
        signature = None
        if match:
            ticker = match.group(1)
            signature = match.group('sig')
        else:
            legacy_match = re.match(r"(.+?)_ppo\.zip$", path.name)
            if not legacy_match:
                continue
            ticker = legacy_match.group(1)
        try:
            model = PPO.load(str(path))
        except Exception as exc:
            logging.warning("Failed to load local PPO model %s: %s", path, exc)
            continue
        if signature:
            setattr(model, "_feature_signature", signature)
        models.setdefault(ticker, {})["PPO"] = model

    return models


def get_latest_local_artifact(exclude_run_id: Optional[str] = None) -> Optional[Path]:
    """Return the most recent local model artifact outside the current run."""
    if not LOCAL_STACKING_DIR.exists():
        return None

    latest_path: Optional[Path] = None
    latest_mtime: float = -1.0

    for run_dir in sorted(LOCAL_STACKING_DIR.glob("*"), reverse=True):
        if not run_dir.is_dir():
            continue
        if exclude_run_id and run_dir.name == str(exclude_run_id):
            continue
        for path in run_dir.glob("*"):
            if path.suffix not in (".joblib", ".zip"):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = path

    return latest_path


def load_local_ppo_model(ticker: str) -> Optional[PPO]:
    """Load the most recent local PPO model for a ticker."""
    candidates = sorted(
        LOCAL_PPO_DIR.glob(f"{ticker}_ppo_*.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            model = PPO.load(str(path))
            return model
        except Exception as exc:
            logging.warning("Failed to load local PPO model %s: %s", path, exc)
    return None


def save_model_to_s3(model, bucket, key):
    """Save a model to S3 using joblib. No-op when S3_ENABLE=0."""
    if not _s3_enabled():
        logging.info("S3 disabled (S3_ENABLE=0); skipping save to %s/%s", bucket, key)
        return
    local_path = f"/tmp/{os_mod.path.basename(key)}"
    joblib.dump(model, local_path)
    s3 = boto3.client("s3")
    s3.upload_file(local_path, bucket, key)
    os_mod.remove(local_path)

def load_model_from_s3(bucket, key):
    """Load a model from S3 using joblib. Returns None when S3 is disabled."""
    if not _s3_enabled():
        logging.info("S3 disabled (S3_ENABLE=0); skipping load of %s/%s", bucket, key)
        return None
    local_path = f"/tmp/{os_mod.path.basename(key)}"
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, local_path)
    model = joblib.load(local_path)
    os_mod.remove(local_path)
    return model

def get_latest_model_key(bucket, prefix, exclude_run_id=None):
    """Get the latest model key from S3; returns None when S3 is disabled."""
    if not _s3_enabled():
        return None
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    keys = [obj['Key'] for obj in response.get('Contents', [])]
    if exclude_run_id:
        keys = [k for k in keys if exclude_run_id not in k]
    if not keys:
        return None
    return sorted(keys)[-1]  # Latest by name (run_id)


def get_latest_model_key_global(bucket, filename_suffix, exclude_run_id=None):
    """Locate the most recent S3 model key across all runs matching the filename suffix."""
    if not _s3_enabled():
        return None

    s3 = boto3.client("s3")
    paginator = s3.get_paginator('list_objects_v2')
    latest_key = None
    latest_time = None

    for page in paginator.paginate(Bucket=bucket, Prefix="models/"):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if exclude_run_id and f"/{exclude_run_id}/" in key:
                continue
            if not key.endswith(filename_suffix):
                continue
            lm = obj.get('LastModified')
            if latest_time is None or (lm and lm > latest_time):
                latest_time = lm
                latest_key = key

    return latest_key

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

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 1 for long, -1 for short, 0 for flat
        self.transaction_cost = 0.0005  # 5 bps per trade, adjust as needed
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        self.risk_penalty_coeff = 0.1  # adjust for risk aversion
        self.action_space = gym.spaces.Discrete(3)  # [0: hold, 1: buy, 2: sell]
        # Drop datetime columns from data for observation space
        from pandas.api.types import is_datetime64_any_dtype

        self.numeric_columns = [
            col for col in self.data.columns
            if not is_datetime64_any_dtype(self.data[col])
        ]
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

import re
import joblib
import boto3
import os as os_mod
from stable_baselines3 import PPO

def load_all_models(bucket, run_id):
    """
    Load all base models for stacking/ensemble. Prioritises local registry and
    augments with S3 artifacts when enabled.
    """
    models = load_local_models(run_id)

    if not _s3_enabled():
        if not models:
            logging.warning("S3 disabled and no local models found for run %s", run_id)
        return models

    s3 = boto3.client("s3")
    prefix = f"models/{run_id}/"
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = os_mod.path.basename(key)
        # ML models: {ticker}_{modelname}.joblib
        m = re.match(r"(.+?)_(RandomForest|XGBoost|LightGBM|SGDClassifier|Perceptron|PassiveAggressive)\.joblib$", filename)
        if m:
            ticker, model_name = m.groups()
            local_path = f"/tmp/{filename}"
            s3.download_file(bucket, key, local_path)
            model = joblib.load(local_path)
            os_mod.remove(local_path)
            models.setdefault(ticker, {})[model_name] = model
            continue

        # PPO RL models: {ticker}_ppo.zip
        m2 = re.match(r"(.+?)_ppo\.zip$", filename)
        if m2:
            ticker = m2.group(1)
            local_path = f"/tmp/{filename}"
            s3.download_file(bucket, key, local_path)
            model = PPO.load(local_path)
            os_mod.remove(local_path)
            models.setdefault(ticker, {})["PPO"] = model

    return models

class RLTradingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        shared_ib = init_ib(host=IBKR_HOST, port=IBKR_PORT, client_id=IBKR_CLIENT_ID)
        self.market_data_adapter = IBKRIngestor(
            host=IBKR_HOST,
            port=IBKR_PORT,
            clientId=shared_ib.client.clientId,
            ib=shared_ib
        )
        # Attach passive Account Summary lookahead cache to the live IB session
        try:
            ib_ref = getattr(self.market_data_adapter, 'ib', None)
            if ib_ref is not None:
                self.acct_la = AccountSummaryLookahead(ib_ref)
        except Exception as _e:
            logging.info(f"[acct_la] init skipped: {_e}")
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 10)
        self.max_drawdown_limit = config.get('max_drawdown_limit', 0.2)  # 20% default
        self.max_position_size = config.get('max_position_size', 1)      # 1 contract/lot default
        self.min_samples_per_model = int(os_mod.getenv(
            'ML_TRAINING_MIN_SAMPLES',
            str(config.get('min_samples_per_model', 1440))
        ))
        self.ml_full_min_samples = int(os_mod.getenv(
            'ML_FULL_MIN_SAMPLES',
            str(config.get('ml_full_min_samples', 1440))
        ))
        self.ml_incremental_min_samples = int(os_mod.getenv(
            'ML_INCREMENTAL_MIN_SAMPLES',
            str(config.get('ml_incremental_min_samples', 120))
        ))
        self.ml_rolling_window = int(os_mod.getenv(
            'ML_ROLLING_WINDOW',
            str(config.get('ml_rolling_window', 1440))
        ))
        self.ml_retrain_every_min = int(os_mod.getenv(
            'ML_RETRAIN_EVERY_MIN',
            str(config.get('ml_retrain_every_min', 60))
        ))
        self.run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.pipeline_corr_id = f"{self.run_id}:{uuid.uuid4().hex}"
        self.local_model_dir = _ensure_dir(LOCAL_STACKING_DIR / self.run_id)
        _ensure_dir(LOCAL_PPO_DIR)
        self.ml_model_store: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.ml_feature_columns: Dict[str, List[str]] = defaultdict(list)
        self.ml_last_incremental_rows: Dict[str, int] = defaultdict(int)
        self.ml_last_incremental_time: Dict[str, datetime] = defaultdict(lambda: datetime(1970, 1, 1))
        self.ml_last_full_retrain_time: Dict[str, datetime] = defaultdict(lambda: datetime(1970, 1, 1))
        self.ppo_live_positions: Dict[str, int] = defaultdict(int)
        self.meta_daily_state: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"day": None, "trades": 0})
        self.micropoll_enable = str(os_mod.getenv('MICROPOLL_ENABLE', '0')).lower() in ('1', 'true', 'yes')
        self.micropoll_every_s = int(os_mod.getenv('MICROPOLL_EVERY_S', '10'))
        self.micropoll_window = os_mod.getenv('MICROPOLL_WINDOW', '90 S')
        self.micropoll_bar_size = os_mod.getenv('MICROPOLL_BAR_SIZE', '5 secs')
        self.micropoll_what = os_mod.getenv('MICROPOLL_WHAT', 'MIDPOINT')
        self._last_poll_ts: Dict[str, Any] = {}
        if self.micropoll_enable:
            try:
                import pandas as pd
                base = Path(os_mod.getenv("DATA_DIR", str(Path.home()/".local/share/m5_trader/data")))
                for t in ["ES1!", "NQ1!", "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD"]:
                    parts = sorted((base / f"symbol={t}").rglob("bars.parquet"))
                    if parts:
                        latest = pd.read_parquet(parts[-1])
                        if not latest.empty and "timestamp" in latest.columns:
                            self._last_poll_ts[t] = pd.to_datetime(latest["timestamp"]).max()
            except Exception as _seed_e:
                logging.info(f"[micropoll] seed last_ts skipped: {_seed_e}")
        self.status = "initialized"
        self.last_error = None
        self.notification_hook = None  # Extensibility: notification system

        # Sequence forecaster integration (PyTorch-based supervised models)
        self.seq_torch = torch
        self._load_seq_artifact = load_sequence_artifact
        self._fit_seq_forecaster = fit_seq_forecaster_for_symbol
        self.seq_enabled = (
            self.seq_torch is not None
            and self._load_seq_artifact is not None
        )
        logging.info("Sequence forecaster enabled flag=%s", self.seq_enabled)
        self.seq_model_dir = Path(
            os_mod.getenv("SEQ_FORECASTER_DIR", str(MODELS_ROOT_DIR / "seq_forecaster"))
        ).expanduser()
        self.seq_models: Dict[str, Optional[SequenceForecasterArtifact]] = {}
        self.seq_feature_names: List[str] = []
        if self.seq_enabled:
            device_str = os_mod.getenv("SEQ_FORECASTER_DEVICE")
            try:
                self.seq_device = self.seq_torch.device(device_str) if device_str else self.seq_torch.device(
                    "cuda" if self.seq_torch.cuda.is_available() else "cpu"
                )
            except Exception as exc:
                logging.warning("Sequence forecaster device fallback to CPU: %s", exc)
                self.seq_device = self.seq_torch.device("cpu")
            logging.info(
                "Sequence forecaster enabled (dir=%s, device=%s)", self.seq_model_dir, self.seq_device
            )
        else:
            self.seq_device = None
            logging.info("Sequence forecaster disabled (missing torch or artifact loader).")

        self.seq_training_enabled = (
            self.seq_enabled
            and callable(self._fit_seq_forecaster)
            and str(os_mod.getenv("SEQ_FORECASTER_TRAIN", "1")).lower() in ("1", "true", "yes")
        )

        def _parse_int_env(name: str, default: int) -> int:
            value = os_mod.getenv(name)
            if value is None or value == "":
                return default
            try:
                return int(value)
            except ValueError:
                logging.warning("Invalid int for %s=%s; using default %s", name, value, default)
                return default

        def _parse_float_env(name: str, default: float) -> float:
            value = os_mod.getenv(name)
            if value is None or value == "":
                return default
            try:
                return float(value)
            except ValueError:
                logging.warning("Invalid float for %s=%s; using default %s", name, value, default)
                return default

        horizons_env = os_mod.getenv("SEQ_FORECASTER_HORIZONS", "1,5,15")
        try:
            parsed = [int(part.strip()) for part in horizons_env.split(",") if part.strip()]
        except ValueError:
            logging.warning("Invalid SEQ_FORECASTER_HORIZONS=%s; using defaults.", horizons_env)
            parsed = [1, 5, 15]
        self.seq_horizons = tuple(sorted({h for h in parsed if h > 0})) or (1, 5, 15)

        self.seq_n_steps = max(_parse_int_env("SEQ_FORECASTER_N_STEPS", 150), 10)
        self.seq_step = max(_parse_int_env("SEQ_FORECASTER_STEP", 1), 1)
        self.seq_train_split = min(max(_parse_float_env("SEQ_FORECASTER_TRAIN_SPLIT", 0.7), 0.1), 0.95)
        self.seq_val_split = min(max(_parse_float_env("SEQ_FORECASTER_VAL_SPLIT", 0.15), 0.05), 0.8)
        if self.seq_train_split + self.seq_val_split >= 0.98:
            self.seq_val_split = max(0.05, 0.98 - self.seq_train_split)

        self.seq_hidden_size = max(_parse_int_env("SEQ_FORECASTER_HIDDEN_SIZE", 128), 16)
        self.seq_num_layers = max(_parse_int_env("SEQ_FORECASTER_NUM_LAYERS", 2), 1)
        self.seq_dropout = min(max(_parse_float_env("SEQ_FORECASTER_DROPOUT", 0.2), 0.0), 0.9)
        self.seq_epochs = max(_parse_int_env("SEQ_FORECASTER_EPOCHS", 10), 1)
        self.seq_batch_size = max(_parse_int_env("SEQ_FORECASTER_BATCH_SIZE", 128), 32)
        self.seq_lr = _parse_float_env("SEQ_FORECASTER_LR", 3e-4)
        self.seq_weight_decay = _parse_float_env("SEQ_FORECASTER_WEIGHT_DECAY", 0.0)
        self.seq_patience = max(_parse_int_env("SEQ_FORECASTER_PATIENCE", 3), 1)
        self.seq_min_train_rows = max(_parse_int_env("SEQ_FORECASTER_MIN_ROWS", max(self.seq_n_steps * 2, 1000)), self.seq_n_steps + 10)
        self.seq_retrain_every_min = max(_parse_int_env("SEQ_FORECASTER_RETRAIN_MIN", 180), 1)

        seed_env = os_mod.getenv("SEQ_FORECASTER_SEED", "42")
        try:
            self.seq_seed: Optional[int] = int(seed_env) if seed_env.strip() else None
        except ValueError:
            logging.warning("Invalid SEQ_FORECASTER_SEED=%s; disabling seed.", seed_env)
            self.seq_seed = None

        self.meta_feature_names: List[str] = []
        self.seq_last_train_time: Dict[str, datetime] = defaultdict(lambda: datetime(1970, 1, 1))

    def _corr(self, ticker: Optional[str] = None) -> str:
        return f"{self.pipeline_corr_id}:{ticker}" if ticker else self.pipeline_corr_id

    def _emit_event(
        self,
        *,
        component: str,
        event: str,
        category: str = "pipeline",
        symbol: Optional[str] = None,
        corr_id: Optional[str] = None,
        message: Optional[str] = None,
        state: Optional[str] = None,
        duration_ms: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        emit_event(
            component=component,
            event=event,
            category=category,
            symbol=symbol,
            corr_id=corr_id or self._corr(symbol),
            run_id=self.run_id,
            message=message,
            state=state,
            duration_ms=duration_ms,
            data=data,
        )

        # Hook IBKR disconnect event to auto-reconnect with backoff
        try:
            ib = getattr(self.market_data_adapter, 'ib', None)
            host = getattr(self.market_data_adapter, 'host', IBKR_HOST)
            port = getattr(self.market_data_adapter, 'port', IBKR_PORT)
            cid  = getattr(self.market_data_adapter, 'clientId', IBKR_CLIENT_ID)

            if ib and hasattr(ib, 'disconnectedEvent'):
                import time as _time
                import logging as _logging

                def _reconnect():
                    for delay in (5, 10, 20, 30, 60, 120):
                        try:
                            # Delegate to adapter's connect logic if available
                            if hasattr(self.market_data_adapter, '_connect'):
                                ok = self.market_data_adapter._connect()
                                if ok:
                                    _logging.warning("Reconnected to IBGW on clientId=%s", cid)
                                    return True
                            else:
                                ib.connect(host, port, clientId=cid, timeout=IBKR_CONNECT_TIMEOUT)
                                if ib.isConnected():
                                    _logging.warning("Reconnected to IBGW on clientId=%s", cid)
                                    return True
                        except Exception as e:
                            _logging.warning("Reconnect failed (%ss): %s", delay, e)
                        _time.sleep(delay)
                    return False

                def _on_disconnected():
                    _logging.warning("IB disconnected; attempting auto-reconnect")
                    _reconnect()

                # Register only once
                try:
                    ib.disconnectedEvent += _on_disconnected
                except Exception:
                    pass
        except Exception:
            # Do not let optional hooks break initialization
            pass

    def _register_seq_features(self, artifact: SequenceForecasterArtifact) -> None:
        if artifact is None:
            return
        new_names = {f"seq_r{int(h)}" for h in artifact.horizons}
        if not new_names:
            return
        merged = set(self.seq_feature_names)
        merged.update(new_names)
        try:
            self.seq_feature_names = sorted(merged, key=lambda name: int(name.split("h", 1)[1]))
        except Exception:
            # Fallback to lexical order if parsing fails
            self.seq_feature_names = sorted(merged)

    def _get_seq_artifact(self, ticker: str) -> Optional[SequenceForecasterArtifact]:
        if not self.seq_enabled:
            return None
        if ticker in self.seq_models:
            return self.seq_models[ticker]

        model_path = self.seq_model_dir / ticker / "model.pt"
        if not model_path.exists():
            logging.debug("Sequence forecaster artifact missing for %s (%s)", ticker, model_path)
            self.seq_models[ticker] = None
            return None

        try:
            artifact = self._load_seq_artifact(model_path, device=self.seq_device)
        except Exception as exc:
            logging.warning("Failed to load sequence forecaster for %s: %s", ticker, exc)
            self.seq_models[ticker] = None
            return None

        self._register_seq_features(artifact)
        self.seq_models[ticker] = artifact
        logging.info("Loaded sequence forecaster for %s from %s", ticker, model_path)
        return artifact

    def _compute_sequence_forecasts(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> Optional[pd.DataFrame]:
        if not self.seq_enabled or self.seq_torch is None:
            return None

        artifact = self._get_seq_artifact(ticker)
        if artifact is None:
            return None

        feature_cols = artifact.feature_cols
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            logging.warning(
                "Sequence forecaster skipped for %s: missing feature columns %s", ticker, missing
            )
            return None

        df_feat = (
            df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=0, how="any")
        )
        if df_feat.empty:
            logging.debug("Sequence forecaster skipped for %s: all feature rows have NaNs", ticker)
            return None

        if len(df_feat) < artifact.n_steps:
            logging.debug(
                "Sequence forecaster skipped for %s: rows=%s < n_steps=%s",
                ticker,
                len(df_feat),
                artifact.n_steps,
            )
            return None

        values = df_feat.astype(np.float32).values
        windows = np.lib.stride_tricks.sliding_window_view(
            values, (artifact.n_steps, len(feature_cols))
        )
        windows = windows.squeeze(1)
        scaled = artifact.scaler.transform(windows)

        device = self.seq_device or self.seq_torch.device("cpu")
        tensor = self.seq_torch.from_numpy(scaled).to(device)
        try:
            with self.seq_torch.no_grad():
                preds = artifact.model(tensor).detach().cpu().numpy()
        except Exception as exc:
            logging.warning("Sequence forecaster inference failed for %s: %s", ticker, exc)
            return None

        index = df_feat.index[artifact.n_steps - 1 :]
        data = {f"seq_r{int(h)}": preds[:, i] for i, h in enumerate(artifact.horizons)}
        return pd.DataFrame(data, index=index)

    def _micro_poll_symbol(self, ticker: str) -> int:
        """Poll a tiny historical window and append only new rows.

        - Uses adapter connection + contract resolution
        - Fetches small window (e.g., 90 S, 5 secs, MIDPOINT)
        - Resamples to 1‑minute OHLCV and persists to Parquet
        Returns number of 1‑minute rows appended.
        """
        if not self.micropoll_enable:
            return 0
        try:
            # Ensure connection
            if not self.market_data_adapter.ensure_connected():
                return 0
            # Resolve contract via adapter (handles canonicalization)
            canonical = self.market_data_adapter._canonical_symbol(ticker)
            contract = self.market_data_adapter._get_contract(canonical)
            ib = self.market_data_adapter.ib

            def _pull(what):
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=self.micropoll_window,
                    barSizeSetting=self.micropoll_bar_size,
                    whatToShow=what,
                    useRTH=False,
                    formatDate=1,
                )
                return util.df(bars)

            df = _pull(self.micropoll_what)
            # Optional fallback: if requesting TRADES during off-hours yields empty, fall back to MIDPOINT
            if (df is None or df.empty) and self.micropoll_what.upper() == 'TRADES':
                df = _pull('MIDPOINT')
            if df is None or df.empty:
                return 0
            # Filter new rows beyond last seen timestamp
            last_ts = self._last_poll_ts.get(ticker)
            if last_ts is not None:
                df = df[df['date'] > last_ts]
            if df.empty:
                return 0
            self._last_poll_ts[ticker] = df['date'].max()

            # Normalize columns and resample 5‑sec bars to 1‑min
            df = df.rename(columns={'date': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            cols = [c for c in ['timestamp','open','high','low','close','volume'] if c in df.columns]
            df = df[cols]
            g = df.set_index('timestamp').sort_index()
            o = g['open'].resample('1min').first() if 'open' in g.columns else None
            h = g['high'].resample('1min').max()   if 'high' in g.columns else None
            l = g['low'].resample('1min').min()    if 'low' in g.columns else None
            c = g['close'].resample('1min').last() if 'close' in g.columns else None
            v = g['volume'].resample('1min').sum() if 'volume' in g.columns else None
            parts = []
            if o is not None: parts.append(o.rename('open'))
            if h is not None: parts.append(h.rename('high'))
            if l is not None: parts.append(l.rename('low'))
            if c is not None: parts.append(c.rename('close'))
            if v is not None: parts.append(v.rename('volume'))
            if not parts:
                return 0
            df1 = pd.concat(parts, axis=1).dropna(how='all').reset_index()
            if df1.empty:
                return 0
            persist_bars(ticker, df1)
            logging.info(f"[micropoll] {ticker}: appended {len(df1)} rows (window={self.micropoll_window}, bar={self.micropoll_bar_size}, what={self.micropoll_what})")
            return len(df1)
        except Exception as e:
            logging.warning(f"[micropoll] {ticker}: error: {e}")
            return 0

    def run(self):
        self.status = "running"
        log_trade_results(f"pipeline_start: run_id={self.run_id}")
        self._emit_event(component="pipeline", category="lifecycle", event="run_start", corr_id=self._corr())
        try:
            self._main_loop()
            self.status = "completed"
            log_trade_results(f"pipeline_complete: run_id={self.run_id}")
            self._emit_event(component="pipeline", category="lifecycle", event="run_complete", corr_id=self._corr(), state="OK")
        except Exception as e:
            self.status = "failed"
            self.last_error = str(e)
            log_trade_results(f"pipeline_error: run_id={self.run_id}, error={self.last_error}, traceback={traceback.format_exc()}")
            logger.error(f"Pipeline failed: {e}")
            self.notify(f"Pipeline failed: {e}")
            self._emit_event(
                component="pipeline",
                category="lifecycle",
                event="run_failed",
                corr_id=self._corr(),
                state="ERROR",
                message=str(e),
                data={"traceback": traceback.format_exc()},
            )
            raise

    def _main_loop(self):
        global last_retrain_time
        retries = 0
        last_hb = time.monotonic()
        while retries < self.max_retries:
            try:
                if _news_enabled:
                    news_start = time.time()
                    ma_local, ibkr_local, news_local = fetch_union_news()
                    globals()['ma_df'] = ma_local
                    globals()['ibkr_df'] = ibkr_local
                    globals()['news_df'] = news_local
                    self._emit_event(
                        component="news",
                        category="data",
                        event="fetch_complete",
                        duration_ms=(time.time() - news_start) * 1000,
                        data={
                            "marketaux_rows": int(len(ma_local)),
                            "ibkr_rows": int(len(ibkr_local)),
                            "combined_rows": int(len(news_local)),
                        },
                    )

                logging.info("Fetching market data...")
                # Canonical adapter-supported tickers
                tickers = list(TICKERS)
                last_check_times = {ticker: 0 for ticker in tickers}
                raw_data = {}
                for idx, ticker in enumerate(tickers):
                    # Lightweight heartbeat every 30s to keep socket sticky
                    now_mono = time.monotonic()
                    heartbeat_interval = self.config.get("heartbeat_interval", 30)
                    heartbeat_enabled = (
                        not self.config.get("use_mock_data", True)
                        and not self.config.get("disable_ib_heartbeat", False)
                        and getattr(self.market_data_adapter, 'ib', None) is not None
                    )
                    if heartbeat_enabled and (now_mono - last_hb) >= heartbeat_interval:
                        try:
                            self.market_data_adapter.ib.reqCurrentTime()
                        except Exception as _hb_e:
                            logging.warning(f"Heartbeat error: {_hb_e}")
                        finally:
                            last_hb = time.monotonic()
                    start_time = time.time()
                    corr_id = self._corr(ticker)
                    try:
                        use_mock = self.config.get("use_mock_data", False)
                        source = "mock" if use_mock else "ibkr"
                        if use_mock:
                            logging.warning("use_mock_data requested for %s but mock generator is unavailable; skipping.", ticker)
                            continue
                        else:
                            df = self.market_data_adapter.fetch_data(ticker)
                            logging.info(f"{ticker} raw data shape: {df.shape}")
                            if df is None or df.empty:
                                logging.warning(f"No data for {ticker}, skipping.")
                                self._emit_event(
                                    component="ingest",
                                    category="data",
                                    event="fetch_empty",
                                    symbol=ticker,
                                    corr_id=corr_id,
                                    state="WARN",
                                    duration_ms=(time.time() - start_time) * 1000,
                                    data={"source": source},
                                )
                                continue
                            raw_data[ticker] = df
                            logging.info(f"Market data fetched successfully.")
                            print(f"{ticker} columns: {df.columns}")

                            # --- Regime Detection and Alerting ---
                            regime = detect_regime(df['close'])
                            if regime == "high_vol":
                                send_sms_alert(f"High volatility regime detected for {ticker}!")
                                logging.info(f"Regime change: High volatility detected for {ticker}")
                                print(f"High volatility regime detected for {ticker}, triggering retraining/model switch.")
                                # Trigger retraining/model switch
                                regime = detect_regime(df['close'])
                    # (model selection / training handled in scheduled block below)
                                # Optionally, you could load a different model here if you have one for high-vol regimes

                        self._emit_event(
                            component="ingest",
                            category="data",
                            event="fetch_complete",
                            symbol=ticker,
                            corr_id=corr_id,
                            duration_ms=(time.time() - start_time) * 1000,
                            data={
                                "rows": int(len(df)) if df is not None else 0,
                                "source": source,
                            },
                        )

                        # --- PPO Training for Each Ticker Sequentially ---
                        current_time = time.time()
                        if (current_time - last_retrain_time) > RETRAIN_INTERVAL or new_data_available(data_dir, ticker, last_check_times[ticker]):
                            print(f"Training PPO for {ticker} (scheduled or new data)...")
                            self.train_ppo(df, ticker)
                            last_check_times[ticker] = current_time
                            last_retrain_time = current_time
                            # (rest of your retraining logic)
                        else:
                            # Option B: Micro-poll historicals (permission-agnostic) to keep features fresh
                            try:
                                now_m = time.monotonic()
                                cadence_key = f"{ticker}_cadence"
                                last_cadence = self._last_poll_ts.get(cadence_key)
                                if (last_cadence is None) or (now_m - last_cadence >= self.micropoll_every_s):
                                    # Optional per-symbol offset to stagger pacing
                                    off_env = f"MICROPOLL_OFFSET_{ticker.replace('!','')}"
                                    off = int(os_mod.getenv(off_env, "0"))
                                    if off:
                                        time.sleep(off)
                                    appended = self._micro_poll_symbol(ticker)
                                    # mark cadence regardless of appended count
                                    self._last_poll_ts[cadence_key] = time.monotonic()
                                    if appended:
                                        last_check_times[ticker] = time.time()
                            except Exception as _mp_e:
                                logging.warning(f"Micropoll error for {ticker}: {_mp_e}")
                            print(f"No new data or scheduled retrain for {ticker}, skipping retraining.")

                        # Log resource usage after PPO training
                        cpu = psutil.cpu_percent()
                        mem = psutil.virtual_memory().percent
                        print(f"Resource usage after {ticker}: CPU {cpu}%, RAM {mem}%")

                        # Progress logging
                        elapsed = time.time() - start_time
                        print(f"Finished PPO for {ticker} ({idx+1}/{len(tickers)}) in {elapsed:.1f} seconds.")

                        # Dynamic sleep based on resource usage
                        if cpu > 80 or mem > 80:
                            print("High resource usage detected, sleeping for 30 seconds...")
                            time.sleep(30)
                        else:
                            time.sleep(10)  # Default sleep

                    except Exception as e:
                        print(f"ERROR during PPO training for {ticker}: {e}")
                        traceback.print_exc()
                        print(f"Skipping {ticker} and continuing to next ticker...")
                        self._emit_event(
                            component="ingest",
                            category="data",
                            event="fetch_error",
                            symbol=ticker,
                            corr_id=corr_id,
                            state="ERROR",
                            message=str(e),
                            data={"phase": "ppo_training"},
                        )

                print("All tickers processed. Check above for any errors or resource warnings.")

                logging.info("Engineering features...")

                # Load last trained timestamp/index
                try:
                    with open(LAST_TRAINED_FILE, "r") as f:
                        last_trained_time = f.read().strip()
                        if last_trained_time:
                            # Convert to int (or pd.to_datetime if using datetime)
                            last_trained_time = _coerce_epoch(last_trained_time)
                except FileNotFoundError:
                    last_trained_time = None

                features = {}
                labels = {}

                for ticker, df in raw_data.items():
                    X, y = generate_features(df)
                    logging.info(f"{ticker} features shape: {X.shape}, labels shape: {y.shape}")
                    if X is None or X.empty or y is None or y.empty:
                        logging.warning(f"No features/labels for {ticker} after feature engineering, skipping.")
                        continue
                    features[ticker] = X
                    labels[ticker] = y
                    try:
                        feature_store.update(ticker, X)
                        cache_cap = int(self.config.get("feature_store_max_rows", 5000))
                        cached_df = feature_store.get(ticker)
                        if cached_df is not None and len(cached_df) > cache_cap:
                            feature_store._store[ticker] = cached_df.tail(cache_cap)
                    except Exception as exc:
                        logging.warning(f"Failed to cache real-time features for {ticker}: {exc}")
                    self._emit_event(
                        component="features",
                        category="ml",
                        event="feature_snapshot",
                        symbol=ticker,
                        corr_id=self._corr(ticker),
                        data={
                            "features_rows": int(len(X)),
                            "labels_rows": int(len(y)),
                        },
                    )
                logging.info("Features engineered successfully.")

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
                from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
                from sklearn.ensemble import RandomForestClassifier
                logging.warning("ML SYSTEM: Running ML model training and prediction for all tickers.")
                from sklearn.metrics import accuracy_score, confusion_matrix


                base_model_factories = {
                    "SGDClassifier": lambda: SGDClassifier(loss="log_loss", random_state=42, max_iter=1000, tol=1e-3),
                    "Perceptron": lambda: Perceptron(),
                    "PassiveAggressive": lambda: PassiveAggressiveClassifier(),
                    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42)
                }

                fitted_models: Dict[str, Dict[str, Any]] = {}
                results: Dict[str, Dict[str, Any]] = {}
                now_utc = datetime.utcnow()
                full_retrain_timestamp = None

                for ticker in features:
                    X_all = feature_store.get(ticker)
                    if X_all is None or X_all.empty:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue

                    y_all = labels[ticker]
                    if y_all is None or y_all.empty:
                        logging.warning(f"No labels for {ticker} (features={len(X_all)}). Skipping ML for this ticker.")
                        continue

                    total_rows = len(X_all)
                    new_rows = total_rows - self.ml_last_incremental_rows.get(ticker, 0)
                    incremental_ready = new_rows >= self.ml_incremental_min_samples

                    X_window = X_all.tail(self.ml_rolling_window)
                    y_window = y_all.loc[X_window.index]
                    window_timestamp = None
                    if "timestamp" in X_window.columns:
                        window_timestamp = X_window["timestamp"].max()

                    last_full_time = self.ml_last_full_retrain_time.get(ticker, datetime(1970, 1, 1))
                    full_ready = (
                        len(X_window) >= self.ml_full_min_samples
                        and (now_utc - last_full_time) >= timedelta(minutes=self.ml_retrain_every_min)
                    )

                    if full_ready:
                        window_rows = len(X_window)
                        if window_rows < self.min_samples_per_model:
                            logging.warning(
                                f"Window below min samples for {ticker}: rows={window_rows} (min={self.min_samples_per_model}). Skipping."
                            )
                            self._emit_event(
                                component="ml",
                                category="training",
                                event="retrain_skipped",
                                symbol=ticker,
                                corr_id=self._corr(ticker),
                                state="WARN",
                                data={
                                    "reason": "window_min_samples",
                                    "window_rows": int(window_rows),
                                    "min_required": int(self.min_samples_per_model),
                                },
                            )
                            continue

                        models = {name: factory() for name, factory in base_model_factories.items()}
                        split_idx = max(int(window_rows * 0.90), 1)
                        train_ratio = split_idx / window_rows
                        min_train_rows = max(1, int(round(self.min_samples_per_model * train_ratio)))
                        self._emit_event(
                            component="ml",
                            category="training",
                            event="full_retrain_start",
                            symbol=ticker,
                            corr_id=self._corr(ticker),
                            data={
                                "total_rows": int(total_rows),
                                "window_rows": int(len(X_window)),
                                "new_rows": int(new_rows),
                                "split_index": int(split_idx),
                                "min_samples": int(self.min_samples_per_model),
                            },
                        )
                        X_train = X_window.iloc[:split_idx].copy()
                        X_test = X_window.iloc[split_idx:].copy()
                        y_train = y_window.iloc[:split_idx].copy()
                        y_test = y_window.iloc[split_idx:].copy()

                        cols_to_drop = [
                            col for col in X_train.columns
                            if is_datetime64_any_dtype(X_train[col]) or is_datetime64tz_dtype(X_train[col])
                            or col in ("datetime", "timestamp")
                        ]
                        if cols_to_drop:
                            X_train = X_train.drop(columns=cols_to_drop)
                            X_test = X_test.drop(columns=cols_to_drop)

                        if X_train.empty or X_test.empty or y_train.empty or y_test.empty:
                            logging.warning(
                                f"Split produced insufficient data for {ticker}: "
                                f"X_train={len(X_train)}, X_test={len(X_test)}, "
                                f"y_train={len(y_train)}, y_test={len(y_test)}. Skipping."
                            )
                            self._emit_event(
                                component="ml",
                                category="training",
                                event="retrain_skipped",
                                symbol=ticker,
                                corr_id=self._corr(ticker),
                                state="WARN",
                                data={
                                    "reason": "split_insufficient",
                                    "X_train": int(len(X_train)),
                                    "X_test": int(len(X_test)),
                                    "y_train": int(len(y_train)),
                                    "y_test": int(len(y_test)),
                                },
                            )
                            continue

                        if len(X_train) < min_train_rows or len(y_train) < min_train_rows:
                            logging.warning(
                                f"Not enough samples for {ticker}: X={len(X_train)}, y={len(y_train)} "
                                f"(min={min_train_rows}). Skipping."
                            )
                            self._emit_event(
                                component="ml",
                                category="training",
                                event="retrain_skipped",
                                symbol=ticker,
                                corr_id=self._corr(ticker),
                                state="WARN",
                                data={
                                    "reason": "min_samples",
                                    "samples": int(len(X_train)),
                                    "min_required": int(min_train_rows),
                                },
                            )
                            continue

                        ticker_results = {}
                        feature_columns = list(X_train.columns)

                        for name, model in models.items():
                            classes = np.unique(y_train)
                            if len(classes) < 2 and hasattr(model, 'partial_fit'):
                                logging.warning(f"Skipping model update for {ticker}/{name}: only one class present.")
                                continue

                            model.fit(X_train, y_train)

                            self.ml_model_store[ticker][name] = model
                            fitted_models.setdefault(ticker, {})[name] = model
                            self.ml_feature_columns[ticker] = feature_columns

                            save_model_local(model, self.run_id, f"{ticker}_{name}.joblib")
                            save_key = f"models/{self.run_id}/{ticker}_{name}.joblib"
                            save_model_to_s3(model, self.config['s3_bucket'], save_key)

                            preds = model.predict(X_test)
                            acc = accuracy_score(y_test, preds)
                            cm = confusion_matrix(y_test, preds)
                            wins = int((preds == 1).sum())
                            losses = int((preds == 0).sum())
                            total_trades = len(preds)
                            win_rate = (preds == y_test).sum() / total_trades if total_trades > 0 else 0
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

                            logging.info(
                                f"[{ticker}][{name}] Trades: {total_trades}, Wins: {wins}, Losses: {losses}, "
                                f"Accuracy: {acc:.2%}, Win Rate: {win_rate:.2%}, PnL: {pnl}"
                            )
                            self._emit_event(
                                component="ml",
                                category="training",
                                event="model_trained",
                                symbol=ticker,
                                corr_id=self._corr(ticker),
                                data={
                                    "model": name,
                                    "accuracy": acc,
                                    "win_rate": win_rate,
                                    "trades": total_trades,
                                    "wins": wins,
                                    "losses": losses,
                                },
                            )

                        if ticker_results:
                            results[ticker] = ticker_results
                            self._emit_event(
                                component="ml",
                                category="training",
                                event="full_retrain_complete",
                                symbol=ticker,
                                corr_id=self._corr(ticker),
                                data={
                                    "models": list(ticker_results.keys()),
                                    "window_rows": int(len(X_window)),
                                    "new_rows": int(new_rows),
                                },
                            )

    
                        if self.seq_training_enabled:
                            df_seq = X_all.reset_index(drop=True).copy()
                            if "close" not in df_seq.columns:
                                logging.warning(
                                    "Sequence forecaster skipped for %s: missing 'close' column in features.",
                                    ticker,
                                )
                            else:
                                seq_feature_cols = [
                                    col
                                    for col in df_seq.columns
                                    if col != "close"
                                    and not is_datetime64_any_dtype(df_seq[col])
                                    and not is_datetime64tz_dtype(df_seq[col])
                                    and np.issubdtype(df_seq[col].dtype, np.number)
                                ]
                                if not seq_feature_cols:
                                    logging.warning(
                                        "Sequence forecaster skipped for %s: no numeric feature columns after filtering.",
                                        ticker,
                                    )
                                else:
                                    try:
                                        if len(df_seq) < self.seq_min_train_rows:
                                            logging.debug(
                                                "Sequence forecaster skipped for %s: only %s rows (min=%s).",
                                                ticker,
                                                len(df_seq),
                                                self.seq_min_train_rows,
                                            )
                                            continue
                                        last_seq_train = self.seq_last_train_time.get(ticker, datetime(1970, 1, 1))
                                        if (now_utc - last_seq_train) < timedelta(minutes=self.seq_retrain_every_min):
                                            logging.debug(
                                                "Sequence forecaster throttled for %s: %.1f min since last run (min interval %s).",
                                                ticker,
                                                (now_utc - last_seq_train).total_seconds() / 60.0,
                                                self.seq_retrain_every_min,
                                            )
                                            continue
                                        seq_metrics = self._fit_seq_forecaster(
                                            df_seq,
                                            seq_feature_cols,
                                            symbol=ticker,
                                            horizons=self.seq_horizons,
                                            n_steps=self.seq_n_steps,
                                            step=self.seq_step,
                                            out_dir=str(self.seq_model_dir),
                                            train_split=self.seq_train_split,
                                            val_split=self.seq_val_split,
                                            hidden_size=self.seq_hidden_size,
                                            num_layers=self.seq_num_layers,
                                            dropout=self.seq_dropout,
                                            epochs=self.seq_epochs,
                                            batch_size=self.seq_batch_size,
                                            lr=self.seq_lr,
                                            weight_decay=self.seq_weight_decay,
                                            patience=self.seq_patience,
                                            device=self.seq_device,
                                            seed=self.seq_seed,
                                        )
                                        self.seq_last_train_time[ticker] = now_utc
                                        self.seq_models.pop(ticker, None)
                                        artifact = self._get_seq_artifact(ticker)
                                        self._emit_event(
                                            component="sequence_forecaster",
                                            category="ml",
                                            event="train_complete",
                                            symbol=ticker,
                                            corr_id=self._corr(ticker),
                                            data=seq_metrics,
                                        )
                                        if artifact is not None:
                                            logging.info(
                                                "Sequence forecaster trained for %s (horizons=%s)",
                                                ticker,
                                                artifact.horizons,
                                            )
                                    except Exception as exc:
                                        logging.warning(
                                            "Sequence forecaster training failed for %s: %s",
                                            ticker,
                                            exc,
                                        )
                                        self._emit_event(
                                            component="sequence_forecaster",
                                            category="ml",
                                            event="train_error",
                                            symbol=ticker,
                                            corr_id=self._corr(ticker),
                                            state="ERROR",
                                            message=str(exc),
                                        )
                        self.ml_last_incremental_rows[ticker] = total_rows
                        self.ml_last_incremental_time[ticker] = now_utc
                        self.ml_last_full_retrain_time[ticker] = now_utc
                        if window_timestamp is not None:
                            full_retrain_timestamp = window_timestamp
                        continue

                    if incremental_ready:
                        existing = self.ml_model_store.get(ticker, {})
                        if not existing:
                            logging.info(
                                f"Incremental update for {ticker} skipped: no prior trained models."
                            )
                            self._emit_event(
                                component="ml",
                                category="training",
                                event="incremental_skipped",
                                symbol=ticker,
                                corr_id=self._corr(ticker),
                                state="INFO",
                                data={"reason": "no_models", "new_rows": int(new_rows)},
                            )
                        else:
                            feature_columns = self.ml_feature_columns.get(ticker)
                            if not feature_columns:
                                logging.info(
                                    f"Incremental update for {ticker} skipped: no feature column metadata."
                                )
                                self._emit_event(
                                    component="ml",
                                    category="training",
                                    event="incremental_skipped",
                                    symbol=ticker,
                                    corr_id=self._corr(ticker),
                                    state="INFO",
                                    data={"reason": "no_feature_columns", "new_rows": int(new_rows)},
                                )
                            else:
                                X_inc = X_all.iloc[-new_rows:]
                                y_inc = y_all.loc[X_inc.index]
                                X_inc = X_inc[feature_columns]
                                classes = np.unique(y_inc)
                                if len(classes) == 0:
                                    logging.warning(f"No label variation in incremental window for {ticker}.")
                                    self._emit_event(
                                        component="ml",
                                        category="training",
                                        event="incremental_skipped",
                                        symbol=ticker,
                                        corr_id=self._corr(ticker),
                                        state="WARN",
                                        data={"reason": "no_label_variation", "new_rows": int(new_rows)},
                                    )
                                else:
                                    for name, model in existing.items():
                                        if hasattr(model, 'partial_fit'):
                                            try:
                                                if hasattr(model, 'classes_'):
                                                    model.partial_fit(X_inc, y_inc)
                                                else:
                                                    model.partial_fit(X_inc, y_inc, classes=classes)
                                            except Exception as exc:
                                                logging.warning(f"Incremental update failed for {ticker}/{name}: {exc}")
                                                self._emit_event(
                                                    component="ml",
                                                    category="training",
                                                    event="incremental_error",
                                                    symbol=ticker,
                                                    corr_id=self._corr(ticker),
                                                    state="ERROR",
                                                    message=str(exc),
                                                    data={"model": name},
                                                )
                                    fitted_models[ticker] = existing
                                    self.ml_last_incremental_rows[ticker] = total_rows
                                    self.ml_last_incremental_time[ticker] = now_utc
                                    self._emit_event(
                                        component="ml",
                                        category="training",
                                        event="incremental_update",
                                        symbol=ticker,
                                        corr_id=self._corr(ticker),
                                        data={
                                            "models": list(existing.keys()),
                                            "new_rows": int(new_rows),
                                        },
                                    )
                        continue

                    fitted_models[ticker] = self.ml_model_store.get(ticker, {})
                    logging.info(
                        f"ML update skipped for {ticker}: new_rows={new_rows}, total={total_rows}, full_ready={full_ready}."
                    )
                    self._emit_event(
                        component="ml",
                        category="training",
                        event="ml_update_skipped",
                        symbol=ticker,
                        corr_id=self._corr(ticker),
                        state="INFO",
                        data={
                            "new_rows": int(new_rows),
                            "total_rows": int(total_rows),
                            "full_ready": bool(full_ready),
                            "incremental_ready": bool(incremental_ready),
                        },
                    )

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

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    print("\n=== ML Model Performance Summary ===")
                    print(summary_df.to_string(index=False))
                    self._emit_event(
                        component="ml",
                        category="training",
                        event="performance_summary",
                        data={"rows": summary_rows},
                    )

                if full_retrain_timestamp is not None:
                    with open(LAST_TRAINED_FILE, "w") as f:
                        f.write(str(full_retrain_timestamp))
                # --- RL PPO Training and Inference Block ---
                logging.warning("RL SYSTEM: Evaluating PPO models.")
                for ticker in features:
                    X = feature_store.get(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping PPO evaluation.")
                        continue
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        test_data = X.iloc[split_idx:]
                        self.run_ppo(test_data, ticker)
                        break
                # --- End RL PPO Block ---

                # === Begin Stacking/Meta-Model Block (Fixed) ===
                logging.warning("META-MODEL SYSTEM: Running stacking/meta-model training and prediction.")
                models_from_s3 = load_all_models(self.config['s3_bucket'], self.run_id)

                meta_X = []
                meta_y = []

                meta_signals = None

                seq_preds_by_ticker: Dict[str, pd.DataFrame] = {}
                if self.seq_enabled:
                    for ticker in features:
                        feature_df = feature_store.get(ticker)
                        if feature_df is None or feature_df.empty:
                            self._get_seq_artifact(ticker)
                            continue
                        seq_df = self._compute_sequence_forecasts(ticker, feature_df)
                        if seq_df is not None:
                            seq_preds_by_ticker[ticker] = seq_df

                for ticker in features:
                    X = feature_store.get(ticker)
                    if X is None:
                        logging.warning(f"No real-time features for {ticker}, skipping.")
                        continue
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        y_test = y.iloc[split_idx:]
                    # Collect predictions from all base models
                    preds: List[np.ndarray] = []
                    pred_sources: List[str] = []
                    model_dict = models_from_s3.get(ticker, {})
                    required_models = ["RandomForest", "PPO"]
                    missing_models = [m for m in required_models if m not in model_dict]
                    if missing_models:
                        logging.warning(
                            "Skipping %s in stacking: missing models %s", ticker, missing_models
                        )
                        self._emit_event(
                            component="stacking",
                            category="ml",
                            event="missing_base_models",
                            symbol=ticker,
                            corr_id=self._corr(ticker),
                            state="WARN",
                            data={
                                "expected": required_models,
                                "present": list(model_dict.keys()),
                            },
                        )
                        continue

                    rf_model = model_dict["RandomForest"]
                    cols_to_drop = [
                        col
                        for col in X_test.columns
                        if (
                            is_datetime64_any_dtype(X_test[col])
                            or is_datetime64tz_dtype(X_test[col])
                            or col in ("datetime", "timestamp")
                        )
                    ]
                    X_test_pred = X_test.drop(columns=cols_to_drop) if cols_to_drop else X_test
                    try:
                        rf_preds = np.asarray(rf_model.predict(X_test_pred), dtype=float).flatten()
                    except Exception as exc:
                        logging.warning("RandomForest prediction failed for %s: %s", ticker, exc)
                        continue
                    preds.append(rf_preds)
                    pred_sources.append("RandomForest")

                    ppo_model = model_dict["PPO"]
                    ppo_frame = _ppo_observation_frame(X_test)
                    if ppo_frame is None:
                        logging.warning(
                            "Stacking skipped for %s PPO contribution: missing base price columns.",
                            ticker,
                        )
                        continue
                    rl_preds: List[float] = []
                    env = TradingEnv(ppo_frame)
                    obs, info = env.reset()
                    done = False
                    while not done:
                        action, _ = ppo_model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        rl_preds.append(float(action))
                        done = terminated or truncated
                    preds.append(np.asarray(rl_preds[: len(X_test)], dtype=float).flatten())
                    pred_sources.append("PPO")

                    seq_df = seq_preds_by_ticker.get(ticker)
                    if self.seq_enabled and seq_df is not None and not seq_df.empty:
                        seq_aligned = seq_df.reindex(X_test.index)
                        seq_aligned = seq_aligned.apply(pd.to_numeric, errors="coerce").ffill().fillna(0.0)
                        seq_columns = [
                            col
                            for col in seq_aligned.columns
                            if col.startswith("seq_") and pd.api.types.is_numeric_dtype(seq_aligned[col])
                        ]
                        if seq_columns:
                            merged = list(dict.fromkeys(self.seq_feature_names + seq_columns))
                            self.seq_feature_names = merged
                            for col in seq_columns:
                                preds.append(seq_aligned[col].to_numpy(dtype=float))
                                pred_sources.append(col)
                            logging.info("Seq features appended for %s: %s", ticker, seq_columns)

                    if not preds:
                        logging.warning("Stacking skipped for %s: no predictions available.", ticker)
                        continue

                    lens = [len(p) for p in preds]
                    self._emit_event(
                        component="stacking",
                        category="ml",
                        event="pre_stack",
                        symbol=ticker,
                        corr_id=self._corr(ticker),
                        data={
                            "sources": pred_sources,
                            "lengths": lens,
                            "labels": int(len(y_test)),
                        },
                    )

                    try:
                        min_len = min(lens + [len(y_test)])
                    except ValueError:
                        logging.warning("Stacking skipped for %s: no predictions available.", ticker)
                        continue

                    if min_len <= 0:
                        logging.warning(
                            "Stacking skipped for %s: min_len=%s (sources=%s, lengths=%s)",
                            ticker,
                            min_len,
                            pred_sources,
                            lens,
                        )
                        continue

                    preds = [p[:min_len] for p in preds]
                    y_cut = y_test.iloc[:min_len]

                    try:
                        stack = np.column_stack(preds)
                    except Exception as exc:
                        logging.warning("Stacking failed for %s during column stack: %s", ticker, exc)
                        continue

                    if not self.meta_feature_names:
                        self.meta_feature_names = pred_sources.copy()
                    elif len(self.meta_feature_names) != len(pred_sources):
                        logging.debug(
                            "Meta feature count mismatch for %s (expected %d, got %d)",
                            ticker,
                            len(self.meta_feature_names),
                            len(pred_sources),
                        )

                    meta_X.extend(stack.tolist())
                    meta_y.extend(y_cut.astype(int).tolist())
                    logging.info(
                        "Stacking features for %s: sources=%s, rows=%s",
                        ticker,
                        pred_sources,
                        min_len,
                    )

                # Final check before fitting meta-model
                if len(meta_X) == 0:
                    logger.error("No valid meta_X rows to fit meta-model. Skipping meta-model training for this run.")
                    self._emit_event(
                        component="stacking",
                        category="ml",
                        event="meta_training_skipped",
                        state="ERROR",
                        data={"reason": "no_meta_rows"},
                    )
                else:
                    meta_X_arr = np.array(meta_X)
                    meta_y_arr = np.asarray(meta_y, dtype=int)
                    logging.info(f"meta_X_arr shape: {meta_X_arr.shape}, dtype: {meta_X_arr.dtype}")
                    logging.info(f"meta_X_arr example row: {meta_X_arr[0]}")

                    if meta_X_arr.shape[0] == 0 or meta_y_arr.size == 0:
                        logger.error("No data available for meta-model training.")
                        raise ValueError("No data available for meta-model training.")
                    if meta_X_arr.shape[0] != meta_y_arr.size:
                        logger.error(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y_arr.size}")
                        raise ValueError(f"Meta-model feature/label row mismatch: {meta_X_arr.shape[0]} vs {meta_y_arr.size}")
                    if np.any(pd.isnull(meta_X_arr)) or np.any(pd.isnull(meta_y_arr)):
                        logger.error("NaN values detected in meta-model features or labels.")
                        raise ValueError("NaN values detected in meta-model features or labels.")

                    unique_meta_classes = np.unique(meta_y_arr)
                    meta_model = None
                    prev_meta_model = None
                    meta_model_key = get_latest_model_key(self.config['s3_bucket'], f"models/{self.run_id}/meta_model.joblib")
                    if meta_model_key:
                        prev_meta_model = load_model_from_s3(self.config['s3_bucket'], meta_model_key)

                    if unique_meta_classes.size < 2:
                        logger.warning("Skipping meta-model update: only one class present in meta_y.")
                        self._emit_event(
                            component="stacking",
                            category="ml",
                            event="meta_training_skipped",
                            state="WARN",
                            data={"reason": "single_class"},
                        )
                        meta_model = prev_meta_model
                    else:
                        if prev_meta_model is not None and hasattr(prev_meta_model, "partial_fit"):
                            prev_meta_model.partial_fit(meta_X_arr, meta_y_arr)
                            meta_model = prev_meta_model
                        elif prev_meta_model is not None and not hasattr(prev_meta_model, "partial_fit"):
                            meta_model = LogisticRegression()
                            meta_model.fit(meta_X_arr, meta_y_arr)
                        else:
                            meta_model = LogisticRegression()
                            meta_model.fit(meta_X_arr, meta_y_arr)

                    if meta_model is not None:
                        save_model_to_s3(meta_model, self.config['s3_bucket'], f"models/{self.run_id}/meta_model.joblib")
                        logging.info("Meta-model (stacking) training complete.")
                        self._emit_event(
                            component="stacking",
                            category="ml",
                            event="meta_training_complete",
                            data={
                                "rows": int(meta_X_arr.shape[0]),
                                "feature_dim": int(meta_X_arr.shape[1]) if meta_X_arr.ndim == 2 else None,
                            },
                        )

                        meta_signals = meta_model.predict(meta_X_arr)
                        logging.info(f"Meta-model signals (first 10): {meta_signals[:10]}")
                        self._emit_event(
                            component="stacking",
                            category="ml",
                            event="meta_signals_generated",
                            data={
                                "count": int(len(meta_signals)),
                            },
                        )
                    else:
                        meta_signals = None
                        logging.info("Meta-model unavailable after training; meta signals not generated.")
                # === End Stacking/Meta-Model Block (Fixed) ===

                # === Begin Meta-Model Trading Simulation Block ===
                # For each ticker, simulate trading using meta-model signals

                # We'll assume meta_signals are in the same order as meta_X/meta_y, which are built sequentially for all tickers.
                # We'll split meta_signals back into per-ticker segments for reporting.

                ib = getattr(self.market_data_adapter, 'ib', None)
                if ib is not None and not ib.isConnected():
                    logging.warning("Meta-model trading skipped: IB connection unavailable.")
                    ib = None

                signal_idx = 0
                tickers_with_meta = []
                meta_trades_executed = False
                for ticker in features:
                    X = features[ticker]
                    y = labels[ticker]
                    if len(X) > 20:
                        split_idx = int(len(X) * 0.8)
                        X_test = X.iloc[split_idx:]
                        n = len(X_test)
                        if meta_signals is not None and signal_idx + n <= len(meta_signals):
                            if ib is None:
                                logging.warning("Meta-model trading skipped for %s: IB connection unavailable.", ticker)
                                signal_idx += n
                                continue

                            try:
                                canonical = self.market_data_adapter._canonical_symbol(ticker)
                                contract = self.market_data_adapter._get_contract(canonical)
                            except Exception as exc:
                                logging.warning("Meta-model trading skipped for %s: contract error %s", ticker, exc)
                                signal_idx += n
                                continue

                            ticker_signals = meta_signals[signal_idx:signal_idx + n]
                            prices = X_test['close'].values.astype(float)
                            timestamps = None
                            if 'timestamp' in X_test.columns:
                                timestamps = pd.to_datetime(X_test['timestamp'], utc=True, errors='coerce')
                            elif isinstance(X_test.index, pd.Index) and pd.api.types.is_datetime64_any_dtype(X_test.index):
                                timestamps = pd.to_datetime(X_test.index, utc=True, errors='coerce')
                            if timestamps is None or timestamps.isnull().all():
                                timestamps = pd.Series(pd.Timestamp.utcnow(), index=X_test.index)

                            current_ts = timestamps.iloc[-1] if len(timestamps) else pd.Timestamp.utcnow()
                            current_day = current_ts.date()

                            state = self.meta_daily_state[ticker]
                            if state["day"] != current_day:
                                state["day"] = current_day
                                state["trades"] = 0
                                cached_df = feature_store.get(ticker)
                                if cached_df is not None:
                                    cache_cap = int(self.config.get("feature_store_max_rows", 5000))
                                    feature_store._store[ticker] = cached_df.tail(cache_cap)
                                    logging.debug(
                                        "[meta] feature_store pruned for %s (current day=%s)",
                                        ticker,
                                        current_day,
                                    )

                            if state["trades"] >= MAX_TRADES_PER_DAY:
                                logging.info(
                                    "Meta-model trade skipped for %s: daily trade limit reached (%s)",
                                    ticker,
                                    MAX_TRADES_PER_DAY,
                                )
                                signal_idx += n
                                tickers_with_meta.append(ticker)
                                continue

                            prev_position = self.ppo_live_positions.get(ticker, 0)
                            latest_signal = int(ticker_signals[-1]) if len(ticker_signals) else 0
                            desired_position = 1 if latest_signal == 1 else 0
                            desired_position = int(max(-MAX_POSITION_EXPOSURE, min(MAX_POSITION_EXPOSURE, desired_position)))

                            trade_side = None
                            trade_qty = 0
                            new_position = prev_position

                            if desired_position > prev_position:
                                if prev_position < 0:
                                    qty = min(abs(prev_position), self.max_position_size)
                                else:
                                    qty = min(desired_position - prev_position, self.max_position_size, MAX_POSITION_EXPOSURE - prev_position)
                                qty = int(max(0, qty))
                                if qty > 0:
                                    trade_side = 'BUY'
                                    trade_qty = qty
                                    new_position = prev_position + qty
                            elif desired_position < prev_position:
                                qty = min(prev_position - desired_position, self.max_position_size)
                                qty = int(max(0, qty))
                                if qty > 0:
                                    trade_side = 'SELL'
                                    trade_qty = qty
                                    new_position = prev_position - qty

                            if trade_side and trade_qty > 0:
                                execute_trade(ib, contract, trade_side, trade_qty)
                                reason = "meta_model" if trade_side == 'BUY' else "meta_model_flatten"
                                self._emit_bridge_decision(ticker, trade_side, trade_qty, confidence=0.6, reason=reason)
                                self._emit_event(
                                    component="orders",
                                    category="execution",
                                    event="order_submitted",
                                    symbol=ticker,
                                    corr_id=self._corr(ticker),
                                    data={
                                        "side": trade_side,
                                        "quantity": int(trade_qty),
                                        "position_after": int(new_position),
                                        "reason": reason,
                                        "signal": int(latest_signal),
                                        "timestamp": current_ts.isoformat(),
                                    },
                                )
                                state["trades"] += 1
                                meta_trades_executed = True
                            else:
                                logging.debug(
                                    "Meta-model: no position change for %s (signal=%s, prev=%s)",
                                    ticker,
                                    latest_signal,
                                    prev_position,
                                )

                            self.ppo_live_positions[ticker] = new_position
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

                if not meta_trades_executed:
                    fallback_executed = self._execute_ppo_fallback(raw_data, models_from_s3)
                    if fallback_executed:
                        logging.info("Executed PPO fallback orders (meta stack unavailable).")
                    else:
                        logging.info("No trades executed: meta stack and PPO fallback both idle this run.")

                # === Begin Ensemble Voting Block ===
                logging.warning("ENSEMBLE SYSTEM: Running ensemble voting for all tickers.")
                # For each ticker, collect predictions from all ML models and RL model, then vote

                ensemble_results = {}

                for ticker in features:
                    X = feature_store.get(ticker)
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
                        cols_to_drop = []
                        for col in X_test.columns:
                            if (is_datetime64_any_dtype(X_test[col]) or is_datetime64tz_dtype(X_test[col])
                                    or col in ["datetime", "timestamp"]):
                                cols_to_drop.append(col)
                        if cols_to_drop:
                            X_test_pred = X_test.drop(columns=cols_to_drop)
                        else:
                            X_test_pred = X_test

                        # Get ML model predictions
                        ml_preds = {}
                        for name, model in fitted_models.get(ticker, {}).items():
                            ml_preds[name] = model.predict(X_test_pred)

                        # Get RL model predictions (PPO)
                        rl_model = fitted_models.get(ticker, {}).get("PPO")
                        if rl_model is None:
                            rl_model = models_from_s3.get(ticker, {}).get("PPO")
                        if rl_model is None:
                            rl_model = load_local_ppo_model(ticker)

                        rl_preds = []
                        if rl_model is not None:
                            # IMPORTANT: Build PPO observation frame from raw OHLCV so the
                            # observation shape matches the PPO training setup (5,)
                            base_df = raw_data.get(ticker)
                            ppo_frame = _ppo_observation_frame(base_df) if base_df is not None else None
                            if ppo_frame is None or ppo_frame.empty:
                                logging.warning("Ensemble PPO vote skipped for %s: no OHLCV frame available", ticker)
                                rl_preds = np.zeros(len(X_test_pred))
                            else:
                                env = TradingEnv(ppo_frame)
                                if getattr(rl_model, 'observation_space', None) is not None and rl_model.observation_space.shape != env.observation_space.shape:
                                    logging.warning(
                                        "Skipping PPO predictions for %s in ensemble: obs shape mismatch %s != %s",
                                        ticker,
                                        rl_model.observation_space.shape,
                                        env.observation_space.shape,
                                    )
                                    rl_preds = np.zeros(len(X_test_pred))
                                else:
                                    obs, info = env.reset()
                                    done = False
                                    while not done:
                                        action, _ = rl_model.predict(obs, deterministic=True)
                                        rl_preds.append(action)
                                        obs, reward, terminated, truncated, info = env.step(action)
                                        done = terminated or truncated
                                    # Align length with X_test
                                    rl_preds = np.array(rl_preds[:len(X_test)]).flatten()
                        else:
                            logging.warning("Ensemble skipping PPO vote for %s: no PPO model available", ticker)
                            rl_preds = np.zeros(len(X_test_pred))

                        # Align all predictions to the minimum available length
                        min_len = min(
                            len(ml_preds["RandomForest"]),
                            len(rl_preds)
                        )

                        final_signals = []
                        for i in range(min_len):
                            votes = [
                                ml_preds["RandomForest"][i],
                                rl_preds[i]
                            ]
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
                    if _s3_enabled():
                        upload_file_to_s3(local_csv_path, self.config['s3_bucket'], s3_key)
                    else:
                        logging.info("S3 disabled; skipping upload of %s to s3://%s/%s", local_csv_path, self.config['s3_bucket'], s3_key)
                    # Optional: remove local file after upload to save space
                    os_mod.remove(local_csv_path)
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
                retries += 1
                logging.warning(f"Error in pipeline loop (attempt {retries}): {e}")
                log_trade_results(f"pipeline_retry: run_id={self.run_id}, attempt={retries}, error={str(e)}, traceback={traceback.format_exc()}")
                self.notify(f"Pipeline error (attempt {retries}): {e}")
                prev_key = get_latest_model_key(self.config['s3_bucket'], "models/", exclude_run_id=self.run_id)
                if prev_key:
                    model = load_model_from_s3(self.config['s3_bucket'], prev_key)
                    if model is None and prev_key.endswith(".joblib"):
                        logging.warning("S3 load returned None for %s despite key presence.", prev_key)
                    else:
                        logging.info(f"Rolled back to previous model: {prev_key}")
                        # RL model rollback
                        if prev_key.endswith("_ppo.zip"):
                            from stable_baselines3 import PPO
                            local_ppo_path = "/tmp/" + os_mod.path.basename(prev_key)
                            s3 = boto3.client("s3")
                            s3.download_file(self.config['s3_bucket'], prev_key, local_ppo_path)
                            self.ppo_model = PPO.load(local_ppo_path)
                            logging.info("self.ppo_model has been set to the rolled-back PPO model.")
                            os_mod.remove(local_ppo_path)
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
                    prev_path = get_latest_local_artifact(exclude_run_id=self.run_id)
                    if prev_path:
                        logging.info("Rolled back using local artifact: %s", prev_path)
                        if prev_path.suffix == ".zip":
                            try:
                                self.ppo_model = PPO.load(str(prev_path))
                                logging.info("self.ppo_model has been set to the rolled-back local PPO model.")
                            except Exception as exc:
                                logging.error("Failed to load local PPO rollback artifact %s: %s", prev_path, exc)
                        elif prev_path.suffix == ".joblib":
                            try:
                                model = joblib.load(prev_path)
                            except Exception as exc:
                                logging.error("Failed to load local ML rollback artifact %s: %s", prev_path, exc)
                            else:
                                import re
                                match = re.match(r"(.+?)_([^_]+)\.joblib$", prev_path.name)
                                if match:
                                    ticker, model_name = match.groups()
                                    if not hasattr(self, "ml_models"):
                                        self.ml_models = {}
                                    self.ml_models.setdefault(ticker, {})[model_name] = model
                                    logging.info(f"Rolled back ML model for {ticker} - {model_name} from local cache.")
                                else:
                                    logging.warning("Could not parse ticker/model from %s; rollback skipped.", prev_path)
                    else:
                        logger.error("No previous model found for rollback (S3 disabled and no local checkpoints).")
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

    def _emit_bridge_decision(self, ticker: str, side: str, quantity: int, confidence: float, reason: str) -> None:
        """Append a JSON line decision for the orders bridge."""
        payload = {
            "ts": datetime.utcnow().isoformat(),
            "symbol": ticker,
            "side": side.upper(),
            "qty": int(quantity),
            "confidence": float(confidence),
            "reason": reason,
        }
        try:
            audit_path = Path.home() / "trade_audit_log.jsonl"
            with audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload) + "\n")
        except Exception as exc:
            logging.warning("Failed to append trade decision for bridge: %s", exc)

    def _execute_ppo_fallback(self, raw_data: Dict[str, pd.DataFrame], models_from_registry: Dict[str, Dict[str, Any]]) -> bool:
        """Execute trades using PPO-only signals when meta stack is unavailable."""
        ib = getattr(self.market_data_adapter, 'ib', None)
        if ib is None or not ib.isConnected():
            logging.warning("PPO fallback skipped: IB connection unavailable.")
            return False

        executed = False
        live_window = int(self.config.get('ppo_live_window', 120))

        for ticker, df in raw_data.items():
            model = models_from_registry.get(ticker, {}).get("PPO")
            if model is None:
                model = load_local_ppo_model(ticker)
            if model is None:
                logging.debug("PPO fallback: no model for %s", ticker)
                continue

            data_window = df.tail(live_window)
            ppo_frame = _ppo_observation_frame(data_window)
            if ppo_frame is None or ppo_frame.empty:
                logging.debug("PPO fallback: insufficient data for %s", ticker)
                continue

            env = TradingEnv(ppo_frame)
            if getattr(model, 'observation_space', None) is not None:
                if model.observation_space.shape != env.observation_space.shape:
                    logging.warning(
                        "PPO fallback skipped for %s: obs shape mismatch %s != %s",
                        ticker,
                        model.observation_space.shape,
                        env.observation_space.shape,
                    )
                    continue

            obs, info = env.reset()
            done = False
            last_action = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                last_action = int(action)

            target_position = 0
            if last_action == 1:
                target_position = 1
            elif last_action == 2:
                target_position = -1

            prev_position = self.ppo_live_positions.get(ticker, 0)
            if target_position == prev_position:
                logging.debug("PPO fallback: no position change for %s", ticker)
                continue

            side = 'BUY' if target_position > prev_position else 'SELL'
            quantity = abs(target_position - prev_position)
            if quantity == 0:
                continue

            try:
                canonical = self.market_data_adapter._canonical_symbol(ticker)
                contract = self.market_data_adapter._get_contract(canonical)
            except Exception as exc:
                logging.warning("PPO fallback: failed to qualify contract for %s: %s", ticker, exc)
                continue

            execute_trade(ib, contract, side, quantity)
            self._emit_bridge_decision(ticker, side, quantity, confidence=0.55, reason="ppo_fallback")
            self.ppo_live_positions[ticker] = target_position
            executed = True
            self._emit_event(
                component="orders",
                category="execution",
                event="order_submitted",
                symbol=ticker,
                corr_id=self._corr(ticker),
                data={
                    "side": side,
                    "quantity": int(quantity),
                    "position_after": int(target_position),
                    "reason": "ppo_fallback",
                },
            )

        return executed

    def train_ppo(self, train_data, ticker, save_path=None):
        start = time.time()
        self._emit_event(
            component="ppo",
            category="training",
            event="train_start",
            symbol=ticker,
            corr_id=self._corr(ticker),
            data={"rows": int(len(train_data))},
        )
        train_frame = _ppo_observation_frame(train_data)
        if train_frame is None or train_frame.empty:
            logging.warning(
                "PPO training skipped for %s: missing OHLCV columns or empty frame.",
                ticker,
            )
            return
        env = TradingEnv(train_frame)
        check_env(env)
        ppo_logger = PPOTrainingLogger()
        feature_signature = compute_feature_signature(env.numeric_columns)
        regime_suffix = "highvol" if detect_regime(train_data['close']) == "high_vol" else "normal"
        target_filename = ppo_filename(ticker, regime_suffix, feature_signature)
        warm_key = get_latest_model_key(
            self.config['s3_bucket'],
            f"models/{self.run_id}/{target_filename}",
        )
        if warm_key is None:
            warm_key = get_latest_model_key_global(
                self.config['s3_bucket'],
                target_filename,
                exclude_run_id=self.run_id,
            )

        local_ppo_path = "/tmp/ppo_model.zip"
        model = None
        if warm_key and _s3_enabled():
            s3 = boto3.client("s3")
            try:
                s3.download_file(self.config['s3_bucket'], warm_key, local_ppo_path)
                model = PPO.load(local_ppo_path)
                logging.info("Warm-started PPO from %s", warm_key)
            except Exception as exc:
                logging.warning("Failed to load PPO checkpoint %s: %s", warm_key, exc)
                model = None
            finally:
                if os_mod.path.exists(local_ppo_path):
                    os_mod.remove(local_ppo_path)

        if model is None:
            logging.info("Initializing fresh PPO model (no reusable checkpoint available).")
            model = PPO("MlpPolicy", env, verbose=0)
        train_steps = int(os_mod.getenv("PPO_TRAIN_STEPS", str(self.config.get("ppo_train_steps", 200))))
        model.learn(
            total_timesteps=train_steps,
            callback=ppo_logger,
            progress_bar=False,
        )
        self.ppo_model = model
        ppo_logger.print_summary()
        if save_path is None:
            save_path_path = _ensure_dir(LOCAL_PPO_DIR) / target_filename
        else:
            save_path_path = Path(save_path)
            _ensure_dir(save_path_path.parent)
        model.save(str(save_path_path))

        stack_dir = _ensure_dir(LOCAL_STACKING_DIR / str(self.run_id))
        stack_model_path = stack_dir / f"{ticker}_ppo__{feature_signature}.zip"
        model.save(str(stack_model_path))

        if _s3_enabled():
            s3 = boto3.client("s3")
            s3.upload_file(str(stack_model_path), self.config['s3_bucket'], f"models/{self.run_id}/{stack_model_path.name}")
        else:
            logging.info(
                "S3 disabled; skipping upload of PPO model to s3://%s/models/%s",
                self.config['s3_bucket'],
                stack_model_path.name,
            )

        logging.info(
            "Retraining event: PPO model for %s saved locally to %s and %s",
            ticker,
            save_path_path,
            stack_model_path,
        )
        print("PPO training complete.")
        self._emit_event(
            component="ppo",
            category="training",
            event="train_complete",
            symbol=ticker,
            corr_id=self._corr(ticker),
            duration_ms=(time.time() - start) * 1000,
            data={
                "checkpoint_local": str(save_path_path),
                "checkpoint_stack": str(stack_model_path),
            },
        )

    def run_ppo(self, test_data, ticker):
        from stable_baselines3 import PPO
        start = time.time()
        test_frame = _ppo_observation_frame(test_data)
        if test_frame is None or test_frame.empty:
            logging.warning(
                "PPO run skipped for %s: missing OHLCV columns or empty frame.",
                ticker,
            )
            self.ppo_model = None
            return 0.0
        env = TradingEnv(test_frame)
        regime = detect_regime(test_data['close'])
        regime_suffix = "highvol" if regime == "high_vol" else "normal"
        expected_signature = compute_feature_signature(env.numeric_columns)

        ppo_dir = LOCAL_PPO_DIR
        candidate = ppo_dir / ppo_filename(ticker, regime_suffix, expected_signature)

        if not candidate.exists():
            mismatched = sorted(ppo_dir.glob(ppo_filename(ticker, regime_suffix, "*").replace(expected_signature, "*")))
            mismatched = [path for path in mismatched if path.name != candidate.name]
            if mismatched:
                logging.warning(
                    "Skipping PPO model for %s: no file matching feature signature %s (found %s)",
                    ticker,
                    expected_signature,
                    [path.name for path in mismatched],
                )
                self.ppo_model = None
                return 0.0

            legacy_candidate = ppo_dir / legacy_ppo_filename(ticker, regime_suffix)
            if legacy_candidate.exists():
                candidate = legacy_candidate
            else:
                logging.warning(f"No PPO model found for {ticker} at {candidate}; skipping PPO run.")
                self.ppo_model = None
                return 0.0

        try:
            self.ppo_model = PPO.load(str(candidate))
        except Exception as exc:
            logging.warning(f"Failed to load PPO model %s: %s", candidate, exc)
            self.ppo_model = None
            return 0.0

        if self.ppo_model.observation_space.shape != env.observation_space.shape:
            logging.warning(
                "Discarding PPO model %s: observation space mismatch %s != %s; retrain required.",
                candidate,
                self.ppo_model.observation_space.shape,
                env.observation_space.shape,
            )
            self.ppo_model = None
            return 0.0
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"PPO test run total reward: {total_reward}")
        self._emit_event(
            component="ppo",
            category="evaluation",
            event="run_complete",
            symbol=ticker,
            corr_id=self._corr(ticker),
            duration_ms=(time.time() - start) * 1000,
            data={
                "reward": float(total_reward),
                "rows": int(len(test_data)),
            },
        )
        return total_reward

    def train_ensemble_ppo(self, train_data, n_models=3):
        frame = _ppo_observation_frame(train_data)
        if frame is None or frame.empty:
            logging.warning("PPO ensemble training skipped: invalid training frame.")
            return

        self.ensemble_models = []
        for i in range(n_models):
            env = TradingEnv(frame.copy())
            model = PPO("MlpPolicy", env, verbose=0, seed=i)
            model.learn(total_timesteps=10000)
            self.ensemble_models.append(model)
        print(f"Trained ensemble of {n_models} PPO models.")

    def run_ensemble_ppo(self, test_data):
        frame = _ppo_observation_frame(test_data)
        if frame is None or frame.empty:
            logging.warning("PPO ensemble run skipped: invalid test frame.")
            return 0
        env = TradingEnv(frame)
        obs, info = env.reset()
        done = False
        total_reward = 0
        actions = []
        while not done:
            action_votes = [model.predict(obs)[0] for model in self.ensemble_models]
            majority_action = max(set(action_votes), key=action_votes.count)
            confidence = action_votes.count(majority_action) / len(action_votes)
            if confidence < 0.6:
                majority_action = 0  # Hold if not confident
            obs, reward, terminated, truncated, info = env.step(majority_action)
            done = terminated or truncated
            total_reward += reward

# Example config for testing
default_config = {
    "tickers": list(IBKR_SYMBOLS),
    "use_mock_data": False,
    "feature_params": {"window": 20, "indicators": ["sma", "ema", "rsi"]},
    "log_path": "logs/pipeline.log",
    "s3_bucket": "omega-singularity-ml",
    "max_retries": 3,
    "retry_delay": 10,
    "min_samples_per_model": 1440,
    "ml_full_min_samples": 1440,
    "ml_incremental_min_samples": 120,
    "ml_rolling_window": 1440,
    "ml_retrain_every_min": 60,
    # AI_ASSISTANT_TODO: Set to False before enabling live IBKR data!
    "disable_drift_detection": True,  # TEMP: Disable drift detection for mock data testing
    "drift_detection_threshold": 10.0  # Relaxed for mock data; set to 3.0 for production
}

import time


def main():
    """Run the trading pipeline loop."""
    while True:
        pipeline = RLTradingPipeline(default_config)
        pipeline.run()
        # Sleep for 2 minutes (adjust as needed)
        time.sleep(120)


if __name__ == "__main__":
    main()
