# ============================================================================
# ENHANCED FEATURE ENGINEERING - v2.0 with Performance Tracking
# ============================================================================

import os
import sys
sys.path.append('/home/ubuntu')

import boto3
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from io import BytesIO
import logging

# Import enhanced modules
from monitoring.performance_tracker import get_performance_tracker
from utils.memory_manager import memory_efficient, get_memory_manager
from utils.exception_handler import retry_on_exception, DataFetchError
from utils.audit_logger import audit_log, AuditEventType, AuditSeverity
from config.master_config import config_get

# Initialize components
logger = logging.getLogger(__name__)
performance_tracker = get_performance_tracker()
memory_manager = get_memory_manager()

# === Advanced Market Microstructure Features ===

def compute_order_flow_imbalance(order_book):
    """
    Calculate order flow imbalance from order book.
    order_book: dict with 'bids' and 'asks' (price, size)
    Returns: float (imbalance)
    """
    bid_volume = sum([size for price, size in order_book['bids']])
    ask_volume = sum([size for price, size in order_book['asks']])
    return (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-9)

def compute_queue_position(order_book, my_order_price, my_order_size, side='buy'):
    """
    Estimate queue position for a given order.
    """
    book_side = order_book['bids'] if side == 'buy' else order_book['asks']
    queue_ahead = sum([size for price, size in book_side if (price > my_order_price if side == 'buy' else price < my_order_price)])
    return queue_ahead

def compute_order_book_features(order_book):
    """
    Extracts order book analytics: spread, depth, top-of-book liquidity.
    """
    best_bid = order_book['bids'][0][0] if order_book['bids'] else None
    best_ask = order_book['asks'][0][0] if order_book['asks'] else None
    spread = best_ask - best_bid if best_bid and best_ask else None
    depth = sum([size for price, size in order_book['bids'] + order_book['asks']])
    return {
        'spread': spread,
        'depth': depth,
        'best_bid': best_bid,
        'best_ask': best_ask
    }

# S3 configuration (real values)
S3_BUCKET = "omega-singularity-ml"
RAW_DATA_PREFIX = ""  # Root of the bucket for raw data
PROCESSED_DATA_PREFIX = "processed"

# List of tickers to process
TICKERS = ["ES1!", "NQ1!", "6A", "6E", "6B", "GC"]

s3 = boto3.client("s3")

def list_s3_files(bucket, prefix):
    """List all files in S3 under the given prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

def read_s3_file(bucket, key, data_type='TRADES'):
    """
    Read a file from S3 and return as a pandas DataFrame.
    Supports TRADES, MIDPOINT, etc. by modifying the key.
    """
    # Insert data_type into the filename before .csv or .parquet
    if key.endswith(".csv"):
        key = key.replace(".csv", f"_{data_type}.csv")
    elif key.endswith(".parquet"):
        key = key.replace(".parquet", f"_{data_type}.parquet")
    else:
        raise ValueError("Unsupported file type: " + key)
    obj = s3.get_object(Bucket=bucket, Key=key)
    if key.endswith(".csv"):
        return pd.read_csv(obj["Body"])
    elif key.endswith(".parquet"):
        return pd.read_parquet(BytesIO(obj["Body"].read()))

def add_technical_indicators(df):
    """Add advanced technical indicators and features for ML/RL trading."""
    import time
    start_time = time.time()
    
    # PHASE 4A: Feature reduction optimization
    USE_OPTIMIZED_FEATURES = os.getenv('USE_OPTIMIZED_FEATURES', 'true').lower() == 'true'
    logger.info(f"PHASE4A: Using {'30' if USE_OPTIMIZED_FEATURES else '47'} features mode")
    
    # Ensure required columns exist
    if not all(col in df.columns for col in ["close", "high", "low", "open", "volume"]):
        return df

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()  # Band width (volatility)

    # ATR (Average True Range)
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"]
    ).average_true_range()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        high=df["high"], low=df["low"], close=df["close"]
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # VWAP (Volume Weighted Average Price)
    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    # Volume SMA and spikes
    df["vol_sma_20"] = df["volume"].rolling(window=20).mean()
    df["vol_spike"] = (df["volume"] > 1.5 * df["vol_sma_20"]).astype(int)

    # Time-of-day/session encoding (minute of day, hour, day of week)
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"])
        df["minute_of_day"] = dt.dt.hour * 60 + dt.dt.minute
        df["hour"] = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek
    # If no datetime column, just skip time-based features (no halt)

    # Lagged returns (1, 5, 15 periods)
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    # Rolling correlation with open (as a proxy for related asset, can be replaced)
    df["roll_corr_open"] = df["close"].rolling(window=20).corr(df["open"])

    # === ADVANCED TECHNICAL INDICATORS ===
    
    # Ichimoku Cloud components
    high_9 = df["high"].rolling(window=9).max()
    low_9 = df["low"].rolling(window=9).min()
    df["tenkan_sen"] = (high_9 + low_9) / 2  # Conversion Line
    
    high_26 = df["high"].rolling(window=26).max()
    low_26 = df["low"].rolling(window=26).min()
    df["kijun_sen"] = (high_26 + low_26) / 2  # Base Line
    
    # Williams %R
    if not USE_OPTIMIZED_FEATURES:
        df["williams_r"] = ta.momentum.WilliamsRIndicator(
            high=df["high"], low=df["low"], close=df["close"]
        ).williams_r()
    else:
        logger.debug("PHASE4A: Skipping redundant williams_r calculation")
    
    # Commodity Channel Index (CCI)
    df["cci"] = ta.trend.CCIIndicator(
        high=df["high"], low=df["low"], close=df["close"]
    ).cci()
    
    # Money Flow Index (MFI) - Volume-weighted RSI
    df["mfi"] = ta.volume.MFIIndicator(
        high=df["high"], low=df["low"], close=df["close"], volume=df["volume"]
    ).money_flow_index()
    
    # Aroon Indicators
    aroon = ta.trend.AroonIndicator(high=df["high"], low=df["low"])
    if not USE_OPTIMIZED_FEATURES:
        df["aroon_up"] = aroon.aroon_up()
        df["aroon_down"] = aroon.aroon_down()
        df["aroon_oscillator"] = df["aroon_up"] - df["aroon_down"]
    else:
        # Keep only the oscillator as it's the most informative
        df["aroon_oscillator"] = aroon.aroon_up() - aroon.aroon_down()
        logger.debug("PHASE4A: Skipping redundant aroon_up and aroon_down, keeping oscillator")
    
    # Ultimate Oscillator - Multi-timeframe momentum
    if not USE_OPTIMIZED_FEATURES:
        df["ultimate_osc"] = ta.momentum.UltimateOscillator(
            high=df["high"], low=df["low"], close=df["close"]
        ).ultimate_oscillator()
    else:
        logger.debug("PHASE4A: Skipping redundant ultimate_osc calculation")
    
    # === MARKET MICROSTRUCTURE INDICATORS ===
    
    # Price momentum across different timeframes (beyond existing ret_1, ret_5, ret_15)
    if not USE_OPTIMIZED_FEATURES:
        df["momentum_3"] = df["close"].pct_change(3)
        df["momentum_10"] = df["close"].pct_change(10) 
        df["momentum_20"] = df["close"].pct_change(20)
    else:
        logger.debug("PHASE4A: Skipping redundant momentum_3, momentum_10, momentum_20 calculations")
    
    # Relative Strength across timeframes
    if not USE_OPTIMIZED_FEATURES:
        df["rs_3_10"] = df["momentum_3"] / (df["momentum_10"] + 1e-6)
        df["rs_5_20"] = df["ret_5"] / (df["momentum_20"] + 1e-6)
    else:
        logger.debug("PHASE4A: Skipping redundant rs_3_10 and rs_5_20 calculations")
    
    # On-Balance Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(
        close=df["close"], volume=df["volume"]
    ).on_balance_volume()
    
    # === VOLATILITY-BASED INDICATORS ===
    
    # Keltner Channels
    if not USE_OPTIMIZED_FEATURES:
        keltner = ta.volatility.KeltnerChannel(
            high=df["high"], low=df["low"], close=df["close"]
        )
        df["keltner_high"] = keltner.keltner_channel_hband()
        df["keltner_low"] = keltner.keltner_channel_lband()
        df["keltner_width"] = df["keltner_high"] - df["keltner_low"]
    else:
        logger.debug("PHASE4A: Skipping redundant keltner_high, keltner_low, keltner_width calculations")
    
    # === MARKET REGIME INDICATORS ===
    
    # Trend strength using ADX
    adx = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"])
    df["adx"] = adx.adx()
    if not USE_OPTIMIZED_FEATURES:
        df["adx_pos"] = adx.adx_pos()
        df["adx_neg"] = adx.adx_neg()
    else:
        logger.debug("PHASE4A: Skipping redundant adx_pos and adx_neg, keeping main adx")
    
    # Market efficiency ratio (trending vs choppy)
    df["efficiency_ratio"] = abs(df["close"] - df["close"].shift(10)) / (
        df["close"].diff().abs().rolling(10).sum() + 1e-6
    )
    
    # === CUSTOM MOMENTUM INDICATORS ===
    
    # Rate of Change (ROC) multiple periods
    if not USE_OPTIMIZED_FEATURES:
        df["roc_3"] = df["close"].pct_change(3) * 100
        df["roc_10"] = df["close"].pct_change(10) * 100
        df["roc_20"] = df["close"].pct_change(20) * 100
    else:
        logger.debug("PHASE4A: Skipping redundant roc_3, roc_10, roc_20 calculations")
    
    # Triple EMA (TEMA) for smoother trend following
    ema1 = df["close"].ewm(span=14).mean()
    ema2 = ema1.ewm(span=14).mean()
    ema3 = ema2.ewm(span=14).mean()
    df["tema"] = 3 * ema1 - 3 * ema2 + ema3
    
    # Z-Score for mean reversion signals
    df["zscore_20"] = (df["close"] - df["close"].rolling(20).mean()) / (
        df["close"].rolling(20).std() + 1e-6
    )

    # Clean up
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # PHASE 4A: Performance tracking
    end_time = time.time()
    feature_count = len(df.columns) - 5  # Subtract OHLCV columns
    processing_time = end_time - start_time
    logger.info(f"PHASE4A: Generated {feature_count} features in {processing_time:.3f}s")
    
    return df

def process_ticker(ticker):
    """Process all files for a given ticker."""
    ticker_prefix = f"{RAW_DATA_PREFIX}{ticker}" if RAW_DATA_PREFIX else ticker
    processed_frames = []
    for key in list_s3_files(S3_BUCKET, ticker_prefix):
        df = read_s3_file(S3_BUCKET, key)
        df = add_technical_indicators(df)
        df["asset"] = ticker
        processed_frames.append(df)
    if processed_frames:
        return pd.concat(processed_frames, ignore_index=True)
    return None

def main():
    all_data = []
    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df = process_ticker(ticker)
        if df is not None:
            all_data.append(df)
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        # Save to S3 as CSV
        output_key = f"{PROCESSED_DATA_PREFIX}/features_all_assets.csv"
        csv_buffer = BytesIO()
        final_df.to_csv(csv_buffer, index=False)
        s3.put_object(Bucket=S3_BUCKET, Key=output_key, Body=csv_buffer.getvalue())
        print(f"Processed features saved to s3://{S3_BUCKET}/{output_key}")
    else:
        print("No data processed.")

@memory_efficient(max_mb=200)
@retry_on_exception(max_retries=2, delay=1, exceptions=(DataFetchError,))
def generate_features(df):
    print("MARKET FEATURE ENGINEERING RUNNING")
    print("NEWS FEATURE ENGINEERING RUNNING")
    """
    Given a raw OHLCV DataFrame, add technical indicators and return features and binary up/down labels.
    Features: all columns except 'label'.
    Labels: 1 if next close > current close, else 0.
    Enhanced with performance tracking and memory management.
    """
    # PHASE4A-FIX: Add debug logging for input
    if hasattr(df, 'shape'):
        logger.info(f"[FE] generate_features input shape: {df.shape}")

    with performance_tracker.track_operation("generate_features"):
        # Ensure required columns are present and correct type
        required_columns = ["open", "high", "low", "close", "volume"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"Data validation error: Missing required column: {col}")
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Ensure timestamp column exists and is datetime
    if "timestamp" not in df.columns:
        if "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"])
        else:
            df["timestamp"] = pd.date_range(start="2000-01-01", periods=len(df), freq="min")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # Drop rows with missing values in required columns
    # PHASE4A-FIX: Only drop rows with missing CRITICAL values
    critical_columns = ["close", "volume"]
    initial_len = len(df)
    df = df.dropna(subset=critical_columns + ["timestamp"])
    if len(df) < initial_len:
        logger.info(f"[FE] Dropped {initial_len - len(df)} rows with missing critical values")

    df = add_technical_indicators(df.copy())
    # Ensure a timestamp column exists for incremental learning
    if "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
    elif isinstance(df.index, pd.DatetimeIndex):
        # Handle timezone-aware datetime index properly
        df["timestamp"] = df.index.tz_convert('UTC') if df.index.tz else df.index.tz_localize('UTC')
    elif df.index.name is not None:
        # Use pandas type checking instead of numpy for tz-aware compatibility
        from pandas.api.types import is_datetime64_any_dtype
        if is_datetime64_any_dtype(df.index):
            df["timestamp"] = pd.DatetimeIndex(df.index).tz_localize('UTC') if df.index.tz is None else df.index
    else:
        df["timestamp"] = np.arange(len(df))

    # PHASE4A-FIX: Smart filtering instead of blanket dropna()
    # Only drop rows where essential features are NaN
    essential_features = ["close", "volume", "returns"]
    pre_filter_len = len(df)

    # First, ensure returns column exists
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change()

    # Drop only rows with NaN in essential features
    df = df.dropna(subset=[col for col in essential_features if col in df.columns])

    # Forward-fill remaining NaNs in non-essential columns
    df = df.fillna(method='ffill').fillna(0)

    if len(df) < pre_filter_len:
        logger.info(f"[FE] Smart filter: kept {len(df)}/{pre_filter_len} rows")

    # Binary classification: 1 if next close > current close, else 0
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = df.drop(['label'], axis=1).iloc[:-1]
    labels = df['label'].iloc[:-1]

    # PHASE4A-FIX: Guarantee UTC datetime index and timestamp column
    # Handle various index types
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.DatetimeIndex(features.index)

    # Convert to UTC safely
    try:
        if features.index.tz is None:
            features.index = features.index.tz_localize("UTC")
        else:
            features.index = features.index.tz_convert("UTC")
    except Exception as e:
        logger.warning(f"[FE] Timezone conversion issue: {e}, using naive datetime")
        # If conversion fails, ensure we at least have datetime index
        features.index = pd.DatetimeIndex(features.index.tz_localize(None) if features.index.tz else features.index)

    # Ensure we expose timestamp as datetime (keep as datetime, not int for proper comparisons)
    features["timestamp"] = features.index  # Keep as datetime with UTC timezone

    # PHASE4A-FIX: Log final output shape
    logger.info(f"[FE] generate_features output: features.shape={features.shape}, labels.shape={labels.shape}")

    return features, labels

    print(df.columns)

def classify_news_type(headline, body=None):
    """
    Classifies news into types based on keywords.
    Args:
        headline (str): News headline.
        body (str, optional): News body/content.
    Returns:
        str: News type label ("earnings", "macro", "M&A", "regulatory", "other").
    """
    text = (headline or "") + " " + (body or "")
    text = text.lower()
    if any(word in text for word in ["earnings", "eps", "revenue", "profit", "quarterly results"]):
        return "earnings"
    if any(word in text for word in ["fed", "interest rate", "central bank", "inflation", "cpi", "macro", "gdp", "unemployment", "fomc", "ecb", "boe", "boj"]):
        return "macro"
    if any(word in text for word in ["merger", "acquisition", "buyout", "takeover", "m&a"]):
        return "M&A"
    if any(word in text for word in ["regulation", "regulatory", "sec", "fine", "settlement", "compliance", "lawsuit"]):
        return "regulatory"
    return "other"

def map_macro_news_to_tickers(text):
    """
    Maps macroeconomic or indirect news references to relevant tickers.
    Args:
        text (str): News headline or body.
    Returns:
        List[str]: List of tickers/entities inferred to be impacted.
    """
    text = text.lower()
    tickers = set()
    # Macro terms mapping
    macro_map = {
        "federal reserve": ["ES1!", "NQ1!", "USD", "GC"],
        "interest rate": ["ES1!", "NQ1!", "USD", "GC"],
        "inflation": ["ES1!", "NQ1!", "USD", "GC"],
        "cpi": ["ES1!", "NQ1!", "USD", "GC"],
        "gdp": ["ES1!", "NQ1!", "USD", "GC"],
        "ecb": ["6E"],
        "boe": ["6B"],
        "rba": ["6A"],
        "trade deal": ["ES1!", "NQ1!", "USD", "GC"],
        "oil": ["GC"],
    }
    for macro_term, macro_tickers in macro_map.items():
        if macro_term in text:
            tickers.update(macro_tickers)
    return list(tickers)

def extract_features_from_level2_market_data(market_data, depth=10):
    """
    Extract advanced features from Level 2 (order book) market data.
    Assumes market_data contains 'bids' and 'asks' as lists of [price, size] pairs.
    """
    features = {}

    # Extract top-of-book (Level 1) features
    if market_data.get("bids") and market_data.get("asks"):
        best_bid = market_data["bids"][0][0]
        best_ask = market_data["asks"][0][0]
        features["best_bid"] = best_bid
        features["best_ask"] = best_ask
        features["spread"] = best_ask - best_bid
        features["bid_size"] = market_data["bids"][0][1]
        features["ask_size"] = market_data["asks"][0][1]
    else:
        features["best_bid"] = 0
        features["best_ask"] = 0
        features["spread"] = 0
        features["bid_size"] = 0
        features["ask_size"] = 0

    # Level 2 features: order book depth, imbalance, etc.
    depth = min(len(market_data.get("bids", [])), len(market_data.get("asks", [])), depth)
    features["book_imbalance"] = (
        sum([b[1] for b in market_data.get("bids", [])[:depth]]) -
        sum([a[1] for a in market_data.get("asks", [])[:depth]])
    )
    features["bid_depth"] = sum([b[1] for b in market_data.get("bids", [])[:depth]])
    features["ask_depth"] = sum([a[1] for a in market_data.get("asks", [])[:depth]])

    # Optionally, add more advanced features here

    return features

if __name__ == "__main__":
    main()
