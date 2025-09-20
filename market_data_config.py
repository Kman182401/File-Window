# IBKR TWS/Gateway market data config (Level 2 ready, robust, and efficient)

# ───────────────────────────────── Risk limits (paper / learning) ─────────────────────────────────
# Max daily loss as a percentage of starting equity
MAX_DAILY_LOSS_PCT = 0.02  # 2%

# Max number of trades allowed per day
MAX_TRADES_PER_DAY = 20

# Max position exposure (absolute value)
MAX_POSITION_EXPOSURE = 3  # e.g., never hold more than 3 contracts long or short

# Max order size (contracts per order)
MAX_ORDER_SIZE = 2  # e.g., never send an order for more than 2 contracts

# ───────────────────────────────── IBKR connection ─────────────────────────────────
# IB Gateway publishes API on localhost:4002 (paper trading) or 4001 (live)
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002        # IB Gateway API port (4002 for paper trading)
IBKR_CLIENT_ID = 9002   # default client id (can be overridden by env IBKR_CLIENT_ID)

# Connection/retry settings
IBKR_RETRY_INTERVAL = 5   # seconds between reconnect attempts
IBKR_TIMEOUT = 30         # seconds for API call timeouts

# Market data type
# Options: "REALTIME", "DELAYED", "FROZEN"
IBKR_MARKET_DATA_TYPE = "REALTIME"

# Logging and monitoring
IBKR_LOGGING_ENABLED = True
IBKR_LOG_FILE = "ibkr_market_data.log"
IBKR_LOG_LEVEL = "INFO"   # Options: DEBUG, INFO, WARNING, ERROR

# Dynamic subscription management
IBKR_DYNAMIC_SUBSCRIPTION = True  # Only subscribe to active symbols

# Resource monitoring (optional, for scaling)
IBKR_RESOURCE_MONITORING = True

# Main loop poll interval (optional, for pipeline sleep)
IBKR_POLL_INTERVAL = 1  # seconds

# ───────────────────────────────── Canonical symbols (must match adapter mapping) ─────────────────────────────────
# These are the tickers your pipeline will iterate; the adapter maps each to the correct
# front-month *futures* contract (no options, no spot):
#
# ES1!   → ES (CME)     E-mini S&P 500 futures
# NQ1!   → NQ (CME)     E-mini Nasdaq-100 futures
# GBPUSD → 6B (CME)     British Pound futures
# EURUSD → 6E (CME)     Euro FX futures
# AUDUSD → 6A (CME)     Australian Dollar futures
# XAUUSD → GC (COMEX)   Gold futures
#
IBKR_SYMBOLS = ["ES1!", "NQ1!", "XAUUSD", "EURUSD", "GBPUSD", "AUDUSD"]
