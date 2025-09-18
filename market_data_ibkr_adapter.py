# ============================================================================
# ENHANCED IBKR ADAPTER - v2.0 with Health Monitoring and Error Recovery
# ============================================================================

from ib_insync import IB, util, Contract, Future, MarketOrder, LimitOrder, Order, Trade, Position, PortfolioItem
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Callable
import logging
import time
from utils.persist_market_data import persist_bars
import asyncio
import os
from dataclasses import dataclass
from enum import Enum
from utils.ibkr_pacing import PaceGate

# Import enhanced modules
import sys

from utils.exception_handler import retry_on_exception, circuit_breaker, IBKRConnectionError, DataFetchError
from utils.ibkr_health_monitor import IBKRHealthMonitor
from monitoring.performance_tracker import get_performance_tracker
from utils.audit_logger import audit_log, AuditEventType, AuditSeverity
from config.master_config import config_get

# Initialize components
logger = logging.getLogger(__name__)
performance_tracker = get_performance_tracker()
health_monitor = None  # Will be initialized per connection
pace_gate = PaceGate(rate_per_sec=3.0, burst=3, dupe_cooldown_s=15.0)


def get_ibkr_front_month_contract(
    ib: IB,
    symbol: str,
    exchange: str,
    currency: str,
    asof: Optional[datetime] = None,
    tradingClass: Optional[str] = None
):
    """
    Return the nearest non-expired *futures* contract for the given root as of a reference date.

    Parameters
    ----------
    ib : IB
        Connected ib_insync.IB instance.
    symbol : str
        Futures root, e.g. 'ES','NQ','6E','6B','6A','GC'.
    exchange : str
        Primary exchange, e.g. 'CME', 'COMEX'.
    currency : str
        Contract currency, e.g. 'USD'.
    asof : datetime or None
        Reference date to choose the first non-expired contract (UTC).
    tradingClass : str or None
        Explicit tradingClass filter (recommended for currency futures like 6E/6B/6A).
    """
    ex = exchange.upper()
    # Try reasonable synonyms for CME/GLOBEX
    if ex in ("GLOBEX", "CME"):
        exchange_candidates = ["CME", "GLOBEX"]
    else:
        exchange_candidates = [ex]

    ref_ymd = int((asof or datetime.utcnow()).strftime("%Y%m%d"))

    for ex_name in exchange_candidates:
        # Prefer querying with tradingClass when provided (currency futures)
        template = (
            Future(symbol=symbol, exchange=ex_name, currency=currency, tradingClass=tradingClass)
            if tradingClass else
            Future(symbol=symbol, exchange=ex_name, currency=currency)
        )
        details = ib.reqContractDetails(template)
        if not details:
            continue

        # Optionally filter to the right tradingClass
        if tradingClass:
            details = [d for d in details if getattr(d.contract, "tradingClass", None) == tradingClass]
            if not details:
                continue

        def expiry_int(cd):
            exp = cd.contract.lastTradeDateOrContractMonth or ""
            # Normalize YYYYMM → YYYYMM01 so it can be int-compared with YYYYMMDD
            exp = (exp + "01") if len(exp) == 6 else exp
            try:
                return int(exp)
            except ValueError:
                return 99999999

        # sort by expiry and pick first FUT not expired as of ref_ymd
        details.sort(key=expiry_int)
        for cd in details:
            exp_i = expiry_int(cd)
            if exp_i >= ref_ymd and getattr(cd.contract, "secType", "FUT") == "FUT":
                ib.qualifyContracts(cd.contract)
                return cd.contract

    raise RuntimeError(
        f"No valid (non-expired) futures contract found for {symbol} "
        f"on {exchange_candidates} as of {ref_ymd} (tradingClass={tradingClass})."
    )


class IBKRIngestor:
    # Symbol alias mapping: alternative forms -> canonical forms
    SYMBOL_ALIASES = {
        "GC": "XAUUSD",
        "6E": "EURUSD", 
        "6B": "GBPUSD",
        "6A": "AUDUSD",
        "XAUUSD": "XAUUSD",
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD", 
        "AUDUSD": "AUDUSD",
        "ES1!": "ES1!",
        "NQ1!": "NQ1!"
    }
    
    def _canonical_symbol(self, symbol: str) -> str:
        """Convert symbol to canonical form, case-insensitive lookup."""
        normalized = symbol.strip().upper()
        return self.SYMBOL_ALIASES.get(normalized, symbol)

    def __init__(self, host=None, port=None, clientId=None, ib=None):
        import os
        import time
        import threading

        # Get connection params from config or environment
        self.host = host or config_get("ibkr.host") or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or config_get("ibkr.port") or os.getenv("IBKR_PORT", "4002"))
        self.clientId = int(clientId or config_get("ibkr.client_id") or os.getenv("IBKR_CLIENT_ID", "9002"))

        # Initialize health monitor ONLY if not in single-socket mode AND not using external IB
        self.health_monitor = None

        # Check if we should skip monitor (single-socket mode or external IB provided)
        disable_monitor = os.getenv("DISABLE_HEALTH_MONITOR", "0") in ("1", "true", "True", "YES", "yes")

        if not disable_monitor and ib is None:
            # Only spawn monitor if NOT in single-socket mode AND not using external IB
            global health_monitor
            self.health_monitor = IBKRHealthMonitor(self.host, self.port, self.clientId)
            health_monitor = self.health_monitor
            self.health_monitor.start_monitoring()
            print(f"[IBKRIngestor] Health monitor started (host={self.host}, port={self.port})")
        else:
            print(f"[IBKRIngestor] Health monitor disabled (single-socket mode={disable_monitor}, external IB={ib is not None})")

        # Use provided IB connection or create new one
        if ib is not None:
            self.ib = ib
            self.connected = ib.isConnected()
            self.external_ib = True
            print(f"[IBKRIngestor] Using external IB connection (connected={self.connected})")
        else:
            self.ib = IB()
            self.connected = False
            self.external_ib = False
        self.auto_reconnect = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = config_get("ibkr.retry_attempts", 10)
        self.reconnect_delay = 5  # seconds
        self._connection_lock = threading.Lock()
        
        # Audit log connection attempt
        audit_log(
            AuditEventType.CONNECTION_ESTABLISHED,
            AuditSeverity.INFO,
            "IBKRIngestor",
            f"Attempting connection to {self.host}:{self.port}",
            details={"client_id": self.clientId}
        )
        
        # Attach event hooks if using an external IB instance
        if self.external_ib:
            try:
                self._attach_event_handlers()
            except Exception:
                pass

        # Initial connection with enhanced error handling (skip if using external IB)
        if not self.external_ib:
            self._connect()

        # Start auto-reconnect monitor thread (can be disabled for testing or external IB)
        if not os.getenv('DISABLE_HEALTH_MONITOR') and not self.external_ib:
            self._monitor_thread = threading.Thread(target=self._connection_monitor, daemon=True)
            self._monitor_thread.start()
        else:
            print("[IBKRIngestor] Connection monitor thread disabled")
        
    def _attach_event_handlers(self):
        """Attach IB error/disconnect event hooks once per IB instance."""
        if getattr(self, "_events_hooked", False):
            return
        try:
            def _on_error(reqId, code, msg, _):
                if code == 1100:
                    logging.warning("[IB] 1100 LOST: %s", msg)
                elif code in (1101, 1102):
                    logging.warning("[IB] %s RECONNECTED: %s", code, msg)
                    # small settle before issuing large requests again
                    time.sleep(1.5)
            self.ib.errorEvent += _on_error

            def _on_disconnected():
                logging.warning("[IB] disconnectedEvent → scheduling reconnect")
                # monitor thread handles reconnect; this gives immediate visibility
            self.ib.disconnectedEvent += _on_disconnected

            self._events_hooked = True
        except Exception:
            # Do not let event hook failure break the adapter
            pass

    def _connect(self):
        """Internal connection method with retry logic."""
        # If using external IB connection, don't try to reconnect
        if self.external_ib:
            self.connected = self.ib.isConnected()
            return self.connected

        with self._connection_lock:
            try:
                if self.connected and self.ib.isConnected():
                    return True
                    
                print(f"[IBKRIngestor] connecting to {self.host}:{self.port} clientId={self.clientId}")
                
                # Disconnect if already connected but not working
                if self.ib.isConnected():
                    self.ib.disconnect()
                
                # PHASE4A-FIX: IB connection using proper async handling
                print(f"[IBKRIngestor] PHASE4A-FIX: IB connection using proper async handling")
                
                try:
                    # Ensure ib_insync event loop is running
                    util.startLoop()
                    print(f"[IBKRIngestor] Started ib_insync event loop")
                    
                    # Now use standard sync connect (which works properly with startLoop)
                    self.ib.connect(self.host, self.port, clientId=self.clientId, timeout=12)
                    print(f"[IBKRIngestor] Connected successfully using util.startLoop + sync connect")
                    # Attach event hooks and allow brief post-login settle
                    try:
                        self._attach_event_handlers()
                    except Exception:
                        pass
                    time.sleep(1.5)
                    
                except Exception as connect_error:
                    print(f"[IBKRIngestor] Connection failed: {connect_error}")
                    raise
                self.connected = True
                self.reconnect_attempts = 0
                print(f"[IBKRIngestor] successfully connected to IB Gateway")
                return True
                
            except Exception as e:
                self.connected = False
                self.reconnect_attempts += 1
                print(f"[IBKRIngestor] ERROR: Failed to connect to IB Gateway at {self.host}:{self.port} - {e}")
                
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    print(f"[IBKRIngestor] MAX RECONNECT ATTEMPTS REACHED ({self.max_reconnect_attempts})")
                    raise
                return False
    
    def _connection_monitor(self):
        """Background thread to monitor and maintain connection."""
        import time
        
        while self.auto_reconnect:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                if not self.connected or not self.ib.isConnected():
                    print("[IBKRIngestor] Connection lost, attempting reconnect...")
                    self._connect()
                    
            except Exception as e:
                print(f"[IBKRIngestor] Connection monitor error: {e}")
                
    def ensure_connected(self):
        """Ensure connection is active before operations."""
        # If using external IB connection, just check if it's connected
        if self.external_ib:
            self.connected = self.ib.isConnected()
            if not self.connected:
                print(f"[IBKRIngestor] External IB connection lost! Connected={self.connected}")
                # Don't try to reconnect - let the single socket handler deal with it
                raise RuntimeError("External IB connection lost - single socket handler should manage reconnection")
            return self.connected

        if not self.connected or not self.ib.isConnected():
            print("[IBKRIngestor] Connection not active, reconnecting...")
            return self._connect()
        return True

    def disconnect(self):
        """Disconnect from IB Gateway and stop auto-reconnect."""
        self.auto_reconnect = False  # Stop the monitor thread
        try:
            if self.connected and self.ib.isConnected():
                self.ib.disconnect()
        finally:
            self.connected = False

    @retry_on_exception(max_retries=3, delay=2, exceptions=(IBKRConnectionError, DataFetchError))
    def fetch_data(
        self,
        ticker,
        duration="3 D",  # INCREASED: Was 1D, now 3D for technical indicators
        barSize="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=1,
        endDateTime="",
        asof: Optional[datetime] = None
    ):
        """
        Fetch historical bars for the mapped futures contract.
        Returns a DataFrame with columns: ['timestamp','open','high','low','close','volume'] (UTC).
        Enhanced with retry logic, memory management, and performance tracking.
        """
        # Start performance tracking
        with performance_tracker.track_operation(f"fetch_data_{ticker}"):
            # Ensure connection is active
            if not self.ensure_connected():
                error_msg = f"Unable to connect to IB Gateway for {ticker}"
                audit_log(
                    AuditEventType.CONNECTION_LOST,
                    AuditSeverity.ERROR,
                    "IBKRIngestor",
                    error_msg,
                    ticker=ticker
                )
                raise IBKRConnectionError(error_msg)
            
        if not self.connected:
            raise RuntimeError(f"IBKR not connected. Cannot fetch data for {ticker}.")

        canonical_ticker = self._canonical_symbol(ticker)
        contract = self._get_contract(canonical_ticker, asof=asof)
        if contract is None:
            raise RuntimeError(f"No contract available for {ticker}.")

        print(f"Requesting contract: {contract}")  # debug log

        try:
            # Pacing gate (per contract, ticktype) with dupe suppression
            try:
                # Robust fingerprint across naming variants and unqualified contracts
                sym  = getattr(contract, 'localSymbol', None) or getattr(contract, 'symbol', '?')
                exp  = getattr(contract, 'lastTradeDateOrContractMonth', '') or getattr(getattr(contract, 'comboLegsDescrip', None), '', '')
                exch = getattr(contract, 'exchange', '')
                dstr = locals().get('durationStr',  locals().get('duration',  ''))
                bss  = locals().get('barSizeSetting', locals().get('barSize', ''))
                edt  = locals().get('endDateTime', '')
                fp   = f"{sym}-{exp}-{exch}-{whatToShow}-{dstr}-{bss}-{edt}"
                cid  = getattr(contract, 'conId', None)
                lane = (cid or (sym, exp, exch), whatToShow)
                pace_gate.acquire(lane, fp)
            except Exception:
                pass  # never block if pacing helper fails

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=endDateTime,
                durationStr=duration,
                barSizeSetting=barSize,
                whatToShow=whatToShow,
                useRTH=useRTH,
                formatDate=formatDate,
            )
            
            # ADDED: Delay to avoid IB rate limits
            import time
            time.sleep(0.35)  # 350ms delay between requests
            
            df = util.df(bars)
            if df is None or df.empty:
                raise RuntimeError(f"No data returned for {ticker} (contract: {contract}).")

            # Ensure a UTC 'timestamp' column exists
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            elif df.index.name == "date":
                df = df.reset_index().rename(columns={"date": "timestamp"})
            else:
                df["timestamp"] = df.index

            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

            keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            cols = [c for c in keep_cols if c in df.columns]
            
            try:
                _sym = locals().get('symbol') or locals().get('ticker') or 'UNKNOWN'
                if df is not None and not df.empty and _sym != 'UNKNOWN':
                    persist_bars(_sym, df)
            except Exception as _e:
                print(f"[persist] warning: {_e}")

            return df[cols]
        except Exception as e:
            raise RuntimeError(f"Exception during IBKR data fetch for {ticker}: {e}")

    def _get_contract(self, ticker, asof: Optional[datetime] = None):
        """
        Map your system's canonical symbols to IBKR futures roots (front month).
        Futures only (no options, no spot).
        """
        # ----- Equity index futures (CME) -----
        if ticker == "ES1!":
            # ES - E-mini S&P 500 (CME)
            return get_ibkr_front_month_contract(self.ib, symbol="ES", exchange="CME",
                                                 currency="USD", asof=asof, tradingClass="ES")

        elif ticker == "NQ1!":
            # NQ - E-mini Nasdaq-100 (CME)
            return get_ibkr_front_month_contract(self.ib, symbol="NQ", exchange="CME",
                                                 currency="USD", asof=asof, tradingClass="NQ")

        # ----- FX futures (CME) -----
        elif ticker == "GBPUSD":
            # 6B - British Pound (CME)
            return get_ibkr_front_month_contract(self.ib, symbol="6B", exchange="CME",
                                                 currency="USD", asof=asof, tradingClass="6B")

        elif ticker == "EURUSD":
            # 6E - Euro FX (CME)
            return get_ibkr_front_month_contract(self.ib, symbol="6E", exchange="CME",
                                                 currency="USD", asof=asof, tradingClass="6E")

        elif ticker == "AUDUSD":
            # 6A - Australian Dollar (CME)
            return get_ibkr_front_month_contract(self.ib, symbol="6A", exchange="CME",
                                                 currency="USD", asof=asof, tradingClass="6A")

        # ----- Metals futures (COMEX) -----
        elif ticker == "XAUUSD":
            # GC - Gold (COMEX)
            return get_ibkr_front_month_contract(self.ib, symbol="GC", exchange="COMEX",
                                                 currency="USD", asof=asof, tradingClass="GC")

        else:
            raise ValueError(f"Unknown ticker: {ticker}.")
    
    # ============================================================================
    # ORDER PLACEMENT AND PORTFOLIO MANAGEMENT
    # ============================================================================
    
    def place_market_order(self, ticker: str, quantity: float, action: str = "BUY") -> Optional[Trade]:
        """
        Place a market order for the specified ticker
        
        Args:
            ticker: Symbol ticker (e.g., "ES1!", "NQ1!")
            quantity: Number of contracts (positive number, direction determined by action)
            action: "BUY" or "SELL"
            
        Returns:
            Trade object if successful, None if failed
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot place order - not connected to IB Gateway")
                return None
            
            # Get contract
            contract = self.map_ticker_to_contract(ticker)
            if not contract:
                logger.error(f"Failed to get contract for {ticker}")
                return None
            
            # Create market order
            order = MarketOrder(action=action, totalQuantity=abs(quantity))
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Log order placement
            logger.info(f"Market order placed: {action} {quantity} {ticker}")
            audit_log(
                AuditEventType.SYSTEM_START,
                AuditSeverity.INFO,
                "IBKRIngestor",
                f"Market order placed: {action} {quantity} {ticker}",
                ticker=ticker,
                quantity=quantity,
                action=action,
                order_type="MARKET"
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place market order for {ticker}: {e}")
            audit_log(
                AuditEventType.SYSTEM_START,
                AuditSeverity.ERROR,
                "IBKRIngestor",
                f"Failed to place market order: {e}",
                ticker=ticker,
                error=str(e)
            )
            return None
    
    def place_limit_order(self, ticker: str, quantity: float, price: float, action: str = "BUY") -> Optional[Trade]:
        """
        Place a limit order for the specified ticker
        
        Args:
            ticker: Symbol ticker
            quantity: Number of contracts
            price: Limit price
            action: "BUY" or "SELL"
            
        Returns:
            Trade object if successful, None if failed
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot place order - not connected to IB Gateway")
                return None
            
            # Get contract
            contract = self.map_ticker_to_contract(ticker)
            if not contract:
                logger.error(f"Failed to get contract for {ticker}")
                return None
            
            # Create limit order
            order = LimitOrder(action=action, totalQuantity=abs(quantity), lmtPrice=price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Log order placement
            logger.info(f"Limit order placed: {action} {quantity} {ticker} @ ${price}")
            audit_log(
                AuditEventType.SYSTEM_START,
                AuditSeverity.INFO,
                "IBKRIngestor",
                f"Limit order placed: {action} {quantity} {ticker} @ ${price}",
                ticker=ticker,
                quantity=quantity,
                action=action,
                price=price,
                order_type="LIMIT"
            )
            
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place limit order for {ticker}: {e}")
            audit_log(
                AuditEventType.SYSTEM_START,
                AuditSeverity.ERROR,
                "IBKRIngestor",
                f"Failed to place limit order: {e}",
                ticker=ticker,
                error=str(e)
            )
            return None
    
    def cancel_order(self, trade: Trade) -> bool:
        """
        Cancel an order
        
        Args:
            trade: Trade object to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot cancel order - not connected to IB Gateway")
                return False
            
            # Cancel order
            self.ib.cancelOrder(trade.order)
            
            logger.info(f"Order cancelled: {trade.order.orderId}")
            audit_log(
                AuditEventType.SYSTEM_START,
                AuditSeverity.INFO,
                "IBKRIngestor",
                f"Order cancelled: {trade.order.orderId}",
                order_id=trade.order.orderId
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    def get_orders(self) -> List[Trade]:
        """
        Get all orders (open and filled)
        
        Returns:
            List of Trade objects
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot get orders - not connected to IB Gateway")
                return []
            
            # Get all trades
            trades = self.ib.trades()
            return trades
            
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    def get_open_orders(self) -> List[Trade]:
        """
        Get only open orders
        
        Returns:
            List of Trade objects for open orders
        """
        try:
            # Get all trades and filter for open ones
            all_trades = self.get_orders()
            open_trades = [trade for trade in all_trades if trade.orderStatus.status in ['PreSubmitted', 'Submitted']]
            return open_trades
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    def get_positions(self) -> List[Position]:
        """
        Get current positions
        
        Returns:
            List of Position objects
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot get positions - not connected to IB Gateway")
                return []
            
            # Get positions
            positions = self.ib.positions()
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def get_portfolio(self) -> List[PortfolioItem]:
        """
        Get portfolio items
        
        Returns:
            List of PortfolioItem objects
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot get portfolio - not connected to IB Gateway")
                return []
            
            # Get portfolio
            portfolio = self.ib.portfolio()
            return portfolio
            
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return []
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account summary information
        
        Returns:
            Dictionary with account information
        """
        try:
            # Ensure connection
            if not self.ensure_connected():
                logger.error("Cannot get account summary - not connected to IB Gateway")
                return {}
            
            # Get account summary
            account_values = self.ib.accountSummary()
            
            # Convert to dictionary
            summary = {}
            for item in account_values:
                summary[item.tag] = {
                    'value': item.value,
                    'currency': item.currency,
                    'account': item.account
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return {}
    
    def get_position_for_ticker(self, ticker: str) -> Optional[Position]:
        """
        Get position for a specific ticker
        
        Args:
            ticker: Symbol ticker (e.g., "ES1!")
            
        Returns:
            Position object if found, None otherwise
        """
        try:
            # Get contract for ticker
            contract = self.map_ticker_to_contract(ticker)
            if not contract:
                return None
            
            # Get all positions
            positions = self.get_positions()
            
            # Find position for this contract
            for position in positions:
                if (position.contract.symbol == contract.symbol and
                    position.contract.exchange == contract.exchange and
                    position.contract.secType == contract.secType):
                    return position
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get position for {ticker}: {e}")
            return None
    
    def get_order_status(self, trade: Trade) -> str:
        """
        Get current status of an order
        
        Args:
            trade: Trade object
            
        Returns:
            Order status string
        """
        try:
            # Update trade status
            self.ib.sleep(0.1)  # Allow for status updates
            return trade.orderStatus.status
            
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return "Unknown"
    
    def wait_for_order_fill(self, trade: Trade, timeout: int = 60) -> bool:
        """
        Wait for an order to be filled
        
        Args:
            trade: Trade object to monitor
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if filled, False if timeout or error
        """
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = self.get_order_status(trade)
                
                if status == "Filled":
                    return True
                elif status in ["Cancelled", "ApiCancelled", "Inactive"]:
                    return False
                
                time.sleep(0.5)  # Check every 500ms
            
            logger.warning(f"Order fill timeout after {timeout} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error waiting for order fill: {e}")
            return False
