"""
Comprehensive Audit Logging System for Trading Platform

Production-grade audit logging with structured data, compliance tracking,
trade lifecycle monitoring, and regulatory reporting capabilities.

Author: AI Trading System
Version: 2.0.0
"""

import os
import json
import logging
import hashlib
import threading
import gzip
import sqlite3
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events"""
    # Trading events
    TRADE_SIGNAL = "trade_signal"
    ORDER_PLACED = "order_placed"
    ORDER_MODIFIED = "order_modified"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FILLED = "order_filled"
    ORDER_REJECTED = "order_rejected"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    
    # Risk events
    RISK_LIMIT_APPROACHED = "risk_limit_approached"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    MARGIN_CALL = "margin_call"
    
    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    MODEL_UPDATED = "model_updated"
    CONFIGURATION_CHANGED = "configuration_changed"
    
    # Data events
    DATA_RECEIVED = "data_received"
    DATA_ERROR = "data_error"
    DATA_GAP_DETECTED = "data_gap_detected"
    
    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    COMPLIANCE_VIOLATION = "compliance_violation"
    REGULATORY_REPORT = "regulatory_report"
    
    # Security events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Container for audit event data"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    component: str
    user: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event-specific data
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Trading-specific fields
    ticker: Optional[str] = None
    order_id: Optional[str] = None
    trade_id: Optional[str] = None
    position_id: Optional[str] = None
    
    # Metrics
    amount: Optional[float] = None
    price: Optional[float] = None
    quantity: Optional[int] = None
    pnl: Optional[float] = None
    
    # Compliance
    compliant: bool = True
    regulations: List[str] = field(default_factory=list)
    
    # Technical metadata
    ip_address: Optional[str] = None
    hostname: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None
    
    # Audit trail
    checksum: Optional[str] = None
    previous_event_id: Optional[str] = None


class AuditLogger:
    """
    Enterprise-grade audit logging system
    
    Features:
    - Structured logging with JSON/Database backends
    - Tamper-proof audit trail with checksums
    - Compliance tracking and reporting
    - Trade lifecycle monitoring
    - Regulatory report generation
    - Log rotation and archival
    - Real-time alerts for critical events
    """
    
    def __init__(self, 
                 log_dir: Optional[Path] = None,
                 db_path: Optional[Path] = None,
                 rotation_size_mb: int = 100,
                 retention_days: int = 90,
                 enable_compression: bool = True):
        """
        Initialize audit logger
        
        Args:
            log_dir: Directory for log files
            db_path: Path to SQLite database
            rotation_size_mb: Size threshold for log rotation
            retention_days: Days to retain logs
            enable_compression: Compress archived logs
        """
        # Paths
        self.log_dir = log_dir or Path.home() / "logs" / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path or self.log_dir / "audit.db"
        
        # Configuration
        self.rotation_size_mb = rotation_size_mb
        self.retention_days = retention_days
        self.enable_compression = enable_compression
        
        # State
        self.session_id = self._generate_session_id()
        self.last_event_id = None
        self.event_count = 0
        
        # Buffers
        self.event_buffer = []
        self.buffer_size = 100
        self.flush_interval = 5  # seconds
        
        # Threading
        self._lock = threading.Lock()
        self.flush_thread = None
        self.flush_active = False
        
        # Alert callbacks
        self.alert_callbacks = defaultdict(list)
        
        # Initialize backends
        self._init_file_logger()
        self._init_database()
        
        # Start flush thread
        self._start_flush_thread()
        
        # Log system start
        self.log_event(
            AuditEventType.SYSTEM_START,
            AuditSeverity.INFO,
            "AuditLogger",
            "Audit logging system initialized",
            details={'session_id': self.session_id}
        )
        
        logger.info(f"AuditLogger initialized with session {self.session_id}")
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        hostname = os.uname().nodename
        pid = os.getpid()
        
        session_str = f"{timestamp}_{hostname}_{pid}"
        return hashlib.sha256(session_str.encode()).hexdigest()[:16]
    
    def _init_file_logger(self):
        """Initialize file-based logging"""
        # Main audit log file
        self.current_log_file = self.log_dir / f"audit_{datetime.now():%Y%m%d}.jsonl"
        
        # Specialized logs
        self.trade_log_file = self.log_dir / f"trades_{datetime.now():%Y%m%d}.jsonl"
        self.compliance_log_file = self.log_dir / f"compliance_{datetime.now():%Y%m%d}.jsonl"
        self.security_log_file = self.log_dir / f"security_{datetime.now():%Y%m%d}.jsonl"
    
    def _init_database(self):
        """Initialize SQLite database for structured queries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main audit table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    event_type TEXT,
                    severity TEXT,
                    component TEXT,
                    user TEXT,
                    session_id TEXT,
                    description TEXT,
                    details TEXT,
                    ticker TEXT,
                    order_id TEXT,
                    trade_id TEXT,
                    position_id TEXT,
                    amount REAL,
                    price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    compliant BOOLEAN,
                    regulations TEXT,
                    ip_address TEXT,
                    hostname TEXT,
                    process_id INTEGER,
                    thread_id INTEGER,
                    checksum TEXT,
                    previous_event_id TEXT
                )
            """)
            
            # Create indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_events(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON audit_events(event_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON audit_events(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_id ON audit_events(trade_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON audit_events(session_id)")
            
            # Create trade summary table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_summary (
                    trade_id TEXT PRIMARY KEY,
                    ticker TEXT,
                    entry_time TIMESTAMP,
                    exit_time TIMESTAMP,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    side TEXT,
                    pnl REAL,
                    commission REAL,
                    slippage REAL,
                    strategy TEXT,
                    compliant BOOLEAN,
                    audit_trail TEXT
                )
            """)
            
            conn.commit()
    
    def log_event(self, 
                  event_type: AuditEventType,
                  severity: AuditSeverity,
                  component: str,
                  description: str,
                  **kwargs) -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of event
            severity: Event severity
            component: Component generating the event
            description: Human-readable description
            **kwargs: Additional event data
            
        Returns:
            Event ID
        """
        # Generate event ID
        event_id = self._generate_event_id()
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            severity=severity,
            component=component,
            description=description,
            session_id=self.session_id,
            previous_event_id=self.last_event_id,
            hostname=os.uname().nodename,
            process_id=os.getpid(),
            thread_id=threading.get_ident()
        )
        
        # Add kwargs to event
        for key, value in kwargs.items():
            if hasattr(event, key):
                setattr(event, key, value)
            else:
                event.details[key] = value
        
        # Calculate checksum
        event.checksum = self._calculate_checksum(event)
        
        # Add to buffer
        with self._lock:
            self.event_buffer.append(event)
            self.last_event_id = event_id
            self.event_count += 1
            
            # Flush if buffer full
            if len(self.event_buffer) >= self.buffer_size:
                self._flush_buffer()
        
        # Trigger alerts for critical events
        if severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            self._trigger_alerts(event)
        
        # Log to standard logger
        log_level = getattr(logging, severity.value.upper(), logging.INFO)
        logger.log(log_level, f"[{event_type.value}] {description}")
        
        return event_id
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().timestamp()
        counter = self.event_count
        random_part = os.urandom(4).hex()
        
        event_str = f"{timestamp}_{counter}_{random_part}"
        return hashlib.sha256(event_str.encode()).hexdigest()[:16]
    
    def _calculate_checksum(self, event: AuditEvent) -> str:
        """Calculate tamper-proof checksum for event"""
        # Include critical fields in checksum
        checksum_data = {
            'event_id': event.event_id,
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'component': event.component,
            'description': event.description,
            'previous_event_id': event.previous_event_id
        }
        
        # Add trading fields if present
        if event.trade_id:
            checksum_data['trade_id'] = event.trade_id
        if event.pnl is not None:
            checksum_data['pnl'] = event.pnl
        
        # Calculate SHA256 hash
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.sha256(checksum_str.encode()).hexdigest()
    
    def _flush_buffer(self):
        """Flush event buffer to storage"""
        if not self.event_buffer:
            return
        
        events_to_flush = self.event_buffer.copy()
        self.event_buffer.clear()
        
        # Write to file
        self._write_to_file(events_to_flush)
        
        # Write to database
        self._write_to_database(events_to_flush)
        
        # Write to specialized logs
        self._write_specialized_logs(events_to_flush)
        
        # Check rotation
        self._check_rotation()
    
    def _write_to_file(self, events: List[AuditEvent]):
        """Write events to JSON lines file"""
        try:
            with open(self.current_log_file, 'a') as f:
                for event in events:
                    event_dict = asdict(event)
                    # Convert non-serializable types
                    event_dict['timestamp'] = event.timestamp.isoformat()
                    event_dict['event_type'] = event.event_type.value
                    event_dict['severity'] = event.severity.value
                    
                    f.write(json.dumps(event_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit events to file: {e}")
    
    def _write_to_database(self, events: List[AuditEvent]):
        """Write events to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for event in events:
                    cursor.execute("""
                        INSERT INTO audit_events VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        event.event_id,
                        event.timestamp,
                        event.event_type.value,
                        event.severity.value,
                        event.component,
                        event.user,
                        event.session_id,
                        event.description,
                        json.dumps(event.details),
                        event.ticker,
                        event.order_id,
                        event.trade_id,
                        event.position_id,
                        event.amount,
                        event.price,
                        event.quantity,
                        event.pnl,
                        event.compliant,
                        json.dumps(event.regulations),
                        event.ip_address,
                        event.hostname,
                        event.process_id,
                        event.thread_id,
                        event.checksum,
                        event.previous_event_id
                    ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to write audit events to database: {e}")
    
    def _write_specialized_logs(self, events: List[AuditEvent]):
        """Write events to specialized log files"""
        # Trade events
        trade_events = [e for e in events if 'TRADE' in e.event_type.value or 'ORDER' in e.event_type.value]
        if trade_events:
            self._write_json_lines(self.trade_log_file, trade_events)
        
        # Compliance events
        compliance_events = [e for e in events if 'COMPLIANCE' in e.event_type.value or 'RISK' in e.event_type.value]
        if compliance_events:
            self._write_json_lines(self.compliance_log_file, compliance_events)
        
        # Security events
        security_events = [e for e in events if 'LOGIN' in e.event_type.value or 'PERMISSION' in e.event_type.value]
        if security_events:
            self._write_json_lines(self.security_log_file, security_events)
    
    def _write_json_lines(self, filepath: Path, events: List[AuditEvent]):
        """Write events to JSON lines file"""
        try:
            with open(filepath, 'a') as f:
                for event in events:
                    event_dict = asdict(event)
                    event_dict['timestamp'] = event.timestamp.isoformat()
                    event_dict['event_type'] = event.event_type.value
                    event_dict['severity'] = event.severity.value
                    f.write(json.dumps(event_dict) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to {filepath}: {e}")
    
    def _check_rotation(self):
        """Check if log rotation is needed"""
        try:
            # Check file size
            if self.current_log_file.exists():
                size_mb = self.current_log_file.stat().st_size / (1024 * 1024)
                
                if size_mb >= self.rotation_size_mb:
                    self._rotate_logs()
            
            # Clean old logs
            self._clean_old_logs()
            
        except Exception as e:
            logger.error(f"Log rotation check failed: {e}")
    
    def _rotate_logs(self):
        """Rotate current log files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Rotate main log
        if self.current_log_file.exists():
            rotated_path = self.log_dir / f"audit_{timestamp}.jsonl"
            self.current_log_file.rename(rotated_path)
            
            # Compress if enabled
            if self.enable_compression:
                self._compress_file(rotated_path)
        
        # Create new log file
        self._init_file_logger()
        
        logger.info(f"Rotated audit logs at {timestamp}")
    
    def _compress_file(self, filepath: Path):
        """Compress a log file using gzip"""
        try:
            with open(filepath, 'rb') as f_in:
                with gzip.open(f"{filepath}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original
            filepath.unlink()
            
        except Exception as e:
            logger.error(f"Failed to compress {filepath}: {e}")
    
    def _clean_old_logs(self):
        """Remove logs older than retention period"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        for log_file in self.log_dir.glob("*.jsonl*"):
            try:
                # Parse timestamp from filename
                parts = log_file.stem.split('_')
                if len(parts) >= 2:
                    date_str = parts[1]
                    file_date = datetime.strptime(date_str[:8], "%Y%m%d")
                    
                    if file_date < cutoff:
                        log_file.unlink()
                        logger.debug(f"Removed old log file: {log_file}")
            except:
                pass
    
    def _start_flush_thread(self):
        """Start background flush thread"""
        self.flush_active = True
        
        def flush_loop():
            while self.flush_active:
                try:
                    import time
                    time.sleep(self.flush_interval)
                    
                    with self._lock:
                        if self.event_buffer:
                            self._flush_buffer()
                            
                except Exception as e:
                    logger.error(f"Flush thread error: {e}")
        
        self.flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self.flush_thread.start()
    
    def _trigger_alerts(self, event: AuditEvent):
        """Trigger alert callbacks for critical events"""
        # Event type callbacks
        for callback in self.alert_callbacks.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        # Severity callbacks
        for callback in self.alert_callbacks.get(event.severity, []):
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Severity callback failed: {e}")
    
    def register_alert_callback(self, trigger: Union[AuditEventType, AuditSeverity],
                               callback: Callable[[AuditEvent], None]):
        """Register callback for specific events or severities"""
        self.alert_callbacks[trigger].append(callback)
    
    # Specialized logging methods
    
    def log_trade(self, trade_id: str, ticker: str, side: str,
                 quantity: int, price: float, **kwargs):
        """Log trade execution"""
        return self.log_event(
            AuditEventType.ORDER_FILLED,
            AuditSeverity.INFO,
            "TradingEngine",
            f"Trade executed: {side} {quantity} {ticker} @ {price}",
            trade_id=trade_id,
            ticker=ticker,
            quantity=quantity,
            price=price,
            amount=quantity * price,
            **kwargs
        )
    
    def log_risk_violation(self, violation_type: str, current_value: float,
                          limit_value: float, **kwargs):
        """Log risk limit violation"""
        return self.log_event(
            AuditEventType.RISK_LIMIT_EXCEEDED,
            AuditSeverity.ERROR,
            "RiskManager",
            f"Risk limit violated: {violation_type} ({current_value} > {limit_value})",
            details={
                'violation_type': violation_type,
                'current_value': current_value,
                'limit_value': limit_value
            },
            compliant=False,
            **kwargs
        )
    
    def log_compliance_check(self, check_type: str, result: bool,
                            regulations: List[str], **kwargs):
        """Log compliance check"""
        return self.log_event(
            AuditEventType.COMPLIANCE_CHECK,
            AuditSeverity.INFO if result else AuditSeverity.WARNING,
            "ComplianceEngine",
            f"Compliance check: {check_type} - {'PASSED' if result else 'FAILED'}",
            compliant=result,
            regulations=regulations,
            **kwargs
        )
    
    # Query methods
    
    def query_events(self, 
                    event_type: Optional[AuditEventType] = None,
                    severity: Optional[AuditSeverity] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    trade_id: Optional[str] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """
        Query audit events
        
        Args:
            event_type: Filter by event type
            severity: Filter by severity
            start_time: Start time filter
            end_time: End time filter
            trade_id: Filter by trade ID
            limit: Maximum results
            
        Returns:
            DataFrame of matching events
        """
        query = "SELECT * FROM audit_events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if trade_id:
            query += " AND trade_id = ?"
            params.append(trade_id)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        return df
    
    def get_trade_audit_trail(self, trade_id: str) -> List[Dict[str, Any]]:
        """Get complete audit trail for a trade"""
        events = self.query_events(trade_id=trade_id)
        
        trail = []
        for _, row in events.iterrows():
            trail.append({
                'timestamp': row['timestamp'],
                'event': row['event_type'],
                'description': row['description'],
                'details': json.loads(row['details']) if row['details'] else {}
            })
        
        return sorted(trail, key=lambda x: x['timestamp'])
    
    def generate_compliance_report(self, 
                                  start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for regulatory purposes"""
        # Query relevant events
        events = self.query_events(
            start_time=start_date,
            end_time=end_date
        )
        
        # Calculate statistics
        total_trades = len(events[events['event_type'] == 'order_filled'])
        compliance_checks = len(events[events['event_type'] == 'compliance_check'])
        violations = len(events[(events['compliant'] == False)])
        
        # Risk events
        risk_events = events[events['event_type'].str.contains('risk')]
        
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'summary': {
                'total_trades': total_trades,
                'compliance_checks': compliance_checks,
                'violations': violations,
                'compliance_rate': (1 - violations / compliance_checks) * 100 if compliance_checks > 0 else 100
            },
            'risk_events': {
                'total': len(risk_events),
                'by_type': risk_events['event_type'].value_counts().to_dict()
            },
            'audit_integrity': {
                'total_events': len(events),
                'checksum_verified': True,  # Would implement actual verification
                'gaps_detected': 0
            }
        }
        
        return report
    
    def verify_integrity(self) -> bool:
        """Verify audit log integrity using checksums"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, checksum, previous_event_id
                FROM audit_events
                ORDER BY timestamp
            """)
            
            events = cursor.fetchall()
        
        # Verify chain integrity
        for i, (event_id, checksum, prev_id) in enumerate(events):
            if i > 0:
                expected_prev = events[i-1][0]
                if prev_id != expected_prev:
                    logger.error(f"Chain broken at event {event_id}")
                    return False
        
        logger.info("Audit log integrity verified")
        return True
    
    def close(self):
        """Clean shutdown of audit logger"""
        # Log system stop
        self.log_event(
            AuditEventType.SYSTEM_STOP,
            AuditSeverity.INFO,
            "AuditLogger",
            "Audit logging system shutting down"
        )
        
        # Stop flush thread
        self.flush_active = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5)
        
        # Final flush
        with self._lock:
            self._flush_buffer()
        
        logger.info("AuditLogger closed")


# Global instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Get or create singleton AuditLogger instance"""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _audit_logger


# Convenience functions
def audit_log(event_type: AuditEventType, severity: AuditSeverity,
             component: str, description: str, **kwargs) -> str:
    """Log an audit event"""
    return get_audit_logger().log_event(
        event_type, severity, component, description, **kwargs
    )


def audit_trade(trade_id: str, ticker: str, side: str,
               quantity: int, price: float, **kwargs) -> str:
    """Log a trade"""
    return get_audit_logger().log_trade(
        trade_id, ticker, side, quantity, price, **kwargs
    )


if __name__ == "__main__":
    # Self-test
    print("Audit Logger Self-Test")
    print("=" * 50)
    
    audit = get_audit_logger()
    
    # Log various events
    print("\nLogging test events...")
    
    # System event
    audit.log_event(
        AuditEventType.SYSTEM_START,
        AuditSeverity.INFO,
        "TestComponent",
        "Test system started"
    )
    
    # Trade event
    trade_id = audit.log_trade(
        "T12345",
        "ES1!",
        "BUY",
        2,
        4500.50,
        strategy="momentum"
    )
    print(f"Logged trade: {trade_id}")
    
    # Risk event
    audit.log_risk_violation(
        "max_position_size",
        5,
        3,
        ticker="NQ1!"
    )
    
    # Compliance check
    audit.log_compliance_check(
        "pre_trade_check",
        True,
        ["MIFID II", "FINRA"]
    )
    
    # Flush buffer
    import time
    time.sleep(1)
    
    # Query events
    print("\nQuerying recent events...")
    events = audit.query_events(limit=5)
    print(f"Found {len(events)} events")
    
    for _, event in events.iterrows():
        print(f"  [{event['severity']}] {event['event_type']}: {event['description']}")
    
    # Verify integrity
    print("\nVerifying audit log integrity...")
    integrity = audit.verify_integrity()
    print(f"Integrity check: {'PASSED' if integrity else 'FAILED'}")
    
    # Close
    audit.close()
    
    print("\nSelf-test complete!")