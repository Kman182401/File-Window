"""
Comprehensive Audit Logging System for Phase 4A Trading Platform

This module provides comprehensive audit logging capabilities for all trading decisions,
optimization changes, system state changes, and compliance requirements.

Key Features:
- Complete audit trail of all trading decisions and reasoning
- Optimization change tracking with before/after metrics
- System state change monitoring
- Compliance reporting and validation
- Secure audit log storage with integrity verification
- Integration with performance monitoring and diagnostic systems
- Automated compliance report generation

Author: Security Compliance Auditor Agent
Version: 2.0.0 (Phase 4A Enhanced)
"""

import os
import json
import logging
import sqlite3
import hashlib
import uuid
import threading
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager
import psutil


class AuditEventType(Enum):
    """Types of events that can be audited"""
    TRADING_DECISION = "trading_decision"
    OPTIMIZATION_CHANGE = "optimization_change"
    SYSTEM_STATE_CHANGE = "system_state_change"
    PERFORMANCE_CHANGE = "performance_change"
    SECURITY_EVENT = "security_event"
    COMPLIANCE_CHECK = "compliance_check"
    DATA_ACCESS = "data_access"
    MODEL_UPDATE = "model_update"
    CONFIGURATION_CHANGE = "configuration_change"
    ERROR_EVENT = "error_event"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance status for audit events"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"
    EXEMPT = "exempt"


@dataclass
class AuditLogEntry:
    """Comprehensive audit log entry"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AuditEventType = AuditEventType.TRADING_DECISION
    severity: AuditSeverity = AuditSeverity.INFO
    
    # Event details
    title: str = ""
    description: str = ""
    component: str = ""
    operation: str = ""
    
    # Context
    session_id: Optional[str] = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "system"
    
    # Trading specific
    symbol: Optional[str] = None
    action: Optional[str] = None  # BUY, SELL, HOLD
    quantity: Optional[float] = None
    price: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    model_version: Optional[str] = None
    
    # System state
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Dict[str, Any] = field(default_factory=dict)
    system_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Impact assessment
    impact_level: str = "low"
    risk_assessment: str = ""
    compliance_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    
    # Technical details
    stack_trace: Optional[str] = None
    error_code: Optional[str] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Integrity
    checksum: Optional[str] = None


class ComprehensiveAuditLogger:
    """
    Comprehensive audit logging system with full compliance capabilities
    """
    
    def __init__(self, audit_path: Optional[str] = None, 
                 max_memory_entries: int = 10000,
                 auto_compliance_check: bool = True):
        """
        Initialize comprehensive audit logger
        
        Args:
            audit_path: Directory for audit logs
            max_memory_entries: Maximum entries to keep in memory
            auto_compliance_check: Whether to automatically run compliance checks
        """
        self.audit_path = Path(audit_path or Path.home() / "logs" / "audit")
        self.max_memory_entries = max_memory_entries
        self.auto_compliance_check = auto_compliance_check
        
        # Ensure audit directory exists
        self.audit_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for fast access
        self.memory_logs = deque(maxlen=max_memory_entries)
        self.session_logs = defaultdict(list)
        
        # Statistics and monitoring
        self.event_counts = defaultdict(int)
        self.compliance_violations = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.Lock()
        
        # Database for persistent storage
        self.db_path = self.audit_path / "audit_comprehensive.db"
        self._init_database()
        
        # Setup enhanced logging
        self._setup_enhanced_logging()
        
        # Compliance rules
        self.compliance_rules = self._init_compliance_rules()
        
        logger = logging.getLogger(__name__)
        logger.info("ComprehensiveAuditLogger initialized")
    
    def _init_database(self):
        """Initialize comprehensive audit database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_events (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        component TEXT,
                        operation TEXT,
                        session_id TEXT,
                        correlation_id TEXT,
                        user_id TEXT DEFAULT 'system',
                        symbol TEXT,
                        action TEXT,
                        quantity REAL,
                        price REAL,
                        confidence REAL,
                        reasoning TEXT,
                        model_version TEXT,
                        before_state TEXT,
                        after_state TEXT,
                        system_metrics TEXT,
                        impact_level TEXT,
                        risk_assessment TEXT,
                        compliance_status TEXT,
                        stack_trace TEXT,
                        error_code TEXT,
                        tags TEXT,
                        metadata TEXT,
                        checksum TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS compliance_checks (
                        check_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        check_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        violations_found INTEGER DEFAULT 0,
                        details TEXT,
                        recommendations TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_integrity (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        total_events INTEGER NOT NULL,
                        integrity_hash TEXT NOT NULL,
                        verified BOOLEAN DEFAULT TRUE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes for performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_correlation ON audit_events(correlation_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_symbol ON audit_events(symbol)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_audit_compliance ON audit_events(compliance_status)')
                
                conn.commit()
                
        except Exception as e:
            print(f"Failed to initialize audit database: {e}")
    
    def _setup_enhanced_logging(self):
        """Setup enhanced Python logging integration"""
        # Create audit-specific logger
        self.logger = logging.getLogger("comprehensive_audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler for audit logs
        log_file = self.audit_path / f"audit_{datetime.now():%Y%m%d}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Formatter for audit logs
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [AUDIT] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def _init_compliance_rules(self) -> Dict[str, Any]:
        """Initialize compliance rules and validation criteria"""
        return {
            "trading_decision": {
                "required_fields": ["symbol", "action", "confidence", "reasoning"],
                "confidence_threshold": 0.6,  # Minimum confidence for trades
                "max_position_size": 3.0,     # Maximum contracts per position
                "max_daily_trades": 20,       # Maximum trades per day
                "required_risk_assessment": True
            },
            "optimization_change": {
                "required_fields": ["component", "description", "before_state", "after_state"],
                "impact_assessment_required": True,
                "rollback_plan_required": True,
                "approval_required_for": ["aggressive", "critical"]
            },
            "system_state_change": {
                "required_fields": ["component", "operation", "impact_level"],
                "change_window_validation": True,
                "backup_verification": True
            },
            "data_access": {
                "log_all_access": True,
                "sensitive_data_flag": True,
                "access_pattern_monitoring": True
            }
        }
    
    def log_event(self, event_type: AuditEventType, title: str, description: str = "",
                  component: str = "", operation: str = "",
                  severity: AuditSeverity = AuditSeverity.INFO,
                  **kwargs) -> str:
        """
        Log a comprehensive audit event
        
        Args:
            event_type: Type of event being audited
            title: Event title
            description: Detailed description
            component: Component generating the event
            operation: Specific operation
            severity: Event severity
            **kwargs: Additional event-specific fields
            
        Returns:
            Event ID
        """
        # Create audit entry
        entry = AuditLogEntry(
            event_type=event_type,
            title=title,
            description=description,
            component=component,
            operation=operation,
            severity=severity,
            **{k: v for k, v in kwargs.items() if hasattr(AuditLogEntry, k)}
        )
        
        # Collect system metrics automatically
        entry.system_metrics = self._collect_system_metrics()
        
        # Generate integrity checksum
        entry.checksum = self._generate_checksum(entry)
        
        # Validate compliance
        if self.auto_compliance_check:
            entry.compliance_status = self._check_compliance(entry)
        
        # Store in memory and database
        with self._lock:
            self.memory_logs.append(entry)
            self.event_counts[event_type.value] += 1
            
            if entry.session_id:
                self.session_logs[entry.session_id].append(entry)
        
        # Persist to database
        self._persist_audit_entry(entry)
        
        # Log to file system
        self._log_to_file(entry)
        
        # Check for compliance violations
        if entry.compliance_status == ComplianceStatus.NON_COMPLIANT:
            self.compliance_violations.append(entry)
            self._alert_compliance_violation(entry)
        
        return entry.event_id
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics for audit context"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }
        except Exception:
            return {"collection_error": True, "timestamp": time.time()}
    
    def _generate_checksum(self, entry: AuditLogEntry) -> str:
        """Generate integrity checksum for audit entry"""
        # Create deterministic string representation
        data_to_hash = {
            "event_id": entry.event_id,
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type.value,
            "title": entry.title,
            "description": entry.description,
            "component": entry.component,
            "correlation_id": entry.correlation_id
        }
        
        # Sort keys for consistency
        data_str = json.dumps(data_to_hash, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def _check_compliance(self, entry: AuditLogEntry) -> ComplianceStatus:
        """Check entry against compliance rules"""
        event_type_key = entry.event_type.value
        
        if event_type_key not in self.compliance_rules:
            return ComplianceStatus.COMPLIANT
        
        rules = self.compliance_rules[event_type_key]
        
        # Check required fields
        required_fields = rules.get("required_fields", [])
        for field in required_fields:
            if not getattr(entry, field, None):
                return ComplianceStatus.NON_COMPLIANT
        
        # Specific validations by event type
        if event_type_key == "trading_decision":
            if entry.confidence and entry.confidence < rules.get("confidence_threshold", 0.6):
                return ComplianceStatus.REQUIRES_REVIEW
            
            if entry.quantity and entry.quantity > rules.get("max_position_size", 3.0):
                return ComplianceStatus.NON_COMPLIANT
        
        elif event_type_key == "optimization_change":
            if rules.get("impact_assessment_required") and not entry.risk_assessment:
                return ComplianceStatus.REQUIRES_REVIEW
        
        return ComplianceStatus.COMPLIANT
    
    def _persist_audit_entry(self, entry: AuditLogEntry):
        """Persist audit entry to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO audit_events 
                    (event_id, timestamp, event_type, severity, title, description,
                     component, operation, session_id, correlation_id, user_id,
                     symbol, action, quantity, price, confidence, reasoning, 
                     model_version, before_state, after_state, system_metrics,
                     impact_level, risk_assessment, compliance_status, stack_trace,
                     error_code, tags, metadata, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.event_id,
                    entry.timestamp.isoformat(),
                    entry.event_type.value,
                    entry.severity.value,
                    entry.title,
                    entry.description,
                    entry.component,
                    entry.operation,
                    entry.session_id,
                    entry.correlation_id,
                    entry.user_id,
                    entry.symbol,
                    entry.action,
                    entry.quantity,
                    entry.price,
                    entry.confidence,
                    entry.reasoning,
                    entry.model_version,
                    json.dumps(entry.before_state) if entry.before_state else None,
                    json.dumps(entry.after_state) if entry.after_state else None,
                    json.dumps(entry.system_metrics) if entry.system_metrics else None,
                    entry.impact_level,
                    entry.risk_assessment,
                    entry.compliance_status.value,
                    entry.stack_trace,
                    entry.error_code,
                    json.dumps(entry.tags) if entry.tags else None,
                    json.dumps(entry.metadata) if entry.metadata else None,
                    entry.checksum
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to persist audit entry: {e}")
    
    def _log_to_file(self, entry: AuditLogEntry):
        """Log entry to file system"""
        log_message = f"[{entry.event_type.value.upper()}] {entry.title}"
        if entry.symbol:
            log_message += f" | Symbol: {entry.symbol}"
        if entry.action:
            log_message += f" | Action: {entry.action}"
        if entry.component:
            log_message += f" | Component: {entry.component}"
        if entry.correlation_id:
            log_message += f" | ID: {entry.correlation_id[:8]}"
        
        # Log with appropriate level
        level_map = {
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.LOW: logging.INFO,
            AuditSeverity.MEDIUM: logging.WARNING,
            AuditSeverity.HIGH: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL
        }
        
        self.logger.log(level_map.get(entry.severity, logging.INFO), log_message)
    
    def _alert_compliance_violation(self, entry: AuditLogEntry):
        """Alert on compliance violations"""
        print(f"🚨 COMPLIANCE VIOLATION: {entry.title}")
        print(f"   Event ID: {entry.event_id}")
        print(f"   Type: {entry.event_type.value}")
        print(f"   Component: {entry.component}")
        print(f"   Status: {entry.compliance_status.value}")
        
        # Log violation to separate database table
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO compliance_checks 
                    (check_id, timestamp, check_type, status, violations_found, details)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    str(uuid.uuid4()),
                    entry.timestamp.isoformat(),
                    "automatic_violation_detection",
                    "violation_found",
                    1,
                    json.dumps({
                        "event_id": entry.event_id,
                        "violation_type": entry.event_type.value,
                        "details": entry.title
                    })
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to log compliance violation: {e}")
    
    # Convenience methods for specific event types
    def log_trading_decision(self, symbol: str, action: str, confidence: float,
                           reasoning: str, quantity: Optional[float] = None,
                           price: Optional[float] = None, model_version: Optional[str] = None,
                           **kwargs) -> str:
        """Log trading decision with full audit trail"""
        return self.log_event(
            AuditEventType.TRADING_DECISION,
            f"Trading Decision: {action} {symbol}",
            f"Decision to {action} {symbol} with {confidence:.3f} confidence",
            component="trading_engine",
            operation="make_trading_decision",
            severity=AuditSeverity.MEDIUM if confidence > 0.8 else AuditSeverity.LOW,
            symbol=symbol,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            quantity=quantity,
            price=price,
            model_version=model_version,
            risk_assessment=f"Confidence: {confidence:.3f}, Action: {action}",
            **kwargs
        )
    
    def log_optimization_change(self, component: str, description: str,
                              before_metrics: Optional[Dict[str, float]] = None,
                              after_metrics: Optional[Dict[str, float]] = None,
                              impact_assessment: Optional[str] = None,
                              rollback_plan: Optional[str] = None,
                              **kwargs) -> str:
        """Log optimization change with full context"""
        return self.log_event(
            AuditEventType.OPTIMIZATION_CHANGE,
            f"Optimization: {component}",
            description,
            component=component,
            operation="optimization_change",
            severity=AuditSeverity.MEDIUM,
            before_state=before_metrics or {},
            after_state=after_metrics or {},
            impact_level="medium",
            risk_assessment=impact_assessment or "Optimization change applied",
            metadata={"rollback_plan": rollback_plan} if rollback_plan else {},
            **kwargs
        )
    
    def log_system_state_change(self, component: str, operation: str,
                              before_state: Dict[str, Any],
                              after_state: Dict[str, Any],
                              impact_level: str = "low",
                              **kwargs) -> str:
        """Log system state change"""
        return self.log_event(
            AuditEventType.SYSTEM_STATE_CHANGE,
            f"System Change: {component}.{operation}",
            f"State change in {component} during {operation}",
            component=component,
            operation=operation,
            severity=AuditSeverity.MEDIUM if impact_level in ['high', 'critical'] else AuditSeverity.LOW,
            before_state=before_state,
            after_state=after_state,
            impact_level=impact_level,
            **kwargs
        )
    
    def log_performance_change(self, component: str, metric_name: str,
                             before_value: float, after_value: float,
                             improvement_percent: float,
                             **kwargs) -> str:
        """Log performance change"""
        severity = AuditSeverity.HIGH if abs(improvement_percent) > 50 else AuditSeverity.MEDIUM
        
        return self.log_event(
            AuditEventType.PERFORMANCE_CHANGE,
            f"Performance Change: {component}.{metric_name}",
            f"Performance change: {before_value:.3f} → {after_value:.3f} ({improvement_percent:+.1f}%)",
            component=component,
            operation="performance_monitoring",
            severity=severity,
            before_state={metric_name: before_value},
            after_state={metric_name: after_value},
            metadata={"improvement_percent": improvement_percent},
            **kwargs
        )
    
    def log_error_event(self, component: str, error_message: str,
                       stack_trace: Optional[str] = None,
                       error_code: Optional[str] = None,
                       **kwargs) -> str:
        """Log error event"""
        return self.log_event(
            AuditEventType.ERROR_EVENT,
            f"Error: {component}",
            error_message,
            component=component,
            operation="error_handling",
            severity=AuditSeverity.HIGH,
            stack_trace=stack_trace,
            error_code=error_code,
            **kwargs
        )
    
    def generate_compliance_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        if not time_window:
            time_window = timedelta(hours=24)
        
        cutoff = datetime.now() - time_window
        cutoff_str = cutoff.isoformat()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "time_window": str(time_window),
            "compliance_summary": {},
            "violations": [],
            "event_breakdown": {},
            "trading_analysis": {},
            "recommendations": []
        }
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Overall compliance summary
                cursor = conn.execute('''
                    SELECT compliance_status, COUNT(*) as count
                    FROM audit_events
                    WHERE timestamp > ?
                    GROUP BY compliance_status
                ''', (cutoff_str,))
                
                compliance_summary = {}
                total_events = 0
                for row in cursor.fetchall():
                    status, count = row
                    compliance_summary[status] = count
                    total_events += count
                
                report["compliance_summary"] = {
                    "total_events": total_events,
                    "compliance_breakdown": compliance_summary,
                    "compliance_rate": (compliance_summary.get("compliant", 0) / total_events * 100) if total_events > 0 else 100
                }
                
                # Get violations
                cursor = conn.execute('''
                    SELECT event_id, timestamp, event_type, title, component, compliance_status
                    FROM audit_events
                    WHERE timestamp > ? AND compliance_status != 'compliant'
                    ORDER BY timestamp DESC
                    LIMIT 50
                ''', (cutoff_str,))
                
                violations = []
                for row in cursor.fetchall():
                    violations.append({
                        "event_id": row[0],
                        "timestamp": row[1],
                        "event_type": row[2],
                        "title": row[3],
                        "component": row[4],
                        "status": row[5]
                    })
                
                report["violations"] = violations
                
                # Event breakdown
                cursor = conn.execute('''
                    SELECT event_type, COUNT(*) as count
                    FROM audit_events
                    WHERE timestamp > ?
                    GROUP BY event_type
                    ORDER BY count DESC
                ''', (cutoff_str,))
                
                event_breakdown = {}
                for row in cursor.fetchall():
                    event_breakdown[row[0]] = row[1]
                
                report["event_breakdown"] = event_breakdown
                
                # Trading analysis
                cursor = conn.execute('''
                    SELECT symbol, action, COUNT(*) as trade_count, AVG(confidence) as avg_confidence
                    FROM audit_events
                    WHERE timestamp > ? AND event_type = 'trading_decision' AND symbol IS NOT NULL
                    GROUP BY symbol, action
                    ORDER BY trade_count DESC
                ''', (cutoff_str,))
                
                trading_analysis = {}
                for row in cursor.fetchall():
                    symbol, action, count, avg_conf = row
                    key = f"{symbol}_{action}"
                    trading_analysis[key] = {
                        "symbol": symbol,
                        "action": action,
                        "count": count,
                        "avg_confidence": round(avg_conf, 3) if avg_conf else 0
                    }
                
                report["trading_analysis"] = trading_analysis
                
        except Exception as e:
            print(f"Failed to generate compliance report: {e}")
        
        # Generate recommendations
        recommendations = []
        
        compliance_rate = report["compliance_summary"].get("compliance_rate", 100)
        if compliance_rate < 95:
            recommendations.append("Compliance rate below 95% - review audit procedures")
        
        violation_count = len(report["violations"])
        if violation_count > 10:
            recommendations.append(f"{violation_count} compliance violations detected - immediate review required")
        
        if report["event_breakdown"].get("error_event", 0) > 50:
            recommendations.append("High error event count - system stability review recommended")
        
        report["recommendations"] = recommendations
        
        return report
    
    @contextmanager
    def audit_session(self, session_name: str):
        """Context manager for audit session tracking"""
        session_id = f"{session_name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        
        self.log_event(
            AuditEventType.SYSTEM_STATE_CHANGE,
            f"Audit Session Started: {session_name}",
            f"Beginning audit session: {session_name}",
            component="audit_logger",
            operation="session_start",
            session_id=session_id
        )
        
        try:
            yield session_id
        except Exception as e:
            self.log_event(
                AuditEventType.ERROR_EVENT,
                f"Audit Session Error: {session_name}",
                f"Error in audit session: {str(e)}",
                component="audit_logger",
                operation="session_error",
                session_id=session_id,
                severity=AuditSeverity.HIGH,
                stack_trace=str(e)
            )
            raise
        finally:
            self.log_event(
                AuditEventType.SYSTEM_STATE_CHANGE,
                f"Audit Session Ended: {session_name}",
                f"Ending audit session: {session_name}",
                component="audit_logger",
                operation="session_end",
                session_id=session_id
            )


# Global comprehensive audit logger
_comprehensive_audit_logger = None


def get_comprehensive_audit_logger() -> ComprehensiveAuditLogger:
    """Get or create singleton ComprehensiveAuditLogger instance"""
    global _comprehensive_audit_logger
    
    if _comprehensive_audit_logger is None:
        _comprehensive_audit_logger = ComprehensiveAuditLogger()
    
    return _comprehensive_audit_logger


# Legacy compatibility functions
def log_trade_results(message):
    """
    Legacy function for backward compatibility
    Logs trade results or audit messages.
    """
    audit_logger = get_comprehensive_audit_logger()
    audit_logger.log_event(
        AuditEventType.TRADING_DECISION,
        "Legacy Trade Result",
        message,
        component="legacy_system",
        operation="trade_result"
    )


# Enhanced convenience functions
def audit_trading_decision(symbol: str, action: str, confidence: float, 
                          reasoning: str, **kwargs):
    """Audit a trading decision"""
    return get_comprehensive_audit_logger().log_trading_decision(
        symbol, action, confidence, reasoning, **kwargs
    )


def audit_optimization_change(component: str, description: str, **kwargs):
    """Audit an optimization change"""
    return get_comprehensive_audit_logger().log_optimization_change(
        component, description, **kwargs
    )


def audit_system_change(component: str, operation: str, before_state: Dict, 
                       after_state: Dict, **kwargs):
    """Audit a system state change"""
    return get_comprehensive_audit_logger().log_system_state_change(
        component, operation, before_state, after_state, **kwargs
    )


def audit_performance_change(component: str, metric_name: str, 
                           before_value: float, after_value: float, **kwargs):
    """Audit a performance change"""
    improvement = ((after_value - before_value) / before_value * 100) if before_value != 0 else 0
    return get_comprehensive_audit_logger().log_performance_change(
        component, metric_name, before_value, after_value, improvement, **kwargs
    )


def audit_error(component: str, error_message: str, **kwargs):
    """Audit an error event"""
    return get_comprehensive_audit_logger().log_error_event(
        component, error_message, **kwargs
    )


if __name__ == "__main__":
    # Self-test
    print("Comprehensive Audit Logger Self-Test")
    print("=" * 50)
    
    audit_logger = ComprehensiveAuditLogger()
    
    # Test trading decision logging
    print("Testing trading decision logging...")
    audit_trading_decision(
        "ES1!", "BUY", 0.85, "Strong bullish signal detected",
        quantity=2.0, price=4500.25, model_version="PPO_v2.1"
    )
    
    # Test optimization change logging
    print("Testing optimization change logging...")
    audit_optimization_change(
        "feature_engineering", "Optimized technical indicators calculation",
        before_metrics={"latency_ms": 150, "memory_mb": 500},
        after_metrics={"latency_ms": 120, "memory_mb": 450},
        impact_assessment="20% latency improvement, 10% memory reduction"
    )
    
    # Test system state change
    print("Testing system state change logging...")
    audit_system_change(
        "ibkr_gateway", "restart",
        {"status": "running", "connections": 1},
        {"status": "restarted", "connections": 0}
    )
    
    # Test performance change
    print("Testing performance change logging...")
    audit_performance_change(
        "model_inference", "prediction_latency_ms", 250.0, 180.0
    )
    
    # Test error logging
    print("Testing error logging...")
    audit_error(
        "data_ingestion", "Failed to fetch market data",
        error_code="TIMEOUT_ERROR",
        stack_trace="TimeoutError: Connection timeout after 30s"
    )
    
    # Test session context
    print("Testing session context...")
    with audit_logger.audit_session("test_session") as session_id:
        audit_logger.log_event(
            AuditEventType.SYSTEM_STATE_CHANGE,
            "Test Event in Session",
            "This is a test event within a session context",
            session_id=session_id
        )
        print(f"  Session ID: {session_id}")
    
    # Generate compliance report
    print("Generating compliance report...")
    compliance_report = audit_logger.generate_compliance_report(timedelta(minutes=5))
    print(f"  Total events: {compliance_report['compliance_summary']['total_events']}")
    print(f"  Compliance rate: {compliance_report['compliance_summary']['compliance_rate']:.1f}%")
    print(f"  Violations: {len(compliance_report['violations'])}")
    
    print("\nComprehensive audit logging self-test complete!")