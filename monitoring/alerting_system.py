"""
Alerting System for Trading Platform

Provides real-time alerts for critical events, errors, and risk conditions.
Supports multiple alert channels and severity levels.
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = 1      # Informational
    WARNING = 2   # Warning condition
    ERROR = 3     # Error condition
    CRITICAL = 4  # Critical - immediate action required
    EMERGENCY = 5 # System failure imminent


class AlertType(Enum):
    """Types of alerts"""
    SYSTEM_ERROR = "system_error"
    CONNECTION_LOST = "connection_lost"
    RISK_LIMIT = "risk_limit"
    DAILY_LOSS = "daily_loss"
    POSITION_LIMIT = "position_limit"
    UNUSUAL_ACTIVITY = "unusual_activity"
    MODEL_DRIFT = "model_drift"
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class Alert:
    """Represents a single alert"""
    timestamp: datetime = field(default_factory=datetime.now)
    severity: AlertSeverity = AlertSeverity.INFO
    alert_type: AlertType = AlertType.SYSTEM_ERROR
    title: str = ""
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    alert_id: str = ""
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class AlertChannel:
    """Base class for alert channels"""
    
    def send(self, alert: Alert) -> bool:
        """Send alert through this channel"""
        raise NotImplementedError


class LoggingChannel(AlertChannel):
    """Send alerts to logging system"""
    
    def send(self, alert: Alert) -> bool:
        """Log the alert"""
        log_message = f"[ALERT] {alert.severity.name} - {alert.title}: {alert.message}"
        
        if alert.severity == AlertSeverity.CRITICAL or alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(log_message)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        return True


class FileChannel(AlertChannel):
    """Write alerts to file"""
    
    def __init__(self, filepath: str = "alerts.jsonl"):
        self.filepath = filepath
    
    def send(self, alert: Alert) -> bool:
        """Write alert to file"""
        try:
            alert_dict = {
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.name,
                'type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'details': alert.details,
                'source': alert.source,
                'alert_id': alert.alert_id
            }
            
            with open(self.filepath, 'a') as f:
                f.write(json.dumps(alert_dict) + '\n')
            
            return True
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
            return False


class AlertingSystem:
    """Main alerting system"""
    
    def __init__(self):
        self.channels: List[AlertChannel] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_counter = 0
        self.rate_limits: Dict[str, deque] = {}
        
        # Default channels
        self.add_channel(LoggingChannel())
        self.add_channel(FileChannel())
        
        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None
        
        # Alert conditions
        self.conditions: Dict[str, Callable] = {}
    
    def add_channel(self, channel: AlertChannel):
        """Add an alert channel"""
        self.channels.append(channel)
    
    def add_condition(self, name: str, condition_func: Callable):
        """Add an alert condition to monitor"""
        self.conditions[name] = condition_func
    
    def create_alert(self, 
                    severity: AlertSeverity,
                    alert_type: AlertType,
                    title: str,
                    message: str,
                    details: Optional[Dict[str, Any]] = None,
                    source: str = "") -> Alert:
        """Create a new alert"""
        self.alert_counter += 1
        
        alert = Alert(
            severity=severity,
            alert_type=alert_type,
            title=title,
            message=message,
            details=details or {},
            source=source,
            alert_id=f"ALERT_{self.alert_counter:06d}"
        )
        
        return alert
    
    def send_alert(self, alert: Alert) -> bool:
        """Send alert through all channels"""
        # Check rate limiting
        if not self._check_rate_limit(alert):
            logger.debug(f"Alert rate limited: {alert.title}")
            return False
        
        # Store in history
        self.alert_history.append(alert)
        
        # Send through all channels
        success = True
        for channel in self.channels:
            try:
                if not channel.send(alert):
                    success = False
            except Exception as e:
                logger.error(f"Failed to send alert through channel {channel.__class__.__name__}: {e}")
                success = False
        
        return success
    
    def _check_rate_limit(self, alert: Alert, max_per_minute: int = 10) -> bool:
        """Check if alert passes rate limiting"""
        key = f"{alert.alert_type.value}_{alert.severity.name}"
        
        if key not in self.rate_limits:
            self.rate_limits[key] = deque(maxlen=max_per_minute)
        
        now = datetime.now()
        recent = self.rate_limits[key]
        
        # Remove old entries
        while recent and (now - recent[0]) > timedelta(minutes=1):
            recent.popleft()
        
        # Check limit
        if len(recent) >= max_per_minute:
            return False
        
        recent.append(now)
        return True
    
    def check_risk_limits(self, 
                         current_position: float,
                         daily_pnl: float,
                         max_position: float,
                         max_daily_loss: float) -> List[Alert]:
        """Check risk management limits"""
        alerts = []
        
        # Position limit check
        if abs(current_position) > max_position * 0.9:
            alert = self.create_alert(
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.POSITION_LIMIT,
                title="Position Limit Warning",
                message=f"Position {current_position} approaching limit {max_position}",
                details={
                    'current_position': current_position,
                    'max_position': max_position,
                    'utilization': abs(current_position) / max_position
                }
            )
            alerts.append(alert)
        
        # Daily loss check
        if daily_pnl < 0 and abs(daily_pnl) > max_daily_loss * 0.8:
            severity = AlertSeverity.CRITICAL if abs(daily_pnl) >= max_daily_loss else AlertSeverity.WARNING
            
            alert = self.create_alert(
                severity=severity,
                alert_type=AlertType.DAILY_LOSS,
                title="Daily Loss Limit Alert",
                message=f"Daily loss {daily_pnl} approaching limit {-max_daily_loss}",
                details={
                    'daily_pnl': daily_pnl,
                    'max_daily_loss': max_daily_loss,
                    'percentage': abs(daily_pnl) / max_daily_loss * 100
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def check_system_health(self,
                           cpu_usage: float,
                           memory_usage: float,
                           disk_usage: float,
                           latency_ms: float) -> List[Alert]:
        """Check system health metrics"""
        alerts = []
        
        # CPU usage
        if cpu_usage > 90:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR if cpu_usage > 95 else AlertSeverity.WARNING,
                alert_type=AlertType.PERFORMANCE,
                title="High CPU Usage",
                message=f"CPU usage at {cpu_usage:.1f}%",
                details={'cpu_usage': cpu_usage}
            )
            alerts.append(alert)
        
        # Memory usage
        if memory_usage > 85:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR if memory_usage > 95 else AlertSeverity.WARNING,
                alert_type=AlertType.PERFORMANCE,
                title="High Memory Usage", 
                message=f"Memory usage at {memory_usage:.1f}%",
                details={'memory_usage': memory_usage}
            )
            alerts.append(alert)
        
        # Disk usage
        if disk_usage > 80:
            alert = self.create_alert(
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.PERFORMANCE,
                title="High Disk Usage",
                message=f"Disk usage at {disk_usage:.1f}%",
                details={'disk_usage': disk_usage}
            )
            alerts.append(alert)
        
        # Latency
        if latency_ms > 1000:
            alert = self.create_alert(
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.PERFORMANCE,
                title="High Latency",
                message=f"System latency at {latency_ms:.0f}ms",
                details={'latency_ms': latency_ms}
            )
            alerts.append(alert)
        
        return alerts
    
    def check_data_quality(self,
                          missing_data_pct: float,
                          stale_data_seconds: float,
                          validation_errors: int) -> List[Alert]:
        """Check data quality metrics"""
        alerts = []
        
        # Missing data
        if missing_data_pct > 5:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR if missing_data_pct > 20 else AlertSeverity.WARNING,
                alert_type=AlertType.DATA_QUALITY,
                title="Missing Market Data",
                message=f"{missing_data_pct:.1f}% of expected data missing",
                details={'missing_data_pct': missing_data_pct}
            )
            alerts.append(alert)
        
        # Stale data
        if stale_data_seconds > 60:
            alert = self.create_alert(
                severity=AlertSeverity.ERROR if stale_data_seconds > 300 else AlertSeverity.WARNING,
                alert_type=AlertType.DATA_QUALITY,
                title="Stale Market Data",
                message=f"Data is {stale_data_seconds:.0f} seconds old",
                details={'stale_data_seconds': stale_data_seconds}
            )
            alerts.append(alert)
        
        # Validation errors
        if validation_errors > 10:
            alert = self.create_alert(
                severity=AlertSeverity.WARNING,
                alert_type=AlertType.DATA_QUALITY,
                title="Data Validation Errors",
                message=f"{validation_errors} validation errors detected",
                details={'validation_errors': validation_errors}
            )
            alerts.append(alert)
        
        return alerts
    
    def start_monitoring(self, interval: int = 60):
        """Start monitoring thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                # Check all registered conditions
                for name, condition_func in self.conditions.items():
                    try:
                        alerts = condition_func()
                        if alerts:
                            for alert in alerts:
                                self.send_alert(alert)
                    except Exception as e:
                        logger.error(f"Error checking condition {name}: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def get_recent_alerts(self, 
                         severity: Optional[AlertSeverity] = None,
                         alert_type: Optional[AlertType] = None,
                         hours: int = 24) -> List[Alert]:
        """Get recent alerts filtered by criteria"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered = []
        for alert in self.alert_history:
            if alert.timestamp < cutoff:
                continue
            
            if severity and alert.severity != severity:
                continue
            
            if alert_type and alert.alert_type != alert_type:
                continue
            
            filtered.append(alert)
        
        return filtered
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        recent = self.get_recent_alerts(hours=24)
        
        summary = {
            'total_24h': len(recent),
            'by_severity': {},
            'by_type': {},
            'unresolved': 0
        }
        
        for severity in AlertSeverity:
            count = sum(1 for a in recent if a.severity == severity)
            if count > 0:
                summary['by_severity'][severity.name] = count
        
        for alert_type in AlertType:
            count = sum(1 for a in recent if a.alert_type == alert_type)
            if count > 0:
                summary['by_type'][alert_type.value] = count
        
        summary['unresolved'] = sum(1 for a in recent if not a.resolved)
        
        return summary


# Global alerting system instance
_alerting_system = None

def get_alerting_system() -> AlertingSystem:
    """Get or create global alerting system"""
    global _alerting_system
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
        _alerting_system.start_monitoring()
    return _alerting_system