"""
Automated Troubleshooting System for Phase 4A Trading Platform

This module provides advanced root cause analysis, issue correlation, automated
remediation suggestions, and health check validation after optimizations.

Key Features:
- Root cause analysis for performance and system issues
- Issue correlation matrix linking symptoms to causes
- Automated remediation command generation and execution
- Health check validation after optimizations
- Pattern-based problem diagnosis
- System state monitoring and comparison
- Rollback capability for failed optimizations
- Integration with performance monitoring and diagnostic logging

Author: Security Compliance Auditor Agent
Version: 1.0.0
"""

import os
import re
import json
import time
import subprocess
import threading
import sqlite3
import uuid
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import traceback
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import logging

# Import our monitoring systems
try:
    from monitoring.performance_tracker import get_performance_tracker, PerformanceTracker
    from logging.diagnostic_system import get_diagnostic_system, DiagnosticSystem, LogLevel, LogCategory
except ImportError:
    # Fallback for standalone testing
    PerformanceTracker = None
    DiagnosticSystem = None


class TroubleshootingLevel(Enum):
    """Levels of troubleshooting intervention"""
    MONITORING = "monitoring"          # Just monitor and log
    DIAGNOSTIC = "diagnostic"          # Run diagnostics and analyze
    CONSERVATIVE = "conservative"      # Safe, reversible fixes
    AGGRESSIVE = "aggressive"          # More invasive fixes
    CRITICAL = "critical"              # Emergency interventions


class IssueCategory(Enum):
    """Categories of issues the troubleshooter can handle"""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    NETWORK = "network"
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"
    IBKR_CONNECTION = "ibkr_connection"
    SYSTEM_RESOURCE = "system_resource"
    CONFIGURATION = "configuration"
    SECURITY = "security"


class IssueSeverity(Enum):
    """Severity levels for issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class RemediationResult(Enum):
    """Results of remediation attempts"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_MANUAL_INTERVENTION = "needs_manual_intervention"


@dataclass
class TroubleshootingIssue:
    """Representation of a system issue requiring troubleshooting"""
    issue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    category: IssueCategory = IssueCategory.PERFORMANCE
    severity: IssueSeverity = IssueSeverity.MEDIUM
    title: str = ""
    description: str = ""
    symptoms: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    correlation_ids: List[str] = field(default_factory=list)
    
    # Metrics and context
    metrics: Dict[str, float] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    error_patterns: List[str] = field(default_factory=list)
    
    # Remediation
    suggested_actions: List[str] = field(default_factory=list)
    automated_fixes: List[str] = field(default_factory=list)
    manual_steps: List[str] = field(default_factory=list)
    rollback_commands: List[str] = field(default_factory=list)
    
    # Status tracking
    status: str = "open"
    attempted_fixes: List[Dict[str, Any]] = field(default_factory=list)
    resolution_notes: str = ""


@dataclass
class RemediationAction:
    """Representation of a remediation action"""
    action_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    issue_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    action_type: str = ""
    command: str = ""
    description: str = ""
    expected_outcome: str = ""
    risk_level: TroubleshootingLevel = TroubleshootingLevel.CONSERVATIVE
    timeout_seconds: int = 60
    requires_confirmation: bool = False
    rollback_command: Optional[str] = None


@dataclass
class SystemHealthCheck:
    """System health check result"""
    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    check_name: str = ""
    status: str = "unknown"
    score: float = 0.0  # 0-100 health score
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    baseline_comparison: Optional[Dict[str, float]] = None


class AutomatedTroubleshooter:
    """
    Advanced automated troubleshooting system with root cause analysis,
    correlation detection, and automated remediation capabilities.
    """
    
    def __init__(self, troubleshooting_level: TroubleshootingLevel = TroubleshootingLevel.CONSERVATIVE,
                 max_concurrent_fixes: int = 2,
                 auto_execute_fixes: bool = False,
                 health_check_interval: int = 300):
        """
        Initialize automated troubleshooter
        
        Args:
            troubleshooting_level: Maximum intervention level
            max_concurrent_fixes: Maximum concurrent remediation attempts
            auto_execute_fixes: Whether to automatically execute safe fixes
            health_check_interval: Seconds between health checks
        """
        self.troubleshooting_level = troubleshooting_level
        self.max_concurrent_fixes = max_concurrent_fixes
        self.auto_execute_fixes = auto_execute_fixes
        self.health_check_interval = health_check_interval
        
        # Storage
        self.log_path = Path.home() / "logs" / "troubleshooting"
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Issue tracking
        self.active_issues = {}  # issue_id -> TroubleshootingIssue
        self.issue_history = deque(maxlen=1000)
        self.correlation_matrix = defaultdict(lambda: defaultdict(int))
        
        # Health monitoring
        self.health_checks = {}
        self.health_baselines = {}
        self.last_health_check = None
        
        # Remediation tracking
        self.active_remediations = {}
        self.remediation_history = deque(maxlen=500)
        
        # Threading
        self._lock = threading.Lock()
        self.monitoring_active = False
        self.monitor_thread = None
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_fixes + 1)
        
        # Database
        self.db_path = self.log_path / "troubleshooting.db"
        self._init_database()
        
        # Integration with monitoring systems
        self.performance_tracker = None
        self.diagnostic_system = None
        self._setup_monitoring_integration()
        
        # Initialize troubleshooting rules and patterns
        self.troubleshooting_rules = self._init_troubleshooting_rules()
        self.correlation_patterns = self._init_correlation_patterns()
        
        logger = logging.getLogger(__name__)
        logger.info(f"AutomatedTroubleshooter initialized (level: {troubleshooting_level.value})")
    
    def _init_database(self):
        """Initialize SQLite database for troubleshooting audit"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS issues (
                        issue_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        category TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        symptoms TEXT,
                        root_causes TEXT,
                        affected_components TEXT,
                        correlation_ids TEXT,
                        metrics TEXT,
                        system_state TEXT,
                        suggested_actions TEXT,
                        status TEXT DEFAULT 'open',
                        resolution_notes TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS remediation_actions (
                        action_id TEXT PRIMARY KEY,
                        issue_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        command TEXT,
                        description TEXT,
                        risk_level TEXT,
                        result TEXT,
                        output TEXT,
                        error_message TEXT,
                        duration_seconds REAL,
                        rollback_performed BOOLEAN DEFAULT FALSE,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (issue_id) REFERENCES issues (issue_id)
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS health_checks (
                        check_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        check_name TEXT NOT NULL,
                        status TEXT NOT NULL,
                        score REAL NOT NULL,
                        details TEXT,
                        recommendations TEXT,
                        baseline_comparison TEXT,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS issue_correlations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        issue1_category TEXT NOT NULL,
                        issue2_category TEXT NOT NULL,
                        correlation_strength REAL NOT NULL,
                        frequency INTEGER NOT NULL,
                        last_seen TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                conn.execute('CREATE INDEX IF NOT EXISTS idx_issues_timestamp ON issues(timestamp)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_actions_issue_id ON remediation_actions(issue_id)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_health_timestamp ON health_checks(timestamp)')
                
                conn.commit()
        except Exception as e:
            print(f"Failed to initialize troubleshooting database: {e}")
    
    def _setup_monitoring_integration(self):
        """Setup integration with performance and diagnostic monitoring"""
        try:
            if PerformanceTracker:
                self.performance_tracker = get_performance_tracker()
                # Register callback for performance alerts
                self.performance_tracker.register_alert_callback(self._handle_performance_alert)
            
            if DiagnosticSystem:
                self.diagnostic_system = get_diagnostic_system()
                # Register callback for issue detection
                self.diagnostic_system.register_issue_callback(self._handle_diagnostic_issue)
                
        except Exception as e:
            print(f"Warning: Could not setup monitoring integration: {e}")
    
    def _init_troubleshooting_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize troubleshooting rules for different issue types"""
        return {
            "high_latency": {
                "category": IssueCategory.PERFORMANCE,
                "triggers": ["latency > 1000ms", "response_time > 2s"],
                "root_causes": [
                    "High CPU usage",
                    "Memory pressure",
                    "I/O bottleneck",
                    "Network latency",
                    "Database contention",
                    "Inefficient algorithms"
                ],
                "diagnostic_commands": [
                    "top -b -n 1 | head -20",
                    "free -h",
                    "iostat -x 1 3",
                    "netstat -i",
                    "ps aux --sort=-%cpu | head -10"
                ],
                "remediation_actions": [
                    {
                        "type": "optimization",
                        "command": "python3 /home/ubuntu/immediate_optimizations.py",
                        "description": "Run immediate performance optimizations",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "memory_cleanup", 
                        "command": "python3 /home/ubuntu/memory_management_system.py --cleanup",
                        "description": "Clean up memory usage",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "process_restart",
                        "command": "systemctl restart trading-system",
                        "description": "Restart trading system",
                        "risk_level": TroubleshootingLevel.AGGRESSIVE
                    }
                ]
            },
            
            "memory_exhaustion": {
                "category": IssueCategory.MEMORY,
                "triggers": ["memory_usage > 5GB", "out_of_memory_error"],
                "root_causes": [
                    "Memory leak",
                    "Large dataset processing",
                    "Inefficient data structures",
                    "Accumulating cache"
                ],
                "diagnostic_commands": [
                    "free -h",
                    "ps aux --sort=-%mem | head -10",
                    "python3 -c \"import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')\"",
                    "du -sh /tmp/* 2>/dev/null | sort -hr | head -5"
                ],
                "remediation_actions": [
                    {
                        "type": "memory_cleanup",
                        "command": "python3 /home/ubuntu/memory_management_system.py --emergency-cleanup",
                        "description": "Emergency memory cleanup",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "cache_clear",
                        "command": "sync && echo 3 > /proc/sys/vm/drop_caches",
                        "description": "Clear system caches",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "swap_enable",
                        "command": "swapon -a",
                        "description": "Enable swap if available",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    }
                ]
            },
            
            "ibkr_connection_failure": {
                "category": IssueCategory.IBKR_CONNECTION,
                "triggers": ["connection_timeout", "authentication_failure", "ibkr_error"],
                "root_causes": [
                    "IB Gateway not running",
                    "Network connectivity issues",
                    "Authentication problems",
                    "Port configuration errors",
                    "Container configuration issues"
                ],
                "diagnostic_commands": [
                    "docker ps --filter 'name=ibkr'",
                    "docker logs ibkr-ibkr-gateway-1 --tail 20",
                    "netstat -tuln | grep ':4002'",
                    "ping -c 3 127.0.0.1",
                    "python3 /home/ubuntu/test_ib_connection.py"
                ],
                "remediation_actions": [
                    {
                        "type": "container_restart",
                        "command": "docker compose -f /home/ubuntu/ibkr/compose.yml restart",
                        "description": "Restart IBKR Gateway container",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "connection_test",
                        "command": "python3 /home/ubuntu/test_ib_connection.py",
                        "description": "Test IBKR connection",
                        "risk_level": TroubleshootingLevel.MONITORING
                    },
                    {
                        "type": "port_check",
                        "command": "docker exec ibkr-ibkr-gateway-1 netstat -tuln | grep 4002",
                        "description": "Verify port configuration",
                        "risk_level": TroubleshootingLevel.DIAGNOSTIC
                    }
                ]
            },
            
            "data_quality_issues": {
                "category": IssueCategory.DATA_QUALITY,
                "triggers": ["nan_values_detected", "data_corruption", "missing_data"],
                "root_causes": [
                    "Data source unavailable",
                    "Network interruption during fetch",
                    "Data format changes",
                    "API rate limiting",
                    "Corrupted cache files"
                ],
                "diagnostic_commands": [
                    "python3 /home/ubuntu/test_data_fetch.py",
                    "aws s3 ls s3://omega-singularity-ml/",
                    "python3 -c \"from feature_engineering import validate_data_quality; validate_data_quality()\"",
                    "ls -la /tmp/market_data_*"
                ],
                "remediation_actions": [
                    {
                        "type": "data_validation",
                        "command": "python3 /home/ubuntu/test_data_fetch.py",
                        "description": "Validate data sources",
                        "risk_level": TroubleshootingLevel.MONITORING
                    },
                    {
                        "type": "cache_clear",
                        "command": "rm -f /tmp/market_data_cache_*",
                        "description": "Clear data cache",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "s3_sync",
                        "command": "aws s3 sync s3://omega-singularity-ml/latest/ /tmp/recovery/",
                        "description": "Sync data from S3",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    }
                ]
            },
            
            "model_performance_degradation": {
                "category": IssueCategory.MODEL_PERFORMANCE,
                "triggers": ["accuracy_drop", "prediction_errors", "model_drift"],
                "root_causes": [
                    "Market regime change",
                    "Model overfitting",
                    "Data distribution shift",
                    "Feature engineering issues",
                    "Training data corruption"
                ],
                "diagnostic_commands": [
                    "python3 /home/ubuntu/verify_trading_intelligence.py",
                    "python3 -c \"from model_manager import check_model_health; check_model_health()\"",
                    "ls -la /home/ubuntu/models/",
                    "python3 /home/ubuntu/test_production_readiness.py"
                ],
                "remediation_actions": [
                    {
                        "type": "model_validation",
                        "command": "python3 /home/ubuntu/verify_trading_intelligence.py",
                        "description": "Validate model performance",
                        "risk_level": TroubleshootingLevel.MONITORING
                    },
                    {
                        "type": "feature_refresh",
                        "command": "python3 /home/ubuntu/feature_engineering.py --refresh",
                        "description": "Refresh feature engineering",
                        "risk_level": TroubleshootingLevel.CONSERVATIVE
                    },
                    {
                        "type": "model_retrain",
                        "command": "python3 /home/ubuntu/rl_trainer.py --quick-retrain",
                        "description": "Quick model retraining",
                        "risk_level": TroubleshootingLevel.AGGRESSIVE
                    }
                ]
            }
        }
    
    def _init_correlation_patterns(self) -> Dict[str, List[str]]:
        """Initialize issue correlation patterns"""
        return {
            "memory_exhaustion": ["high_latency", "model_performance_degradation", "system_instability"],
            "high_latency": ["memory_exhaustion", "cpu_overload", "io_bottleneck"],
            "ibkr_connection_failure": ["data_quality_issues", "trading_failures", "authentication_errors"],
            "data_quality_issues": ["model_performance_degradation", "trading_errors", "feature_engineering_failures"],
            "model_performance_degradation": ["trading_losses", "prediction_errors", "feature_drift"],
            "cpu_overload": ["high_latency", "system_instability", "process_timeouts"],
            "network_issues": ["ibkr_connection_failure", "data_fetch_failures", "api_timeouts"],
            "disk_space_low": ["log_rotation_failures", "model_save_failures", "cache_issues"]
        }
    
    def _handle_performance_alert(self, alert_info: Dict[str, Any]):
        """Handle performance alert from monitoring system"""
        try:
            issue = self._create_issue_from_performance_alert(alert_info)
            self.register_issue(issue)
        except Exception as e:
            print(f"Failed to handle performance alert: {e}")
    
    def _handle_diagnostic_issue(self, issue_info: Dict[str, Any]):
        """Handle issue from diagnostic system"""
        try:
            issue = self._create_issue_from_diagnostic(issue_info)
            self.register_issue(issue)
        except Exception as e:
            print(f"Failed to handle diagnostic issue: {e}")
    
    def _create_issue_from_performance_alert(self, alert_info: Dict[str, Any]) -> TroubleshootingIssue:
        """Create troubleshooting issue from performance alert"""
        metric_name = alert_info.get('metric_name', 'unknown')
        current_value = alert_info.get('current_value', 0)
        threshold = alert_info.get('threshold', 0)
        severity_map = {'HIGH': IssueSeverity.HIGH, 'CRITICAL': IssueSeverity.CRITICAL}
        
        issue = TroubleshootingIssue(
            category=IssueCategory.PERFORMANCE,
            severity=severity_map.get(alert_info.get('severity', 'HIGH'), IssueSeverity.MEDIUM),
            title=f"Performance Alert: {metric_name}",
            description=f"Performance metric {metric_name} exceeded threshold: {current_value} > {threshold}",
            symptoms=[f"{metric_name} = {current_value}", f"Threshold = {threshold}"],
            metrics={metric_name: current_value, 'threshold': threshold},
            correlation_ids=[alert_info.get('correlation_id', str(uuid.uuid4()))]
        )
        
        return issue
    
    def _create_issue_from_diagnostic(self, issue_info: Dict[str, Any]) -> TroubleshootingIssue:
        """Create troubleshooting issue from diagnostic system"""
        pattern_name = issue_info.get('pattern_name', 'Unknown Issue')
        issue_type = issue_info.get('issue_type', 'unknown')
        
        category_map = {
            'performance_degradation': IssueCategory.PERFORMANCE,
            'memory_leak': IssueCategory.MEMORY,
            'connection_error': IssueCategory.NETWORK,
            'data_quality': IssueCategory.DATA_QUALITY,
            'model_drift': IssueCategory.MODEL_PERFORMANCE
        }
        
        severity_map = {
            'CRITICAL': IssueSeverity.CRITICAL,
            'HIGH': IssueSeverity.HIGH,
            'MEDIUM': IssueSeverity.MEDIUM,
            'LOW': IssueSeverity.LOW
        }
        
        issue = TroubleshootingIssue(
            category=category_map.get(issue_type, IssueCategory.PERFORMANCE),
            severity=severity_map.get(issue_info.get('severity', 'MEDIUM'), IssueSeverity.MEDIUM),
            title=pattern_name,
            description=issue_info.get('message', ''),
            symptoms=[issue_info.get('message', '')],
            affected_components=[issue_info.get('component', 'unknown')],
            correlation_ids=[issue_info.get('correlation_id', str(uuid.uuid4()))],
            suggested_actions=issue_info.get('fix_commands', [])
        )
        
        return issue
    
    def register_issue(self, issue: TroubleshootingIssue) -> str:
        """
        Register a new issue for troubleshooting
        
        Args:
            issue: TroubleshootingIssue to register
            
        Returns:
            Issue ID
        """
        with self._lock:
            self.active_issues[issue.issue_id] = issue
            self.issue_history.append(issue)
        
        # Persist to database
        self._persist_issue(issue)
        
        # Analyze root causes
        self._analyze_root_causes(issue)
        
        # Check for correlations with other issues
        self._update_issue_correlations(issue)
        
        # Automatically start remediation if appropriate
        if self.auto_execute_fixes and issue.severity != IssueSeverity.CRITICAL:
            self._schedule_automated_remediation(issue)
        
        print(f"🔧 Issue registered: {issue.title} (ID: {issue.issue_id[:8]}, Severity: {issue.severity.value})")
        
        return issue.issue_id
    
    def _persist_issue(self, issue: TroubleshootingIssue):
        """Persist issue to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO issues 
                    (issue_id, timestamp, category, severity, title, description,
                     symptoms, root_causes, affected_components, correlation_ids,
                     metrics, system_state, suggested_actions, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    issue.issue_id,
                    issue.timestamp.isoformat(),
                    issue.category.value,
                    issue.severity.value,
                    issue.title,
                    issue.description,
                    json.dumps(issue.symptoms),
                    json.dumps(issue.root_causes),
                    json.dumps(issue.affected_components),
                    json.dumps(issue.correlation_ids),
                    json.dumps(issue.metrics),
                    json.dumps(issue.system_state),
                    json.dumps(issue.suggested_actions),
                    issue.status
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to persist issue: {e}")
    
    def _analyze_root_causes(self, issue: TroubleshootingIssue):
        """Analyze potential root causes for the issue"""
        # Get troubleshooting rules for this category
        category_key = issue.category.value.lower().replace('_', '')
        
        # Map categories to rule keys
        rule_mapping = {
            'performance': 'high_latency',
            'memory': 'memory_exhaustion',
            'ibkrconnection': 'ibkr_connection_failure',
            'dataquality': 'data_quality_issues',
            'modelperformance': 'model_performance_degradation'
        }
        
        rule_key = rule_mapping.get(category_key)
        if not rule_key or rule_key not in self.troubleshooting_rules:
            return
        
        rule = self.troubleshooting_rules[rule_key]
        
        # Add potential root causes from rules
        issue.root_causes.extend(rule.get('root_causes', []))
        
        # Run diagnostic commands to gather more information
        diagnostic_results = self._run_diagnostic_commands(rule.get('diagnostic_commands', []))
        issue.system_state.update(diagnostic_results)
        
        # Generate automated fixes
        remediation_actions = rule.get('remediation_actions', [])
        for action in remediation_actions:
            if action['risk_level'].value <= self.troubleshooting_level.value:
                issue.automated_fixes.append(action['command'])
                if action.get('description'):
                    issue.suggested_actions.append(action['description'])
    
    def _run_diagnostic_commands(self, commands: List[str]) -> Dict[str, Any]:
        """Run diagnostic commands and collect results"""
        results = {}
        
        for command in commands:
            try:
                # Skip comments
                if command.strip().startswith('#'):
                    continue
                
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                results[command] = {
                    'returncode': result.returncode,
                    'stdout': result.stdout.strip(),
                    'stderr': result.stderr.strip()
                }
                
            except subprocess.TimeoutExpired:
                results[command] = {'error': 'timeout'}
            except Exception as e:
                results[command] = {'error': str(e)}
        
        return results
    
    def _update_issue_correlations(self, issue: TroubleshootingIssue):
        """Update issue correlation matrix"""
        current_category = issue.category.value
        
        # Look for other active issues to correlate with
        for other_issue in self.active_issues.values():
            if other_issue.issue_id == issue.issue_id:
                continue
            
            other_category = other_issue.category.value
            
            # Update correlation matrix
            self.correlation_matrix[current_category][other_category] += 1
            self.correlation_matrix[other_category][current_category] += 1
            
            # Check for known correlation patterns
            if other_category in self.correlation_patterns.get(current_category, []):
                print(f"🔗 Correlated issues detected: {current_category} ↔ {other_category}")
                
                # Add correlation information to both issues
                issue.root_causes.append(f"Correlated with {other_category} issue: {other_issue.title}")
                other_issue.root_causes.append(f"Correlated with {current_category} issue: {issue.title}")
    
    def _schedule_automated_remediation(self, issue: TroubleshootingIssue):
        """Schedule automated remediation for an issue"""
        if len(self.active_remediations) >= self.max_concurrent_fixes:
            print(f"⏸️  Remediation queue full, deferring fix for issue {issue.issue_id[:8]}")
            return
        
        # Filter automated fixes by risk level
        safe_fixes = []
        for fix_command in issue.automated_fixes:
            # Simple risk assessment based on command content
            if any(risky in fix_command.lower() for risky in ['rm ', 'delete', 'drop', 'kill -9']):
                continue
            safe_fixes.append(fix_command)
        
        if not safe_fixes:
            print(f"⚠️  No safe automated fixes available for issue {issue.issue_id[:8]}")
            return
        
        # Schedule remediation
        self.executor.submit(self._execute_remediation, issue, safe_fixes)
    
    def _execute_remediation(self, issue: TroubleshootingIssue, fix_commands: List[str]):
        """Execute remediation actions for an issue"""
        remediation_id = str(uuid.uuid4())
        
        with self._lock:
            self.active_remediations[remediation_id] = {
                'issue_id': issue.issue_id,
                'start_time': datetime.now(),
                'commands': fix_commands
            }
        
        print(f"🔧 Starting automated remediation for issue {issue.issue_id[:8]}")
        
        try:
            # Take system snapshot before remediation
            pre_snapshot = self._take_system_snapshot()
            
            results = []
            for i, command in enumerate(fix_commands):
                print(f"  Executing fix {i+1}/{len(fix_commands)}: {command}")
                
                action = RemediationAction(
                    issue_id=issue.issue_id,
                    action_type="automated_fix",
                    command=command,
                    description=f"Automated fix {i+1}",
                    risk_level=TroubleshootingLevel.CONSERVATIVE
                )
                
                result = self._execute_remediation_action(action)
                results.append(result)
                
                # Stop if a critical fix failed
                if result == RemediationResult.FAILED and "critical" in command.lower():
                    break
                
                # Wait between fixes
                time.sleep(2)
            
            # Take system snapshot after remediation
            post_snapshot = self._take_system_snapshot()
            
            # Validate remediation effectiveness
            validation_result = self._validate_remediation(issue, pre_snapshot, post_snapshot)
            
            # Update issue status
            if validation_result['effective']:
                issue.status = "resolved"
                issue.resolution_notes = f"Resolved by automated remediation: {validation_result['summary']}"
                print(f"✅ Issue {issue.issue_id[:8]} resolved successfully")
            else:
                issue.status = "remediation_failed"
                issue.resolution_notes = f"Automated remediation ineffective: {validation_result['summary']}"
                print(f"❌ Automated remediation failed for issue {issue.issue_id[:8]}")
            
            # Record attempted fixes
            issue.attempted_fixes.append({
                'timestamp': datetime.now().isoformat(),
                'remediation_id': remediation_id,
                'commands': fix_commands,
                'results': [r.value for r in results],
                'validation': validation_result
            })
            
        except Exception as e:
            print(f"💥 Remediation failed with exception: {e}")
            issue.status = "remediation_error"
            issue.resolution_notes = f"Remediation error: {str(e)}"
            
        finally:
            # Clean up
            with self._lock:
                if remediation_id in self.active_remediations:
                    del self.active_remediations[remediation_id]
            
            # Update issue in database
            self._persist_issue(issue)
    
    def _execute_remediation_action(self, action: RemediationAction) -> RemediationResult:
        """Execute a single remediation action"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                action.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=action.timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Log action to database
            self._log_remediation_action(action, result, duration)
            
            if result.returncode == 0:
                return RemediationResult.SUCCESS
            else:
                print(f"  Command failed with return code {result.returncode}")
                print(f"  stderr: {result.stderr[:200]}")
                return RemediationResult.FAILED
                
        except subprocess.TimeoutExpired:
            print(f"  Command timed out after {action.timeout_seconds} seconds")
            return RemediationResult.FAILED
            
        except Exception as e:
            print(f"  Command execution error: {e}")
            return RemediationResult.FAILED
    
    def _log_remediation_action(self, action: RemediationAction, 
                               result: subprocess.CompletedProcess, 
                               duration: float):
        """Log remediation action to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO remediation_actions 
                    (action_id, issue_id, timestamp, action_type, command, 
                     description, risk_level, result, output, error_message, 
                     duration_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    action.action_id,
                    action.issue_id,
                    action.timestamp.isoformat(),
                    action.action_type,
                    action.command,
                    action.description,
                    action.risk_level.value,
                    'success' if result.returncode == 0 else 'failed',
                    result.stdout[:1000],  # Limit output size
                    result.stderr[:1000] if result.stderr else None,
                    duration
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to log remediation action: {e}")
    
    def _take_system_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of system state"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {},
            'health_checks': {}
        }
        
        try:
            # System metrics
            snapshot['system_metrics'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            # Process information
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['cpu_percent'] > 1 or proc.info['memory_percent'] > 1:
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            snapshot['top_processes'] = sorted(processes, 
                                            key=lambda x: x['cpu_percent'], 
                                            reverse=True)[:10]
            
        except Exception as e:
            print(f"Failed to take system snapshot: {e}")
        
        return snapshot
    
    def _validate_remediation(self, issue: TroubleshootingIssue, 
                            pre_snapshot: Dict[str, Any], 
                            post_snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if remediation was effective"""
        validation = {
            'effective': False,
            'summary': 'No improvement detected',
            'metrics_improved': [],
            'metrics_degraded': [],
            'recommendations': []
        }
        
        try:
            pre_metrics = pre_snapshot.get('system_metrics', {})
            post_metrics = post_snapshot.get('system_metrics', {})
            
            improvements = []
            degradations = []
            
            # Check specific improvements based on issue category
            if issue.category == IssueCategory.PERFORMANCE:
                if 'cpu_percent' in pre_metrics and 'cpu_percent' in post_metrics:
                    cpu_improvement = pre_metrics['cpu_percent'] - post_metrics['cpu_percent']
                    if cpu_improvement > 5:  # 5% improvement
                        improvements.append(f"CPU usage reduced by {cpu_improvement:.1f}%")
                    elif cpu_improvement < -5:
                        degradations.append(f"CPU usage increased by {abs(cpu_improvement):.1f}%")
            
            elif issue.category == IssueCategory.MEMORY:
                if 'memory_percent' in pre_metrics and 'memory_percent' in post_metrics:
                    memory_improvement = pre_metrics['memory_percent'] - post_metrics['memory_percent']
                    if memory_improvement > 2:  # 2% improvement
                        improvements.append(f"Memory usage reduced by {memory_improvement:.1f}%")
                    elif memory_improvement < -2:
                        degradations.append(f"Memory usage increased by {abs(memory_improvement):.1f}%")
            
            # Overall assessment
            validation['metrics_improved'] = improvements
            validation['metrics_degraded'] = degradations
            
            if improvements and not degradations:
                validation['effective'] = True
                validation['summary'] = f"Remediation successful: {'; '.join(improvements)}"
            elif improvements and degradations:
                validation['effective'] = len(improvements) > len(degradations)
                validation['summary'] = f"Mixed results: {len(improvements)} improvements, {len(degradations)} degradations"
            elif degradations:
                validation['effective'] = False
                validation['summary'] = f"Remediation may have caused issues: {'; '.join(degradations)}"
                validation['recommendations'].append("Consider rollback")
            
            # Run specific health checks
            if issue.category == IssueCategory.IBKR_CONNECTION:
                validation['recommendations'].append("Run IBKR connection test")
            
        except Exception as e:
            validation['summary'] = f"Validation failed: {str(e)}"
        
        return validation
    
    def run_health_check(self, check_name: str = "system_health") -> SystemHealthCheck:
        """Run comprehensive system health check"""
        health_check = SystemHealthCheck(
            check_name=check_name,
            timestamp=datetime.now()
        )
        
        try:
            scores = []
            details = {}
            
            # System resource health
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # CPU score (lower usage = higher score)
            cpu_score = max(0, 100 - cpu_percent)
            scores.append(cpu_score)
            details['cpu'] = {'usage_percent': cpu_percent, 'score': cpu_score}
            
            # Memory score
            memory_score = max(0, 100 - memory_percent)
            scores.append(memory_score)
            details['memory'] = {'usage_percent': memory_percent, 'score': memory_score}
            
            # Disk score
            disk_score = max(0, 100 - disk_percent)
            scores.append(disk_score)
            details['disk'] = {'usage_percent': disk_percent, 'score': disk_score}
            
            # Process health (check for zombie processes, high CPU processes)
            process_issues = 0
            high_cpu_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        process_issues += 1
                    if proc.info['cpu_percent'] > 50:
                        high_cpu_processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            process_score = max(0, 100 - (process_issues * 10) - (len(high_cpu_processes) * 5))
            scores.append(process_score)
            details['processes'] = {
                'zombie_count': process_issues, 
                'high_cpu_count': len(high_cpu_processes),
                'score': process_score
            }
            
            # IBKR connection health
            ibkr_score = self._check_ibkr_health()
            scores.append(ibkr_score)
            details['ibkr'] = {'score': ibkr_score}
            
            # Trading system health
            trading_score = self._check_trading_system_health()
            scores.append(trading_score)
            details['trading_system'] = {'score': trading_score}
            
            # Overall health score
            overall_score = sum(scores) / len(scores) if scores else 0
            health_check.score = overall_score
            health_check.details = details
            
            # Determine status
            if overall_score >= 90:
                health_check.status = "excellent"
            elif overall_score >= 75:
                health_check.status = "good"
            elif overall_score >= 60:
                health_check.status = "fair"
            elif overall_score >= 40:
                health_check.status = "poor"
            else:
                health_check.status = "critical"
            
            # Generate recommendations
            recommendations = []
            if cpu_percent > 80:
                recommendations.append("High CPU usage detected - consider optimization")
            if memory_percent > 85:
                recommendations.append("High memory usage - consider cleanup or restart")
            if disk_percent > 90:
                recommendations.append("Low disk space - cleanup recommended")
            if process_issues > 0:
                recommendations.append(f"{process_issues} zombie processes detected - system restart may be needed")
            if ibkr_score < 80:
                recommendations.append("IBKR connection issues detected")
            if trading_score < 80:
                recommendations.append("Trading system performance issues detected")
            
            health_check.recommendations = recommendations
            
        except Exception as e:
            health_check.status = "error"
            health_check.score = 0
            health_check.details = {'error': str(e)}
        
        # Store health check
        self.health_checks[check_name] = health_check
        self.last_health_check = health_check
        
        # Persist to database
        self._persist_health_check(health_check)
        
        return health_check
    
    def _check_ibkr_health(self) -> float:
        """Check IBKR connection health"""
        try:
            # Try to run connection test
            result = subprocess.run(
                "python3 /home/ubuntu/test_ib_connection.py",
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "connected: True" in result.stdout:
                return 100.0
            else:
                return 30.0
                
        except Exception:
            return 0.0
    
    def _check_trading_system_health(self) -> float:
        """Check trading system health"""
        try:
            # Check if trading intelligence is working
            result = subprocess.run(
                "timeout 30 python3 /home/ubuntu/verify_trading_intelligence.py",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Count successful tests
                success_count = result.stdout.count("✅") + result.stdout.count("PASS")
                total_tests = success_count + result.stdout.count("❌") + result.stdout.count("FAIL")
                
                if total_tests > 0:
                    return (success_count / total_tests) * 100
                else:
                    return 50.0  # Neutral score if can't determine
            else:
                return 20.0
                
        except Exception:
            return 0.0
    
    def _persist_health_check(self, health_check: SystemHealthCheck):
        """Persist health check to database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT INTO health_checks 
                    (check_id, timestamp, check_name, status, score, details, recommendations)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    health_check.check_id,
                    health_check.timestamp.isoformat(),
                    health_check.check_name,
                    health_check.status,
                    health_check.score,
                    json.dumps(health_check.details),
                    json.dumps(health_check.recommendations)
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to persist health check: {e}")
    
    def generate_troubleshooting_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Generate comprehensive troubleshooting report"""
        if not time_window:
            time_window = timedelta(hours=24)
        
        cutoff = datetime.now() - time_window
        
        # Get recent issues
        recent_issues = [
            issue for issue in self.issue_history 
            if issue.timestamp > cutoff
        ]
        
        # Analyze issue patterns
        category_counts = Counter(issue.category.value for issue in recent_issues)
        severity_counts = Counter(issue.severity.value for issue in recent_issues)
        status_counts = Counter(issue.status for issue in recent_issues)
        
        # Calculate resolution rates
        total_issues = len(recent_issues)
        resolved_issues = len([i for i in recent_issues if i.status == 'resolved'])
        resolution_rate = (resolved_issues / total_issues * 100) if total_issues > 0 else 0
        
        # Get health check data
        latest_health = self.last_health_check
        
        # Get correlation data
        top_correlations = []
        for cat1, correlations in self.correlation_matrix.items():
            for cat2, count in correlations.most_common(5):
                if count > 1 and cat1 != cat2:
                    top_correlations.append((cat1, cat2, count))
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'time_window': str(time_window),
            'summary': {
                'total_issues': total_issues,
                'resolved_issues': resolved_issues,
                'open_issues': len(self.active_issues),
                'resolution_rate_percent': resolution_rate,
                'automated_fixes_attempted': len([i for i in recent_issues if i.attempted_fixes])
            },
            'issue_breakdown': {
                'by_category': dict(category_counts),
                'by_severity': dict(severity_counts),
                'by_status': dict(status_counts)
            },
            'top_correlations': top_correlations,
            'current_health': {
                'overall_score': latest_health.score if latest_health else 0,
                'status': latest_health.status if latest_health else 'unknown',
                'recommendations': latest_health.recommendations if latest_health else []
            },
            'active_issues': [
                {
                    'id': issue.issue_id[:8],
                    'title': issue.title,
                    'category': issue.category.value,
                    'severity': issue.severity.value,
                    'age_hours': (datetime.now() - issue.timestamp).total_seconds() / 3600
                }
                for issue in self.active_issues.values()
            ]
        }
        
        return report
    
    def start_monitoring(self):
        """Start automatic health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Run health check
                    health_check = self.run_health_check()
                    
                    # Check for new issues based on health check
                    if health_check.score < 70:
                        # Create issue for poor health
                        issue = TroubleshootingIssue(
                            category=IssueCategory.SYSTEM_RESOURCE,
                            severity=IssueSeverity.HIGH if health_check.score < 40 else IssueSeverity.MEDIUM,
                            title=f"System health degraded (Score: {health_check.score:.1f})",
                            description=f"Overall system health score has dropped to {health_check.score:.1f}",
                            symptoms=[f"Health score: {health_check.score:.1f}", f"Status: {health_check.status}"],
                            metrics={'health_score': health_check.score},
                            suggested_actions=health_check.recommendations
                        )
                        
                        # Only register if we don't already have a similar active issue
                        similar_issue = any(
                            'system health' in active_issue.title.lower() 
                            for active_issue in self.active_issues.values()
                        )
                        
                        if not similar_issue:
                            self.register_issue(issue)
                    
                except Exception as e:
                    print(f"Health monitoring error: {e}")
                
                time.sleep(self.health_check_interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"Automated troubleshooting monitoring started (interval: {self.health_check_interval}s)")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        print("Automated troubleshooting monitoring stopped")
    
    def get_issue(self, issue_id: str) -> Optional[TroubleshootingIssue]:
        """Get issue by ID"""
        return self.active_issues.get(issue_id)
    
    def resolve_issue(self, issue_id: str, resolution_notes: str = ""):
        """Manually resolve an issue"""
        if issue_id in self.active_issues:
            issue = self.active_issues[issue_id]
            issue.status = "resolved"
            issue.resolution_notes = resolution_notes
            
            # Move from active to history
            del self.active_issues[issue_id]
            self.issue_history.append(issue)
            
            # Update in database
            self._persist_issue(issue)
            
            print(f"✅ Issue {issue_id[:8]} manually resolved: {resolution_notes}")


# Global troubleshooter instance
_automated_troubleshooter = None


def get_automated_troubleshooter() -> AutomatedTroubleshooter:
    """Get or create singleton AutomatedTroubleshooter instance"""
    global _automated_troubleshooter
    
    if _automated_troubleshooter is None:
        _automated_troubleshooter = AutomatedTroubleshooter()
        _automated_troubleshooter.start_monitoring()
    
    return _automated_troubleshooter


if __name__ == "__main__":
    # Self-test
    print("Automated Troubleshooter Self-Test")
    print("=" * 50)
    
    troubleshooter = AutomatedTroubleshooter(
        troubleshooting_level=TroubleshootingLevel.CONSERVATIVE,
        auto_execute_fixes=False  # Don't auto-execute in test
    )
    
    # Test issue registration
    print("Testing issue registration...")
    
    test_issue = TroubleshootingIssue(
        category=IssueCategory.PERFORMANCE,
        severity=IssueSeverity.HIGH,
        title="High CPU Usage Test",
        description="Test issue for high CPU usage",
        symptoms=["CPU usage > 80%", "System sluggish"],
        metrics={"cpu_usage": 85.5}
    )
    
    issue_id = troubleshooter.register_issue(test_issue)
    print(f"  Registered test issue: {issue_id[:8]}")
    
    # Test health check
    print("Running health check...")
    health_check = troubleshooter.run_health_check()
    print(f"  Health score: {health_check.score:.1f}")
    print(f"  Status: {health_check.status}")
    print(f"  Recommendations: {len(health_check.recommendations)}")
    
    # Test report generation
    print("Generating troubleshooting report...")
    report = troubleshooter.generate_troubleshooting_report(timedelta(minutes=5))
    print(f"  Total issues: {report['summary']['total_issues']}")
    print(f"  Open issues: {report['summary']['open_issues']}")
    print(f"  Current health score: {report['current_health']['overall_score']:.1f}")
    
    # Test manual resolution
    print("Testing manual resolution...")
    troubleshooter.resolve_issue(issue_id, "Test resolution - issue was simulated")
    print(f"  Resolved issue {issue_id[:8]}")
    
    print("\nSelf-test complete!")