"""Omni-monitor orchestrator tying together system, performance, and data health.

This module introduces a single event logger that writes a unified JSONL stream and
backs it with an SQLite table for ad-hoc analysis.  The monitor itself samples key
subsystems (resource usage, IBKR gateway health, pipeline ingest freshness, and
performance metrics) on a configurable interval.  Additional instrumentation can
re-use the shared :func:`emit_event` helper so that pipeline components emit
structured events that land in the same log.

The design is intentionally lightweight and dependency-free so it can run alongside
existing monitors without disrupting them.  Downstream services (Grafana/Loki,
Prometheus exporters, etc.) can tail the JSONL file or read directly from the
SQLite database as needed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import psutil

try:
    from performance_monitoring_system import performance_monitor
except ImportError:  # pragma: no cover - optional dependency during tests
    performance_monitor = None  # type: ignore

try:
    from monitoring.ingest_monitor import (
        check_port,
        check_ibkr_connectivity,
        list_latest_parquets,
        pipeline_pids,
    )
except ImportError:  # pragma: no cover
    check_port = check_ibkr_connectivity = list_latest_parquets = pipeline_pids = None  # type: ignore

try:
    from utils.ibkr_health_monitor import IBKRHealthMonitor
except ImportError:  # pragma: no cover
    IBKRHealthMonitor = None  # type: ignore

EVENT_LOG_FILENAME = os.getenv("OMNI_MONITOR_LOG", str(Path.home() / "logs/omni_monitor.jsonl"))
EVENT_DB_FILENAME = os.getenv("OMNI_MONITOR_DB", str(Path.home() / "logs/omni_monitor.db"))


# ---------------------------------------------------------------------------
# Event logging primitives
# ---------------------------------------------------------------------------

@dataclass
class OmniEvent:
    ts: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    level: str = "INFO"
    component: str = "system"
    category: str = "general"
    event: str = "status"
    message: Optional[str] = None
    run_id: Optional[str] = None
    corr_id: Optional[str] = None
    symbol: Optional[str] = None
    state: Optional[str] = None
    duration_ms: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        # Ensure nested data survives SQLite insertion by serialising explicitly
        record["data"] = json.dumps(record["data"], default=str)
        return record


class EventLogger:
    """Thread-safe logger that writes to JSONL and mirrors into SQLite."""

    def __init__(self, jsonl_path: str, db_path: str) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._initialised_db = False

    # -- persistence helpers -------------------------------------------------

    def _ensure_db(self) -> None:
        if self._initialised_db:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT,
                    component TEXT,
                    category TEXT,
                    event TEXT,
                    message TEXT,
                    run_id TEXT,
                    corr_id TEXT,
                    symbol TEXT,
                    state TEXT,
                    duration_ms REAL,
                    data TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_component ON events(component);
                """
            )
            conn.commit()
        self._initialised_db = True

    # -- public API ----------------------------------------------------------

    def log_event(self, event: OmniEvent) -> None:
        record = event.to_record()
        with self._lock:
            # JSONL append
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            # SQLite mirror
            self._ensure_db()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO events (
                        ts, level, component, category, event, message,
                        run_id, corr_id, symbol, state, duration_ms, data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["ts"],
                        record["level"],
                        record["component"],
                        record["category"],
                        record["event"],
                        record["message"],
                        record["run_id"],
                        record["corr_id"],
                        record["symbol"],
                        record["state"],
                        record["duration_ms"],
                        record["data"],
                    ),
                )
                conn.commit()


# Global singleton used by emit_event helpers
_EVENT_LOGGER: Optional[EventLogger] = None


def get_event_logger() -> EventLogger:
    global _EVENT_LOGGER
    if _EVENT_LOGGER is None:
        _EVENT_LOGGER = EventLogger(EVENT_LOG_FILENAME, EVENT_DB_FILENAME)
    return _EVENT_LOGGER


def emit_event(
    *,
    level: str = "INFO",
    component: str,
    category: str = "general",
    event: str,
    message: Optional[str] = None,
    run_id: Optional[str] = None,
    corr_id: Optional[str] = None,
    symbol: Optional[str] = None,
    state: Optional[str] = None,
    duration_ms: Optional[float] = None,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience helper for instrumentation points.

    Parameters mirror :class:`OmniEvent`.  All inputs are optional except
    ``component`` and ``event`` which identify the source and action.  ``data``
    should remain JSON-serialisable.
    """

    event_obj = OmniEvent(
        level=level.upper(),
        component=component,
        category=category,
        event=event,
        message=message,
        run_id=run_id,
        corr_id=corr_id,
        symbol=symbol,
        state=state,
        duration_ms=duration_ms,
        data=data or {},
    )
    get_event_logger().log_event(event_obj)


# ---------------------------------------------------------------------------
# Omni monitor orchestration
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    interval_seconds: int = 15
    host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    port: int = int(os.getenv("IBKR_PORT", "4002"))
    data_dir: Path = Path(os.getenv("DATA_DIR", str(Path.home() / ".local/share/m5_trader/data")))
    symbols: Iterable[str] = (
        "ES1!",
        "NQ1!",
        "XAUUSD",
        "EURUSD",
        "GBPUSD",
        "AUDUSD",
    )


class OmniMonitor:
    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self.config = config or MonitorConfig()
        self._stop_event = threading.Event()

        self._ibkr_monitor: Optional[Any] = None
        if IBKRHealthMonitor is not None:
            self._ibkr_monitor = IBKRHealthMonitor(
                host=self.config.host,
                port=self.config.port,
                client_id=int(os.getenv("IBKR_CLIENT_ID", "9002")),
                check_interval=self.config.interval_seconds,
            )

    # -- collection helpers --------------------------------------------------

    def _collect_resources(self) -> None:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        disk = psutil.disk_usage(str(Path.home()))
        get_event_logger()  # ensure initialised
        emit_event(
            component="resources",
            category="system",
            event="snapshot",
            data={
                "memory_used_mb": round(vm.used / (1024 * 1024), 2),
                "memory_percent": vm.percent,
                "cpu_percent": cpu,
                "disk_used_percent": round(disk.percent, 2),
                "disk_free_gb": round(disk.free / (1024 ** 3), 2),
                "process_count": len(psutil.pids()),
            },
        )

    def _collect_performance(self) -> None:
        if performance_monitor is None:
            return
        try:
            dashboard = performance_monitor.get_performance_dashboard()
        except Exception as exc:  # pragma: no cover - defensive
            emit_event(
                level="ERROR",
                component="performance",
                category="metrics",
                event="dashboard_error",
                message=str(exc),
            )
            return

        emit_event(
            component="performance",
            category="metrics",
            event="dashboard",
            data=dashboard,
        )
        self._evaluate_performance_slos(dashboard)

    def _collect_ingest(self) -> None:
        if not list_latest_parquets or not check_port or not check_ibkr_connectivity:
            return
        is_listening = check_port(self.config.host, self.config.port)
        ibkr_status = check_ibkr_connectivity(self.config.host, self.config.port, client_id=9001, timeout=7)
        parquet_status = list_latest_parquets(Path(self.config.data_dir), list(self.config.symbols))

        emit_event(
            component="ingest",
            category="data",
            event="status",
            data={
                "gateway_listening": is_listening,
                "ibkr": ibkr_status,
                "parquets": parquet_status,
                "pipeline_pids": pipeline_pids() if pipeline_pids else [],
            },
        )
        self._evaluate_ingest_slos(parquet_status)

    def _collect_ibkr_health(self) -> None:
        if self._ibkr_monitor is None:
            return
        try:
            metrics = self._ibkr_monitor.perform_health_check()
        except Exception as exc:  # pragma: no cover
            emit_event(
                level="ERROR",
                component="ibkr",
                category="connection",
                event="health_check_failed",
                message=str(exc),
            )
            return
        status = getattr(metrics, "status", None)
        if hasattr(status, "value"):
            status = status.value
        emit_event(
            component="ibkr",
            category="connection",
            event="health",
            state=status,
            data={
                "port_accessible": bool(getattr(metrics, "port_accessible", False)),
                "gateway_process": bool(getattr(metrics, "gateway_process_running", False)),
                "docker": bool(getattr(metrics, "docker_container_healthy", False)),
                "last_heartbeat": str(getattr(metrics, "last_heartbeat", None)),
                "reconnect_count": getattr(metrics, "reconnect_count", None),
            },
        )

    def _evaluate_performance_slos(self, dashboard: Dict[str, Any]) -> None:
        slo_p95 = os.getenv("SLO_PIPELINE_P95_MS")
        try:
            slo_threshold = float(slo_p95) if slo_p95 else None
        except ValueError:
            slo_threshold = None
        if not slo_threshold:
            return
        p95 = (
            dashboard.get("pipeline_metrics", {})
            .get("pipeline_latency", {})
            .get("p95")
        )
        if p95 is None:
            return
        if p95 > slo_threshold:
            emit_event(
                component="performance",
                category="slo",
                event="pipeline_latency_breach",
                state="WARN",
                data={"p95_ms": p95, "threshold_ms": slo_threshold},
            )

    def _evaluate_ingest_slos(self, parquet_status: Iterable[Dict[str, Any]]) -> None:
        slo_staleness = os.getenv("SLO_INGEST_STALENESS_MIN")
        try:
            threshold = float(slo_staleness) if slo_staleness else None
        except ValueError:
            threshold = None
        if not threshold:
            return
        for entry in parquet_status:
            age = entry.get("age_min")
            symbol = entry.get("symbol")
            if age is None or symbol is None:
                continue
            if age > threshold:
                emit_event(
                    component="ingest",
                    category="slo",
                    event="parquet_stale",
                    symbol=symbol,
                    state="WARN",
                    data={"age_min": age, "threshold_min": threshold},
                )

    # -- public control ------------------------------------------------------

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            start = time.time()
            self._collect_resources()
            self._collect_performance()
            self._collect_ingest()
            self._collect_ibkr_health()
            # Honor interval while accounting for work time
            elapsed = time.time() - start
            sleep_for = max(0.0, self.config.interval_seconds - elapsed)
            self._stop_event.wait(sleep_for)

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    monitor = OmniMonitor()
    try:
        monitor.run_forever()
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
