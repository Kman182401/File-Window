import os
import time
import atexit
import subprocess
from typing import Any, Dict, List


_NVML_OK = False
_NV = None
_last: Dict[str, Any] = {"ts": 0.0, "data": {"available": False, "reason": "uninitialized"}}
_backoff_until = 0.0


def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


_SMI_MIN_PERIOD = _getenv_float("GPU_METRICS_SMI_MIN_PERIOD_SEC", 5.0)
_BACKOFF_MAX = _getenv_float("GPU_METRICS_BACKOFF_MAX_SEC", 60.0)
_MOCK = _getenv_int("GPU_METRICS_MOCK", 0) == 1
_ENABLED = _getenv_int("GPU_METRICS_ENABLED", 1) == 1


def _init_nvml_once() -> None:
    global _NVML_OK, _NV
    if _NVML_OK or not _ENABLED or _MOCK:
        return
    try:
        import pynvml as nv  # type: ignore

        nv.nvmlInit()
        atexit.register(lambda: _safe_nvml_shutdown(nv))
        _NV = nv
        _NVML_OK = True
    except Exception:
        _NVML_OK = False
        _NV = None


def _safe_nvml_shutdown(nv) -> None:
    try:
        nv.nvmlShutdown()
    except Exception:
        pass


def detect_gpu_support() -> Dict[str, Any]:
    if _MOCK:
        return {"vendor": "nvidia", "available": True, "reason": "mock"}
    if not _ENABLED:
        return {"vendor": "none", "available": False, "reason": "disabled"}
    _init_nvml_once()
    if _NVML_OK:
        return {"vendor": "nvidia", "available": True, "reason": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout.strip():
            return {"vendor": "nvidia", "available": True, "reason": "smi_only"}
    except Exception:
        pass
    return {"vendor": "none", "available": False, "reason": "no-nvml"}


def _mock_sample() -> Dict[str, Any]:
    now = time.time()
    g = {
        "index": 0,
        "name": "Mock GPU",
        "uuid": "GPU-MOCK",
        "util_pct": 35.0,
        "mem_used_mb": 900.0,
        "mem_total_mb": 3072.0,
        "mem_pct": 29.3,
        "temp_c": 65.0,
        "power_w": 75.0,
        "sm_clock_mhz": 1400,
        "mem_clock_mhz": 4000,
        "fan_percent": 30.0,
        "pstate": "P2",
        "throttle_reasons": 0,
        "processes": [],
    }
    return {
        "available": True,
        "gpus": [g],
        "aggregate": {
            "count": 1,
            "max_util_pct": g["util_pct"],
            "max_mem_pct": g["mem_pct"],
            "max_temp_c": g["temp_c"],
            "avg_util_pct": g["util_pct"],
            "avg_mem_pct": g["mem_pct"],
            "avg_temp_c": g["temp_c"],
        },
        "reason": "mock",
        "ts": now,
    }


def _collect_via_nvml() -> Dict[str, Any]:
    nv = _NV
    assert nv is not None
    count = nv.nvmlDeviceGetCount()
    gpus: List[Dict[str, Any]] = []
    import psutil  # local import to avoid mandatory dependency at import time

    for i in range(count):
        h = nv.nvmlDeviceGetHandleByIndex(i)
        name = nv.nvmlDeviceGetName(h).decode("utf-8", errors="ignore") if isinstance(nv.nvmlDeviceGetName(h), bytes) else nv.nvmlDeviceGetName(h)
        uuid = nv.nvmlDeviceGetUUID(h).decode("utf-8", errors="ignore") if isinstance(nv.nvmlDeviceGetUUID(h), bytes) else nv.nvmlDeviceGetUUID(h)

        util = nv.nvmlDeviceGetUtilizationRates(h)
        mem = nv.nvmlDeviceGetMemoryInfo(h)

        try:
            temp_c = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        except Exception:
            temp_c = 0.0

        try:
            power_w = float(nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
        except Exception:
            power_w = 0.0

        try:
            sm_clock = int(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
        except Exception:
            sm_clock = 0

        try:
            mem_clock = int(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM))
        except Exception:
            mem_clock = 0

        try:
            fan_percent = float(nv.nvmlDeviceGetFanSpeed(h))
        except Exception:
            fan_percent = 0.0

        try:
            pstate = f"P{nv.nvmlDeviceGetPerformanceState(h)}"
        except Exception:
            pstate = "P?"

        try:
            thr = nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
        except Exception:
            thr = 0

        procs: List[Dict[str, Any]] = []
        # Try multiple NVML variants for running processes to maximize compatibility
        proc_lists = []
        for fn_name in (
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses",
        ):
            try:
                fn = getattr(nv, fn_name)
                proc_lists.extend(fn(h))
                break
            except Exception:
                continue
        for fn_name in (
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            try:
                fn = getattr(nv, fn_name)
                proc_lists.extend(fn(h))
                break
            except Exception:
                continue

        # Map PIDs to process names; guard aggressively
        pid_to_name: Dict[int, str] = {}
        try:
            for p in psutil.process_iter(["pid", "name"]):
                pid_to_name[p.info["pid"]] = p.info.get("name") or ""
        except Exception:
            pass

        for p in proc_lists[:16]:  # cap to avoid bloating payloads
            try:
                pid = int(getattr(p, "pid", 0))
                used_mb = float(getattr(p, "usedGpuMemory", 0) / (1024 * 1024))
                procs.append({"pid": pid, "name": pid_to_name.get(pid, ""), "used_mem_mb": used_mb})
            except Exception:
                continue

        total_mb = float(mem.total) / (1024 * 1024)
        used_mb = float(mem.used) / (1024 * 1024)
        mem_pct = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

        gpus.append(
            {
                "index": i,
                "name": name,
                "uuid": uuid,
                "util_pct": float(util.gpu),
                "mem_used_mb": used_mb,
                "mem_total_mb": total_mb,
                "mem_pct": mem_pct,
                "temp_c": temp_c,
                "power_w": power_w,
                "sm_clock_mhz": sm_clock,
                "mem_clock_mhz": mem_clock,
                "fan_percent": fan_percent,
                "pstate": pstate,
                "throttle_reasons": int(thr),
                "processes": procs,
            }
        )

    return _aggregate_result(gpus, reason=None)


def _collect_via_smi_fallback() -> Dict[str, Any]:
    try:
        q = (
            "index,name,uuid,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,power.draw,clocks.sm,clocks.mem,fan.speed"
        )
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=" + q,
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if res.returncode != 0:
            return {"available": False, "reason": "smi_error", "gpus": [], "aggregate": {}}
        lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
        gpus: List[Dict[str, Any]] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 11:
                continue
            try:
                idx = int(parts[0])
                name = parts[1]
                uuid = parts[2]
                util = float(parts[3])
                mem_used_mb = float(parts[4])
                mem_total_mb = float(parts[5])
                mem_pct = (mem_used_mb / mem_total_mb * 100.0) if mem_total_mb > 0 else 0.0
                temp_c = float(parts[6])
                power_w = float(parts[7])
                sm_clock = int(parts[8])
                mem_clock = int(parts[9])
                fan = float(parts[10])
                gpus.append(
                    {
                        "index": idx,
                        "name": name,
                        "uuid": uuid,
                        "util_pct": util,
                        "mem_used_mb": mem_used_mb,
                        "mem_total_mb": mem_total_mb,
                        "mem_pct": mem_pct,
                        "temp_c": temp_c,
                        "power_w": power_w,
                        "sm_clock_mhz": sm_clock,
                        "mem_clock_mhz": mem_clock,
                        "fan_percent": fan,
                        "pstate": "",
                        "throttle_reasons": 0,
                        "processes": [],
                    }
                )
            except Exception:
                continue
        if not gpus:
            return {"available": False, "reason": "smi_no_gpus", "gpus": [], "aggregate": {}}
        return _aggregate_result(gpus, reason="smi")
    except Exception:
        return {"available": False, "reason": "smi_exception", "gpus": [], "aggregate": {}}


def _aggregate_result(gpus: List[Dict[str, Any]], reason: Any) -> Dict[str, Any]:
    now = time.time()
    agg = {
        "count": len(gpus),
        "max_util_pct": max((g.get("util_pct", 0.0) for g in gpus), default=0.0),
        "max_mem_pct": max((g.get("mem_pct", 0.0) for g in gpus), default=0.0),
        "max_temp_c": max((g.get("temp_c", 0.0) for g in gpus), default=0.0),
        "avg_util_pct": sum((g.get("util_pct", 0.0) for g in gpus)) / len(gpus) if gpus else 0.0,
        "avg_mem_pct": sum((g.get("mem_pct", 0.0) for g in gpus)) / len(gpus) if gpus else 0.0,
        "avg_temp_c": sum((g.get("temp_c", 0.0) for g in gpus)) / len(gpus) if gpus else 0.0,
    }
    data = {"available": True, "gpus": gpus, "aggregate": agg, "reason": reason, "ts": now}
    return data


def collect_gpu_metrics() -> Dict[str, Any]:
    global _last, _backoff_until
    if not _ENABLED:
        return {"available": False, "reason": "disabled", "gpus": [], "aggregate": {}}
    if _MOCK:
        return _mock_sample()

    now = time.time()

    _init_nvml_once()
    if _NVML_OK:
        try:
            data = _collect_via_nvml()
            _last = {"ts": now, "data": data}
            return data
        except Exception:
            _backoff_until = min(now + _BACKOFF_MAX, now + 60.0)

    if now < _backoff_until:
        return {**_last["data"], "reason": "backoff"}

    if (now - _last["ts"]) < _SMI_MIN_PERIOD:
        return {**_last["data"], "reason": "cached"}

    data = _collect_via_smi_fallback()
    _last = {"ts": now, "data": data}
    return data

