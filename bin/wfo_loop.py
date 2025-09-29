from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import os
import time
import json
import yaml
import importlib
import logging
import random
import gc
import multiprocessing as mp
from datetime import datetime, UTC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("wfo_loop")

ROOT = Path.home() / "File-Window"
CFG  = Path.home() / "M5-Trader/wfo/wfo_config.yaml"
LOG  = Path.home() / "logs/wfo_loop.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
from monitoring.client.omega_trading_status import write_trading_snapshot
from monitoring.client.omega_polybar_callback import write_learning_snapshot

def run_wfo_once(dry_run: bool, rl_fast_smoke: bool, rl_overrides: dict | None):
    cfg = yaml.safe_load(CFG.read_text())
    r = importlib.import_module("wfo.runner")

    symbols = cfg.get("symbols", ["ES","NQ","GC"])
    is_days = int(cfg.get("is_days", 20))
    oos_days = int(cfg.get("oos_days", 5))
    step_days = int(cfg.get("step_days", 2))
    cycles_min = int(cfg.get("cycles_min", 2))
    embargo_days = int(cfg.get("embargo_days", 1))
    lookahead = int(cfg.get("label_lookahead_bars", 390))
    cpcv_folds = int(cfg.get("cpcv",{}).get("n_folds", 3))

    out = r.run_wfo(
        symbols=symbols,
        is_days=is_days,
        oos_days=oos_days,
        step_days=step_days,
        cycles_min=cycles_min,
        embargo_days=embargo_days,
        label_lookahead_bars=lookahead,
        cpcv_folds=cpcv_folds,
        config_path=str(CFG),
        dry_run=dry_run,
        rl_fast_smoke=rl_fast_smoke,
        rl_fast_overrides=rl_overrides,
    )
    return out

def _json_default(obj):
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def _log_event(payload: dict) -> None:
    """Append structured payload to the JSONL log and stdout."""
    payload.setdefault("start_method", _current_start_method())
    line = json.dumps(payload, default=_json_default)
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    logger.info(
        "event=%s %s",
        payload.get("event", "summary"),
        json.dumps(payload, sort_keys=True, default=_json_default),
    )


def log_summary(tag: str, out: dict, *, run_id: str, seed: int | None, rl_overrides: dict | None):
    s = out.get("summary", {}) or {}
    metadata = out.get("run_metadata") or out.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    per_cycle = out.get("per_cycle_records") or []

    base_ctx = {
        "ts": time.time(),
        "event": "summary",
        "tag": tag,
        "run_id": run_id,
        "cycle": "aggregate",
        "fold": "aggregate",
        "symbol": "ALL",
        "seed": seed,
    }
    rec = {
        **base_ctx,
        "white_rc_p": s.get("white_rc_p"),
        "spa_p": s.get("spa_p"),
        "go_no_go": s.get("go_no_go"),
        "sharpe_median": s.get("sharpe_median"),
        "cycles": s.get("cycles"),
        "dsr": out.get("dsr"),
        "output_dir": out.get("output_dir"),
        "strategy_seeds": metadata.get("strategy_seeds"),
        "rl_overrides": rl_overrides,
        "n_envs": (rl_overrides or {}).get("n_envs"),
    }
    _log_event(rec)

    for cycle_rec in per_cycle:
        cycle_payload = {
            "ts": time.time(),
            "event": "cycle_metrics",
            "tag": tag,
            "run_id": run_id,
            "cycle": cycle_rec.get("cycle"),
            "fold": cycle_rec.get("config", "aggregate"),
            "symbol": cycle_rec.get("symbol"),
            "seed": metadata.get("strategy_seeds", {}).get(cycle_rec.get("config")),
            "sharpe": cycle_rec.get("sharpe"),
            "sortino": cycle_rec.get("sortino"),
            "max_drawdown": cycle_rec.get("max_drawdown"),
            "turnover": cycle_rec.get("turnover"),
            "expectancy": cycle_rec.get("expectancy"),
            "n_envs": (rl_overrides or {}).get("n_envs"),
        }
        _log_event(cycle_payload)

    _update_telemetry(tag=tag, summary=s, out=out, run_id=run_id, seed=seed, rl_overrides=rl_overrides)
    return s


def _update_telemetry(*, tag: str, summary: dict, out: dict, run_id: str, seed: int | None, rl_overrides: dict | None) -> None:
    try:
        write_trading_snapshot(
            sharpe=summary.get("sharpe_median"),
            max_drawdown=summary.get("max_drawdown_min"),
            pnl_session=summary.get("total_return"),
            extras={
                "tag": tag,
                "run_id": run_id,
                "go_no_go": summary.get("go_no_go"),
                "pnl_units": "return",
            },
        )
    except Exception:
        logger.exception("run_id=%s tag=%s message=failed writing trading snapshot", run_id, tag)

    try:
        metrics = {
            "approx_kl": (out.get("dsr") or {}).get("p_value"),
            "entropy": out.get("white_rc_p"),
            "clipfrac": out.get("spa_p"),
            "cycles": summary.get("cycles"),
        }
        eval_payload = {
            "eval_sharpe": summary.get("sharpe_median"),
            "best_eval_sharpe": summary.get("sharpe_median"),
            "best_ckpt_steps": summary.get("cycles"),
            "go_no_go": summary.get("go_no_go"),
        }
        total_timesteps = (rl_overrides or {}).get("total_timesteps")
        extras = {
            "tag": tag,
            "run_id": run_id,
            "seed": seed,
        }
        write_learning_snapshot(
            algo="wfo",
            total_timesteps=total_timesteps,
            metrics=metrics,
            evaluation=eval_payload,
            extras=extras,
        )
    except Exception:
        logger.exception("run_id=%s tag=%s message=failed writing learning snapshot", run_id, tag)


def _sleep_with_jitter(delay: float) -> None:
    jitter = random.uniform(0, delay * 0.3) if delay > 0 else 0.0
    time.sleep(delay + jitter)


def _cleanup_after_failure() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _current_start_method() -> str | None:
    try:
        return mp.get_start_method(allow_none=True)
    except Exception:
        return None

if __name__ == "__main__":
    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    interval_sec = int(os.environ.get("WFO_INTERVAL_SEC", "3600"))

    # Smoke pass: fast + cheap, just to test gates
    smoke_rl = {"n_envs": 2, "total_timesteps": 5000, "batch_size": 1024}

    # Promotion pass: real RL training but bounded
    promo_rl = {"n_envs": 4, "total_timesteps": 200_000, "batch_size": 2048}

    rng = random.SystemRandom()
    backoff = 60
    while True:
        run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        smoke_seed = rng.randrange(0, 2**31 - 1)
        promo_seed = rng.randrange(0, 2**31 - 1)
        smoke_overrides = {**smoke_rl, "seed": smoke_seed}
        promo_overrides = {**promo_rl, "seed": promo_seed}
        try:
            out_smoke = run_wfo_once(dry_run=True, rl_fast_smoke=True, rl_overrides=smoke_overrides)
            s = log_summary("smoke", out_smoke, run_id=run_id, seed=smoke_seed, rl_overrides=smoke_overrides)

            if s.get("go_no_go"):
                _log_event(
                    {
                        "ts": time.time(),
                        "event": "promotion_start",
                        "tag": "promotion",
                        "run_id": run_id,
                        "cycle": "aggregate",
                        "fold": "aggregate",
                        "symbol": "ALL",
                        "seed": promo_seed,
                        "n_envs": promo_overrides.get("n_envs"),
                        "message": "Gates passed — starting promotion run with RL training.",
                    }
                )
                out_rl = run_wfo_once(dry_run=False, rl_fast_smoke=False, rl_overrides=promo_overrides)
                log_summary("promotion", out_rl, run_id=run_id, seed=promo_seed, rl_overrides=promo_overrides)
            else:
                _log_event(
                    {
                        "ts": time.time(),
                        "event": "promotion_skip",
                        "tag": "promotion",
                        "run_id": run_id,
                        "cycle": "aggregate",
                        "fold": "aggregate",
                        "symbol": "ALL",
                        "seed": promo_seed,
                        "n_envs": promo_overrides.get("n_envs"),
                        "message": "Gates failed — skipping RL training this cycle.",
                    }
                )

            time.sleep(interval_sec)
            backoff = 60
        except Exception:
            logger.exception("run_id=%s tag=loop_failure message=Exception raised during WFO loop", run_id)
            _cleanup_after_failure()
            _sleep_with_jitter(backoff)
            backoff = min(backoff * 2, 1800)
