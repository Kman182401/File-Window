import os, time, json, yaml, importlib, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

ROOT = Path.home() / "File-Window"
CFG  = Path.home() / "M5-Trader/wfo/wfo_config.yaml"
LOG  = Path.home() / "logs/wfo_loop.jsonl"
LOG.parent.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

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

def log_summary(tag: str, out: dict):
    s = out.get("summary", {}) or {}
    rec = {
        "ts": time.time(),
        "tag": tag,
        "white_rc_p": s.get("white_rc_p"),
        "spa_p": s.get("spa_p"),
        "go_no_go": s.get("go_no_go"),
        "sharpe_median": s.get("sharpe_median"),
        "dsr": out.get("dsr"),
        "output_dir": out.get("output_dir"),
    }
    with LOG.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    logging.info("%s: %s", tag, rec)
    return s

if __name__ == "__main__":
    interval_sec = int(os.environ.get("WFO_INTERVAL_SEC", "3600"))

    # Smoke pass: fast + cheap, just to test gates
    smoke_rl = {"n_envs": 2, "total_timesteps": 5000, "batch_size": 1024}

    # Promotion pass: real RL training but bounded
    promo_rl = {"n_envs": 4, "total_timesteps": 200_000, "batch_size": 2048}

    backoff = 60
    while True:
        try:
            out_smoke = run_wfo_once(dry_run=True, rl_fast_smoke=True, rl_overrides=smoke_rl)
            s = log_summary("smoke", out_smoke)

            if s.get("go_no_go"):
                logging.info("Gates passed — starting promotion run with RL training.")
                out_rl = run_wfo_once(dry_run=False, rl_fast_smoke=False, rl_overrides=promo_rl)
                log_summary("promotion", out_rl)
            else:
                logging.info("Gates failed — skipping RL training this cycle.")

            time.sleep(interval_sec)
            backoff = 60
        except Exception as e:
            logging.exception("Loop error: %s", e)
            time.sleep(backoff)
            backoff = min(backoff * 2, 1800)
