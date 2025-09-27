#!/usr/bin/env python3
"""Codex orchestrator CLI for offline RL training, CPCV selection, WFO validation, and shadow learners."""

# Examples:
#   python run_pipeline.py train_offline --symbol ES
#   python run_pipeline.py select_with_cpcv --symbol ES
#   python run_pipeline.py validate_wfo --symbols ES --is_days 5 --oos_days 1
#   python run_pipeline.py paper_and_shadow --symbol ES --deploy-canary

from __future__ import annotations

import argparse
import json
import os
import sys
import importlib
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

# Enforce dry-run defaults for safety
os.environ.setdefault("DRY_RUN", "1")
os.environ.setdefault("ALLOW_ORDERS", "0")

from wfo.runner import run_wfo  # noqa: E402
from wfo.utils import enable_determinism  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_yaml(path: Path | str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def merge_dicts(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = deepcopy(base)
    if not override:
        return result
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_minute_data(data_path: Path, symbol: str, fallback_days: int = 5) -> pd.DataFrame:
    if data_path.is_dir():
        parquet_files = sorted(data_path.glob(f"{symbol}*.parquet"))
        if parquet_files:
            logger.info("Loading minute data from %s", parquet_files[0])
            df = pd.read_parquet(parquet_files[0])
            if "timestamp" not in df.columns:
                df["timestamp"] = pd.date_range(start=datetime.utcnow(), periods=len(df), freq="1min")
            return df
    logger.warning("Minute data not found for %s at %s; generating synthetic sample", symbol, data_path)
    bars = fallback_days * 390
    index = pd.date_range(end=datetime.utcnow(), periods=bars, freq="1min")
    prices = 100 + np.cumsum(np.random.randn(bars) * 0.1)
    df = pd.DataFrame({
        "timestamp": index,
        "close": prices,
        "volume": np.random.randint(100, 1000, size=bars),
    })
    df["returns"] = df["close"].pct_change().fillna(0.0)
    return df


def ensure_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(start=datetime.utcnow(), periods=len(df), freq="1min")
    if "returns" not in df.columns and "close" in df.columns:
        df["returns"] = df["close"].pct_change().fillna(0.0)
    if "returns" not in df.columns:
        df["returns"] = 0.0
    return df


def save_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str))


def _resolve_legacy_main():
    try:
        mod = importlib.import_module("src.rl_trading_pipeline")
        return getattr(mod, "main")
    except Exception:
        mod = importlib.import_module("rl_trading_pipeline")
        return getattr(mod, "main")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_train_offline(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    data_cfg = config.get("datasets", {})
    minute_path = Path(data_cfg.get("minute_bars_path", "data/minute_bars"))
    df = ensure_returns(load_minute_data(minute_path, args.symbol, data_cfg.get("lookback_days", 5)))

    if config.get("deterministic_debug"):
        logger.info("Deterministic debug enabled (training may slow down)")
        enable_determinism(int(config.get("deterministic_seed", 42)))

    rl_defaults = config.get("rl_defaults", {})
    output_root = Path(args.output_dir or "artifacts/offline_train") / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root.mkdir(parents=True, exist_ok=True)

    rl_strategies = [s for s in config.get("strategies", []) if s.get("type") == "rl_policy"]
    if not rl_strategies:
        logger.error("No RL strategies defined in %s", args.config)
        return

    from wfo_rl import RLAdapter, RLSpec, make_env_from_df

    for strat in rl_strategies:
        rl_cfg = merge_dicts(rl_defaults, strat.get("rl", {}))
        reward_kwargs = strat.get("reward", {})
        spec = RLSpec(
            algo=strat.get("algo", "SAC"),
            policy=strat.get("policy", "MlpPolicy"),
            train_timesteps=int(rl_cfg.get("train_timesteps", 50_000)),
            n_envs=int(rl_cfg.get("n_envs", 1)),
            seed=int(rl_cfg.get("seed", 42)),
            policy_kwargs=rl_cfg.get("policy_kwargs") or {},
            algo_kwargs=rl_cfg.get("algo_kwargs") or {},
            vecnormalize_obs=bool(rl_cfg.get("vecnormalize_obs", True)),
            vecnormalize_reward=bool(rl_cfg.get("vecnormalize_reward", True)),
            use_imitation_warmstart=bool(rl_cfg.get("use_imitation_warmstart", False)),
            imitation_kwargs=rl_cfg.get("imitation_kwargs"),
            warmstart_epochs=int(rl_cfg.get("warmstart_epochs", 3)),
        )
        adapter = RLAdapter(spec, fast_smoke=bool(args.fast_smoke))
        env_fn = make_env_from_df(df, costs_bps=float(reward_kwargs.get("costs_bps", 0.0)), reward_kwargs=reward_kwargs, eval_mode=False)

        strategy_dir = output_root / strat.get("name", spec.algo)
        strategy_dir.mkdir(parents=True, exist_ok=True)
        try:
            model, vecnorm_path = adapter.fit_on_is(env_fn, str(strategy_dir))
            if hasattr(model, "save"):
                model.save(str(strategy_dir / "policy.zip"))
            save_metadata(
                strategy_dir / "metadata.json",
                {
                    "strategy": strat,
                    "rl": rl_cfg,
                    "reward": reward_kwargs,
                    "trained_at": datetime.utcnow().isoformat(),
                    "vecnormalize_stats": str(vecnorm_path) if vecnorm_path else None,
                },
            )
            logger.info("Offline training completed for %s", strat.get("name", spec.algo))
        except RuntimeError as exc:
            logger.error("Training failed for %s: %s", strat.get("name", spec.algo), exc)


def cmd_select_with_cpcv(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    data_cfg = config.get("datasets", {})
    minute_path = Path(data_cfg.get("minute_bars_path", "data/minute_bars"))
    df = ensure_returns(load_minute_data(minute_path, args.symbol, data_cfg.get("lookback_days", 5)))

    from algorithm_selector import select_strategies_with_cpcv

    cpcv_cfg = config.get("cpcv", {})
    cpcv_cfg.setdefault("log_dir", "artifacts/cpcv_selection")
    cpcv_cfg.setdefault("dsr_threshold", 0.05)
    cpcv_cfg.setdefault("session_minutes", config.get("session_minutes", {}))
    cpcv_cfg.setdefault("symbol", args.symbol)
    cpcv_cfg.setdefault("deterministic_debug", config.get("deterministic_debug", False))
    cpcv_cfg.setdefault("deterministic_seed", config.get("deterministic_seed", 42))
    shortlist = select_strategies_with_cpcv(df, config.get("strategies", []), cpcv_cfg)

    output_path = Path(args.output or Path(cpcv_cfg["log_dir"]) / "shortlist.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_metadata(output_path, {"shortlist": shortlist})
    logger.info("CPCV shortlist saved to %s", output_path)


def cmd_validate_wfo(args: argparse.Namespace) -> None:
    config_preview = load_yaml(args.config) if args.config else {}
    if config_preview.get("deterministic_debug"):
        logger.info("Deterministic debug enabled (training may slow down)")
        enable_determinism(int(config_preview.get("deterministic_seed", 42)))

    config_path = Path(args.config) if args.config else None
    symbols = args.symbols.split(",") if isinstance(args.symbols, str) else args.symbols
    result = run_wfo(
        symbols=symbols,
        is_days=args.is_days,
        oos_days=args.oos_days,
        step_days=args.step_days,
        cycles_min=args.cycles_min,
        embargo_days=args.embargo_days,
        label_lookahead_bars=args.label_lookahead_bars,
        cpcv_folds=args.cpcv_folds,
        config_path=config_path,
        dry_run=True,
        rl_fast_smoke=bool(args.rl_fast_smoke),
        rl_fast_overrides={"train_timesteps": args.rl_fast_timesteps} if args.rl_fast_timesteps else None,
    )
    logger.info("WFO summary saved to %s", result["output_dir"])


def cmd_paper_and_shadow(args: argparse.Namespace) -> None:
    config = load_yaml(args.config)
    if config.get("deterministic_debug"):
        logger.info("Deterministic debug enabled (training may slow down)")
        enable_determinism(int(config.get("deterministic_seed", 42)))
    data_cfg = config.get("datasets", {})
    minute_path = Path(data_cfg.get("minute_bars_path", "data/minute_bars"))
    df = ensure_returns(load_minute_data(minute_path, args.symbol, data_cfg.get("lookback_days", 5)))

    from ensemble_rl_coordinator import EnsembleRLCoordinator
    from enhanced_trading_environment import EnhancedTradingEnvironment, EnhancedTradingConfig
    from online_learning_system import OnlineLearningSystem

    # Prepare ensemble and online learning components
    env = EnhancedTradingEnvironment(data=df, config=EnhancedTradingConfig(use_dict_obs=False))
    coordinator = EnsembleRLCoordinator(env, enable_feature_flag=True)
    champion = next((s for s in config.get("strategies", []) if s.get("type") == "rl_policy"), None)
    if champion:
        coordinator.set_champion_strategy(champion)
    online = OnlineLearningSystem(enable_feature_flag=True)
    online.attach_ensemble_coordinator(coordinator)

    rl_candidates = [s for s in config.get("strategies", []) if s.get("type") == "rl_policy"]
    shadow_conf = rl_candidates[1] if len(rl_candidates) > 1 else rl_candidates[0]
    reward_conf = shadow_conf.get("reward", {})
    wfo_params = {
        "symbols": [args.symbol],
        "is_days": args.is_days,
        "oos_days": args.oos_days,
        "step_days": args.step_days,
        "cycles_min": args.cycles_min,
        "embargo_days": args.embargo_days,
        "label_lookahead_bars": args.label_lookahead_bars,
        "cpcv_folds": args.cpcv_folds,
        "config_path": Path(args.wfo_config) if args.wfo_config else None,
        "dsr_threshold": config.get("cpcv", {}).get("dsr_threshold", 0.05),
        "dry_run": True,
    }
    evaluation = online.run_shadow_champion_cycle(
        recent_data=df,
        rl_config=shadow_conf.get("rl"),
        reward_config=reward_conf,
        wfo_params=wfo_params,
        symbols=config.get("canary", {}).get("symbols", [args.symbol]),
        deploy_canary=args.deploy_canary,
    )
    logger.info("Shadow evaluation result: %s", evaluation.evaluation if evaluation else "no candidate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Codex pipeline orchestrator")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train_offline", help="Train RL policies offline with VecNormalize")
    train.add_argument("--config", default="configs/train_system.yaml")
    train.add_argument("--symbol", default="ES")
    train.add_argument("--output-dir", default=None)
    train.add_argument("--fast-smoke", action="store_true")
    train.set_defaults(func=cmd_train_offline)

    cpcv = sub.add_parser("select_with_cpcv", help="Rank strategies using CPCV + DSR")
    cpcv.add_argument("--config", default="configs/train_system.yaml")
    cpcv.add_argument("--symbol", default="ES")
    cpcv.add_argument("--output", default=None)
    cpcv.set_defaults(func=cmd_select_with_cpcv)

    wfo = sub.add_parser("validate_wfo", help="Run walk-forward validation with reality checks")
    wfo.add_argument("--symbols", default="ES")
    wfo.add_argument("--is_days", type=int, default=5)
    wfo.add_argument("--oos_days", type=int, default=1)
    wfo.add_argument("--step_days", type=int, default=1)
    wfo.add_argument("--cycles_min", type=int, default=1)
    wfo.add_argument("--embargo_days", type=int, default=1)
    wfo.add_argument("--label_lookahead_bars", type=int, default=1)
    wfo.add_argument("--cpcv_folds", type=int, default=20)
    wfo.add_argument("--rl_fast_smoke", type=int, default=0)
    wfo.add_argument("--rl_fast_timesteps", type=int, default=None)
    wfo.add_argument("--config", default="wfo/wfo_config.yaml")
    wfo.set_defaults(func=cmd_validate_wfo)

    paper = sub.add_parser("paper_and_shadow", help="Evaluate champion/challenger via shadow learner")
    paper.add_argument("--config", default="configs/train_system.yaml")
    paper.add_argument("--wfo-config", default="wfo/wfo_config.yaml")
    paper.add_argument("--symbol", default="ES")
    paper.add_argument("--is_days", type=int, default=5)
    paper.add_argument("--oos_days", type=int, default=1)
    paper.add_argument("--step_days", type=int, default=1)
    paper.add_argument("--cycles_min", type=int, default=1)
    paper.add_argument("--embargo_days", type=int, default=1)
    paper.add_argument("--label_lookahead_bars", type=int, default=1)
    paper.add_argument("--cpcv_folds", type=int, default=5)
    paper.add_argument("--deploy-canary", action="store_true")
    paper.set_defaults(func=cmd_paper_and_shadow)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        logger.info("No subcommand specified; delegating to legacy rl_trading_pipeline.main()")
        legacy_main = _resolve_legacy_main()
        legacy_main()
        return

    args.func(args)


if __name__ == "__main__":
    main()
