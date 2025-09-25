#!/usr/bin/env python3
import argparse
import sys
import json
import logging
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from enhanced_trading_environment import EnhancedTradingConfig, EnhancedTradingEnvironment
from online_learning_system import EvaluationHarness

log = logging.getLogger("walk_forward")


def load_config(path: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "experiment_name": "ppo_experiment",
        "clip_range_vf": 0.2,
        "total_timesteps": 5000,
        "eval_episodes": 100,
        "seeds": [101, 202, 303, 404, 505],
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "target_kl": 0.03,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "rc_iterations": 1000,
        "policy": "MlpLstmPolicy",
        "n_envs": 1,
        "market_data_csv": None,
        "market_data_length": 5000,
        "environment": {
            "config": {
                "lookback_window": 64,
                "use_dict_obs": False,
                "use_continuous_actions": False,
                "use_multi_component_reward": True,
                "use_domain_randomization": True,
                "max_episode_length": 512,
                "random_start": True,
            },
            "use_domain_randomization_train": True,
            "use_domain_randomization_eval": False,
        },
    }
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    defaults.update(loaded)
    return defaults


def load_market_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    csv_path = cfg.get("market_data_csv")
    if csv_path:
        candidate = Path(csv_path).expanduser()
        if candidate.exists():
            data = pd.read_csv(candidate)
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            required = {"open", "high", "low", "close", "volume"}
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(f"Market data file missing columns: {sorted(missing)}")
            return data.reset_index(drop=True)
        log.warning("Market data CSV %s not found; falling back to synthetic data", candidate)
    length = int(cfg.get("market_data_length", 5000))
    return generate_synthetic_data(length)


def generate_synthetic_data(length: int) -> pd.DataFrame:
    rng = np.random.default_rng(12345)
    returns = rng.normal(0.0, 0.0015, size=length)
    close = 100 + np.cumsum(returns)
    open_prices = close + rng.normal(0.0, 0.05, size=length)
    high = np.maximum(open_prices, close) + np.abs(rng.normal(0.0, 0.1, size=length))
    low = np.minimum(open_prices, close) - np.abs(rng.normal(0.0, 0.1, size=length))
    volume = rng.lognormal(mean=11.0, sigma=0.35, size=length)
    timestamp = pd.date_range("2020-01-01", periods=length, freq="5T")
    data = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    return data.reset_index(drop=True)


def build_env_config(config_dict: Dict[str, Any]) -> EnhancedTradingConfig:
    valid_keys = {field.name for field in fields(EnhancedTradingConfig)}
    filtered = {k: v for k, v in (config_dict or {}).items() if k in valid_keys}
    return EnhancedTradingConfig(**filtered)


def make_env_factory(
    market_data: pd.DataFrame,
    env_settings: Dict[str, Any],
    training: bool,
    seed: int,
):
    def _init():
        config = build_env_config(env_settings.get("config", {}))
        flag_key = "use_domain_randomization_train" if training else "use_domain_randomization_eval"
        if flag_key in env_settings:
            config.use_domain_randomization = bool(env_settings[flag_key])
        env = EnhancedTradingEnvironment(data=market_data.copy(), config=config)
        env.reset(seed=seed)
        return env

    return _init


def evaluate_policy(
    model: RecurrentPPO,
    eval_env: VecNormalize,
    episodes: int,
) -> Dict[str, Any]:
    harness = EvaluationHarness()
    step_returns: List[float] = []
    episode_returns: List[float] = []

    obs = eval_env.reset()
    lstm_states = None
    episode_start = np.ones((eval_env.num_envs,), dtype=bool)
    current_episode: List[float] = []

    while len(episode_returns) < episodes:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, reward, done, _ = eval_env.step(action)
        reward_scalar = float(np.array(reward).ravel()[0])
        step_returns.append(reward_scalar)
        current_episode.append(reward_scalar)
        episode_start = np.array(done, dtype=bool)
        if episode_start.any():
            episode_returns.append(float(np.sum(current_episode)))
            current_episode = []
            obs = eval_env.reset()
            lstm_states = None
            episode_start = np.ones((eval_env.num_envs,), dtype=bool)

    episode_array = np.array(episode_returns, dtype=np.float64)
    benchmark = np.zeros_like(episode_array)
    returns_matrix = np.column_stack([benchmark, episode_array])

    metrics = {
        "episode_returns": episode_returns,
        "step_returns": step_returns,
        "episode_mean": float(np.mean(episode_array)),
        "episode_std": float(np.std(episode_array, ddof=1)) if episode_array.size > 1 else 0.0,
        "sharpe": float(harness.compute_sharpe_ratio(episode_array)),
        "dsr": float(harness.deflated_sharpe_ratio(episode_array, benchmark_sr=0.0, num_trials=1)),
        "rc_pvalue": float(
            harness.reality_check_pvalue(
                returns_matrix,
                benchmark_index=0,
                n_iterations=int(max(1000, episodes)),
            )
        ),
    }
    return metrics


def train_and_evaluate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seeds = cfg.get("seeds") or [cfg.get("seed", 42)]
    seeds = [int(s) for s in seeds]
    market_data = load_market_data(cfg)
    env_settings = cfg.get("environment", {})
    artifacts_dir = Path("artifacts/walk_forward")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    aggregate_episode_returns: List[float] = []
    aggregate_step_returns: List[float] = []
    seed_summaries: List[Dict[str, Any]] = []
    vec_paths: List[str] = []

    for seed in seeds:
        set_random_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_env = DummyVecEnv([
            make_env_factory(market_data, env_settings, training=True, seed=seed)
        ])
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=cfg["gamma"],
        )

        model = RecurrentPPO(
            policy=cfg.get("policy", "MlpLstmPolicy"),
            env=train_env,
            learning_rate=cfg["learning_rate"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            gamma=cfg["gamma"],
            gae_lambda=cfg["gae_lambda"],
            clip_range=cfg["clip_range"],
            clip_range_vf=cfg.get("clip_range_vf"),
            normalize_advantage=True,
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            target_kl=cfg.get("target_kl"),
            verbose=0,
            seed=seed,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        log.info(
            "[%s] Training seed %d for %d timesteps (clip_range_vf=%s, target_kl=%s)",
            cfg["experiment_name"],
            seed,
            cfg["total_timesteps"],
            str(cfg.get("clip_range_vf")),
            str(cfg.get("target_kl")),
        )
        model.learn(total_timesteps=int(cfg["total_timesteps"]), progress_bar=False)

        vec_path = artifacts_dir / f"vecnorm_{cfg['experiment_name']}_seed{seed}.pkl"
        train_env.save(vec_path)
        vec_paths.append(str(vec_path.resolve()))
        train_env.close()

        eval_env = DummyVecEnv([
            make_env_factory(market_data, env_settings, training=False, seed=seed + 1000)
        ])
        eval_env = VecNormalize.load(vec_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

        metrics = evaluate_policy(model, eval_env, int(cfg["eval_episodes"]))

        aggregate_episode_returns.extend(metrics["episode_returns"])
        aggregate_step_returns.extend(metrics["step_returns"])
        seed_summaries.append(
            {
                "seed": seed,
                "episode_return_mean": metrics["episode_mean"],
                "episode_return_std": metrics["episode_std"],
                "oos_sharpe": metrics["sharpe"],
                "deflated_sharpe_ratio": metrics["dsr"],
                "reality_check_pvalue": metrics["rc_pvalue"],
                "episodes": len(metrics["episode_returns"]),
                "steps": len(metrics["step_returns"]),
            }
        )

        eval_env.close()
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    harness = EvaluationHarness()
    episode_array = np.array(aggregate_episode_returns, dtype=np.float64)
    benchmark = np.zeros_like(episode_array)
    returns_matrix = np.column_stack([benchmark, episode_array])

    sharpe = harness.compute_sharpe_ratio(episode_array)
    dsr = harness.deflated_sharpe_ratio(
        episode_array,
        benchmark_sr=0.0,
        num_trials=max(1, len(seeds)),
    )
    rc_iterations = int(max(cfg.get("rc_iterations", 1000), 1000))
    rc_pvalue = harness.reality_check_pvalue(
        returns_matrix,
        benchmark_index=0,
        n_iterations=rc_iterations,
    )

    step_array = np.array(aggregate_step_returns, dtype=np.float64)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment": cfg["experiment_name"],
        "clip_range_vf": cfg.get("clip_range_vf"),
        "total_timesteps": int(cfg["total_timesteps"]),
        "eval_episodes": int(cfg["eval_episodes"]),
        "seeds": seeds,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics": {
            "oos_sharpe": float(sharpe),
            "deflated_sharpe_ratio": float(dsr),
            "reality_check_pvalue": float(rc_pvalue),
            "step_return_mean": float(np.mean(step_array)) if step_array.size else 0.0,
            "step_return_std": float(np.std(step_array, ddof=1)) if step_array.size > 1 else 0.0,
            "episode_return_mean": float(np.mean(episode_array)),
            "episode_return_std": float(np.std(episode_array, ddof=1)) if episode_array.size > 1 else 0.0,
        },
        "counts": {
            "steps": len(aggregate_step_returns),
            "episodes": len(aggregate_episode_returns),
        },
        "artifacts": {
            "vecnormalize_paths": vec_paths,
        },
        "seed_metrics": seed_summaries,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO walk-forward evaluation")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("promotion_reports"),
        help="Directory for promotion reports",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    cfg = load_config(args.config)
    report = train_and_evaluate(cfg)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"promotion_report_{cfg['experiment_name']}.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log.info(
        "OOS Sharpe: %.4f | DSR: %.4f | RC p-value: %.4f",
        report["metrics"]["oos_sharpe"],
        report["metrics"]["deflated_sharpe_ratio"],
        report["metrics"]["reality_check_pvalue"],
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
