#!/usr/bin/env python3
import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from online_learning_system import EvaluationHarness

log = logging.getLogger("walk_forward")


def load_config(path: Path) -> dict:
    defaults = {
        "experiment_name": "ppo_experiment",
        "env_id": "CartPole-v1",
        "clip_range_vf": 0.2,
        "total_timesteps": 5000,
        "eval_episodes": 5,
        "seed": 42,
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 64,
        "n_epochs": 4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "rc_iterations": 200,
    }
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    defaults.update(data)
    return defaults


def make_env(env_id: str, seed: int):
    def _init():
        import gymnasium as gym

        env = gym.make(env_id)
        env.reset(seed=seed)
        return env

    return _init


def train_and_evaluate(cfg: dict) -> dict:
    seed = int(cfg["seed"])
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = DummyVecEnv([make_env(cfg["env_id"], seed + i) for i in range(1)])
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=cfg["gamma"],
    )

    policy_kwargs = {
        "lstm_hidden_size": 64,
        "n_lstm_layers": 1,
        "shared_lstm": True,
    }

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        clip_range_vf=cfg["clip_range_vf"],
        normalize_advantage=True,
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        target_kl=None,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    log.info(
        "Training RecurrentPPO (%s) for %d timesteps (clip_range_vf=%s)",
        cfg["experiment_name"],
        cfg["total_timesteps"],
        str(cfg["clip_range_vf"]),
    )
    model.learn(total_timesteps=int(cfg["total_timesteps"]), progress_bar=False)

    artifacts_dir = Path("artifacts/walk_forward")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    vec_path = artifacts_dir / f"vecnorm_{cfg['experiment_name']}.pkl"
    train_env.save(vec_path)

    eval_env = DummyVecEnv([make_env(cfg["env_id"], seed + 100 + i) for i in range(1)])
    eval_env = VecNormalize.load(vec_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    harness = EvaluationHarness()
    step_returns = []
    episode_returns = []

    episodes = int(cfg["eval_episodes"])
    completed = 0
    obs = eval_env.reset()
    lstm_states = None
    episode_start = np.ones((eval_env.num_envs,), dtype=bool)
    current_episode = []

    while completed < episodes:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        obs, reward, done, info = eval_env.step(action)
        reward_scalar = float(np.array(reward).ravel()[0])
        step_returns.append(reward_scalar)
        current_episode.append(reward_scalar)
        episode_start = np.array(done, dtype=bool)
        if episode_start[0]:
            episode_returns.append(float(np.sum(current_episode)))
            current_episode = []
            completed += 1
            obs = eval_env.reset()
            lstm_states = None
            episode_start = np.ones((eval_env.num_envs,), dtype=bool)

    returns_array = np.array(step_returns, dtype=np.float64)
    benchmark = np.zeros_like(returns_array)
    returns_matrix = np.column_stack([benchmark, returns_array])

    sharpe = harness.compute_sharpe_ratio(returns_array)
    dsr = harness.deflated_sharpe_ratio(returns_array, benchmark_sr=0.0, num_trials=1)
    rc_pvalue = harness.reality_check_pvalue(
        returns_matrix,
        benchmark_index=0,
        n_iterations=int(cfg["rc_iterations"]),
    )

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "experiment": cfg["experiment_name"],
        "clip_range_vf": cfg["clip_range_vf"],
        "total_timesteps": int(cfg["total_timesteps"]),
        "eval_episodes": episodes,
        "seed": seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "metrics": {
            "oos_sharpe": float(sharpe),
            "deflated_sharpe_ratio": float(dsr),
            "reality_check_pvalue": float(rc_pvalue),
            "step_return_mean": float(np.mean(returns_array)),
            "step_return_std": float(np.std(returns_array, ddof=1)) if len(returns_array) > 1 else 0.0,
            "episode_return_mean": float(np.mean(episode_returns)) if episode_returns else 0.0,
        },
        "counts": {
            "steps": len(step_returns),
            "episodes": len(episode_returns),
        },
        "artifacts": {
            "vecnormalize_path": str(vec_path.resolve()),
        },
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
