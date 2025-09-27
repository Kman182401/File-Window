"""RL adapter wrapping Stable-Baselines3 training for WFO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy deps
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

try:  # pragma: no cover - optional heavy deps
    import gymnasium as gym  # noqa: F401
    from stable_baselines3 import PPO, SAC, TD3  # type: ignore
    from sb3_contrib import RecurrentPPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore
    SB3_AVAILABLE = True
except Exception:  # pragma: no cover
    SB3_AVAILABLE = False
    DummyVecEnv = None  # type: ignore
    VecNormalize = None  # type: ignore

from .imitation_learning import pretrain_policy_via_behavior_cloning


@dataclass
class RLSpec:
    """Configuration describing how to instantiate and train an RL model."""

    algo: str
    policy: str = "MlpPolicy"
    train_timesteps: int = 50_000
    n_envs: int = 1
    seed: int = 42
    policy_kwargs: Optional[Dict[str, Any]] = None
    algo_kwargs: Optional[Dict[str, Any]] = None
    vecnormalize_obs: bool = True
    vecnormalize_reward: bool = True
    use_imitation_warmstart: bool = False
    imitation_kwargs: Optional[Dict[str, Any]] = None
    warmstart_epochs: int = 5


class RLAdapter:
    """Helper that encapsulates SB3/Contrib training and deterministic replay."""

    def __init__(self, spec: RLSpec, fast_smoke: bool = False):
        self.spec = spec
        self.fast = bool(fast_smoke)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_on_is(self, make_env_fn: Callable[[], Any], log_dir: str) -> Tuple[Any, Optional[Any]]:
        """Train the requested algorithm on the in-sample data."""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not available for RL training")

        steps = self.spec.train_timesteps
        if self.fast:
            steps = min(steps, 1_000)

        vec_env = self._build_vec_env(make_env_fn, self.spec.n_envs, training=True)
        algo_cls = self._resolve_algo()
        policy_kwargs = self.spec.policy_kwargs or {}
        algo_kwargs = dict(seed=self.spec.seed, **(self.spec.algo_kwargs or {}))
        model = algo_cls(
            self._resolve_policy_name(),
            vec_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            **algo_kwargs,
        )

        self._maybe_run_behavior_cloning(model, make_env_fn)
        model.learn(total_timesteps=steps, progress_bar=False)

        vecnorm = vec_env if isinstance(vec_env, VecNormalize) else None
        if vecnorm is not None:
            vecnorm.training = False
        return model, vecnorm

    def score_on_oos(
        self,
        model: Any,
        make_env_fn: Callable[[], Any],
        vecnorm_stats: Optional[Any],
    ) -> np.ndarray:
        """Score a trained model on out-of-sample data and return per-bar returns."""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not available for RL evaluation")

        if DummyVecEnv is None:
            raise RuntimeError("Vectorised environment utilities unavailable")

        if isinstance(vecnorm_stats, VecNormalize):
            base_env = DummyVecEnv([make_env_fn])
            eval_env = VecNormalize(
                base_env,
                training=False,
                norm_obs=self.spec.vecnormalize_obs,
                norm_reward=False,
            )
            eval_env.obs_rms = getattr(vecnorm_stats, "obs_rms", None)
            eval_env.ret_rms = getattr(vecnorm_stats, "ret_rms", None)
            eval_env.clip_obs = getattr(vecnorm_stats, "clip_obs", 10.0)
            eval_env.clip_reward = getattr(vecnorm_stats, "clip_reward", np.inf)
        else:
            eval_env = DummyVecEnv([make_env_fn])

        observations, _ = eval_env.reset()
        state = None
        rewards: list[float] = []
        while True:
            if hasattr(model, "predict"):
                action, state = model.predict(observations, state=state, deterministic=True)
            else:  # pragma: no cover - custom agents
                action = model.act(observations)
            observations, reward, terminated, truncated, _ = eval_env.step(action)
            rewards.extend(np.asarray(reward, dtype=float).flatten().tolist())
            if bool(np.any(terminated) or np.any(truncated)):
                break
        if hasattr(eval_env, "close"):
            eval_env.close()
        return np.asarray(rewards, dtype=float)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_vec_env(self, make_env_fn: Callable[[], Any], n_envs: int, *, training: bool) -> Any:
        if DummyVecEnv is None or VecNormalize is None:
            raise RuntimeError("Vectorised environment wrappers unavailable")
        venv = DummyVecEnv([make_env_fn for _ in range(n_envs)])
        venv = VecNormalize(
            venv,
            training=training,
            norm_obs=self.spec.vecnormalize_obs,
            norm_reward=self.spec.vecnormalize_reward if training else False,
        )
        return venv

    def _resolve_algo(self):
        mapping = {
            "RecurrentPPO": RecurrentPPO,
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
        }
        try:
            return mapping[self.spec.algo]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unsupported RL algo '{self.spec.algo}'") from exc

    def _resolve_policy_name(self) -> str:
        if self.spec.policy:
            return self.spec.policy
        if self.spec.algo == "RecurrentPPO":
            return "MlpLstmPolicy"
        return "MlpPolicy"

    def _maybe_run_behavior_cloning(self, model: Any, make_env_fn: Callable[[], Any]) -> None:
        if not self.spec.use_imitation_warmstart:
            return
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch not available for imitation warm-start")
        df_source = getattr(make_env_fn, "_df", None)
        teacher_conf = dict(self.spec.imitation_kwargs or {})
        teacher_conf.setdefault("make_env_fn", make_env_fn)
        dataset = pretrain_policy_via_behavior_cloning(df_source, teacher_conf)
        if not dataset:
            return
        observations = dataset.get("observations")
        actions = dataset.get("actions")
        if observations is None or actions is None:
            return
        if len(observations) == 0:
            return
        obs_tensor = torch.as_tensor(observations, dtype=torch.float32)
        act_tensor = torch.as_tensor(actions, dtype=torch.float32)
        self._apply_behavior_cloning(model, obs_tensor, act_tensor)

    def _apply_behavior_cloning(self, model: Any, obs_tensor, act_tensor) -> None:
        algo = self.spec.algo
        epochs = max(1, int(self.spec.warmstart_epochs))
        if algo in ("SAC", "TD3"):
            actor = getattr(model, "actor", None)
            if actor is None or not hasattr(actor, "optimizer"):
                return
            optimizer = actor.optimizer
            actor.train()
            device = next(actor.parameters()).device
            obs_device = obs_tensor.to(device)
            act_device = act_tensor.to(device)
            for _ in range(epochs):
                optimizer.zero_grad()
                pred = actor(obs_device)
                loss = torch.nn.functional.mse_loss(pred, act_device[:, : pred.shape[-1]])
                loss.backward()
                optimizer.step()
        elif algo == "PPO":
            policy = getattr(model, "policy", None)
            if policy is None or not hasattr(policy, "optimizer"):
                return
            optimizer = policy.optimizer
            policy.train()
            device = next(policy.parameters()).device
            obs_device = obs_tensor.to(device)
            act_device = act_tensor.to(device)
            for _ in range(epochs):
                optimizer.zero_grad()
                dist = policy.get_distribution(obs_device)
                mean_action = dist.mean
                loss = torch.nn.functional.mse_loss(
                    mean_action,
                    act_device[:, : mean_action.shape[-1]],
                )
                loss.backward()
                optimizer.step()
        elif algo == "RecurrentPPO":
            policy = getattr(model, "policy", None)
            if policy is None or not hasattr(policy, "optimizer"):
                return
            optimizer = policy.optimizer
            policy.train()
            batch = obs_tensor.shape[0]
            device = next(policy.parameters()).device
            obs_device = obs_tensor.to(device)
            act_device = act_tensor.to(device)
            for _ in range(epochs):
                optimizer.zero_grad()
                loss_accum = torch.zeros((), device=device)
                for idx in range(batch):
                    obs_sample = obs_device[idx : idx + 1]
                    episode_starts = torch.ones((1, 1), dtype=torch.float32, device=device)
                    lstm_states = tuple(state.to(device) for state in policy.initial_state(batch_size=1))
                    dist = policy.get_distribution(obs_sample, lstm_states, episode_starts)
                    mean_action = dist.mean.squeeze(0)
                    target = act_device[idx][: mean_action.shape[-1]]
                    loss_accum = loss_accum + torch.nn.functional.mse_loss(mean_action, target)
                (loss_accum / batch).backward()
                optimizer.step()
        # Other algos: noop warm-start


__all__ = ["RLSpec", "RLAdapter", "SB3_AVAILABLE"]
