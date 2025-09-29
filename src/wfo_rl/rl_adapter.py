"""RL adapter wrapping Stable-Baselines3 training for WFO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import logging
import numpy as np

try:  # pragma: no cover - optional heavy deps
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore

if TORCH_AVAILABLE:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

try:  # pragma: no cover - optional heavy deps
    import gymnasium as gym  # noqa: F401
    from stable_baselines3 import PPO, SAC, TD3  # type: ignore
    from sb3_contrib import RecurrentPPO  # type: ignore
    from stable_baselines3.common.env_util import make_vec_env  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # type: ignore
    SB3_AVAILABLE = True
except Exception:  # pragma: no cover
    SB3_AVAILABLE = False
    DummyVecEnv = None  # type: ignore
    SubprocVecEnv = None  # type: ignore
    VecNormalize = None  # type: ignore
    make_vec_env = None  # type: ignore

from .imitation_learning import pretrain_policy_via_behavior_cloning


logger = logging.getLogger(__name__)


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
    device: str = "auto"
    compile_policy: bool = False


class RLAdapter:
    """Helper that encapsulates SB3/Contrib training and deterministic replay."""

    def __init__(self, spec: RLSpec, fast_smoke: bool = False):
        self.spec = spec
        self.fast = bool(fast_smoke)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_on_is(self, make_env_fn: Callable[[], Any], log_dir: str) -> Tuple[Any, Optional[Path]]:
        """Train the requested algorithm on the in-sample data."""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not available for RL training")

        log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)
        steps = self.spec.train_timesteps
        if self.fast:
            steps = min(steps, 1_000)

        vec_env = self._build_vec_env(make_env_fn, self.spec.n_envs, training=True)
        try:
            algo_cls = self._resolve_algo()
            policy_kwargs = dict(self.spec.policy_kwargs or {})
            algo_kwargs = dict(self.spec.algo_kwargs or {})
            algo_kwargs.setdefault("seed", self.spec.seed)
            algo_kwargs.setdefault("device", self._resolve_device())
            if self.spec.algo == "TD3":
                for key in ("use_sde", "sde_sample_freq", "use_sde_at_warmup"):
                    policy_kwargs.pop(key, None)
                    algo_kwargs.pop(key, None)
            model = algo_cls(
                self._resolve_policy_name(),
                vec_env,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir,
                **algo_kwargs,
            )

            self._maybe_compile_policy(model)

            self._maybe_run_behavior_cloning(model, make_env_fn)
            model.learn(total_timesteps=steps, progress_bar=False)

            stats_path: Optional[Path] = None
            if isinstance(vec_env, VecNormalize):
                vec_env.training = False
                stats_path = log_root / "vecnormalize.pkl"
                vec_env.save(str(stats_path))
            return model, stats_path
        finally:
            if hasattr(vec_env, "close"):
                try:
                    vec_env.close()
                except Exception:
                    pass

    def score_on_oos(
        self,
        model: Any,
        make_env_fn: Callable[[], Any],
        vecnorm_stats: Optional[Union[str, Path]],
    ) -> np.ndarray:
        """Score a trained model on out-of-sample data and return per-bar returns."""
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not available for RL evaluation")

        if DummyVecEnv is None:
            raise RuntimeError("Vectorised environment utilities unavailable")

        stats_path: Optional[Path] = Path(vecnorm_stats) if vecnorm_stats else None
        base_env = DummyVecEnv([make_env_fn])
        try:
            if stats_path and stats_path.exists() and VecNormalize is not None:
                eval_env = VecNormalize.load(str(stats_path), base_env)
                eval_env.training = False
                eval_env.norm_reward = False
            else:
                eval_env = VecNormalize(
                    base_env,
                    training=False,
                    norm_obs=self.spec.vecnormalize_obs,
                    norm_reward=False,
                ) if VecNormalize is not None and self.spec.vecnormalize_obs else base_env

            reset_result = eval_env.reset()
            if isinstance(reset_result, tuple):
                observations, _ = reset_result
            else:
                observations = reset_result
                _ = None
            state = None
            rewards: list[float] = []
            while True:
                if hasattr(model, "predict"):
                    action, state = model.predict(observations, state=state, deterministic=True)
                else:  # pragma: no cover - custom agents
                    action = model.act(observations)
                step_result = eval_env.step(action)
                if isinstance(step_result, tuple):
                    if len(step_result) == 5:
                        observations, reward, terminated, truncated, _ = step_result
                    elif len(step_result) == 4:
                        observations, reward, done, _ = step_result
                        terminated = done
                        truncated = np.zeros_like(done, dtype=bool)
                    else:  # pragma: no cover - defensive
                        observations, reward, terminated = step_result  # type: ignore[misc]
                        truncated = np.zeros_like(terminated, dtype=bool)
                else:  # pragma: no cover - highly unlikely
                    observations = step_result
                    reward = np.zeros((1,), dtype=float)
                    terminated = np.ones((1,), dtype=bool)
                    truncated = np.ones((1,), dtype=bool)
                rewards.extend(np.asarray(reward, dtype=float).flatten().tolist())
                if bool(np.any(terminated) or np.any(truncated)):
                    break
            return np.asarray(rewards, dtype=float)
        finally:
            if 'eval_env' in locals() and hasattr(eval_env, "close"):
                try:
                    eval_env.close()
                except Exception:
                    pass
            elif hasattr(base_env, "close"):
                try:
                    base_env.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_vec_env(self, make_env_fn: Callable[[], Any], n_envs: int, *, training: bool) -> Any:
        if not SB3_AVAILABLE:
            raise RuntimeError("Stable-Baselines3 not available for vectorised environments")
        if make_vec_env is not None and SubprocVecEnv is not None and n_envs > 1:
            venv = make_vec_env(
                make_env_fn,
                n_envs=n_envs,
                seed=self.spec.seed,
                vec_env_cls=SubprocVecEnv,
            )
        elif DummyVecEnv is not None:
            venv = DummyVecEnv([make_env_fn for _ in range(n_envs)])
        else:
            raise RuntimeError("Vectorised environment wrappers unavailable")

        if VecNormalize is None:
            return venv

        norm_kwargs = dict(
            training=training,
            norm_obs=self.spec.vecnormalize_obs,
            norm_reward=self.spec.vecnormalize_reward if training else False,
        )
        if self.spec.vecnormalize_obs:
            try:
                import gymnasium as gym  # delay for optional dependency

                obs_space = venv.observation_space
                if isinstance(obs_space, gym.spaces.Dict):
                    box_keys = [key for key, space in obs_space.spaces.items() if isinstance(space, gym.spaces.Box)]
                    norm_kwargs["norm_obs_keys"] = box_keys
            except Exception:
                pass

        try:
            return VecNormalize(venv, **norm_kwargs)
        except Exception:
            # Fallback: attempt VecNormalize without key filtering to avoid silently disabling normalization
            return VecNormalize(
                venv,
                training=training,
                norm_obs=self.spec.vecnormalize_obs,
                norm_reward=self.spec.vecnormalize_reward if training else False,
            )

        return venv

    def _resolve_device(self) -> str:
        if self.spec.device != "auto":
            return self.spec.device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _maybe_compile_policy(self, model: Any) -> None:
        if not self.spec.compile_policy or not TORCH_AVAILABLE:
            return
        if self._resolve_device() == "cuda" and not self._gpu_supports_triton():
            logger.info(
                "Skipping torch.compile for %s: CUDA compute capability %s < 7.0",
                model.__class__.__name__,
                getattr(torch.cuda, "get_device_capability", lambda *_: ("?", "?"))(),
            )
            return
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            return
        try:  # pragma: no cover - best effort
            model.policy = compile_fn(model.policy)
        except Exception as exc:
            logger.debug("torch.compile failed for %s: %s", model.__class__.__name__, exc)

    def _gpu_supports_triton(self) -> bool:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        return major >= 7

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
