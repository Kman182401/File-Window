"""RL adapter wrapping SB3 algorithms for IS/OOS training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np

# Optional imports: allow skips when SB3/torch missing
try:  # pragma: no cover - heavy optional deps
    import gymnasium as gym  # noqa: F401
    from stable_baselines3 import PPO, SAC, TD3  # type: ignore
    from sb3_contrib import RecurrentPPO  # type: ignore
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # type: ignore
    SB3_AVAILABLE = True
except Exception:  # pragma: no cover
    SB3_AVAILABLE = False
    DummyVecEnv = None  # type: ignore
    VecNormalize = None  # type: ignore

try:  # optional user-provided agent
    from recurrent_ppo_agent import RecurrentPPOAgent  # type: ignore
    LOCAL_PPO_AVAILABLE = True
except Exception:  # pragma: no cover
    LOCAL_PPO_AVAILABLE = False


@dataclass
class RLSpec:
    algo: str
    policy: str = "MlpPolicy"
    train_timesteps: int = 50_000
    n_envs: int = 1
    seed: int = 42
    vecnormalize_obs: bool = True
    vecnormalize_reward: bool = True
    policy_kwargs: Optional[Dict[str, Any]] = None
    algo_kwargs: Optional[Dict[str, Any]] = None


class RLAdapter:
    """Helper that encapsulates SB3/Local training and deterministic replay."""

    def __init__(self, spec: RLSpec, fast_smoke: bool = False):
        self.spec = spec
        self.fast = fast_smoke

    def _build_vec_env(self, make_env_fn: Callable[[], Any], n_envs: int, *, training: bool) -> Any:
        if not SB3_AVAILABLE or DummyVecEnv is None or VecNormalize is None:
            raise RuntimeError("Stable-Baselines3 not available")
        venv = DummyVecEnv([make_env_fn for _ in range(n_envs)])
        venv = VecNormalize(
            venv,
            training=training,
            norm_obs=self.spec.vecnormalize_obs,
            norm_reward=self.spec.vecnormalize_reward,
        )
        return venv

    def fit_on_is(self, make_env_fn: Callable[[], Any], log_dir: str) -> Tuple[Any, Optional[VecNormalize]]:
        if not SB3_AVAILABLE and not LOCAL_PPO_AVAILABLE:
            raise RuntimeError("RL libraries not installed")

        steps = min(self.spec.train_timesteps, 5_000) if self.fast else self.spec.train_timesteps
        policy_kwargs = self.spec.policy_kwargs or {}
        algo_kwargs = dict(seed=self.spec.seed, **(self.spec.algo_kwargs or {}))

        if self.spec.algo == "LocalRecurrentPPO":
            if not LOCAL_PPO_AVAILABLE:
                raise RuntimeError("Local recurrent PPO agent not available")
            env = make_env_fn()
            agent = RecurrentPPOAgent(**algo_kwargs)
            agent.train(env, total_timesteps=steps, policy_kwargs=policy_kwargs)
            model = agent.freeze()
            vecnorm = None
        else:
            env = self._build_vec_env(make_env_fn, self.spec.n_envs, training=True)
            algo = self._resolve_algo()
            model = algo(
                self._resolve_policy(),
                env,
                **algo_kwargs,
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_dir,
            )
            model.learn(total_timesteps=steps, progress_bar=False)
            vecnorm = env if isinstance(env, VecNormalize) else None
        return model, vecnorm

    def score_on_oos(
        self,
        model: Any,
        make_env_fn: Callable[[], Any],
        vecnorm_stats: Optional[VecNormalize],
    ) -> np.ndarray:
        if not SB3_AVAILABLE and not hasattr(model, "predict"):
            raise RuntimeError("RL evaluation unavailable")

        if SB3_AVAILABLE and DummyVecEnv is not None and VecNormalize is not None and isinstance(vecnorm_stats, VecNormalize):
            venv = DummyVecEnv([make_env_fn])
            eval_env = VecNormalize(
                venv,
                training=False,
                norm_obs=self.spec.vecnormalize_obs,
                norm_reward=False,
            )
            eval_env.obs_rms = vecnorm_stats.obs_rms
            eval_env.ret_rms = vecnorm_stats.ret_rms
        else:
            eval_env = DummyVecEnv([make_env_fn]) if SB3_AVAILABLE and DummyVecEnv is not None else make_env_fn()

        returns: list[float] = []
        obs = eval_env.reset()[0] if SB3_AVAILABLE and DummyVecEnv is not None else eval_env.reset()
        state = None
        while True:
            if hasattr(model, "predict"):
                if self.spec.algo == "RecurrentPPO":
                    action, state = model.predict(obs, state=state, deterministic=True)
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                action = model.act(obs)

            step_out = eval_env.step(action)
            if SB3_AVAILABLE:
                obs, reward, terminated, truncated, _ = step_out
                done = bool(np.any(terminated) or np.any(truncated))
                returns.append(float(reward))
            else:
                obs, reward, done, _ = step_out
                returns.append(float(reward))
            if done:
                break
        return np.asarray(returns, dtype=float)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _resolve_algo(self):
        if not SB3_AVAILABLE:
            raise RuntimeError("SB3 algorithms unavailable")
        mapping = {
            "RecurrentPPO": RecurrentPPO,
            "PPO": PPO,
            "SAC": SAC,
            "TD3": TD3,
        }
        try:
            return mapping[self.spec.algo]
        except KeyError:
            raise ValueError(f"Unsupported RL algo '{self.spec.algo}'")

    def _resolve_policy(self) -> str:
        if self.spec.algo == "RecurrentPPO":
            return self.spec.policy or "MlpLstmPolicy"
        return self.spec.policy or "MlpPolicy"
