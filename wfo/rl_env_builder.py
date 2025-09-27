"""Utility for constructing simple trading environments for RL adapters."""

from __future__ import annotations

from typing import Callable

import numpy as np

try:  # pragma: no cover - optional heavy dep
    import gymnasium as gym
except Exception:  # pragma: no cover
    gym = None


class TradingEnv(gym.Env if gym else object):  # type: ignore[misc]
    """Very light-weight environment replaying pre-computed returns.

    Observation: feature vector at current bar.
    Action: continuous position in [-1, 1].
    Reward: position * return - cost.
    """

    metadata = {"render_modes": []}

    def __init__(self, df, costs_bps: float = 0.0):  # pragma: no cover - smoke sized
        if gym is None:
            raise RuntimeError("gymnasium is required for RL environments")
        super().__init__()
        self.df = df.reset_index(drop=True)
        feat_cols = [c for c in df.columns if c not in ("timestamp", "returns", "price")]
        self.features = (
            df[feat_cols].to_numpy(dtype=np.float32)
            if feat_cols
            else df[["returns"]].to_numpy(dtype=np.float32)
        )
        self.returns = df.get("returns", np.zeros(len(df), dtype=np.float32)).to_numpy(dtype=np.float32)
        self.costs = costs_bps * 1e-4
        n = self.features.shape[1]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(n,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._idx = 0
        self._pos_prev = 0.0

    def reset(self, *, seed=None, options=None):  # pragma: no cover - smoke sized
        super().reset(seed=seed)
        self._idx = 0
        self._pos_prev = 0.0
        return self.features[self._idx], {}

    def step(self, action):  # pragma: no cover - smoke sized
        self._idx += 1
        pos = float(np.clip(action, -1.0, 1.0))
        ret = self.returns[self._idx] if self._idx < len(self.returns) else 0.0
        cost = abs(pos - self._pos_prev) * self.costs
        reward = pos * ret - cost
        self._pos_prev = pos
        terminated = self._idx >= (len(self.features) - 1)
        truncated = False
        obs = self.features[self._idx] if not terminated else self.features[-1]
        return obs, reward, terminated, truncated, {}


def make_env_from_df(df, costs_bps: float = 0.0) -> Callable[[], TradingEnv]:
    return lambda: TradingEnv(df, costs_bps=costs_bps)
