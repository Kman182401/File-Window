"""Simple behaviour-cloning utilities for warm-starting RL policies."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    series = pd.Series(signal)
    return series.rolling(window, min_periods=1).mean().to_numpy()


def generate_teacher_actions(df: pd.DataFrame, teacher_conf: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Generate signed direction/size actions using a lightweight teacher policy."""
    if df.empty:
        return np.zeros((0, 2), dtype=np.float32)
    teacher_conf = teacher_conf or {}
    if "close" in df.columns:
        close = df["close"]
    elif "price" in df.columns:
        close = df["price"]
    else:
        close = pd.Series(np.zeros(len(df)), dtype=float)
    close_arr = close.to_numpy(dtype=float)

    ma_fast = int(teacher_conf.get("ma_fast", 20))
    ma_slow = int(teacher_conf.get("ma_slow", 50))
    twap_weight = float(teacher_conf.get("twap_weight", 0.2))
    participation = float(teacher_conf.get("participation", 0.1))

    fast_ma = _moving_average(close_arr, ma_fast)
    slow_ma = _moving_average(close_arr, ma_slow)
    ma_signal = np.sign(fast_ma - slow_ma)

    progress = np.linspace(-1.0, 1.0, num=len(df))
    twap_signal = participation * progress

    combined = (1 - twap_weight) * ma_signal + twap_weight * twap_signal
    combined = np.clip(combined, -1.0, 1.0)

    direction = np.sign(combined)
    size = np.minimum(np.abs(combined), 1.0)
    actions = np.stack([direction, size], axis=1).astype(np.float32)
    return actions


def _flatten_observation(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                parts.append(value.astype(np.float32).ravel())
            else:
                parts.append(np.array([value], dtype=np.float32))
        return np.concatenate(parts, axis=0)
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32).ravel()
    return np.array([float(obs)], dtype=np.float32)


def pretrain_policy_via_behavior_cloning(
    df_is: Optional[pd.DataFrame],
    teacher_conf: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Return an observation/action dataset produced by the teacher policy."""
    if df_is is None or df_is.empty:
        return None
    teacher_conf = teacher_conf.copy() if teacher_conf else {}
    make_env_fn = teacher_conf.get("make_env_fn")

    actions = generate_teacher_actions(df_is, teacher_conf)
    if actions.size == 0:
        return None

    observations = []
    if callable(make_env_fn):
        env = make_env_fn()
        obs, _ = env.reset()
        for idx, action in enumerate(actions):
            observations.append(_flatten_observation(obs))
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
                term = bool(np.asarray(terminated).astype(bool).any())
                trunc = bool(np.asarray(truncated).astype(bool).any())
                done = term or trunc
            else:  # pragma: no cover - compatibility guard
                obs, _, done, _ = step_out
            if done:
                break
        if hasattr(env, "close"):
            env.close()
    else:
        feature_cols = [c for c in df_is.columns if c not in ("timestamp", "returns")]
        if not feature_cols:
            observations = [np.zeros(2, dtype=np.float32) for _ in range(len(actions))]
        else:
            features = df_is[feature_cols].to_numpy(dtype=np.float32)
            observations = [feat.ravel() for feat in features[: len(actions)]]

    if not observations:
        return None

    obs_arr = np.asarray(observations, dtype=np.float32)
    act_arr = actions[: obs_arr.shape[0]]
    if obs_arr.shape[0] == 0 or act_arr.shape[0] == 0:
        return None
    return {
        "observations": obs_arr,
        "actions": act_arr.astype(np.float32),
    }


__all__ = [
    "generate_teacher_actions",
    "pretrain_policy_via_behavior_cloning",
]
