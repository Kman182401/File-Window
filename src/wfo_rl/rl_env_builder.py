"""Environment builders for offline RL training/evaluation."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from enhanced_trading_environment import EnhancedTradingEnvironment, EnhancedTradingConfig


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.reset_index(drop=True).copy()
    if "returns" not in clean.columns:
        if "close" in clean.columns:
            clean["returns"] = clean["close"].pct_change().fillna(0.0)
        else:
            clean["returns"] = 0.0
    if "price" not in clean.columns and "close" in clean.columns:
        clean["price"] = clean["close"]
    return clean


def make_env_from_df(
    df: pd.DataFrame,
    costs_bps: float,
    reward_kwargs: Optional[Dict[str, Any]] = None,
    eval_mode: bool = False,
    config: Optional[EnhancedTradingConfig] = None,
) -> Callable[[], EnhancedTradingEnvironment]:
    """Return a callable that instantiates the enhanced trading environment."""
    reward_kwargs = reward_kwargs.copy() if reward_kwargs else {}
    reward_kwargs.setdefault("lambda_var", 0.0)
    reward_kwargs.setdefault("lambda_dd", 0.0)
    reward_kwargs.setdefault("h_var", 60)
    reward_kwargs.setdefault("hindsight_H", 0)
    reward_kwargs.setdefault("hindsight_weight", 0.0)
    reward_kwargs.setdefault("use_hindsight_in_training", False)

    prepared = _prepare_dataframe(df)

    def _factory() -> EnhancedTradingEnvironment:
        env = EnhancedTradingEnvironment(
            data=prepared.copy(),
            config=config,
            costs_bps=costs_bps,
            lambda_var=reward_kwargs.get("lambda_var", 0.0),
            lambda_dd=reward_kwargs.get("lambda_dd", 0.0),
            h_var=int(reward_kwargs.get("h_var", 60)),
            hindsight_H=int(reward_kwargs.get("hindsight_H", 0)),
            hindsight_weight=float(reward_kwargs.get("hindsight_weight", 0.0)),
            use_hindsight_in_training=bool(reward_kwargs.get("use_hindsight_in_training", False) and not eval_mode),
            eval_mode=eval_mode,
        )
        # Flatten dict observations so SB3 policies and VecNormalize see a single Box space.
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = FlattenObservation(env)
        if eval_mode:
            env.set_eval_mode(True)
        else:
            env.set_eval_mode(False)
        return env

    setattr(_factory, "_df", prepared)
    setattr(_factory, "_reward_kwargs", reward_kwargs)
    setattr(_factory, "_eval_mode", eval_mode)
    return _factory


__all__ = ["make_env_from_df"]
