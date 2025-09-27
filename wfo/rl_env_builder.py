"""Backwards compatibility shim for RL environment builders."""

from wfo_rl.rl_env_builder import make_env_from_df  # noqa: F401

__all__ = ["make_env_from_df"]
