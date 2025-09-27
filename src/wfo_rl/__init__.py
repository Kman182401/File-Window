"""Convenience exports for RL utilities used by the walk-forward stack."""

from .rl_adapter import RLAdapter, RLSpec
try:  # pragma: no cover - optional gym dependency
    from .rl_env_builder import make_env_from_df
except Exception:  # pragma: no cover
    make_env_from_df = None
from .imitation_learning import pretrain_policy_via_behavior_cloning, generate_teacher_actions
from .supervised_baselines import logistic_positions

__all__ = [
    "RLAdapter",
    "RLSpec",
    "pretrain_policy_via_behavior_cloning",
    "generate_teacher_actions",
    "logistic_positions",
]

if make_env_from_df is not None:
    __all__.append("make_env_from_df")
