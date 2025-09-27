"""Backwards compatibility shim re-exporting the new RL adapter."""

from wfo_rl.rl_adapter import RLAdapter, RLSpec  # noqa: F401

__all__ = ["RLAdapter", "RLSpec"]
