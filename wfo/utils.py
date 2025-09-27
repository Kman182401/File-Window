"""Utility helpers for reproducibility and diagnostics."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def enable_determinism(seed: int) -> None:
    """Force deterministic behavior across supported libraries.

    Warning: Enabling deterministic algorithms can slow down training, as noted in
    the PyTorch docs and SB3 guidance on reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
    except ImportError:
        pass


__all__ = ["enable_determinism"]
