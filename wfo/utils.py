"""Utility helpers for reproducibility and diagnostics."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


SYMBOL_MARKET_MAP = {
    "ES": "US_FUTURES",
    "NQ": "US_FUTURES",
    "GC": "US_FUTURES",
    "6E": "US_FUTURES",
    "6B": "US_FUTURES",
    "6A": "US_FUTURES",
}

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


def resolve_session_minutes(symbol: str, default: Optional[int] = None) -> Optional[int]:
    """Infer session length in minutes for a symbol using market hours.

    Falls back to "default" if the symbol is unknown or if market data is unavailable.
    """

    market_name = SYMBOL_MARKET_MAP.get(symbol.upper())
    if not market_name:
        return default
    try:  # pragma: no cover - optional dependency
        from market_hours_detector import MarketHoursDetector
    except ImportError:  # pragma: no cover
        return default

    detector = MarketHoursDetector()
    market = detector.markets.get(market_name)
    if market is None:
        return default

    open_minutes = market.open_time.hour * 60 + market.open_time.minute
    close_minutes = market.close_time.hour * 60 + market.close_time.minute
    if open_minutes <= close_minutes:
        minutes = close_minutes - open_minutes
    else:
        minutes = (24 * 60 - open_minutes) + close_minutes

    if minutes <= 0:
        minutes += 24 * 60

    if market_name == "US_FUTURES":
        maintenance_start = detector.cme_maintenance_start
        maintenance_end = detector.cme_maintenance_end
        maintenance = (maintenance_end.hour * 60 + maintenance_end.minute) - (maintenance_start.hour * 60 + maintenance_start.minute)
        if maintenance > 0:
            minutes = max(0, minutes - maintenance)

    return minutes if minutes > 0 else default



__all__ = ["enable_determinism", "resolve_session_minutes"]
