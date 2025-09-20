import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def make_returns(
    df: pd.DataFrame,
    close_col: str = "close",
    horizons: Tuple[int, ...] = (1, 5, 15),
) -> Dict[int, pd.Series]:
    """Compute forward returns for each horizon."""
    y: Dict[int, pd.Series] = {}
    closes = df[close_col].astype(float)
    for h in horizons:
        y[h] = closes.shift(-h) / closes - 1.0
    return y


def make_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    horizons: Tuple[int, ...] = (1, 5, 15),
    n_steps: int = 150,
    step: int = 1,
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """Turn a feature DataFrame into sliding windows and aligned returns."""
    if step <= 0:
        raise ValueError("step must be a positive integer")

    unique_horizons = sorted(set(int(h) for h in horizons))
    if any(h <= 0 for h in unique_horizons):
        raise ValueError("All horizons must be positive integers")
    if len(unique_horizons) != len(horizons):
        logger.warning("Duplicate horizons detected: collapsing to unique values %s", unique_horizons)

    if len(set(feature_cols)) != len(feature_cols):
        logger.warning("Duplicate feature columns detected; duplicates will be merged")

    if "close" in feature_cols:
        logger.warning("'close' included in feature_cols; ensure you are not leaking the target")

    horizons = tuple(unique_horizons)

    df = df.copy()
    df = df[feature_cols + ["close"]].dropna()
    y_df = make_returns(df, "close", horizons)

    max_h = max(horizons)
    df = df.iloc[:-max_h]
    for h in horizons:
        y_df[h] = y_df[h].iloc[:-max_h]

    X_src = df[feature_cols].astype(np.float32).values
    num_rows = len(df)
    if num_rows <= n_steps:
        raise ValueError(f"Not enough rows for n_steps={n_steps} (have {num_rows}).")

    X = np.lib.stride_tricks.sliding_window_view(X_src, n_steps, axis=0)

    y = {
        h: y_df[h].iloc[n_steps - 1 :].to_numpy(dtype=np.float32)
        for h in horizons
    }
    X = X[::step]
    L = X.shape[0]
    for h in horizons:
        y[h] = y[h][::step][:L]
    return X, y


def chrono_split(
    length: int, train: float = 0.7, val: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological train/val/test indices."""
    if not 0.0 < train < 1.0:
        raise ValueError("train fraction must be between 0 and 1")
    if not 0.0 <= val < 1.0:
        raise ValueError("val fraction must be between 0 and 1")
    if not 0.0 < train + val < 1.0:
        raise ValueError("train + val must be between 0 and 1")

    split1 = int(length * train)
    split2 = int(length * (train + val))
    idx = np.arange(length)
    return idx[:split1], idx[split1:split2], idx[split2:]


class Standardizer:
    """Fit on train only; apply everywhere else to avoid leakage."""

    def __init__(self) -> None:
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> None:
        if X_train.ndim != 3:
            raise ValueError("Expected X_train with shape [N, T, F].")
        reshaped = X_train.reshape(-1, X_train.shape[-1])
        self.mu = reshaped.mean(axis=0).astype(np.float32)
        self.sigma = reshaped.std(axis=0).astype(np.float32) + 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Standardizer must be fit before transform().")
        return ((X - self.mu) / self.sigma).astype(np.float32)

    def state_dict(self) -> Dict[str, np.ndarray]:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("Standardizer must be fit before obtaining state.")
        return {"mu": self.mu, "sigma": self.sigma}

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        mu = state.get("mu")
        sigma = state.get("sigma")
        if mu is None or sigma is None:
            raise ValueError("State dict must contain 'mu' and 'sigma'.")
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sigma = np.asarray(sigma, dtype=np.float32)
