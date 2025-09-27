"""Supervised learning baselines for WFO pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    from sklearn.linear_model import LogisticRegression
    SK_AVAILABLE = True
except Exception:  # pragma: no cover
    SK_AVAILABLE = False


def logistic_positions(is_df, oos_df, target_col: str = "label", **kwargs: Any) -> np.ndarray:
    """Fit logistic regression on IS data and emit OOS positions in [-1, 1]."""
    if not SK_AVAILABLE:
        raise RuntimeError("scikit-learn not available")
    feature_cols = [c for c in is_df.columns if c not in ("timestamp", "returns", "price", target_col)]
    if not feature_cols:
        raise ValueError("No feature columns available for logistic baseline")
    if target_col not in is_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from IS data")
    X = is_df[feature_cols].to_numpy()
    y = is_df[target_col].to_numpy()
    missing = [col for col in feature_cols if col not in oos_df.columns]
    if missing:
        raise ValueError(f"OOS data missing feature columns: {missing}")
    model = LogisticRegression(max_iter=200, **kwargs)
    model.fit(X, y)
    X_oos = oos_df[feature_cols].to_numpy()
    proba = model.predict_proba(X_oos)[:, 1]
    return (proba * 2.0 - 1.0).astype(float)


__all__ = ["logistic_positions"]
