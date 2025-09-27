"""Supervised baselines for WFO evaluation."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LogisticRegression
    SK_AVAILABLE = True
except Exception:  # pragma: no cover
    SK_AVAILABLE = False


def logistic_positions(df, target_col: str = "label", **kwargs: Any) -> np.ndarray:
    """Fit logistic regression and emit signed positions in [-1, 1]."""
    if not SK_AVAILABLE:
        raise RuntimeError("scikit-learn not available")
    feature_cols = [c for c in df.columns if c not in ("timestamp", "returns", "price", target_col)]
    if not feature_cols:
        raise ValueError("No feature columns available for logistic baseline")
    X = df[feature_cols].to_numpy()
    y = df[target_col].to_numpy()
    model = LogisticRegression(max_iter=200, **kwargs)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    return (proba * 2.0 - 1.0).astype(float)
