"""Supervised learning baselines for WFO pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    SK_AVAILABLE = True
except Exception:  # pragma: no cover
    SK_AVAILABLE = False


def logistic_positions(
    is_df,
    oos_df,
    target_col: str = "label",
    calibration: str = "sigmoid",
    calibration_cv: int = 3,
    **kwargs: Any,
) -> np.ndarray:
    """Fit logistic regression (with probability calibration) and emit OOS positions.

    Calibration improves probability estimates, which leads to more stable position
    sizing when mapping probabilities to [-1, 1] signals. The defaults use Platt's
    sigmoid, as it is robust with limited folds; switch to ``calibration="isotonic"``
    once the IS sample is large enough (cf. Niculescu-Mizil & Caruana, 2005).
    """

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

    base_model = LogisticRegression(max_iter=200, **kwargs)
    calibrator = CalibratedClassifierCV(base_model, method=calibration, cv=calibration_cv)
    calibrator.fit(X, y)

    X_oos = oos_df[feature_cols].to_numpy()
    proba = calibrator.predict_proba(X_oos)[:, 1]
    return (proba * 2.0 - 1.0).astype(float)


__all__ = ["logistic_positions"]
