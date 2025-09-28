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
    *,
    side_col: str = "side",
    meta_label_col: str = "meta_label",
    use_meta: bool = False,
    activation_threshold: float = 0.5,
    sample_weight_col: str | None = None,
    probability_floor: float = 0.0,
    feature_blacklist: tuple[str, ...] = ("timestamp", "returns", "price"),
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

    blacklist = set(feature_blacklist + (target_col,))
    feature_cols = [c for c in is_df.columns if c not in blacklist]
    if not feature_cols:
        raise ValueError("No feature columns available for logistic baseline")
    if target_col not in is_df.columns:
        raise ValueError(f"Target column '{target_col}' missing from IS data")

    train_df = is_df.dropna(subset=[target_col])
    if train_df.empty:
        return np.zeros(len(oos_df), dtype=float)

    X = train_df[feature_cols].to_numpy()
    y = train_df[target_col].to_numpy()

    classes = np.unique(y)
    if classes.size < 2:
        return np.zeros(len(oos_df), dtype=float)

    missing = [col for col in feature_cols if col not in oos_df.columns]
    if missing:
        raise ValueError(f"OOS data missing feature columns: {missing}")

    sample_weight = None
    if sample_weight_col and sample_weight_col in train_df.columns:
        sample_weight = train_df[sample_weight_col].to_numpy(dtype=float)
        if np.all(sample_weight == 0):
            sample_weight = None

    base_model = LogisticRegression(max_iter=200, **kwargs)
    X_oos = oos_df[feature_cols].to_numpy()

    if sample_weight is None and calibration_cv > 1:
        calibrator = CalibratedClassifierCV(base_model, method=calibration, cv=calibration_cv)
        try:
            calibrator.fit(X, y)
            proba = calibrator.predict_proba(X_oos)[:, 1]
        except ValueError:
            base_model.fit(X, y)
            proba = base_model.predict_proba(X_oos)[:, 1]
    else:
        base_model.fit(X, y, sample_weight=sample_weight)
        proba = base_model.predict_proba(X_oos)[:, 1]

    proba = np.clip(proba, probability_floor, 1.0)

    if use_meta:
        if side_col not in oos_df.columns:
            raise ValueError(f"OOS data missing side column '{side_col}' for meta-label sizing")
        if meta_label_col not in train_df.columns:
            raise ValueError(f"IS data missing meta label column '{meta_label_col}'")
        mask = proba >= activation_threshold
        size = np.where(mask, proba, 0.0)
        return (oos_df[side_col].to_numpy(dtype=float) * size).astype(float)

    return (proba * 2.0 - 1.0).astype(float)


__all__ = ["logistic_positions"]
