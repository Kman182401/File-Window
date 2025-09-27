"""Label engineering helpers for CPCV/WFO workflows."""

from __future__ import annotations

from typing import Hashable

import numpy as np
import pandas as pd


def ensure_forward_label(
    df: pd.DataFrame,
    horizon: int = 1,
    *,
    label_col: Hashable = "label",
    price_col: Hashable = "close",
    returns_col: Hashable = "returns",
) -> None:
    """Ensure ``df`` has a binary forward-return label column.

    The label is 1 when the forward return over ``horizon`` bars is positive and 0
    otherwise. Missing lookahead values default to 0 so short windows do not break
    lightweight smoke tests. The function mutates ``df`` in place.
    """

    if df is None or df.empty:
        return

    horizon = max(1, int(horizon))

    if label_col in df.columns and df[label_col].notna().all():
        return

    if price_col in df.columns:
        price = pd.Series(df[price_col], dtype="float64", copy=False)
        forward_price = price.shift(-horizon)
        forward_price = forward_price.fillna(price)
        forward_return = forward_price.div(price).sub(1.0)
    elif returns_col in df.columns:
        returns = pd.Series(df[returns_col], dtype="float64", copy=False)
        forward_return = sum(
            returns.shift(-step).fillna(0.0) for step in range(1, horizon + 1)
        )
    else:
        df[label_col] = df.get(label_col, 0)
        df[label_col] = df[label_col].fillna(0).astype(int)
        return

    labels = (forward_return > 0).astype(int)
    df[label_col] = labels.to_numpy(dtype=int)


__all__ = ["ensure_forward_label"]
