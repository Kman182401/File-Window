"""Label engineering helpers for CPCV/WFO workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Optional

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
    """Ensure ``df`` has a binary forward-return label column."""

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


def estimate_volatility(
    price: pd.Series,
    *,
    span: int = 50,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """Return an exponentially-weighted volatility proxy (Lopez de Prado, Ch.3)."""

    pct = price.pct_change()
    vol = pct.ewm(span=span, min_periods=min_periods or span).std()
    vol = vol.fillna(method="bfill").fillna(0.0)
    return vol


@dataclass
class TripleBarrierConfig:
    pt_multiplier: float = 1.0
    sl_multiplier: float = 1.0
    max_holding: int = 50
    volatility_span: int = 50
    price_col: str = "close"
    volatility_col: Optional[str] = None
    side_col: Optional[str] = None
    label_col: str = "label"
    meta_label_col: str = "meta_label"
    ret_col: str = "ret"
    t1_col: str = "t1"
    sample_weight_col: str = "sample_weight"


def triple_barrier_labels(df: pd.DataFrame, config: TripleBarrierConfig) -> pd.DataFrame:
    """Compute triple-barrier + meta-label outputs for each observation.

    Implementation follows Lopez de Prado's triple-barrier method (Advances in Financial
    Machine Learning, p.72) and meta-labeling workflow (ibid., pp.77–80).
    """

    if df is None or df.empty:
        return pd.DataFrame(index=getattr(df, "index", None))

    price = df[config.price_col].astype(float)
    n = len(price)

    if config.volatility_col and config.volatility_col in df.columns:
        vol = df[config.volatility_col].astype(float).to_numpy()
    else:
        vol = estimate_volatility(price, span=config.volatility_span).to_numpy()

    side = None
    if config.side_col and config.side_col in df.columns:
        side = df[config.side_col].fillna(0.0).to_numpy(dtype=float)

    label = np.zeros(n, dtype=float)
    meta = np.zeros(n, dtype=float)
    ret = np.zeros(n, dtype=float)
    t1 = np.full(n, np.nan)
    exit_index = np.full(n, np.nan)

    pt_mult = max(0.0, float(config.pt_multiplier))
    sl_mult = max(0.0, float(config.sl_multiplier))
    max_holding = max(1, int(config.max_holding))

    price_values = price.to_numpy(dtype=float)
    index = price.index.to_numpy()

    for i in range(n):
        sigma = vol[i]
        if not np.isfinite(sigma) or sigma == 0:
            continue

        direction = side[i] if side is not None else 1.0
        if direction == 0:
            continue

        start_price = price_values[i]
        vertical_end = min(n - 1, i + max_holding)
        path_slice = slice(i, vertical_end + 1)
        path_prices = price_values[path_slice]

        path_ret = direction * (path_prices / start_price - 1.0)
        upper = pt_mult * sigma if pt_mult > 0 else np.inf
        lower = -sl_mult * sigma if sl_mult > 0 else -np.inf

        hit_idx = None
        for offset, r_val in enumerate(path_ret):
            if r_val >= upper:
                hit_idx = i + offset
                break
            if r_val <= lower:
                hit_idx = i + offset
                break

        if hit_idx is None:
            hit_idx = vertical_end
        exit_index[i] = hit_idx
        t1[i] = index[hit_idx]

        realised_ret = price_values[hit_idx] / start_price - 1.0
        aligned_ret = direction * realised_ret
        ret[i] = realised_ret

        if side is None:
            label[i] = np.sign(realised_ret)
            meta[i] = 1.0 if realised_ret > 0 else 0.0
        else:
            label[i] = direction
            meta[i] = 1.0 if aligned_ret > 0 else 0.0

    weights = _meta_sample_weights(exit_index, ret)

    out = pd.DataFrame(
        {
            config.t1_col: pd.to_datetime(t1, errors="coerce"),
            config.ret_col: ret,
            config.label_col: label,
            config.meta_label_col: meta,
            config.sample_weight_col: weights,
        },
        index=price.index,
    )
    if config.side_col and config.side_col in df.columns:
        out[config.side_col] = df[config.side_col]
    out["exit_index"] = exit_index
    return out


def _meta_sample_weights(exit_indices: Iterable[float], returns: Iterable[float]) -> np.ndarray:
    exit_arr = np.asarray(exit_indices, dtype=float)
    n = exit_arr.shape[0]
    concurrency = np.zeros(n, dtype=float)

    for i, exit_idx in enumerate(exit_arr):
        if not np.isfinite(exit_idx):
            continue
        start = i
        end = int(exit_idx)
        concurrency[start : end + 1] += 1.0

    concurrency[concurrency == 0] = 1.0

    weights = np.zeros(n, dtype=float)
    for i, exit_idx in enumerate(exit_arr):
        if not np.isfinite(exit_idx):
            continue
        end = int(exit_idx)
        contrib = np.sum(1.0 / concurrency[i : end + 1])
        weights[i] = contrib

    abs_ret = np.abs(np.asarray(returns, dtype=float))
    weights *= abs_ret
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


__all__ = [
    "ensure_forward_label",
    "estimate_volatility",
    "TripleBarrierConfig",
    "triple_barrier_labels",
]
