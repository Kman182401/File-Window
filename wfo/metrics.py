"""Performance metrics used in the WFO tooling."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


def _to_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def sharpe_ratio(returns, risk_free: float = 0.0, period_scaler: float | None = None) -> float:
    """Compute the Sharpe ratio.

    Args:
        returns: Sequence of per-period returns.
        risk_free: Per-period risk-free rate (already aligned with returns frequency).
        period_scaler: Optional factor (e.g. sqrt(252)) to annualise. If None, no scaling.
    """
    r = _to_array(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    sr = np.mean(excess) / std
    if period_scaler is not None:
        sr *= period_scaler
    return float(sr)


def sortino_ratio(returns, risk_free: float = 0.0, period_scaler: float | None = None) -> float:
    r = _to_array(returns)
    if r.size < 2:
        return 0.0
    excess = r - risk_free
    downside = excess[excess < 0]
    if downside.size == 0:
        return np.inf
    denom = np.sqrt(np.mean(downside ** 2))
    if denom == 0:
        return np.inf
    sr = np.mean(excess) / denom
    if period_scaler is not None:
        sr *= period_scaler
    return float(sr)


def max_drawdown(equity_curve) -> float:
    eq = _to_array(equity_curve)
    if eq.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(eq)
    running_max[running_max == 0] = np.finfo(float).eps
    drawdowns = (eq - running_max) / running_max
    return float(np.min(drawdowns))


def calmar_ratio(returns, equity_curve) -> float:
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return 0.0
    compounded = np.prod(1 + _to_array(returns)) - 1
    return float(compounded / abs(mdd))


def turnover(signals) -> float:
    s = np.asarray(signals, dtype=float)
    if s.size < 2:
        return 0.0
    return float(np.nansum(np.abs(np.diff(s))))


def expectancy(trade_pnls) -> float:
    pnls = _to_array(trade_pnls)
    if pnls.size == 0:
        return 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = wins.size / pnls.size if pnls.size else 0
    avg_win = wins.mean() if wins.size else 0
    avg_loss = losses.mean() if losses.size else 0
    return float(win_rate * avg_win + (1 - win_rate) * avg_loss)


@dataclass
class DSRResult:
    z_score: float
    p_value: float


def deflated_sharpe_ratio(
    returns,
    risk_free: float = 0.0,
    period_scaler: float | None = None,
    trials_effective: float = 1.0,
) -> DSRResult:
    """Bailey & López de Prado's Deflated Sharpe Ratio."""
    r = _to_array(returns)
    if r.size < 2:
        return DSRResult(z_score=0.0, p_value=1.0)

    sr = sharpe_ratio(r, risk_free=risk_free, period_scaler=period_scaler)
    if sr == 0.0:
        return DSRResult(z_score=0.0, p_value=1.0)

    skew = stats.skew(r, bias=False)
    kurt = stats.kurtosis(r, fisher=False, bias=False)
    t = r.size
    if t <= 2:
        return DSRResult(z_score=0.0, p_value=1.0)

    numerator = sr * np.sqrt(t - 1)
    denominator = np.sqrt(1 - skew * sr + ((kurt - 1) / 4.0) * (sr ** 2))
    if denominator == 0:
        return DSRResult(z_score=0.0, p_value=1.0)

    if trials_effective <= 0:
        trials_effective = 1.0

    adjustment = (np.log(trials_effective)) / np.sqrt(t - 1)
    z = (numerator - adjustment) / denominator
    p = 1 - stats.norm.cdf(z)
    return DSRResult(z_score=float(z), p_value=float(p))


__all__ = [
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "turnover",
    "expectancy",
    "max_drawdown",
    "deflated_sharpe_ratio",
    "DSRResult",
]
