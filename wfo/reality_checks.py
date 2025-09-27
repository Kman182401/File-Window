"""Reality check tests (White's RC and Hansen's SPA)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Dict, Optional


@dataclass
class RealityCheckResult:
    p_value: float
    survivors: Dict[int, float]


def _circular_block_indices(n: int, block_len: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate indices using circular block bootstrap."""
    if block_len <= 0:
        block_len = 1
    total_needed = size
    starts = rng.integers(0, n, size=max(1, int(np.ceil(total_needed / block_len))))
    indices = []
    for start in starts:
        block = [(start + i) % n for i in range(block_len)]
        indices.extend(block)
        if len(indices) >= total_needed:
            break
    return np.array(indices[:total_needed], dtype=int)


def _diff_matrix(oos_returns_matrix: np.ndarray, benchmark_returns: Optional[np.ndarray]) -> np.ndarray:
    if benchmark_returns is None:
        benchmark = np.zeros(oos_returns_matrix.shape[0])
    else:
        benchmark = np.asarray(benchmark_returns, dtype=float)
        if benchmark.ndim != 1:
            raise ValueError("benchmark_returns must be a 1D array")
    if benchmark.shape[0] != oos_returns_matrix.shape[0]:
        raise ValueError("Benchmark length mismatch")
    return oos_returns_matrix - benchmark[:, None]


def white_reality_check(
    oos_returns_matrix: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    bootstrap: str = "circular",
    block_len: int = 78,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> RealityCheckResult:
    """White's Reality Check with circular block bootstrap."""
    rng = np.random.default_rng(random_state)
    diff = _diff_matrix(np.asarray(oos_returns_matrix, dtype=float), benchmark_returns)
    n, k = diff.shape
    centered = diff - diff.mean(axis=0, keepdims=True)
    obs = np.sqrt(n) * diff.mean(axis=0)
    obs_stat = float(np.max(obs))

    exceed = 0
    survivors = {}
    for _ in range(n_bootstrap):
        if bootstrap != "circular":
            raise ValueError("Only circular bootstrap supported")
        idx = _circular_block_indices(n, block_len, n, rng)
        sample = centered[idx]
        stat = np.sqrt(n) * sample.mean(axis=0)
        if np.max(stat) >= obs_stat:
            exceed += 1
    p_value = (exceed + 1) / (n_bootstrap + 1)

    # Survivors: configs whose individual statistic exceeds bootstrap median
    bootstrap_stats = []
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        idx = _circular_block_indices(n, block_len, n, rng)
        sample = centered[idx]
        bootstrap_stats.append(np.sqrt(n) * sample.mean(axis=0))
    bootstrap_stats = np.stack(bootstrap_stats, axis=0)
    median_stat = np.median(bootstrap_stats, axis=0)
    for i in range(k):
        if obs[i] > median_stat[i]:
            survivors[i] = float(obs[i])

    return RealityCheckResult(p_value=p_value, survivors=survivors)


def hansen_spa(
    oos_returns_matrix: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    bootstrap: str = "circular",
    block_len: int = 78,
    n_bootstrap: int = 500,
    random_state: Optional[int] = None,
) -> RealityCheckResult:
    """Hansen's SPA test following the stationary bootstrap paradigm."""
    rng = np.random.default_rng(random_state)
    diff = _diff_matrix(np.asarray(oos_returns_matrix, dtype=float), benchmark_returns)
    n, k = diff.shape
    mean = diff.mean(axis=0)
    std = diff.std(axis=0, ddof=1)
    std[std == 0] = 1e-8
    t_stat = np.sqrt(n) * mean / std

    centered = diff - mean
    boot_stats = []
    for _ in range(n_bootstrap):
        if bootstrap != "circular":
            raise ValueError("Only circular bootstrap supported")
        idx = _circular_block_indices(n, block_len, n, rng)
        sample = centered[idx]
        boot_mean = sample.mean(axis=0)
        boot_std = sample.std(axis=0, ddof=1)
        boot_std[boot_std == 0] = 1e-8
        boot_stats.append(np.sqrt(n) * boot_mean / boot_std)
    boot_stats = np.stack(boot_stats, axis=0)

    p_values = {}
    for i in range(k):
        greater = np.mean(boot_stats[:, i] >= t_stat[i])
        p_values[i] = float(greater)

    overall_p = float(np.min(list(p_values.values()))) if p_values else 1.0
    survivors = {i: t_stat[i] for i, pv in p_values.items() if pv <= overall_p}
    return RealityCheckResult(p_value=overall_p, survivors=survivors)


__all__ = ["white_reality_check", "hansen_spa", "RealityCheckResult"]
