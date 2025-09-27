"""Utilities for purging and embargoing samples in time-series CV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence
import numpy as np


@dataclass(frozen=True)
class PurgeConfig:
    label_lookahead: int = 0
    embargo: int = 0


def _contiguous_ranges(indices: Sequence[int]) -> List[range]:
    """Return contiguous ranges covering the provided sorted indices."""
    if not indices:
        return []
    start = prev = indices[0]
    ranges: List[range] = []
    for idx in indices[1:]:
        if idx != prev + 1:
            ranges.append(range(start, prev + 1))
            start = idx
        prev = idx
    ranges.append(range(start, prev + 1))
    return ranges


def apply_purge_embargo(
    train_indices: Iterable[int],
    test_indices: Iterable[int],
    config: PurgeConfig,
) -> np.ndarray:
    """Return train indices after applying purge and embargo rules.

    Purge removes training samples whose label window overlaps the test window.
    Embargo removes the bars immediately after the test window to prevent
    information leakage through rapid rebalancing.
    """

    train_indices = np.asarray(sorted(set(train_indices)), dtype=int)
    test_indices = sorted(set(int(i) for i in test_indices))

    if train_indices.size == 0 or not test_indices:
        return train_indices

    blocked = set()

    if config.label_lookahead > 0:
        look = config.label_lookahead
        for idx in test_indices:
            start = idx - look
            end = idx + look
            blocked.update(k for k in range(start, end + 1) if k >= 0)

    if config.embargo > 0:
        for contiguous in _contiguous_ranges(test_indices):
            embargo_start = contiguous.stop
            embargo_end = embargo_start + config.embargo
            blocked.update(k for k in range(embargo_start, embargo_end) if k >= 0)

    if not blocked:
        return train_indices

    mask = ~np.isin(train_indices, list(blocked))
    return train_indices[mask]


def purge_only(train_indices: Iterable[int], test_indices: Iterable[int], lookahead: int) -> np.ndarray:
    return apply_purge_embargo(train_indices, test_indices, PurgeConfig(label_lookahead=lookahead, embargo=0))


def embargo_only(train_indices: Iterable[int], test_indices: Iterable[int], embargo: int) -> np.ndarray:
    return apply_purge_embargo(train_indices, test_indices, PurgeConfig(label_lookahead=0, embargo=embargo))
