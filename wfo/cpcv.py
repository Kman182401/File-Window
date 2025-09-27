"""Combinatorial Purged Cross-Validation implementation."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Generator, Iterable, List, Sequence, Tuple
import numpy as np

from .purging import apply_purge_embargo, PurgeConfig


@dataclass
class CPCVConfig:
    n_groups: int
    test_group_size: int
    embargo: int = 0
    label_lookahead: int = 0
    max_splits: int | None = None
    random_state: int | None = None


class CombinatorialPurgedCV:
    """Lopez de Prado's CPCV splitter with purge and embargo."""

    def __init__(self, config: CPCVConfig):
        if config.test_group_size <= 0:
            raise ValueError("test_group_size must be > 0")
        if config.n_groups <= config.test_group_size:
            raise ValueError("n_groups must be greater than test_group_size")
        self.config = config

    def split(self, timestamps: Sequence) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        indices = np.arange(len(timestamps))
        groups = _split_into_groups(indices, self.config.n_groups)
        combos = list(itertools.combinations(range(len(groups)), self.config.test_group_size))

        if self.config.random_state is not None:
            rng = np.random.default_rng(self.config.random_state)
            rng.shuffle(combos)

        if self.config.max_splits is not None:
            combos = combos[: self.config.max_splits]

        purge_config = PurgeConfig(
            label_lookahead=self.config.label_lookahead,
            embargo=self.config.embargo,
        )

        for combo in combos:
            test_idx = np.concatenate([groups[i] for i in combo])
            candidate_train = np.concatenate([groups[i] for i in range(len(groups)) if i not in combo])
            train_idx = apply_purge_embargo(candidate_train, test_idx, purge_config)
            yield train_idx, test_idx

    def get_n_splits(self) -> int:
        cfg = self.config
        total = _n_combinations(cfg.n_groups, cfg.test_group_size)
        if cfg.max_splits is not None:
            return min(total, cfg.max_splits)
        return total


def _split_into_groups(indices: np.ndarray, n_groups: int) -> List[np.ndarray]:
    groups = [arr for arr in np.array_split(indices, n_groups) if arr.size > 0]
    if not groups:
        raise ValueError("No data to split")
    return groups


def _n_combinations(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)
