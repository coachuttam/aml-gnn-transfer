"""Choose which training labels a run may use, reproducibly.

A bank with few labels still has all its transactions, so masking only removes
labels from training. Every transaction stays in the graph, and validation and test
labels are never touched (AGENTS.md rule 3).

The kept set depends on the seed, the fraction and the number of training
transactions, and on nothing else. So `scratch` and `finetune` runs at the same
fraction and seed see exactly the same labels (rule 4), and the 10% set is
inside the 25% set, which is inside the 50% set (rule 4 again).

Example:
    kept = keep_label_positions(train_size=1000, fraction=0.25, seed=1)
    len(kept)   # 250
"""

from __future__ import annotations

import hashlib

import numpy as np


def keep_label_positions(train_size: int, fraction: float, seed: int) -> np.ndarray:
    """Sorted positions (0-based, in time order) of the training labels to keep.

    One shuffled order is drawn from the seed; a fraction keeps its first part.
    Smaller fractions are therefore always prefixes of larger ones.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Label fraction must be in (0, 1], got %r" % fraction)
    if train_size <= 0:
        raise ValueError("There are no training transactions to keep labels for")
    # RandomState's stream is guaranteed stable across numpy versions.
    shuffled = np.random.RandomState(seed).permutation(train_size)
    keep_count = max(1, int(round(train_size * fraction)))
    return np.sort(shuffled[:keep_count])


def fingerprint(kept_positions: np.ndarray) -> str:
    """Short hash of a kept set, stored in run records so equal masks can be proven."""
    digest = hashlib.sha256(kept_positions.astype(np.int64).tobytes())
    return digest.hexdigest()[:16]
