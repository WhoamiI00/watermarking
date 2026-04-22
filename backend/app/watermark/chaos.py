"""Logistic chaotic map used for block-index scrambling (paper Section 3.2).

    x_{n+1} = mu * x_n * (1 - x_n),   mu in (3.57, 4)

We emit a permutation of length N by generating N values from the map,
then sorting their indices. The initial state comes from the user key
(seed).
"""
from __future__ import annotations

import numpy as np


def logistic_permutation(n: int, seed: int, mu: float = 3.99) -> np.ndarray:
    """Return a length-n permutation derived from a logistic chaotic sequence."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    # derive a stable floating-point seed in (0,1) that is not a fixed point
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    x = 0.1 + 0.8 * float(rng.random())

    # burn-in to escape transient
    for _ in range(200):
        x = mu * x * (1.0 - x)

    seq = np.empty(n, dtype=np.float64)
    for i in range(n):
        x = mu * x * (1.0 - x)
        seq[i] = x

    perm = np.argsort(seq, kind="stable")
    return perm.astype(np.int64)


def derangement_with_min_distance(n: int, seed: int, min_distance: int = 8) -> np.ndarray:
    """Permutation where |perm[i] - i| >= min_distance for every i.

    The paper insists that a block never maps to a neighbouring block so
    that tampering localised to one region can still be recovered from a
    distant block.
    """
    if n <= 1:
        return np.arange(n)
    md = min(min_distance, n // 2)
    base = logistic_permutation(n, seed)
    perm = base.copy()

    # simple repair pass: for any i where the mapping is too close, swap
    # with a partner whose mapping is far enough away
    for i in range(n):
        if abs(int(perm[i]) - i) < md:
            for j in range(n):
                if (
                    abs(int(perm[j]) - j) >= md
                    and abs(int(perm[j]) - i) >= md
                    and abs(int(perm[i]) - j) >= md
                ):
                    perm[i], perm[j] = perm[j], perm[i]
                    break
            else:
                # fall back: rotate
                perm[i] = (i + md) % n
    return perm


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size, dtype=perm.dtype)
    return inv
