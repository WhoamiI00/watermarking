"""Modified multi-scale decomposition from the paper (Section 2.3).

An image is recursively split into equal quadrants until every block
satisfies the homogeneity criterion:

    |p_i - p_avg|  <=  (g_l - 1) * gamma      for every pixel in the block

with minimum block size 4x4. Leaf blocks are labelled homogeneous or
non-homogeneous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Block:
    """Leaf block produced by multi-scale decomposition."""
    y: int           # top-left row
    x: int           # top-left col
    size: int        # square side length (power of two, >= 4)
    homogeneous: bool

    @property
    def slice(self):
        return (slice(self.y, self.y + self.size), slice(self.x, self.x + self.size))


def _is_homogeneous(block: np.ndarray, gamma: float, gray_levels: int = 256) -> bool:
    avg = float(block.mean())
    threshold = (gray_levels - 1) * gamma
    return bool(np.max(np.abs(block.astype(np.float64) - avg)) <= threshold)


def decompose(image: np.ndarray, gamma: float = 0.3, min_size: int = 4) -> List[Block]:
    """Return the ordered list of leaf blocks (top-to-bottom, left-to-right)."""
    assert image.ndim == 2, "expected 2-D grayscale image"
    h, w = image.shape
    assert h == w, "image must be square after normalization"

    blocks: List[Block] = []

    def _recurse(y: int, x: int, size: int):
        sub = image[y:y + size, x:x + size]
        if size <= min_size:
            blocks.append(Block(y, x, size, _is_homogeneous(sub, gamma)))
            return
        if _is_homogeneous(sub, gamma):
            blocks.append(Block(y, x, size, True))
            return
        half = size // 2
        _recurse(y, x, half)
        _recurse(y, x + half, half)
        _recurse(y + half, x, half)
        _recurse(y + half, x + half, half)

    _recurse(0, 0, h)

    # sort top-to-bottom, left-to-right (already natural from recursion, but be explicit)
    blocks.sort(key=lambda b: (b.y, b.x))
    return blocks


def block_map(image_shape, blocks: List[Block]) -> np.ndarray:
    """Visualise the leaf-block grid as an outline image (uint8)."""
    h, w = image_shape
    canvas = np.full((h, w), 255, dtype=np.uint8)
    for b in blocks:
        y, x, s = b.y, b.x, b.size
        canvas[y:y + s, x:x + 1] = 0
        canvas[y:y + s, x + s - 1:x + s] = 0
        canvas[y:y + 1, x:x + s] = 0
        canvas[y + s - 1:y + s, x:x + s] = 0
    return canvas
