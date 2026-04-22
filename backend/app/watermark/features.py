"""Recovery-watermark feature extraction (paper Section 3.1).

For each leaf block emitted by multi-scale decomposition:

* Homogeneous block  ->  8-bit average pixel value.
* Non-homogeneous 4x4 block  ->  44 bits = four 11-bit descriptors,
  one per 2x2 sub-block A_i = {x_i1, x_i2, x_i3, x_i4}:

    f_i[0:6]   high 6 bits of mean of floor(x/4)
    f_i[6:9]   3-bit sub-category code for which two of the four
               pixels are the largest (C(4,2) = 6 classes, encoded 0..5)
    f_i[9:11]  2-bit quantised difference
               floor( ((top2_sum) - (bottom2_sum)) / 32 )
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .multiscale import Block


# ---------------------------------------------------------------------------
# bit helpers
# ---------------------------------------------------------------------------
def _int_to_bits(value: int, width: int) -> np.ndarray:
    value = int(value) & ((1 << width) - 1)
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


# ---------------------------------------------------------------------------
# sub-category encoding
# ---------------------------------------------------------------------------
# The six ways to choose which two of the four pixels are the two largest.
_TOP2_PATTERNS = [
    (0, 1),  # code 0
    (0, 2),  # code 1
    (0, 3),  # code 2
    (1, 2),  # code 3
    (1, 3),  # code 4
    (2, 3),  # code 5
]


def _subcategory_code(pixels4: np.ndarray) -> Tuple[int, Tuple[int, int]]:
    """Return (code, (top_indices)) for a length-4 pixel vector."""
    order = np.argsort(-pixels4, kind="stable")  # descending
    top = tuple(sorted((int(order[0]), int(order[1]))))
    if top in _TOP2_PATTERNS:
        return _TOP2_PATTERNS.index(top), top
    return 0, _TOP2_PATTERNS[0]


# ---------------------------------------------------------------------------
# feature generation
# ---------------------------------------------------------------------------
def homogeneous_feature(block_pixels: np.ndarray) -> np.ndarray:
    """8-bit mean-pixel value (Eq. 13)."""
    avg = int(round(float(block_pixels.mean())))
    avg = max(0, min(255, avg))
    return _int_to_bits(avg, 8)


def nonhomogeneous_feature(block_pixels: np.ndarray) -> np.ndarray:
    """44-bit recovery watermark for a 4x4 non-homogeneous block."""
    assert block_pixels.shape == (4, 4)
    bits = np.zeros(44, dtype=np.uint8)
    for idx, (sy, sx) in enumerate([(0, 0), (0, 2), (2, 0), (2, 2)]):
        sub = block_pixels[sy:sy + 2, sx:sx + 2].astype(np.int32)
        pixels4 = sub.flatten()

        # f[0:6]  high six bits of the mean of floor(x/4)
        mean_hi6 = int(np.floor(pixels4 / 4.0).mean())
        mean_hi6 = max(0, min(63, mean_hi6))

        # f[6:9]  sub-category code
        code, top_idx = _subcategory_code(pixels4)

        # f[9:11] quantised diff (top-2 sum minus bottom-2 sum), /32, two bits
        top_sum = int(np.floor(pixels4[list(top_idx)] / 4.0).sum())
        bot_idx = [i for i in range(4) if i not in top_idx]
        bot_sum = int(np.floor(pixels4[bot_idx] / 4.0).sum())
        diff_q = (top_sum - bot_sum) // 32
        diff_q = max(0, min(3, diff_q))

        off = idx * 11
        bits[off:off + 6] = _int_to_bits(mean_hi6, 6)
        bits[off + 6:off + 9] = _int_to_bits(code, 3)
        bits[off + 9:off + 11] = _int_to_bits(diff_q, 2)
    return bits


def feature_for_block(image: np.ndarray, block: Block) -> np.ndarray:
    pixels = image[block.slice]
    if block.homogeneous:
        return homogeneous_feature(pixels)
    return nonhomogeneous_feature(pixels)


# ---------------------------------------------------------------------------
# pseudo-random encryption (Eq. 16)
# ---------------------------------------------------------------------------
def encrypt_feature(bits: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    mask = rng.integers(0, 2, size=bits.size, dtype=np.uint8)
    return np.bitwise_xor(bits, mask)


def decrypt_feature(bits: np.ndarray, seed: int) -> np.ndarray:
    # XOR is its own inverse
    return encrypt_feature(bits, seed)


# ---------------------------------------------------------------------------
# block-level similarity helpers (tamper comparison)
# ---------------------------------------------------------------------------
def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def split_blocks_by_type(blocks: List[Block]) -> Tuple[List[Block], List[Block]]:
    homo = [b for b in blocks if b.homogeneous]
    nonh = [b for b in blocks if not b.homogeneous]
    return homo, nonh
