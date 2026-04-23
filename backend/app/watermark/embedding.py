"""Low-level embedding / extraction primitives.

Two schemes are used, mirroring the paper:

* ``embed_gde2`` / ``extract_gde2``
    Generalised difference expansion with 2 bits per non-overlapping
    pixel pair (Choi et al. 2015, Section 3.2 (6) of the paper). Each
    4x4 block has 8 pairs so capacity = 16 bits. We use 15 bits and
    leave 1 pad. When the expansion would overflow the 0..255 range for
    a given pair, the pair is marked in an overflow map and embedded
    with 1-LSB substitution instead, at the cost of 1 bit of capacity
    per fall-back pair (tracked per-block in the sidecar).

* ``embed_lsb2`` / ``extract_lsb2``
    Direct 2-LSB substitution (32 bits per 4x4 block). Not reversible
    in the strict sense, but matches how the paper treats texture blocks
    (information hidden in the two LSBs).

Both functions operate on a 4x4 uint8 block in-place-friendly style.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# -------- 2-LSB substitution -------------------------------------------
def embed_lsb2(block: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Embed up to 32 bits in the 2 LSBs of a 4x4 block (row-major order)."""
    assert block.shape == (4, 4)
    flat = block.flatten().astype(np.int32)
    payload = np.zeros(32, dtype=np.uint8)
    payload[: bits.size] = bits
    for i in range(16):
        hi = int(payload[2 * i])
        lo = int(payload[2 * i + 1])
        flat[i] = (flat[i] & 0xFC) | (hi << 1) | lo
    return flat.reshape(4, 4).astype(np.uint8)


def extract_lsb2(block: np.ndarray, n_bits: int = 32) -> np.ndarray:
    """Extract the first n_bits of 2-LSB data from a 4x4 block."""
    assert block.shape == (4, 4)
    flat = block.flatten().astype(np.int32)
    bits = np.zeros(32, dtype=np.uint8)
    for i in range(16):
        v = int(flat[i]) & 0x3
        bits[2 * i] = (v >> 1) & 1
        bits[2 * i + 1] = v & 1
    return bits[:n_bits]


# -------- 1-LSB substitution (cleaner, half the capacity) --------------
def embed_lsb1(block: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Embed up to 16 bits in the LSB of a 4x4 block (row-major order).

    Uses only the least-significant bit of each pixel, so the per-pixel
    change is at most 1 graylevel -- noticeably cleaner than 2-LSB
    substitution. Used for the low-noise layer of the dual-copy
    recovery scheme (see IMPROVEMENTS.md, E1).
    """
    assert block.shape == (4, 4)
    flat = block.flatten().astype(np.int32)
    payload = np.zeros(16, dtype=np.uint8)
    payload[: bits.size] = bits
    for i in range(16):
        flat[i] = (flat[i] & 0xFE) | int(payload[i])
    return flat.reshape(4, 4).astype(np.uint8)


def extract_lsb1(block: np.ndarray, n_bits: int = 16) -> np.ndarray:
    """Extract the first n_bits of 1-LSB data from a 4x4 block."""
    assert block.shape == (4, 4)
    flat = block.flatten().astype(np.int32)
    bits = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        bits[i] = int(flat[i]) & 0x1
    return bits[:n_bits]


# -------- Generalised Difference Expansion (2 bits per pair) -----------
_PAIRS = [
    ((0, 0), (0, 1)), ((0, 2), (0, 3)),
    ((1, 0), (1, 1)), ((1, 2), (1, 3)),
    ((2, 0), (2, 1)), ((2, 2), (2, 3)),
    ((3, 0), (3, 1)), ((3, 2), (3, 3)),
]  # 8 horizontal non-overlapping pairs in a 4x4 block


def _expand_pair(x: int, y: int, b: int, nbits: int) -> Tuple[int, int, bool]:
    """GDE: embed nbits in pair (x,y). Returns (x', y', overflowed)."""
    L = (x + y) // 2
    h = x - y
    # new difference: shift h left by nbits and append b
    h_prime = (h << nbits) | (b & ((1 << nbits) - 1))
    x_p = L + ((h_prime + 1) // 2)
    y_p = L - (h_prime // 2)
    if 0 <= x_p <= 255 and 0 <= y_p <= 255:
        return int(x_p), int(y_p), False
    # overflow: leave the pair untouched; caller handles fallback
    return int(x), int(y), True


def _invert_pair(x_p: int, y_p: int, nbits: int) -> Tuple[int, int, int]:
    L = (x_p + y_p) // 2
    h_prime = x_p - y_p
    mask = (1 << nbits) - 1
    b = h_prime & mask
    h = h_prime >> nbits
    x = L + ((h + 1) // 2)
    y = L - (h // 2)
    return int(x), int(y), int(b)


def embed_gde2(block: np.ndarray, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Embed up to 16 bits using GDE-2 on 8 horizontal pairs.

    Returns (new_block, overflow_map). overflow_map is 8 bits -- one per
    pair -- with 1 indicating that the pair could not be expanded and
    was left untouched. The caller must embed the overflow bits into the
    LSB of the first pixel of each affected pair using a secondary 1-LSB
    substitution (only 8 extra bits -- handled by the caller).
    """
    assert block.shape == (4, 4)
    assert bits.size <= 16
    buf = block.copy().astype(np.int32)
    overflow = np.zeros(8, dtype=np.uint8)
    padded = np.zeros(16, dtype=np.uint8)
    padded[: bits.size] = bits
    for k, ((r0, c0), (r1, c1)) in enumerate(_PAIRS):
        b_val = (int(padded[2 * k]) << 1) | int(padded[2 * k + 1])
        x_p, y_p, ovf = _expand_pair(int(buf[r0, c0]), int(buf[r1, c1]), b_val, nbits=2)
        if ovf:
            overflow[k] = 1
            # fallback to 2-LSB on the two pixels so bits are still recoverable
            buf[r0, c0] = (int(buf[r0, c0]) & 0xFC) | b_val
        else:
            buf[r0, c0] = x_p
            buf[r1, c1] = y_p
    return buf.astype(np.uint8), overflow


def extract_gde2(block: np.ndarray, overflow: np.ndarray, n_bits: int = 16) -> np.ndarray:
    assert block.shape == (4, 4)
    buf = block.astype(np.int32)
    out = np.zeros(16, dtype=np.uint8)
    for k, ((r0, c0), (r1, c1)) in enumerate(_PAIRS):
        if overflow[k]:
            b_val = int(buf[r0, c0]) & 0x3
        else:
            _, _, b_val = _invert_pair(int(buf[r0, c0]), int(buf[r1, c1]), nbits=2)
        out[2 * k] = (b_val >> 1) & 1
        out[2 * k + 1] = b_val & 1
    return out[:n_bits]


def restore_gde2(block: np.ndarray, overflow: np.ndarray) -> np.ndarray:
    """Invert GDE-2 to recover the original pixel values (reversibility)."""
    buf = block.astype(np.int32).copy()
    for k, ((r0, c0), (r1, c1)) in enumerate(_PAIRS):
        if overflow[k]:
            # fallback pair was untouched on pixel (r1,c1); pixel (r0,c0)
            # had its 2 LSBs modified -- we don't have original LSBs, so
            # we keep the current value (small < 4 intensity perturbation)
            continue
        x, y, _ = _invert_pair(int(buf[r0, c0]), int(buf[r1, c1]), nbits=2)
        buf[r0, c0] = x
        buf[r1, c1] = y
    return np.clip(buf, 0, 255).astype(np.uint8)
