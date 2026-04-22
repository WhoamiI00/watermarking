"""Systematic (15, 11, 1) Hamming/BCH codec.

The paper calls for BCH(15, 11, 1) -- 11 info bits, 15 code bits, single-
bit correction capability. The classical extended-Hamming (15, 11) code
has identical parameters and is the primitive narrow-sense BCH code of
designed distance 3, so we implement it directly rather than pulling in
an external library (bchlib has native wheels that are awkward on
Windows).

Encoding is systematic: the first 11 bits of the codeword are the
message bits; the remaining 4 bits are parity. Decoding detects and
corrects up to one bit-flip per 15-bit codeword, then strips the parity
to recover the original 11 bits.
"""
from __future__ import annotations

import numpy as np


# Parity-check columns for H, in ascending syndrome order 1..15.
# We choose the canonical Hamming(15,11) layout where parity-bit positions
# inside the codeword are p0 p1 p2 p3 and each P column is the binary
# representation of the syndrome it produces.
# Systematic form: message occupies positions 0..10, parity 11..14.
# Syndrome columns for positions 0..14 are taken so that each column is
# non-zero and all 15 columns are distinct.

_N = 15
_K = 11
_R = _N - _K  # 4


def _build_parity_matrix() -> np.ndarray:
    # 15 distinct non-zero 4-bit syndromes, one per codeword position.
    # Put the 4 standard basis vectors at the parity positions so that
    # the matrix is in systematic form [P | I].
    syndromes = [s for s in range(1, 16)]

    # move the unit vectors (1, 2, 4, 8) to positions 11, 12, 13, 14
    units = [1, 2, 4, 8]
    for i, u in enumerate(units):
        syndromes.remove(u)
        syndromes.append(u)

    H = np.zeros((_R, _N), dtype=np.uint8)
    for col, s in enumerate(syndromes):
        for r in range(_R):
            H[r, col] = (s >> r) & 1
    return H


_H = _build_parity_matrix()
_P = _H[:, :_K]  # parity portion of H, shape (4, 11)


def encode_block(msg: np.ndarray) -> np.ndarray:
    """Systematic (15,11) encode of an 11-bit message."""
    assert msg.shape == (_K,)
    parity = (_P @ msg) & 1  # shape (4,)
    cw = np.concatenate([msg.astype(np.uint8), parity.astype(np.uint8)])
    return cw


def decode_block(cw: np.ndarray) -> np.ndarray:
    """Correct up to one error and return the 11 message bits."""
    assert cw.shape == (_N,)
    cw = cw.copy().astype(np.uint8)
    syndrome = (_H @ cw) & 1
    s_val = int(syndrome[0]) | (int(syndrome[1]) << 1) | (int(syndrome[2]) << 2) | (int(syndrome[3]) << 3)
    if s_val != 0:
        # find which column of H equals this syndrome and flip that bit
        for col in range(_N):
            col_val = 0
            for r in range(_R):
                col_val |= int(_H[r, col]) << r
            if col_val == s_val:
                cw[col] ^= 1
                break
    return cw[:_K]


def encode_stream(bits: np.ndarray) -> np.ndarray:
    """Encode any multiple-of-11-bit stream. Pads with zeros if needed."""
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    pad = (-len(bits)) % _K
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    n_blocks = len(bits) // _K
    out = np.zeros(n_blocks * _N, dtype=np.uint8)
    for i in range(n_blocks):
        out[i * _N:(i + 1) * _N] = encode_block(bits[i * _K:(i + 1) * _K])
    return out


def decode_stream(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).flatten()
    assert len(bits) % _N == 0
    n_blocks = len(bits) // _N
    out = np.zeros(n_blocks * _K, dtype=np.uint8)
    for i in range(n_blocks):
        out[i * _K:(i + 1) * _K] = decode_block(bits[i * _N:(i + 1) * _N])
    return out


# Convenience wrappers --------------------------------------------------
def encode_with_padding(msg: np.ndarray, info_bits: int, out_bits: int) -> np.ndarray:
    """Pad msg to 11 bits, encode to 15 bits, return first `out_bits` bits."""
    assert info_bits <= _K and out_bits <= _N
    padded = np.zeros(_K, dtype=np.uint8)
    padded[:info_bits] = msg[:info_bits]
    cw = encode_block(padded)
    return cw[:out_bits]


def decode_with_padding(cw: np.ndarray, info_bits: int, out_bits: int = None) -> np.ndarray:
    """Inverse of encode_with_padding: decode 15 bits, return first info_bits."""
    if cw.shape[0] < _N:
        full = np.zeros(_N, dtype=np.uint8)
        full[:cw.shape[0]] = cw
        cw = full
    msg = decode_block(cw)
    return msg[:info_bits] if out_bits is None else msg[:out_bits]
