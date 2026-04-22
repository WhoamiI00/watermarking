"""Evaluation metrics used in the paper (Eqs. 18 and 19 plus TPR / TNR)."""
from __future__ import annotations

import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def normalized_correlation(orig: np.ndarray, reconstructed: np.ndarray) -> float:
    """Paper Eq. 18 -- the reversibility measure."""
    num = float(np.sum(orig.astype(np.float64) * reconstructed.astype(np.float64)))
    den = float(np.sum(orig.astype(np.float64) ** 2))
    if den <= 1e-12:
        return 0.0
    return num / den


def tpr_tnr(
    truth_mask: np.ndarray,
    detected_mask: np.ndarray,
    block_size: int = 4,
) -> tuple[float, float]:
    """Paper Section 4.3 block-level TPR / TNR.

    Aggregates the two 0/255 pixel masks onto a ``block_size x block_size``
    grid (any pixel flipped in a block counts that block as flipped).

    * ``TPR`` = correctly detected tampering blocks / total tampering blocks
      (this is the familiar sensitivity / recall).
    * ``TNR`` = *inaccurately* detected tampering blocks / total tampering
      blocks. This is the paper's idiosyncratic "false-alarm" metric
      (Fig 13(b)) -- it is a ratio of false positives to the number of
      actually-tampered blocks, so it can exceed 1.0 when the detector
      over-flags relative to the true tampering.
    """
    H, W = truth_mask.shape
    Hb, Wb = H // block_size, W // block_size
    t = (truth_mask[:Hb * block_size, :Wb * block_size] > 0).reshape(
        Hb, block_size, Wb, block_size
    ).any(axis=(1, 3))
    d = (detected_mask[:Hb * block_size, :Wb * block_size] > 0).reshape(
        Hb, block_size, Wb, block_size
    ).any(axis=(1, 3))

    tp = int(np.sum(t & d))
    fn = int(np.sum(t & ~d))
    fp = int(np.sum(~t & d))
    total_tampered = tp + fn
    if total_tampered == 0:
        # no tampering in the image: report standard complements so the
        # result is still meaningful (TPR undefined, return 0; TNR=FP rate)
        total_for_tnr = max(int(np.sum(~t)), 1)
        return 0.0, fp / total_for_tnr
    tpr = tp / total_tampered
    tnr = fp / total_tampered
    return tpr, tnr
