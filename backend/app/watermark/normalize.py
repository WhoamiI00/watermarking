"""Image normalisation helpers.

The paper normalises the image before embedding (Section 2.2) and uses
the inscribed disc as the Zernike-moments domain (Section 3.2). We
provide the minimum support needed:

* ``to_square_gray``: convert an uploaded image to a square 4^n x 4^n
  grayscale canvas (zero-padded) so that multi-scale decomposition
  keeps its power-of-two block structure.
* ``inscribed_disc_mask``: binary mask for the largest disc inside the
  square canvas (Fig 6).
* ``zernike_angle``: rough estimate of a rotation angle from the ratio
  of two low-order complex moments (used for rotation correction).
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np


def _next_pow4(n: int) -> int:
    """Smallest integer of the form 4^k that is >= n (minimum 64)."""
    v = 64
    while v < n:
        v *= 4
    return v


def to_square_gray(img: np.ndarray, target: int = 512) -> np.ndarray:
    """Return a ``target x target`` uint8 grayscale image (resized, not padded)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[0] != target or img.shape[1] != target:
        img = cv2.resize(img, (target, target), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


def inscribed_disc_mask(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy = cx = (size - 1) / 2.0
    r = (size - 1) / 2.0
    return ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r).astype(np.uint8)


# -------- lightweight Zernike-style moment for rotation estimate ------
def _geometric_moment(img: np.ndarray, p: int, q: int) -> float:
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return float(np.sum((xx ** p) * (yy ** q) * img.astype(np.float64)))


def estimate_rotation_angle(img: np.ndarray) -> float:
    """Estimate the rotation angle (degrees) via second-order moments.

    This is a cheap Zernike-moment surrogate: it gives the principal
    orientation of the image's intensity distribution, which we can use
    to undo a rotation attack.
    """
    img = img.astype(np.float64)
    m00 = _geometric_moment(img, 0, 0) + 1e-9
    xc = _geometric_moment(img, 1, 0) / m00
    yc = _geometric_moment(img, 0, 1) / m00
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    mu20 = float(np.sum(((xx - xc) ** 2) * img))
    mu02 = float(np.sum(((yy - yc) ** 2) * img))
    mu11 = float(np.sum(((xx - xc) * (yy - yc)) * img))
    theta = 0.5 * math.atan2(2 * mu11, (mu20 - mu02 + 1e-9))
    return math.degrees(theta)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=0)
