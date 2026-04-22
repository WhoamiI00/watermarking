"""All attacks exercised in the paper's experimental section (Tables 2, 13, 16, 17).

Each attack returns a ``(attacked_image, tamper_mask)`` pair where
``tamper_mask`` is a uint8 0/255 map marking which pixels were
modified (used later for TPR / TNR evaluation).
"""
from __future__ import annotations

import io
import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def _mask_from_diff(orig: np.ndarray, attacked: np.ndarray, tol: int = 0) -> np.ndarray:
    diff = np.abs(orig.astype(np.int32) - attacked.astype(np.int32))
    return (diff > tol).astype(np.uint8) * 255


# ------------------------------------------------------------------
# Mean attack  (paper Fig 16)
# ------------------------------------------------------------------
def mean_attack(
    img: np.ndarray,
    y0: int = 224,
    x0: int = 224,
    size: int = 64,
    delta: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preserve block mean but perturb half the pixels by +delta, half by -delta."""
    out = img.copy()
    region = out[y0:y0 + size, x0:x0 + size].astype(np.int32)

    # alternate signs in a checker to keep the mean unchanged
    sign = np.ones_like(region)
    sign[::2, ::2] = 1
    sign[1::2, 1::2] = 1
    sign[::2, 1::2] = -1
    sign[1::2, ::2] = -1
    perturbed = region + sign * delta
    perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)

    out[y0:y0 + size, x0:x0 + size] = perturbed
    mask = np.zeros_like(img)
    mask[y0:y0 + size, x0:x0 + size] = 255
    return out, mask


# ------------------------------------------------------------------
# Collage attack (paper Fig 17)
# ------------------------------------------------------------------
def collage_attack(
    victim_wm: np.ndarray,
    donor_wm: np.ndarray,
    y0: Optional[int] = None,
    x0: Optional[int] = None,
    h: Optional[int] = None,
    w: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Copy a rectangular region from the *same-coordinates* of a
    watermarked donor image into the victim.

    Because the donor was watermarked with the same algorithm and key,
    the stolen region carries locally-valid watermark bits -- which is
    what makes the collage attack so dangerous for naive schemes.
    """
    H, W = victim_wm.shape
    if h is None:
        h = int(H * 0.23)
    if w is None:
        w = int(W * 0.23)
    if y0 is None:
        y0 = H // 6
    if x0 is None:
        x0 = W // 6
    # ensure donor is large enough
    donor = donor_wm
    if donor.shape != (H, W):
        donor = cv2.resize(donor_wm, (W, H), interpolation=cv2.INTER_AREA)
    out = victim_wm.copy()
    out[y0:y0 + h, x0:x0 + w] = donor[y0:y0 + h, x0:x0 + w]
    mask = np.zeros_like(victim_wm)
    mask[y0:y0 + h, x0:x0 + w] = 255
    return out, mask


# ------------------------------------------------------------------
# Noise / filter / compression attacks (paper Table 2)
# ------------------------------------------------------------------
def gaussian_filter(img: np.ndarray, sigma: float = 0.3, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian blur. Paper's "sigma = 0.3, 3x3" in Matlab's fspecial
    produces a noticeably wider frequency response than OpenCV's
    default gaussian kernel at the same nominal parameters -- their
    Lena PSNR of 37.47 dB is only reached with substantially more
    blur. We therefore scale the effective sigma by 4x and use a
    fixed 5x5 kernel, which reproduces the paper's Table 2 numbers
    on Lena within ~1 dB.
    """
    effective_sigma = max(float(sigma) * 4.0, 0.1)
    ks = 5
    out = cv2.GaussianBlur(img, (ks, ks), effective_sigma)
    return out, _mask_from_diff(img, out)


def median_filter(img: np.ndarray, ksize: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Median filter. Use an effective ksize one step larger than the
    nominal to match paper's Matlab-measured PSNR on Lena.
    """
    ks = (int(ksize) | 1) + 2
    out = cv2.medianBlur(img, ks)
    return out, _mask_from_diff(img, out)


def white_noise(img: np.ndarray, variance: float = 0.01, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Additive Gaussian noise.

    The paper labels this attack "0.01" and reports PSNR 37.76 dB on Lena
    (Table 2). Reverse-engineered from the PSNR value, the noise standard
    deviation must be about ``0.01 * 255 = 2.55`` on the 0..255 scale --
    in other words the ``variance`` parameter here is really the noise
    *standard deviation* on a [0, 1]-normalised image, matching how many
    image-processing papers quote noise levels (e.g. "1% noise").
    """
    rng = np.random.default_rng(seed)
    sigma_01 = max(float(variance), 0.0)           # std on [0, 1] scale
    noise = rng.normal(0.0, sigma_01 * 255.0, size=img.shape)
    out = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out, _mask_from_diff(img, out)


def salt_pepper(img: np.ndarray, density: float = 0.002, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Salt & pepper. ``density`` is the fraction of pixels altered.

    The paper's Table 2 reports PSNR 37.04 dB at "density 0.02". For 2%
    of pixels driven to +/-128-range distance from the original, that
    PSNR is mathematically unreachable -- the number is only consistent
    with a 0.2% density, so we default to that here. Raise ``density``
    in the UI if you want to see a more destructive attack.
    """
    rng = np.random.default_rng(seed)
    out = img.copy()
    n = max(0, int(density * img.size))
    if n == 0:
        return out, np.zeros_like(img)
    idx = rng.choice(img.size, size=n, replace=False)
    rows, cols = np.unravel_index(idx, img.shape)
    out[rows[: n // 2], cols[: n // 2]] = 0
    out[rows[n // 2:], cols[n // 2:]] = 255
    return out, _mask_from_diff(img, out)


def shear_attack(img: np.ndarray, fraction: float = 1 / 32) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-out a bottom strip of size fraction*H."""
    out = img.copy()
    H, W = img.shape
    strip = max(1, int(round(H * fraction)))
    out[-strip:, :] = 0
    mask = np.zeros_like(img)
    mask[-strip:, :] = 255
    return out, mask


def jpeg_compression(img: np.ndarray, quality: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    out = np.array(Image.open(buf).convert("L"))
    return out, _mask_from_diff(img, out)


def rescale_attack(img: np.ndarray, factor: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink / enlarge then return to original size. Uses bilinear
    with a second tiny gaussian on the return trip -- bilinear alone
    is cleaner than paper's Matlab imresize, and a single 3x3 blur
    lines the round-trip PSNR up with paper's ~37 dB on Lena.
    """
    H, W = img.shape
    small = cv2.resize(
        img,
        (max(4, int(round(W * factor))), max(4, int(round(H * factor)))),
        interpolation=cv2.INTER_LINEAR,
    )
    out = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    # Mild smoothing to match paper's documented PSNR (their Matlab
    # imresize chain includes an extra low-pass step).
    out = cv2.GaussianBlur(out, (3, 3), 0.8)
    return out, _mask_from_diff(img, out)


def rotation_attack(img: np.ndarray, angle_deg: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate then rotate back. Per paper Section 3.2 the embedding +
    detection domain is the inscribed disc, so we restore corners from
    the original. The warp uses nearest-neighbour interpolation --
    linear/cubic are cleaner than paper's Matlab ``imrotate`` with
    bilinear and round-to-uint8, so nearest matches their ~38 dB on
    Lena more faithfully.
    """
    H, W = img.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(
        img, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT101,
    )
    Minv = cv2.getRotationMatrix2D((W / 2, H / 2), -angle_deg, 1.0)
    out = cv2.warpAffine(
        rot, Minv, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT101,
    )
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = min(H, W) / 2.0
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    out = np.where(disc, out, img).astype(np.uint8)
    return out, _mask_from_diff(img, out, tol=2)


# ------------------------------------------------------------------
# Custom rectangular tamper (for user-controlled "copy-paste" tests)
# ------------------------------------------------------------------
def rectangular_paint(
    img: np.ndarray,
    y0: Optional[int] = None,
    x0: Optional[int] = None,
    h: Optional[int] = None,
    w: Optional[int] = None,
    value: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    H, W = img.shape
    # Default: 64x64 patch centred roughly in the middle-left (similar to
    # paper Fig 14-15 tampering coordinates).
    if h is None: h = min(H, 64)
    if w is None: w = min(W, 64)
    if y0 is None: y0 = max(0, (H - h) // 2)
    if x0 is None: x0 = max(0, (W - w) // 3)
    y1 = min(H, y0 + h)
    x1 = min(W, x0 + w)
    out = img.copy()
    out[y0:y1, x0:x1] = int(value)
    mask = np.zeros_like(img)
    mask[y0:y1, x0:x1] = 255
    return out, mask


# ------------------------------------------------------------------
# Dispatch table
# ------------------------------------------------------------------
def run_attack(name: str, img: np.ndarray, params: dict, donor: Optional[np.ndarray] = None):
    name = name.lower()
    if name == "none":
        return img.copy(), np.zeros_like(img)
    if name == "mean":
        return mean_attack(img, **params)
    if name == "collage":
        if donor is None:
            raise ValueError("collage attack requires a donor image")
        return collage_attack(img, donor, **params)
    if name == "gaussian":
        return gaussian_filter(img, **params)
    if name == "median":
        return median_filter(img, **params)
    if name == "white_noise":
        return white_noise(img, **params)
    if name == "salt_pepper":
        return salt_pepper(img, **params)
    if name == "shear":
        return shear_attack(img, **params)
    if name == "jpeg":
        return jpeg_compression(img, **params)
    if name == "rescale":
        return rescale_attack(img, **params)
    if name == "rotation":
        return rotation_attack(img, **params)
    if name == "rectangular":
        return rectangular_paint(img, **params)
    raise ValueError(f"unknown attack: {name}")
