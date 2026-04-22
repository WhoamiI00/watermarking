"""High-level orchestration service used by the FastAPI layer."""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .watermark import attacks as atk
from .watermark import metrics
from .watermark.normalize import to_square_gray
from .watermark.pipeline import (
    Sidecar,
    decomposition_preview,
    detect_and_recover,
    embed,
    restore_original,
)


def _read_grayscale(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("could not decode image")
    return img


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def encode_pipeline(image_bytes: bytes, key: int = 12345, gamma: float = 0.3, target_size: int = 512):
    """Run normalisation + embedding. Returns serialisable result dict."""
    raw = _read_grayscale(image_bytes)
    norm = to_square_gray(raw, target=target_size)
    decomp_preview = decomposition_preview(norm, gamma=gamma)
    watermarked, sidecar = embed(norm, key=key, gamma=gamma)

    psnr_wm = metrics.psnr(norm, watermarked)
    restored = restore_original(watermarked, sidecar)
    nc = metrics.normalized_correlation(norm, restored)

    n_homo = sum(1 for b in sidecar.blocks if b.homogeneous)
    n_nonh = sum(1 for b in sidecar.blocks if not b.homogeneous)
    n_blocks = len(sidecar.blocks)
    capacity_bits = n_homo * 8 + n_nonh * 44

    return {
        "original": _encode_png(norm),
        "decomposition": _encode_png(decomp_preview),
        "watermarked": _encode_png(watermarked),
        "restored_no_attack": _encode_png(restored),
        "psnr_watermarked": psnr_wm,
        "nc_reversibility": nc,
        "blocks_total": n_blocks,
        "blocks_homogeneous": n_homo,
        "blocks_nonhomogeneous": n_nonh,
        "recovery_bits_total": int(capacity_bits),
        "sidecar": sidecar.to_json(),
    }


def attack_and_recover(
    watermarked_bytes: bytes,
    sidecar_json: dict,
    attack_name: str,
    attack_params: dict,
    donor_bytes: Optional[bytes] = None,
    original_bytes: Optional[bytes] = None,
):
    sidecar = Sidecar.from_json(sidecar_json)
    wm = _read_grayscale(watermarked_bytes)
    if wm.shape != (sidecar.size, sidecar.size):
        wm = cv2.resize(wm, (sidecar.size, sidecar.size), interpolation=cv2.INTER_AREA)

    donor = None
    if donor_bytes:
        donor = _read_grayscale(donor_bytes)

    attacked, truth_mask = atk.run_attack(attack_name, wm, attack_params, donor=donor)

    result = detect_and_recover(attacked, sidecar)
    psnr_attacked = metrics.psnr(wm, attacked)
    psnr_recovered = metrics.psnr(wm, result.recovered_image)
    tpr, tnr = metrics.tpr_tnr(truth_mask, result.detected_mask)

    payload = {
        "watermarked": _encode_png(wm),
        "attacked": _encode_png(attacked),
        "truth_mask": _encode_png(truth_mask),
        "detected_mask": _encode_png(result.detected_mask),
        "recovered": _encode_png(result.recovered_image),
        "psnr_attacked_vs_watermarked": psnr_attacked,
        "psnr_recovered_vs_watermarked": psnr_recovered,
        "tpr": float(tpr),
        "tnr": float(tnr),
        "n_blocks_repaired": result.n_source_tampered,
        "global_attack_detected": bool(result.global_attack_detected),
    }

    if original_bytes:
        original = _read_grayscale(original_bytes)
        if original.shape != (sidecar.size, sidecar.size):
            original = to_square_gray(original, target=sidecar.size)
        payload["psnr_recovered_vs_original"] = metrics.psnr(original, result.recovered_image)

    return payload
