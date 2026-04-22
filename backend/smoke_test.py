"""Smoke test for the watermarking pipeline.

Runs embed + a few representative attacks on the synthetic Lena image
and prints the PSNR/NC numbers alongside the paper's reference values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.watermark import attacks as atk
from app.watermark import metrics
from app.watermark.normalize import to_square_gray
from app.watermark.pipeline import (
    decomposition_preview,
    detect_and_recover,
    embed,
    restore_original,
)


def _report(label, **kv):
    print(f"-- {label}")
    for k, v in kv.items():
        print(f"   {k:32s} {v}")


def main():
    samples = Path(__file__).parent / "app" / "samples"
    img = cv2.imread(str(samples / "test_lena.png"), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "missing test_lena.png"
    img = to_square_gray(img, 512)
    print(f"Test image: {img.shape} min={img.min()} max={img.max()}")

    # 1. Decomposition
    decomp = decomposition_preview(img, gamma=0.3)
    n_blocks = int((decomp == 0).sum())  # rough proxy
    print(f"Decomposition outline generated.")

    # 2. Embed
    watermarked, sidecar = embed(img, key=12345, gamma=0.3)
    n_h = sum(1 for b in sidecar.blocks if b.homogeneous)
    n_n = sum(1 for b in sidecar.blocks if not b.homogeneous)
    psnr_wm = metrics.psnr(img, watermarked)
    print(f"Embed: {len(sidecar.blocks)} leaves (homo={n_h}, nonh={n_n}), "
          f"watermarked PSNR={psnr_wm:.2f} dB (paper: 41.34 dB)")

    # 3. Reversibility (no attack)
    restored = restore_original(watermarked, sidecar)
    nc = metrics.normalized_correlation(img, restored)
    print(f"Reversibility NC={nc:.6f} (paper: 1.0)")

    # 4. Detection/recovery with no tampering
    res_no = detect_and_recover(watermarked, sidecar)
    print(f"No-attack detection: repaired {res_no.n_source_tampered} blocks (expect ~0)")

    # 5. Mean attack
    attacked, mask = atk.mean_attack(watermarked, y0=224, x0=224, size=64, delta=10)
    res = detect_and_recover(attacked, sidecar)
    tpr, tnr = metrics.tpr_tnr(mask, res.detected_mask)
    psnr_rec = metrics.psnr(img, res.recovered_image)
    _report("Mean attack (64x64, delta=10)",
            repaired_blocks=res.n_source_tampered,
            tpr=f"{tpr:.3f}",
            tnr=f"{tnr:.3f}",
            psnr_recovered_vs_original=f"{psnr_rec:.2f} dB  (paper: 69.81)")

    # 6. Collage attack (self-donor)
    donor = cv2.imread(str(samples / "test_lena.png"), cv2.IMREAD_GRAYSCALE)
    donor = to_square_gray(donor, 512)
    donor_wm, _ = embed(donor[:, ::-1].copy(), key=12345, gamma=0.3)  # mirrored donor
    attacked, mask = atk.collage_attack(watermarked, donor_wm,
                                        y0=96, x0=96, h=120, w=120)
    res = detect_and_recover(attacked, sidecar)
    tpr, tnr = metrics.tpr_tnr(mask, res.detected_mask)
    psnr_rec = metrics.psnr(img, res.recovered_image)
    _report("Collage attack (120x120)",
            repaired_blocks=res.n_source_tampered,
            tpr=f"{tpr:.3f}",
            tnr=f"{tnr:.3f}",
            psnr_recovered_vs_original=f"{psnr_rec:.2f} dB  (paper: 63.82)")

    # 7. Gaussian filter
    attacked, mask = atk.gaussian_filter(watermarked, sigma=0.3, ksize=3)
    psnr_att = metrics.psnr(img, attacked)
    _report("Gaussian filter sigma=0.3",
            psnr_vs_original=f"{psnr_att:.2f} dB  (paper: 37.47)")


if __name__ == "__main__":
    main()
