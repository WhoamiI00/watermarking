# Self-Recovery Reversible Image Watermarking

Faithful re-implementation of **Zhang, Sun, Gao & Jin, "Self-recovery
reversible image watermarking algorithm", PLoS ONE 13(6) e0199143
(2018)** with a modern web UI for interactive experimentation.

## What it does

1. Normalises the uploaded image to 512 x 512 grayscale.
2. Recursively multi-scale decomposes it into homogeneous / non-homogeneous
   blocks (Section 2.3 of the paper, gamma = 0.3 by default).
3. Generates a **variable-capacity recovery watermark** per leaf block:
   * homogeneous leaves -> 8-bit mean (Eq. 13)
   * non-homogeneous 4x4 leaves -> 44-bit descriptor (Eqs. 14, 15)
4. BCH(15, 11, 1) error-correcting code on each 11-bit chunk.
5. Logistic chaotic map determines where each recovery watermark goes
   (source block -> distant 4x4 target block).
6. Homogeneous payloads are embedded via **Generalised Difference
   Expansion** (GDE, 2 bits per pair, 8 pairs per 4x4 block).
   Non-homogeneous payloads are embedded via **2-LSB substitution**.
7. At recovery time, every leaf's current feature is compared against
   the BCH-corrected value extracted from its mapped target. Mismatched
   regions are localised and repaired.

## Running it

```bash
# dependencies
pip install -r backend/requirements.txt

# start the server
./run.sh           # Linux / macOS / WSL
run.bat            # Windows cmd
```

Then open <http://127.0.0.1:8000>.

## Using the UI

**Step 1 -- Embed**
Drag any natural image into the drop zone, pick a key + gamma, and
click **Generate watermarked image**. You'll see the original image,
the multi-scale decomposition outline, the watermarked image, and the
reversibly restored version. Paper reference: PSNR 41.34 dB, NC 1.0000.

**Step 2 -- Attack and recover**
Pick any of the built-in attacks:

| Attack           | Paper fig / table | What it tests                                              |
|------------------|-------------------|------------------------------------------------------------|
| Mean attack      | Fig 16            | Mean-preserving perturbation of a 64x64 block              |
| Collage attack   | Fig 17            | Paste a region from another watermarked image              |
| Rectangular paint| (custom)          | Solid-colour tampering                                     |
| Gaussian filter  | Table 2           | 3x3 sigma=0.3                                              |
| Median filter    | Table 2           | 3x3                                                        |
| White noise      | Table 2           | variance 0.01                                              |
| Salt & pepper    | Table 2           | density 0.02                                               |
| Shear 1/32       | Table 2           | Strip the bottom 1/32 of the image                         |
| JPEG Q=50        | Table 2           | Round-trip JPEG compression                                |
| Rescale 70%      | Table 2           | Shrink then upscale                                        |
| Rotation 30 deg  | Table 2           | Rotate then undo-rotate                                    |

You'll see the attacked image, the detected tamper mask, the recovered
image, and a side-by-side table vs the paper's reference PSNR values.

## File layout

```
backend/
  app/
    main.py                 FastAPI endpoints
    service.py              Glue between the API and the watermarking lib
    watermark/
      multiscale.py         Multi-scale decomposition (Section 2.3)
      features.py           Recovery-watermark features (Section 3.1)
      bch.py                Systematic (15, 11) Hamming/BCH codec
      chaos.py              Logistic chaotic permutation (Section 3.2)
      embedding.py          GDE + 2-LSB primitives
      pipeline.py           Embed / extract / detect / recover
      attacks.py            Every attack from Table 2 and Figs 16, 17
      metrics.py            PSNR / NC / TPR / TNR
      normalize.py          Grayscale + square canvas + Zernike-ish helpers
frontend/
  index.html, style.css, app.js
run.bat / run.sh
content/
  Self-recovery reversible image watermarking algorithm.pdf   (the paper)
```

## Notes on fidelity

* **BCH(15, 11, 1)** is implemented as the extended Hamming code with
  the same parameters (single-error correcting, 15 code bits for 11
  information bits) which is mathematically equivalent to the primitive
  narrow-sense BCH of designed distance 3.
* **Zernike moments** are approximated with second-order geometric
  moments. For full rotation correction you'd use true Zernike radial
  polynomials -- this stub is enough to show the principle.
* **Generalised Difference Expansion** is implemented with 2 bits per
  non-overlapping horizontal pair; pairs that would overflow fall back
  to 2-LSB substitution (tracked in a per-block 8-bit overflow map).

## Credits

Original algorithm (c) 2018 Z. Zhang, H. Sun, S. Gao, S. Jin --
<https://doi.org/10.1371/journal.pone.0199143>

This re-implementation is provided for educational and reproducibility
purposes.
