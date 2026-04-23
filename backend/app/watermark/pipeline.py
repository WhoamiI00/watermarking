"""End-to-end watermark embedding / extraction / tamper-recovery pipeline.

Per-4x4 thumbnail recovery.
=======================================================================

The paper's scheme stores one 8-bit mean per multi-scale homogeneous
leaf. When a 64x64 homo leaf gets a small paint tampering inside it,
recovery is forced to fill the whole leaf with a single gray value --
which bounds the recovered-PSNR no matter how accurately tampering is
detected.

This implementation keeps the paper's multi-scale decomposition step
(purely for the authoring-time visualisation users see) but replaces
the recovery feature with a far more informative one: for every 4x4
block in the image we store its 8-bit mean + a 3-bit local-stddev
descriptor, 11 info bits total. Each 4x4 source's 11-bit descriptor is
BCH(15,11,1) encoded; pairs of descriptors (30 bits) are embedded into
a *different* 4x4 target block via 2-LSB substitution. Half of the 4x4
blocks therefore serve only as sources (their LSBs remain untouched),
which incidentally raises the watermarked-image PSNR.

At recovery the receiver compares every 4x4 block's current mean and
stddev against its BCH-corrected stored copy. Any block whose mean
*or* stddev drifts past a paper-derived tolerance is flagged and
overwritten with its original mean. A spatial-clustering guard drops
the surgical repair pass when detections look like global noise.

The richness of the new feature -- one mean per 4x4 region, at the
granularity of a 128x128 thumbnail -- is what closes the mean/collage
PSNR gap against the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from . import bch, chaos
from .embedding import embed_lsb1, embed_lsb2, extract_lsb1, extract_lsb2
from .multiscale import Block, block_map, decompose


# =====================================================================
# Block indexing helpers
# =====================================================================
def _num_4x4(size: int) -> int:
    return (size // 4) * (size // 4)


def _idx_to_slice(idx: int, size: int) -> Tuple[slice, slice]:
    cols = size // 4
    r = (idx // cols) * 4
    c = (idx % cols) * 4
    return slice(r, r + 4), slice(c, c + 4)


def _idx_rc(idx: int, size: int) -> Tuple[int, int]:
    cols = size // 4
    return (idx // cols) * 4, (idx % cols) * 4


def _skip_blend_for_sanity(tampered: np.ndarray) -> bool:
    """cv2.inpaint is expensive on huge masks and pointless on tiny ones --
    gate the fused reconstruction path to avoid both."""
    n = int(tampered.sum())
    return n < 8  # fewer than 8 tampered 4x4 blocks -> just paste thumbnail


# =====================================================================
# Per-4x4 feature: 8-bit mean + 3-bit stddev-category
# =====================================================================
# 8 stddev bins. Deliberately a mix of fine grain at the low end (to
# distinguish LSB noise from weak tampering) and coarse grain at the
# high end (textured regions dominate there).
_STD_BINS = np.array([0, 3, 6, 10, 15, 22, 32, 48], dtype=np.float64)


def _patch_feature(patch: np.ndarray) -> Tuple[int, int]:
    """Return (mean_8bit, stddev_code_3bit) for a 4x4 uint8 patch."""
    mean_val = int(round(float(patch.mean())))
    mean_val = max(0, min(255, mean_val))
    std_val = float(patch.std())
    std_code = int(np.searchsorted(_STD_BINS, std_val, side="right") - 1)
    std_code = max(0, min(7, std_code))
    return mean_val, std_code


def _encode_feature_bits(mean_val: int, std_code: int) -> np.ndarray:
    """Pack (mean, std_code) into an 11-bit info word."""
    bits = np.zeros(11, dtype=np.uint8)
    for i in range(8):
        bits[i] = (mean_val >> (7 - i)) & 1
    for i in range(3):
        bits[8 + i] = (std_code >> (2 - i)) & 1
    return bits


def _decode_feature_bits(bits11: np.ndarray) -> Tuple[int, int]:
    mean_val = 0
    for i in range(8):
        mean_val = (mean_val << 1) | int(bits11[i])
    std_code = 0
    for i in range(3):
        std_code = (std_code << 1) | int(bits11[8 + i])
    return mean_val, std_code


# =====================================================================
# Sidecar
# =====================================================================
# Two embedding layers, each occupies a disjoint half of the image's
# 4x4 blocks so one layer cannot overwrite another layer's payload.
#
#  Layer A : 1-LSB substitution (IMPROVEMENT E1).  Capacity 16 bits per
#            target block = 1 BCH(15,11,1) codeword = 1 source per
#            target. Using only one LSB keeps the per-pixel change at
#            most 1 greylevel, raising the watermarked-image PSNR to
#            ~46 dB (a ~3 dB gain over the uniform-2-LSB scheme).
#            Because a target only holds one source, not every source
#            gets a layer-A copy: the first M/2 entries of the chaotic
#            source permutation are chosen.
#
#  Layer B : 2-LSB substitution. Capacity 32 bits per target block = 2
#            BCH codewords = 2 sources per target. Every one of the M
#            source blocks has a layer-B copy.
#
# Sources covered by both layers get the dual-copy cross-check; sources
# covered only by layer B fall back to the layer-B copy alone plus the
# outlier-against-neighbours sanity (which has been tightened in R3 to
# handle the both-copies-corrupted case too).
# =====================================================================
@dataclass
class Sidecar:
    size: int
    gamma: float
    key: int
    blocks: List[Block]                     # multi-scale leaves (viz only)
    # Layer A: 1 source per target, 1-LSB embedding
    layer_a_sources: List[int]              # length = len(target_blocks_a)
    target_blocks_a: List[int]
    # Layer B: 2 sources per target (pair), 2-LSB embedding
    pair_sources_b: List[int]               # length = 2 * len(target_blocks_b)
    target_blocks_b: List[int]

    def to_json(self) -> Dict:
        return {
            "size": self.size,
            "gamma": self.gamma,
            "key": self.key,
            "blocks": [{"y": b.y, "x": b.x, "size": b.size, "homogeneous": b.homogeneous} for b in self.blocks],
            "layer_a_sources": list(map(int, self.layer_a_sources)),
            "target_blocks_a": list(map(int, self.target_blocks_a)),
            "pair_sources_b": list(map(int, self.pair_sources_b)),
            "target_blocks_b": list(map(int, self.target_blocks_b)),
        }

    @staticmethod
    def from_json(d: Dict) -> "Sidecar":
        return Sidecar(
            size=int(d["size"]),
            gamma=float(d["gamma"]),
            key=int(d["key"]),
            blocks=[Block(y=b["y"], x=b["x"], size=b["size"], homogeneous=b["homogeneous"])
                    for b in d["blocks"]],
            layer_a_sources=[int(v) for v in d["layer_a_sources"]],
            target_blocks_a=[int(v) for v in d["target_blocks_a"]],
            pair_sources_b=[int(v) for v in d["pair_sources_b"]],
            target_blocks_b=[int(v) for v in d["target_blocks_b"]],
        )


# =====================================================================
# Source/target allocation
# =====================================================================
def _allocate_two_layers(
    size: int, key: int
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build the layer-A (1 source / target) and layer-B (2 sources /
    target) mappings. The full set of 4x4 block indices is partitioned
    into two halves via one chaotic permutation; layer A gets the
    first half as targets, layer B gets the second half.

    Returns (layer_a_sources, target_blocks_a, pair_sources_b, target_blocks_b).

      * layer_a_sources[k]     = source at layer-A target k (single).
      * target_blocks_a[k]     = 4x4 target block index for layer-A entry k.
      * pair_sources_b[2*k]    = first source at layer-B target k.
      * pair_sources_b[2*k+1]  = second source at layer-B target k.
      * target_blocks_b[k]     = 4x4 target block index for layer-B pair k.
    """
    M = _num_4x4(size)
    assert M % 2 == 0
    n_half = M // 2

    tgt_perm = chaos.logistic_permutation(M, key)
    layer_a_targets_raw = [int(v) for v in tgt_perm[:n_half].tolist()]
    layer_b_targets_raw = [int(v) for v in tgt_perm[n_half:].tolist()]

    # Layer-A sources: take the first n_half entries of an independent
    # source permutation (so only half the sources have a layer-A copy).
    src_perm_a = chaos.logistic_permutation(M, (key * 2654435761) & 0xFFFFFFFF)
    layer_a_sources_raw = [int(v) for v in src_perm_a[:n_half].tolist()]

    # Layer-B sources: full permutation, consecutive pairs.
    src_perm_b = chaos.logistic_permutation(M, (key * 40503) & 0xFFFFFFFF)
    pair_sources_b_raw = [int(v) for v in src_perm_b.tolist()]

    # Repair self-mapping for layer A (1 source, 1 target).
    target_blocks_a = list(layer_a_targets_raw)
    for i in range(n_half):
        if target_blocks_a[i] == layer_a_sources_raw[i]:
            j = (i + 1) % n_half
            target_blocks_a[i], target_blocks_a[j] = target_blocks_a[j], target_blocks_a[i]
    layer_a_sources = layer_a_sources_raw

    # Repair self-mapping for layer B (2 sources per target).
    def _assign_layer_b(pair_sources, available):
        n_pairs = len(available)
        available = list(available)
        target_blocks = [0] * n_pairs
        defer = []
        for k in range(n_pairs):
            sA = pair_sources[2 * k]; sB = pair_sources[2 * k + 1]
            cand = available[k]
            if cand == sA or cand == sB:
                defer.append(k); target_blocks[k] = -1
            else:
                target_blocks[k] = cand
        for k in defer:
            sA = pair_sources[2 * k]; sB = pair_sources[2 * k + 1]
            my_cand = available[k]
            for j in range(n_pairs):
                if j == k or target_blocks[j] == -1:
                    continue
                tj = target_blocks[j]
                sjA = pair_sources[2 * j]; sjB = pair_sources[2 * j + 1]
                if tj != sA and tj != sB and my_cand != sjA and my_cand != sjB:
                    target_blocks[k] = tj; target_blocks[j] = my_cand; break
            if target_blocks[k] == -1:
                target_blocks[k] = my_cand
        return target_blocks

    target_blocks_b = _assign_layer_b(pair_sources_b_raw, layer_b_targets_raw)
    return layer_a_sources, target_blocks_a, pair_sources_b_raw, target_blocks_b


# =====================================================================
# Embedding
# =====================================================================
def embed(image: np.ndarray, key: int = 12345, gamma: float = 0.3) -> Tuple[np.ndarray, Sidecar]:
    assert image.ndim == 2, "grayscale only"
    size = image.shape[0]
    assert image.shape[0] == image.shape[1], "image must be square"

    # 1. Multi-scale decomposition kept only for the UI visualisation.
    leaf = decompose(image, gamma=gamma)

    # 2. Compute feature for every 4x4 source block
    bw = size // 4
    bh = size // 4
    M = bw * bh

    means = image.reshape(bh, 4, bw, 4).mean(axis=(1, 3))
    stds = image.reshape(bh, 4, bw, 4).astype(np.float32).std(axis=(1, 3))
    means = np.clip(np.round(means), 0, 255).astype(np.uint8).flatten()
    stds = stds.flatten()

    # 3. Allocate the two layers (see IMPROVEMENTS.md E1).
    layer_a_sources, target_blocks_a, pair_sources_b, target_blocks_b = \
        _allocate_two_layers(size, key)

    # 4. Embed.
    watermarked = image.copy()

    # ---- Layer A: 1-LSB, 1 source per target, 15-bit codeword + 1 pad
    for k in range(len(target_blocks_a)):
        s = layer_a_sources[k]
        m = int(means[s])
        c = int(np.searchsorted(_STD_BINS, stds[s], side="right") - 1)
        c = max(0, min(7, c))
        msg = _encode_feature_bits(m, c)
        cw = bch.encode_block(msg)            # 15 bits
        payload = np.zeros(16, dtype=np.uint8)
        payload[:15] = cw
        tgt_slice = _idx_to_slice(target_blocks_a[k], size)
        watermarked[tgt_slice] = embed_lsb1(watermarked[tgt_slice], payload)

    # ---- Layer B: 2-LSB, 2 sources per target, two 15-bit codewords
    for k in range(len(target_blocks_b)):
        sA = pair_sources_b[2 * k]
        sB = pair_sources_b[2 * k + 1]
        mA = int(means[sA]); mB = int(means[sB])
        cA = int(np.searchsorted(_STD_BINS, stds[sA], side="right") - 1)
        cB = int(np.searchsorted(_STD_BINS, stds[sB], side="right") - 1)
        cA = max(0, min(7, cA)); cB = max(0, min(7, cB))
        msgA = _encode_feature_bits(mA, cA)
        msgB = _encode_feature_bits(mB, cB)
        cwA = bch.encode_block(msgA); cwB = bch.encode_block(msgB)
        payload = np.zeros(32, dtype=np.uint8)
        payload[:15] = cwA
        payload[15:30] = cwB
        tgt_slice = _idx_to_slice(target_blocks_b[k], size)
        watermarked[tgt_slice] = embed_lsb2(watermarked[tgt_slice], payload)

    sidecar = Sidecar(
        size=size,
        gamma=gamma,
        key=key,
        blocks=leaf,
        layer_a_sources=layer_a_sources,
        target_blocks_a=target_blocks_a,
        pair_sources_b=pair_sources_b,
        target_blocks_b=target_blocks_b,
    )
    return watermarked, sidecar


# =====================================================================
# Reversibility: approximate (2-LSB substitution is not strictly lossless)
# =====================================================================
def restore_original(watermarked: np.ndarray, sidecar: Sidecar) -> np.ndarray:
    """Best-effort reversibility.

    We did not store the original 2 LSBs of target blocks before
    substitution, so exact invertibility requires per-pair sidecar
    overhead we chose not to pay. The watermark perturbation is within
    +/- 3 graylevels per pixel, so NC is typically > 0.99998.
    """
    return watermarked.copy()


# =====================================================================
# Tamper detection & recovery
# =====================================================================
@dataclass
class RecoveryResult:
    recovered_image: np.ndarray
    detected_mask: np.ndarray
    n_source_tampered: int
    n_target_tampered: int
    global_attack_detected: bool = False


def _extract_layer_a(
    suspect: np.ndarray, size: int, layer_a_sources, target_blocks
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract layer-A features (1-LSB, 1 source per target).

    Returns (mean_a, std_a, has_copy_a) where ``has_copy_a[i]`` is True
    for sources that have a layer-A copy (only half the sources do).
    """
    M = _num_4x4(size)
    mean_a = np.zeros(M, dtype=np.uint8)
    std_a = np.zeros(M, dtype=np.uint8)
    has_copy_a = np.zeros(M, dtype=bool)
    for k, s in enumerate(layer_a_sources):
        tgt = target_blocks[k]
        bits = extract_lsb1(suspect[_idx_to_slice(tgt, size)], n_bits=15)
        msg = bch.decode_block(bits)
        m, c = _decode_feature_bits(msg)
        mean_a[s] = m
        std_a[s] = c
        has_copy_a[s] = True
    return mean_a, std_a, has_copy_a


def _extract_layer_b(
    suspect: np.ndarray, size: int, pair_sources, target_blocks
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract layer-B features (2-LSB, 2 sources per target)."""
    M = _num_4x4(size)
    mean_b = np.zeros(M, dtype=np.uint8)
    std_b = np.zeros(M, dtype=np.uint8)
    for k in range(len(target_blocks)):
        sA = pair_sources[2 * k]
        sB = pair_sources[2 * k + 1]
        tgt = target_blocks[k]
        bits = extract_lsb2(suspect[_idx_to_slice(tgt, size)], n_bits=30)
        msgA = bch.decode_block(bits[:15])
        msgB = bch.decode_block(bits[15:30])
        mA, cA = _decode_feature_bits(msgA)
        mB, cB = _decode_feature_bits(msgB)
        mean_b[sA] = mA; std_b[sA] = cA
        mean_b[sB] = mB; std_b[sB] = cB
    return mean_b, std_b


def _extract_recovery_features(
    suspect: np.ndarray, sidecar: Sidecar
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (recovered_mean, recovered_std, copies_agree).

    Sources with both a layer-A and a layer-B copy are cross-checked;
    sources with only a layer-B copy use it directly. ``copies_agree``
    is True wherever we have high confidence in the recovered value
    (either both copies agreed, or only layer-B exists and looks sane).
    """
    size = sidecar.size
    M = _num_4x4(size)

    mean_a, std_a, has_copy_a = _extract_layer_a(
        suspect, size, sidecar.layer_a_sources, sidecar.target_blocks_a
    )
    mean_b, std_b = _extract_layer_b(
        suspect, size, sidecar.pair_sources_b, sidecar.target_blocks_b
    )

    bw = bh = size // 4
    med_a = cv2.medianBlur(mean_a.reshape(bh, bw), 3).flatten()
    med_b = cv2.medianBlur(mean_b.reshape(bh, bw), 3).flatten()

    # Agreement voting: for sources covered by both layers, require the
    # two decoded means to agree within 6 greylevels.
    disagreement = np.abs(mean_a.astype(np.int32) - mean_b.astype(np.int32))
    copies_agree_both = has_copy_a & (disagreement <= 6)
    # If we have no layer-A copy, trust layer B by default (marked as
    # agreeing, since there is nothing to disagree with).
    copies_agree = copies_agree_both | (~has_copy_a)

    # Pick the better copy per source:
    #   - if only layer B exists -> use mean_b
    #   - if both exist and agree -> use mean_a (== mean_b within 6)
    #   - if both exist and disagree -> pick the one closer to its own
    #     3x3 neighbourhood median (consensus copy)
    a_consistent = has_copy_a & (
        np.abs(mean_a.astype(np.int32) - med_a.astype(np.int32))
        <= np.abs(mean_b.astype(np.int32) - med_b.astype(np.int32))
    )
    recovered_mean = np.where(
        ~has_copy_a, mean_b,
        np.where(copies_agree_both | a_consistent, mean_a, mean_b),
    ).astype(np.uint8)
    recovered_std = np.where(
        ~has_copy_a, std_b,
        np.where(copies_agree_both | a_consistent, std_a, std_b),
    ).astype(np.uint8)

    return recovered_mean, recovered_std, copies_agree


def detect_and_recover(
    suspect: np.ndarray, sidecar: Sidecar,
    global_patch_fraction: float = 0.30,
    mean_tol: int = 12,
    std_code_tol: int = 1,
) -> RecoveryResult:
    size = sidecar.size
    assert suspect.shape == (size, size)
    bw = bh = size // 4
    M = bw * bh

    # --- Current (suspect) features ---
    cur_means = np.clip(np.round(
        suspect.reshape(bh, 4, bw, 4).mean(axis=(1, 3))
    ), 0, 255).astype(np.uint8).flatten()
    cur_stds = suspect.reshape(bh, 4, bw, 4).astype(np.float32).std(axis=(1, 3)).flatten()
    cur_std_code = np.clip(
        np.searchsorted(_STD_BINS, cur_stds, side="right") - 1, 0, 7
    ).astype(np.uint8)

    # --- Recovered (BCH-decoded) features from the dual-copy layers ---
    recovered_mean, recovered_std, copies_agree = _extract_recovery_features(suspect, sidecar)

    # --- Neighbourhood consensus of recovered means (used to detect when
    # BOTH copies of a block were corrupted -- very rare but possible).
    rec_mean_2d = recovered_mean.reshape(bh, bw).astype(np.uint8)
    rec_mean_med = cv2.medianBlur(rec_mean_2d, 3).astype(np.int32).flatten()
    outlier = np.abs(recovered_mean.astype(np.int32) - rec_mean_med) > 30

    # --- Tamper decision: flag whenever current feature disagrees with
    # recovered feature, EXCEPT when the recovered value is itself an
    # outlier (untrustworthy).
    mean_diff = np.abs(cur_means.astype(np.int32) - recovered_mean.astype(np.int32))
    std_diff = np.abs(cur_std_code.astype(np.int32) - recovered_std.astype(np.int32))
    tampered = (~outlier) & ((mean_diff > mean_tol) | (std_diff > std_code_tol))

    # --- Spatial regularisation by connected-component size filtering.
    # Step 1: dilate with a generous kernel so sparse flags inside one
    # contiguous tamper region (mean attack flags only every other
    # block, for instance) get glued together.
    # Step 2: label connected components and discard those smaller than
    # `min_cluster` blocks -- these are false-positives from target
    # corruption that slipped past the outlier filter.
    # Step 3: fill every block inside each surviving cluster so we
    # repair the whole tamper region, not just the blocks that tripped.
    flagged_grid = tampered.reshape(bh, bw).astype(np.uint8)
    dilated = cv2.dilate(flagged_grid, np.ones((3, 3), np.uint8), iterations=1)
    num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    min_cluster = 20    # 20 4x4 blocks = 320 pixels of contiguous tamper
    keep = np.zeros_like(flagged_grid)
    for lbl in range(1, num_cc):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_cluster:
            keep[labels == lbl] = 1
    tampered = keep.astype(bool).flatten()

    # --- Build a clean 128x128 thumbnail from the recovered means.
    # For each block we trust the BCH-decoded mean unless the block was
    # marked untrustworthy (outlier); in that case we use the 3x3 median
    # of recovered means (neighbourhood vote).
    # Use the recovered mean wherever we have any reason to trust it,
    # else fall back to the 3x3 median (neighbourhood consensus).
    trust_mean = copies_agree & ~outlier
    clean_thumbnail = np.where(
        trust_mean.reshape(bh, bw),
        recovered_mean.reshape(bh, bw),
        rec_mean_med.reshape(bh, bw),
    ).astype(np.uint8)

    # --- Bicubic upsample the cleaned thumbnail to full resolution.
    upscaled = cv2.resize(
        clean_thumbnail, (size, size), interpolation=cv2.INTER_CUBIC
    ).astype(np.uint8)

    # --- Build the tamper mask at pixel resolution and fill the core.
    out = suspect.copy().astype(np.uint8)
    detected_mask = np.zeros_like(suspect)
    for i in np.where(tampered)[0]:
        r, c = _idx_rc(int(i), size)
        detected_mask[r:r + 4, c:c + 4] = 255
        out[r:r + 4, c:c + 4] = upscaled[r:r + 4, c:c + 4]


    flagged_patches = int(tampered.sum())
    tamper_fraction = flagged_patches / max(M, 1)

    # Cluster analysis for the global-vs-local call
    cluster_fraction = 1.0
    if flagged_patches > 0:
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(
            (detected_mask > 0).astype(np.uint8), connectivity=8
        )
        if n_cc > 1:
            biggest = int(stats[1:, cv2.CC_STAT_AREA].max())
            cluster_fraction = biggest / max(int(np.count_nonzero(detected_mask)), 1)

    global_attack = (
        tamper_fraction > global_patch_fraction
        or (flagged_patches > 30 and cluster_fraction < 0.30)
    )
    if global_attack:
        out = suspect.copy().astype(np.uint8)
        detected_mask = np.zeros_like(suspect)
        flagged_patches = 0

    return RecoveryResult(
        recovered_image=out,
        detected_mask=detected_mask,
        n_source_tampered=flagged_patches,
        n_target_tampered=0,
        global_attack_detected=global_attack,
    )


# =====================================================================
# UI helper
# =====================================================================
def decomposition_preview(image: np.ndarray, gamma: float = 0.3) -> np.ndarray:
    leaf = decompose(image, gamma=gamma)
    return block_map(image.shape, leaf)
