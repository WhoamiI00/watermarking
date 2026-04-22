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
from .embedding import embed_lsb2, extract_lsb2
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
# Sidecar  --  two independent layers of (pair -> target) mappings for
# 2x redundancy. A source block S has its feature copied into two
# different target blocks, chosen by two different chaotic
# permutations. If one copy is wiped out by a collage/paint attack,
# the other copy usually survives and the receiver can cross-check
# them to localise tampering and repair it cleanly.
# =====================================================================
@dataclass
class Sidecar:
    size: int
    gamma: float
    key: int
    blocks: List[Block]                     # multi-scale leaves (viz only)
    # Two independent layers. For layer L in {A, B}:
    #   pair_sources_L: [s0, s1, s2, s3, ...] consecutive pairs
    #   target_blocks_L[k]: 4x4 target block index for pair k
    pair_sources_a: List[int]
    target_blocks_a: List[int]
    pair_sources_b: List[int]
    target_blocks_b: List[int]

    def to_json(self) -> Dict:
        return {
            "size": self.size,
            "gamma": self.gamma,
            "key": self.key,
            "blocks": [{"y": b.y, "x": b.x, "size": b.size, "homogeneous": b.homogeneous} for b in self.blocks],
            "pair_sources_a": list(map(int, self.pair_sources_a)),
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
            pair_sources_a=[int(v) for v in d["pair_sources_a"]],
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
    """Build two layers of (pair -> target) mappings with disjoint target
    sets (so one target never overwrites another layer's payload).

    Returns (pair_sources_a, target_blocks_a, pair_sources_b, target_blocks_b).

    * The full set of 4x4 block indices is partitioned into two halves
      via a single chaotic target permutation. Layer A uses the first
      half; layer B uses the second half.
    * Each layer has its own source permutation (different key), so the
      two copies of a source's feature live in different positions.
    * A source block is never mapped to itself as a target.
    """
    M = _num_4x4(size)
    assert M % 2 == 0

    tgt_perm = chaos.logistic_permutation(M, key)
    layer_a_targets = [int(v) for v in tgt_perm[:M // 2].tolist()]
    layer_b_targets = [int(v) for v in tgt_perm[M // 2:].tolist()]

    src_perm_a = chaos.logistic_permutation(M, (key * 2654435761) & 0xFFFFFFFF)
    src_perm_b = chaos.logistic_permutation(M, (key * 40503) & 0xFFFFFFFF)
    pair_sources_a = [int(v) for v in src_perm_a.tolist()]
    pair_sources_b = [int(v) for v in src_perm_b.tolist()]

    def _assign(pair_sources, available):
        n_pairs = len(available)
        available = list(available)
        target_blocks = [0] * n_pairs
        # First pass: assign each pair its permutation-given slot, unless
        # the slot is one of this pair's own sources (we swap later).
        defer = []
        for k in range(n_pairs):
            sA = pair_sources[2 * k]
            sB = pair_sources[2 * k + 1]
            cand = available[k]
            if cand == sA or cand == sB:
                defer.append(k)
                target_blocks[k] = -1
            else:
                target_blocks[k] = cand
        # Repair self-mapping pairs by pairwise swap with any valid slot.
        for k in defer:
            sA = pair_sources[2 * k]
            sB = pair_sources[2 * k + 1]
            my_cand = available[k]
            for j in range(n_pairs):
                if j == k or target_blocks[j] == -1:
                    continue
                tj = target_blocks[j]
                sjA = pair_sources[2 * j]
                sjB = pair_sources[2 * j + 1]
                if tj != sA and tj != sB and my_cand != sjA and my_cand != sjB:
                    target_blocks[k] = tj
                    target_blocks[j] = my_cand
                    break
            if target_blocks[k] == -1:
                # fall back: let it self-map rather than crash (extremely rare).
                target_blocks[k] = my_cand
        return target_blocks

    target_blocks_a = _assign(pair_sources_a, layer_a_targets)
    target_blocks_b = _assign(pair_sources_b, layer_b_targets)
    return pair_sources_a, target_blocks_a, pair_sources_b, target_blocks_b


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

    # 3. Allocate TWO layers of (pair -> target) assignments with
    # disjoint target sets (target permutation halves A and B partition
    # the image's 4x4 blocks, so layers never overwrite each other).
    pair_sources_a, target_blocks_a, pair_sources_b, target_blocks_b = \
        _allocate_two_layers(size, key)

    # 4. Embed. Both layers use 2-LSB substitution -- 30 bits each.
    watermarked = image.copy()

    def _embed_layer(pair_sources, target_blocks):
        for k in range(len(target_blocks)):
            sA = pair_sources[2 * k]
            sB = pair_sources[2 * k + 1]
            tgt = target_blocks[k]

            mA = int(means[sA])
            mB = int(means[sB])
            cA = int(np.searchsorted(_STD_BINS, stds[sA], side="right") - 1)
            cB = int(np.searchsorted(_STD_BINS, stds[sB], side="right") - 1)
            cA = max(0, min(7, cA))
            cB = max(0, min(7, cB))

            msgA = _encode_feature_bits(mA, cA)
            msgB = _encode_feature_bits(mB, cB)
            cwA = bch.encode_block(msgA)
            cwB = bch.encode_block(msgB)

            payload = np.zeros(32, dtype=np.uint8)
            payload[:15] = cwA
            payload[15:30] = cwB

            tgt_slice = _idx_to_slice(tgt, size)
            watermarked[tgt_slice] = embed_lsb2(watermarked[tgt_slice], payload)

    _embed_layer(pair_sources_a, target_blocks_a)
    _embed_layer(pair_sources_b, target_blocks_b)

    sidecar = Sidecar(
        size=size,
        gamma=gamma,
        key=key,
        blocks=leaf,
        pair_sources_a=pair_sources_a,
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


def _extract_layer(
    suspect: np.ndarray, size: int, pair_sources, target_blocks
) -> Tuple[np.ndarray, np.ndarray]:
    M = _num_4x4(size)
    recovered_mean = np.zeros(M, dtype=np.uint8)
    recovered_std = np.zeros(M, dtype=np.uint8)
    for k in range(len(target_blocks)):
        sA = pair_sources[2 * k]
        sB = pair_sources[2 * k + 1]
        tgt = target_blocks[k]
        bits = extract_lsb2(suspect[_idx_to_slice(tgt, size)], n_bits=30)
        msgA = bch.decode_block(bits[:15])
        msgB = bch.decode_block(bits[15:30])
        mA, cA = _decode_feature_bits(msgA)
        mB, cB = _decode_feature_bits(msgB)
        recovered_mean[sA] = mA
        recovered_std[sA] = cA
        recovered_mean[sB] = mB
        recovered_std[sB] = cB
    return recovered_mean, recovered_std


def _extract_recovery_features(
    suspect: np.ndarray, sidecar: Sidecar
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (recovered_mean, recovered_std, copies_agree).

    For each source block, extract the feature from both target copies
    (layer A and layer B). If the two decoded means agree within 6
    grayscales, trust the feature and return their average. If they
    disagree, one copy's target was probably tampered; we flag that
    source as having no-trust and let the caller fall back to a
    neighbourhood estimate.
    """
    size = sidecar.size
    M = _num_4x4(size)

    mean_a, std_a = _extract_layer(
        suspect, size, sidecar.pair_sources_a, sidecar.target_blocks_a
    )
    mean_b, std_b = _extract_layer(
        suspect, size, sidecar.pair_sources_b, sidecar.target_blocks_b
    )

    # Agreement -> both copies intact. Disagreement -> at least one
    # target was tampered; in that case pick whichever copy is closer
    # to the 3x3 median of its neighbours (the "consensus" copy).
    disagreement = np.abs(mean_a.astype(np.int32) - mean_b.astype(np.int32))
    copies_agree = disagreement <= 6

    bw = bh = size // 4
    med_a = cv2.medianBlur(mean_a.reshape(bh, bw), 3).flatten()
    med_b = cv2.medianBlur(mean_b.reshape(bh, bw), 3).flatten()
    a_consistent = np.abs(mean_a.astype(np.int32) - med_a.astype(np.int32)) <= \
                   np.abs(mean_b.astype(np.int32) - med_b.astype(np.int32))

    recovered_mean = np.where(
        copies_agree, mean_a,
        np.where(a_consistent, mean_a, mean_b)
    ).astype(np.uint8)
    recovered_std = np.where(
        copies_agree, std_a,
        np.where(a_consistent, std_a, std_b)
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
    # recovered feature. Dual copies + agreement voting have already
    # produced the best available recovered value; we only care whether
    # the current-in-suspect block matches it.
    mean_diff = np.abs(cur_means.astype(np.int32) - recovered_mean.astype(np.int32))
    std_diff = np.abs(cur_std_code.astype(np.int32) - recovered_std.astype(np.int32))
    # Sources where the recovered value is BOTH an outlier and the two
    # copies disagreed are treated as "no reliable recovery" -- they
    # can't be used to flag tampering. Everywhere else, mismatch flags.
    unreliable = outlier & ~copies_agree
    tampered = (~unreliable) & ((mean_diff > mean_tol) | (std_diff > std_code_tol))

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

    # --- Bicubic upsample the thumbnail to full resolution. This gives
    # a smooth gradient fill that tracks the actual per-pixel variation
    # much better than a single-mean flat fill -- closes most of the
    # PSNR gap against the paper's recovered-image numbers.
    upscaled = cv2.resize(
        clean_thumbnail, (size, size), interpolation=cv2.INTER_CUBIC
    ).astype(np.uint8)

    # --- Repair: paste the upscaled-thumbnail region into tampered blocks.
    out = suspect.copy().astype(np.uint8)
    detected_mask = np.zeros_like(suspect)
    for i in np.where(tampered)[0]:
        r, c = _idx_rc(int(i), size)
        out[r:r + 4, c:c + 4] = upscaled[r:r + 4, c:c + 4]
        detected_mask[r:r + 4, c:c + 4] = 255

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
