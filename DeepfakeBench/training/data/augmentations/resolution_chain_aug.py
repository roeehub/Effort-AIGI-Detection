"""Resolution-chain augmentation — randomize downsample->upsample chain per frame.

Motivated by the 2026-05-15 resolution-chain CPU probe (FACTS at
`analysis/cpu_diagnostics_2026-05-15_resolution_chain/RESULTS_FACTS_2026-05-15.md`):
on the same real frame, P8A and E2B score-range across 20 downsample->upsample
perturbations is 0.89-0.98 (near full [0, 1]); T5C compresses this to 0.54-0.72
but the swing is still production-blocking. Source-resolution dominates kernel
choice for P8A/T5C (~40% vs ~22% variance share); downsampling to mid-sizes
(96, 128) systematically pushes real-frame mean score UP (P8A 0.23 -> 0.70 as
source size shrinks 192 -> 64). The model is using the downsample->upsample
signature as an IQ shortcut.

Applied BEFORE the canonical 224x224 resize in the collate path, same as
`face_scale_jitter`. With probability `p_apply`, the frame is downsampled to a
randomly chosen size in `down_sizes` then immediately upsampled back to the
original (H, W) with the same kernel. The collate then does the canonical
224x224 resize. Net effect on the model input: a re-encoding of the same
content at a randomly chosen effective source-resolution.

The chain is kernel-consistent (same kernel for down + up) by design — mixed
kernels add variants without isolating either axis. Production-realistic
kernels are LINEAR, CUBIC, AREA, LANCZOS4 (cv2's standard set).

Module-level config (set once at trainer init via
`set_resolution_chain_aug_config`); the aug function is stateless per call.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import cv2
import numpy as np


# Default size set: chosen to match the 5-size sweep used in the 2026-05-15
# probe — covers the production-relevant range from heavy downscale (64) to
# mild downscale (192) without exceeding the typical face-crop native size.
DEFAULT_DOWN_SIZES = (64, 96, 128, 160, 192)

_KERNEL_NAME_TO_CV = {
    "LINEAR": cv2.INTER_LINEAR,
    "CUBIC": cv2.INTER_CUBIC,
    "AREA": cv2.INTER_AREA,
    "LANCZOS4": cv2.INTER_LANCZOS4,
}
DEFAULT_KERNELS = tuple(_KERNEL_NAME_TO_CV.keys())


_CONFIG = {
    "enabled": False,
    "p_apply": 0.0,
    "down_sizes": DEFAULT_DOWN_SIZES,
    "kernels": DEFAULT_KERNELS,
}


def set_resolution_chain_aug_config(
    enabled: bool,
    p_apply: float = 0.5,
    down_sizes: Optional[Sequence[int]] = None,
    kernels: Optional[Sequence[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Configure resolution-chain aug at trainer init time. Idempotent."""
    _CONFIG["enabled"] = bool(enabled) and float(p_apply) > 0.0
    _CONFIG["p_apply"] = float(p_apply)
    if down_sizes is not None and len(down_sizes) > 0:
        _CONFIG["down_sizes"] = tuple(int(s) for s in down_sizes if int(s) > 0)
    else:
        _CONFIG["down_sizes"] = DEFAULT_DOWN_SIZES
    if kernels is not None and len(kernels) > 0:
        bad = [k for k in kernels if k not in _KERNEL_NAME_TO_CV]
        if bad:
            raise ValueError(
                f"resolution_chain_aug unknown kernels {bad}. "
                f"Supported: {sorted(_KERNEL_NAME_TO_CV)}"
            )
        _CONFIG["kernels"] = tuple(kernels)
    else:
        _CONFIG["kernels"] = DEFAULT_KERNELS
    log = logger or logging.getLogger(__name__)
    if _CONFIG["enabled"]:
        log.info(
            "ResolutionChainAug ENABLED: p_apply=%.2f sizes=%s kernels=%s",
            _CONFIG["p_apply"],
            list(_CONFIG["down_sizes"]),
            list(_CONFIG["kernels"]),
        )
    else:
        log.info("ResolutionChainAug DISABLED")


def apply_resolution_chain_aug(img: np.ndarray) -> np.ndarray:
    """Return a resolution-chained copy of `img` if enabled+fires, else `img`.

    Fires with probability `p_apply`. When fired:
        1. Pick `size` uniformly from `down_sizes` and `kernel_name` from
           `kernels`.
        2. Resize `img` (H, W) -> (size, size) using that kernel.
        3. Resize that back to (H, W) using the SAME kernel.
        4. Return; the canonical 224x224 resize downstream handles the
           final size.

    Edge cases:
        - If the image is already smaller than `size`, the down step is an
          upsample then a downsample — still kernel-consistent, still tests
          the chain.
        - If the chosen `size` equals min(H, W), the operation is a no-op
          on cv2 with most kernels (still safe).
        - Non-ndarray inputs or low-dim arrays are returned unchanged.
    """
    if not _CONFIG["enabled"] or not isinstance(img, np.ndarray):
        return img
    if img.ndim < 2:
        return img
    if float(np.random.random()) >= _CONFIG["p_apply"]:
        return img

    sizes = _CONFIG["down_sizes"]
    kernels = _CONFIG["kernels"]
    size = int(sizes[np.random.randint(0, len(sizes))])
    kernel_name = kernels[np.random.randint(0, len(kernels))]
    kernel = _KERNEL_NAME_TO_CV[kernel_name]

    h, w = img.shape[:2]
    # Down step
    down = cv2.resize(img, (size, size), interpolation=kernel)
    # Up step back to original (H, W)
    up = cv2.resize(down, (w, h), interpolation=kernel)
    return up


def get_config_snapshot() -> dict:
    """Return a copy of the current module config; for diagnostics/logging."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in _CONFIG.items()}
