"""Band-limited Fourier amplitude randomization.

Per-frame, per-channel multiplicative noise on FFT amplitude in radial bands
of a 16-band partition at 224x224. Phase is preserved; bands flagged as
manipulation-signal-carrying are explicitly preserved. With probability
``p_apply`` the aug fires; otherwise the frame passes through unchanged.

Empirical foundation: ``analysis/fourier_band_overlap_2026-05-06/run_probe.py``
(Probe 6, GREEN verdict). Bands 12-13 are the cleanest cell (shortcut AUC 0.97,
manipulation signal AUC 0.46-0.52); bands 8-10 are the secondary safe zone;
bands 5-6 are explicitly signal-carrying and are preserved. See thread
``processing_signature_shortcut`` and memory ``project_fourier_band_overlap_2026-05-06``.

Configuration is module-level (matching ``face_scale_jitter``'s pattern)
because the collate functions take no extra args. ``set_fourier_aug_config``
is called once at trainer init from the yaml top-level ``fourier_aug`` block.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np


_GLOBAL_CFG = {
    "enabled": False,
    "p_apply": 0.5,
    "bands_randomize": [8, 9, 10, 11, 12, 13],
    "bands_preserve": [5, 6],
    "noise_log_range": [-0.3, 0.3],
    "radial_resolution": 16,
    "image_resolution": 224,
}

_RBINS: Optional[np.ndarray] = None


def _radial_bins() -> np.ndarray:
    """Compute the (RES, RES) array of radial-band indices in 0..N_RADIAL-1.

    Mirrors ``analysis/fourier_band_overlap_2026-05-06/run_probe.py:radial_bins``
    so a band index here maps to the same FFT region as the empirical probe.
    """
    res = int(_GLOBAL_CFG["image_resolution"])
    n_radial = int(_GLOBAL_CFG["radial_resolution"])
    yy, xx = np.mgrid[:res, :res]
    cx = cy = (res - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = r.max()
    return np.clip((r / r_max * n_radial).astype(np.int32), 0, n_radial - 1)


def set_fourier_aug_config(cfg, logger: Optional[logging.Logger] = None) -> None:
    """Configure the fourier-band aug at trainer init time.

    Idempotent: subsequent calls overwrite the prior config. Unknown keys are
    ignored; missing keys retain their previous values. Invalidates the cached
    radial-bin grid so the next call recomputes it for the new resolution.
    """
    global _RBINS
    if cfg is None:
        return
    if not isinstance(cfg, dict):
        return

    for k in list(_GLOBAL_CFG.keys()):
        if k in cfg:
            _GLOBAL_CFG[k] = cfg[k]

    _GLOBAL_CFG["enabled"] = bool(_GLOBAL_CFG.get("enabled", False))
    _RBINS = None

    log = logger or logging.getLogger(__name__)
    if _GLOBAL_CFG["enabled"]:
        log.info(
            "FourierBandAmpAug ENABLED: p_apply=%.2f bands_randomize=%s bands_preserve=%s "
            "noise_log_range=%s radial_resolution=%d image_resolution=%d",
            float(_GLOBAL_CFG["p_apply"]),
            list(_GLOBAL_CFG["bands_randomize"]),
            list(_GLOBAL_CFG["bands_preserve"]),
            list(_GLOBAL_CFG["noise_log_range"]),
            int(_GLOBAL_CFG["radial_resolution"]),
            int(_GLOBAL_CFG["image_resolution"]),
        )
    else:
        log.info("FourierBandAmpAug DISABLED")


def is_fourier_aug_enabled() -> bool:
    """Return True iff the fourier-band aug is enabled."""
    return bool(_GLOBAL_CFG.get("enabled", False))


def get_fourier_aug_config() -> dict:
    """Read-only snapshot of current config (used by tests and diagnostics)."""
    return dict(_GLOBAL_CFG)


def _apply_band_amp_randomization(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply per-channel band-limited amplitude randomization to a HWC image."""
    global _RBINS
    if _RBINS is None:
        _RBINS = _radial_bins()

    bands_pres = set(int(b) for b in _GLOBAL_CFG["bands_preserve"])
    bands_rand = [int(b) for b in _GLOBAL_CFG["bands_randomize"] if int(b) not in bands_pres]
    if not bands_rand:
        return image

    log_lo, log_hi = float(_GLOBAL_CFG["noise_log_range"][0]), float(_GLOBAL_CFG["noise_log_range"][1])
    was_uint8 = image.dtype == np.uint8
    x = image.astype(np.float32, copy=False)
    out = np.empty_like(x)

    for c in range(x.shape[2]):
        F = np.fft.fftshift(np.fft.fft2(x[:, :, c]))
        amp = np.abs(F)
        phase = np.angle(F)
        new_amp = amp.copy()
        for b in bands_rand:
            mask = (_RBINS == b)
            if not mask.any():
                continue
            scale = float(np.exp(rng.uniform(log_lo, log_hi)))
            new_amp[mask] = amp[mask] * scale
        F2 = new_amp * np.exp(1j * phase)
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(F2)))

    np.clip(out, 0.0, 255.0, out=out)
    if was_uint8:
        out = out.astype(np.uint8)
    return out


def apply_fourier_band_aug(img: np.ndarray) -> np.ndarray:
    """Return a Fourier-band-randomized copy of ``img`` if enabled, else ``img``.

    Defensive: any exception is swallowed and the original frame is returned;
    a one-time warning is logged so a misconfigured aug never crashes training.
    The aug is a no-op when disabled, when the image isn't HWC at the configured
    resolution, or when the per-call probability draw misses.
    """
    if not is_fourier_aug_enabled():
        return img
    if not isinstance(img, np.ndarray):
        return img
    if img.ndim != 3 or img.shape[2] not in (1, 3):
        return img
    res = int(_GLOBAL_CFG["image_resolution"])
    if img.shape[0] != res or img.shape[1] != res:
        return img

    try:
        rng = np.random.default_rng()
        if rng.random() > float(_GLOBAL_CFG["p_apply"]):
            return img
        return _apply_band_amp_randomization(img, rng)
    except Exception as exc:  # noqa: BLE001 — bulletproof per task spec §3.3
        logging.getLogger(__name__).warning(
            "FourierBandAmpAug failed on a frame; passing through unchanged: %s", exc
        )
        return img


class FourierBandAmpAug:
    """Callable wrapper matching the smoke-test snippets in the task spec.

    Behavior identical to ``apply_fourier_band_aug``; provided so that
    ``aug = FourierBandAmpAug(); aug(img)`` works as expected.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return apply_fourier_band_aug(image)
