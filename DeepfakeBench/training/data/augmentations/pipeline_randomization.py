"""Pipeline-randomization augmentation — symmetric across labels.

Counters the camera/pipeline-signature shortcut documented in
`memory/project_signature_shortcut_finding.md` (the same identity flips
real↔fake under different processing pipelines), AND the image-quality
shortcut documented in `memory/project_image_quality_shortcut.md` (the
score correlates negatively with sharpness/luminance, and eval data is
4-25× less sharp than training).

Applying these sub-augs to BOTH real AND fake samples — with slightly
different probabilities — breaks the "pipeline → label" correlation AND
the "sharpness → label" correlation that the model would otherwise pick up.

Sub-augs (each fired independently with its own probability):
    1. JPEG roundtrip       — quality ∈ [jpeg_quality_lo, jpeg_quality_hi]
    2. Downscale-upscale    — scale ∈ [downscale_min, downscale_max]
    3. Chroma blur          — light Cb/Cr Gaussian blur
    4. RGB ⇄ YUV roundtrip  — color-space round-trip
    5. Gamma jitter         — gamma ∈ [gamma_lo, gamma_hi]
    6. Gaussian blur (luma) — sigma ∈ [blur_sigma_lo, blur_sigma_hi]   (P22)
    7. Brightness shift     — beta ∈ [brightness_lo, brightness_hi]    (P22)

Defensive guards after each sub-aug:
    - np.nan_to_num to flush any NaN/Inf produced by color math
    - np.clip(x, 0, 255) before reverting to uint8

The class is callable: ``aug(image_uint8_rgb, label)``. Label is an int or
bool (0 = real, 1 = fake); the corresponding probability is used as the
overall gate. Returns the augmented image (or unchanged input if the gate
roll fails).
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from .teams_simulation import _apply_jpeg_roundtrip, _apply_chroma_blur


class PipelineRandomization:
    """Stateless (per-call) random pipeline-style image transform.

    Built once at trainer init from the yaml ``pipeline_randomization`` block,
    then called by the augmentation router after the family pipeline.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        logger: Optional[logging.Logger] = None,
    ):
        cfg = config or {}
        self.enabled = bool(cfg.get('enabled', False))
        self.p_real = float(cfg.get('p_real', 0.55))
        self.p_fake = float(cfg.get('p_fake', 0.45))
        self.jpeg_quality = tuple(cfg.get('jpeg_quality', (40, 95)))
        self.downscale_range = tuple(cfg.get('downscale_range', (0.85, 1.0)))
        self.chroma_blur_ksize = int(cfg.get('chroma_blur_ksize', 3))
        self.gamma_range = tuple(cfg.get('gamma_range', (0.92, 1.08)))
        # P22 — luma Gaussian blur + brightness shift to close the train-eval
        # sharpness/luma gap. Default to no-op so existing callers see no change.
        self.blur_sigma_range = tuple(cfg.get('blur_sigma_range', (0.0, 0.0)))
        self.brightness_range = tuple(cfg.get('brightness_range', (0.0, 0.0)))
        # Per-sub-aug fire probabilities (each independent of the gate).
        self.sub_p = {
            'jpeg': float(cfg.get('jpeg_p', 0.5)),
            'downscale': float(cfg.get('downscale_p', 0.5)),
            'chroma_blur': float(cfg.get('chroma_blur_p', 0.5)),
            'yuv_roundtrip': float(cfg.get('yuv_roundtrip_p', 0.5)),
            'gamma': float(cfg.get('gamma_p', 0.5)),
            'blur': float(cfg.get('blur_p', 0.0)),
            'brightness': float(cfg.get('brightness_p', 0.0)),
        }
        self._logger = logger or logging.getLogger(__name__)
        if self.enabled:
            self._logger.info(
                "PipelineRandomization ENABLED p_real=%.2f p_fake=%.2f "
                "jpeg_q=%s downscale=%s gamma=%s blur_sigma=%s "
                "brightness=%s sub_p=%s",
                self.p_real, self.p_fake, self.jpeg_quality,
                self.downscale_range, self.gamma_range,
                self.blur_sigma_range, self.brightness_range, self.sub_p,
            )
        else:
            self._logger.info("PipelineRandomization DISABLED")

    def __call__(self, image: np.ndarray, label: Optional[int] = None) -> np.ndarray:
        if not self.enabled or not isinstance(image, np.ndarray):
            return image
        # Accept both 0/1 ints and bool / numpy ints.
        is_fake = bool(int(label)) if label is not None else False
        gate_p = self.p_fake if is_fake else self.p_real
        if np.random.random() >= gate_p:
            return image

        out = image
        if out.dtype != np.uint8:
            out = self._sanitize_to_uint8(out)

        if np.random.random() < self.sub_p['gamma']:
            out = self._apply_gamma(out, self.gamma_range)
            out = self._sanitize_to_uint8(out)

        if np.random.random() < self.sub_p['chroma_blur']:
            out = _apply_chroma_blur(out, self.chroma_blur_ksize)
            out = self._sanitize_to_uint8(out)

        if np.random.random() < self.sub_p['yuv_roundtrip']:
            out = self._apply_yuv_roundtrip(out)
            out = self._sanitize_to_uint8(out)

        if np.random.random() < self.sub_p['downscale']:
            out = self._apply_downscale_upscale(out, self.downscale_range)
            out = self._sanitize_to_uint8(out)

        if np.random.random() < self.sub_p['jpeg']:
            out = _apply_jpeg_roundtrip(out, self.jpeg_quality)
            out = self._sanitize_to_uint8(out)

        if self.sub_p['blur'] > 0.0 and np.random.random() < self.sub_p['blur']:
            out = self._apply_gaussian_blur(out, self.blur_sigma_range)
            out = self._sanitize_to_uint8(out)

        if self.sub_p['brightness'] > 0.0 and np.random.random() < self.sub_p['brightness']:
            out = self._apply_brightness_shift(out, self.brightness_range)
            out = self._sanitize_to_uint8(out)

        return out

    @staticmethod
    def _sanitize_to_uint8(img: np.ndarray) -> np.ndarray:
        arr = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        arr = np.clip(arr, 0, 255)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    @staticmethod
    def _apply_gamma(img_rgb: np.ndarray, gamma_range: tuple) -> np.ndarray:
        gamma = float(np.random.uniform(*gamma_range))
        if abs(gamma - 1.0) < 1e-3:
            return img_rgb
        normed = img_rgb.astype(np.float32) / 255.0
        normed = np.clip(normed, 0.0, 1.0)
        adjusted = np.power(normed, gamma) * 255.0
        return adjusted

    @staticmethod
    def _apply_yuv_roundtrip(img_rgb: np.ndarray) -> np.ndarray:
        # Use BGR to leverage cv2's tested conversion paths; convert in/out from RGB.
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
        bgr_back = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return cv2.cvtColor(bgr_back, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _apply_downscale_upscale(img_rgb: np.ndarray, scale_range: tuple) -> np.ndarray:
        h, w = img_rgb.shape[:2]
        scale = float(np.random.uniform(*scale_range))
        if scale >= 0.999:
            return img_rgb
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))
        downscaled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        upscaled = cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)
        return upscaled

    @staticmethod
    def _apply_gaussian_blur(img: np.ndarray, sigma_range: tuple) -> np.ndarray:
        sigma = float(np.random.uniform(*sigma_range))
        if sigma <= 0.05:
            return img
        # Kernel must be odd; cover ±3σ.
        ksize = int(2 * np.ceil(3 * sigma) + 1)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    @staticmethod
    def _apply_brightness_shift(img: np.ndarray, beta_range: tuple) -> np.ndarray:
        beta = float(np.random.uniform(*beta_range))
        if abs(beta) < 0.5:
            return img
        return np.clip(img.astype(np.int32) + int(round(beta)), 0, 255).astype(np.uint8)
