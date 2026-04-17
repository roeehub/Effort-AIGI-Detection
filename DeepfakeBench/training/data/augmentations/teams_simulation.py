"""
TeamsCodecSimulation — Microsoft Teams video pipeline simulation.

Approximates the measured Teams WebRTC codec fingerprint validated against
matched frame pairs (original PNG → Teams passthrough JPG).

Measured deltas (18 samples × 132 matched frame pairs, validated):
    −50.9% sharpness (Laplacian variance)  — codec compression smooths
    +19.0% brightness                       — auto-exposure / gain control
     +3.5% contrast                         — auto-exposure
    −11.8% noise (reduced, not added)       — no additive noise
    −77.4% high-frequency energy            — VP8/VP9 lossy encoding
     −1.2% blockiness                       — slight block edge reduction
     −8.7% bpp (at Q95 JPEG re-encode)     — less compressible

Pipeline stages (applied in order, always together):
    1. Brightness + contrast boost  — Teams auto-exposure / gain control
    2. Light Gaussian blur          — Simulates codec smoothing of detail
    3. JPEG compression             — I-frame encoding at conferencing bitrate
    4. Bilateral deblocking         — VP8/VP9 in-loop deblock filter
    (No sharpening, no additive noise — validated against real Teams data)

Compatible with albumentations 0.4.6 (ImageOnlyTransform API).

Wave 2 — used by R9_F (synthetic-only) and R9_G (combined).

April 2026 sidecar note:
``TeamsAdaptiveCodecSimulation`` is an opt-in Track B prototype that keeps the
legacy single-mode transform intact while adding a family-aware two-mode
mixture for current-bucket evaluation.
"""

from __future__ import annotations

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform


def _apply_brightness_contrast(
    img: np.ndarray,
    brightness_limit: tuple[float, float],
    contrast_limit: tuple[float, float],
) -> np.ndarray:
    result = img.astype(np.float32)
    brightness = np.random.uniform(*brightness_limit) * 255.0
    contrast = 1.0 + np.random.uniform(*contrast_limit)
    result = result * contrast + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def _apply_jpeg_roundtrip(img_rgb: np.ndarray, jpeg_quality: tuple[int, int]) -> np.ndarray:
    quality = int(np.random.uniform(*jpeg_quality))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, encode_param)
    if not ok:
        return img_rgb
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return img_rgb
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def _apply_chroma_blur(img_rgb: np.ndarray, chroma_blur_ksize: int) -> np.ndarray:
    if chroma_blur_ksize <= 0:
        return img_rgb
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    k = chroma_blur_ksize
    ycrcb[:, :, 1] = cv2.GaussianBlur(ycrcb[:, :, 1], (k, k), 0)
    ycrcb[:, :, 2] = cv2.GaussianBlur(ycrcb[:, :, 2], (k, k), 0)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def _apply_bilateral(
    img_rgb: np.ndarray,
    d: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    if d <= 0:
        return img_rgb
    return cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)


def _apply_unsharp(
    img_rgb: np.ndarray,
    sigma_range: tuple[float, float],
    amount_range: tuple[float, float],
) -> np.ndarray:
    sigma = np.random.uniform(*sigma_range)
    amount = np.random.uniform(*amount_range)
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(
        img_rgb.astype(np.float32),
        1.0 + amount,
        blurred.astype(np.float32),
        -amount,
        0.0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


class TeamsCodecSimulation(ImageOnlyTransform):
    """
    Simulate the Microsoft Teams WebRTC video pipeline.

    Calibrated against real Teams passthrough frames.  The dominant effects
    are: (a) significant brightness boost from auto-exposure, (b) contrast
    increase, and (c) lossy video codec compression that reduces sharpness
    and high-frequency energy.  Teams does **not** sharpen or add noise.

    All sub-effects are applied together (not OneOf) because Teams applies
    them to every frame.  The overall ``p`` parameter controls whether the
    block fires for a given image.

    Args:
        brightness_limit:  (min, max) fraction of 255 for additive brightness.
                           Positive-only since Teams consistently brightens.
        contrast_limit:    (min, max) multiplicative contrast factor delta
                           (applied as ``1 + delta``).
        blur_sigma:        (min, max) Gaussian blur sigma to simulate codec
                           smoothing that reduces sharpness / HF energy.
        jpeg_quality:      (min, max) JPEG quality for I-frame simulation.
        chroma_blur_ksize: Kernel size for light Cb/Cr blur (YUV 4:2:0).
                           Set to 0 to disable.
        deblock_d:         Bilateral filter diameter for deblocking.
                           Simulates VP8/VP9 in-loop deblock filter.
                           Set to 0 to disable.
        deblock_sigma_color: Bilateral filter color sigma.
        deblock_sigma_space: Bilateral filter space sigma.
        always_apply:      Force application regardless of probability.
        p:                 Probability of applying this transform.
    """

    def __init__(
        self,
        brightness_limit: tuple[float, float] = (0.02, 0.10),
        contrast_limit: tuple[float, float] = (0.10, 0.22),
        blur_sigma: tuple[float, float] = (0.35, 0.85),
        jpeg_quality: tuple[int, int] = (72, 88),
        chroma_blur_ksize: int = 0,
        deblock_d: int = 5,
        deblock_sigma_color: float = 30.0,
        deblock_sigma_space: float = 30.0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.blur_sigma = blur_sigma
        self.jpeg_quality = jpeg_quality
        self.chroma_blur_ksize = chroma_blur_ksize
        self.deblock_d = deblock_d
        self.deblock_sigma_color = deblock_sigma_color
        self.deblock_sigma_space = deblock_sigma_space

    # ------------------------------------------------------------------
    # albumentations 0.4.6 API
    # ------------------------------------------------------------------

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply the full Teams codec simulation pipeline."""
        # 1. Brightness + Contrast boost (Teams auto-exposure / gain)
        #    Measured: +29% brightness, +18% contrast
        result_u8 = _apply_brightness_contrast(
            img,
            self.brightness_limit,
            self.contrast_limit,
        )

        # 2. Light Gaussian blur (simulates VP8/VP9 codec smoothing)
        #    Measured: -38% sharpness, -87% HF energy
        sigma = np.random.uniform(*self.blur_sigma)
        result_u8 = cv2.GaussianBlur(result_u8, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 3. Light chroma blur (YUV 4:2:0 subsampling)
        result_u8 = _apply_chroma_blur(result_u8, self.chroma_blur_ksize)

        # 4. JPEG compression (conferencing I-frame bitrate)
        result_u8 = _apply_jpeg_roundtrip(result_u8, self.jpeg_quality)

        # 5. Bilateral deblocking (simulates VP8/VP9 in-loop deblock filter)
        #    Smooths 8x8 JPEG block boundaries without destroying edges.
        result_u8 = _apply_bilateral(
            result_u8,
            self.deblock_d,
            self.deblock_sigma_color,
            self.deblock_sigma_space,
        )

        return result_u8

    def get_transform_init_args_names(self):
        return (
            "brightness_limit",
            "contrast_limit",
            "blur_sigma",
            "jpeg_quality",
            "chroma_blur_ksize",
            "deblock_d",
            "deblock_sigma_color",
            "deblock_sigma_space",
        )


class TeamsAdaptiveCodecSimulation(ImageOnlyTransform):
    """
    Sidecar Track B prototype: a two-mode Teams simulation.

    The April 2026 matched-pair rerun showed that ordinary current Teams slices
    are often mixed-to-sharpening, while enhanced-through-Teams remains much
    more blur / HF-loss dominant. This class keeps those two outcomes separate:

    - ordinary mode: exposure lift + mild codec + cleanup/sharpen
    - enhanced mode: exposure lift + blur + stronger codec + deblock

    The router can call ``apply_for_family(..., family_key=...)`` so enhanced
    fake families bias toward the enhanced mode while ordinary families bias
    toward the ordinary mode.
    """

    def __init__(
        self,
        ordinary_mode_probability_non_enhanced: float = 0.75,
        ordinary_mode_probability_enhanced: float = 0.25,
        enhanced_families: tuple[str, ...] = (
            "visomaster_enhanced_fake",
            "deeplive_enhanced_fake",
        ),
        ordinary_brightness_limit: tuple[float, float] = (0.02, 0.09),
        ordinary_contrast_limit: tuple[float, float] = (0.06, 0.18),
        ordinary_jpeg_quality: tuple[int, int] = (80, 92),
        ordinary_deblock_d: int = 3,
        ordinary_deblock_sigma_color: float = 18.0,
        ordinary_deblock_sigma_space: float = 18.0,
        ordinary_unsharp_sigma: tuple[float, float] = (0.55, 1.05),
        ordinary_unsharp_amount: tuple[float, float] = (0.35, 0.90),
        enhanced_brightness_limit: tuple[float, float] = (0.03, 0.11),
        enhanced_contrast_limit: tuple[float, float] = (0.10, 0.24),
        enhanced_blur_sigma: tuple[float, float] = (0.45, 1.10),
        enhanced_jpeg_quality: tuple[int, int] = (68, 84),
        enhanced_deblock_d: int = 5,
        enhanced_deblock_sigma_color: float = 30.0,
        enhanced_deblock_sigma_space: float = 30.0,
        chroma_blur_ksize: int = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.ordinary_mode_probability_non_enhanced = float(
            np.clip(ordinary_mode_probability_non_enhanced, 0.0, 1.0)
        )
        self.ordinary_mode_probability_enhanced = float(
            np.clip(ordinary_mode_probability_enhanced, 0.0, 1.0)
        )
        self.enhanced_families = tuple(enhanced_families)
        self.ordinary_brightness_limit = ordinary_brightness_limit
        self.ordinary_contrast_limit = ordinary_contrast_limit
        self.ordinary_jpeg_quality = ordinary_jpeg_quality
        self.ordinary_deblock_d = ordinary_deblock_d
        self.ordinary_deblock_sigma_color = ordinary_deblock_sigma_color
        self.ordinary_deblock_sigma_space = ordinary_deblock_sigma_space
        self.ordinary_unsharp_sigma = ordinary_unsharp_sigma
        self.ordinary_unsharp_amount = ordinary_unsharp_amount
        self.enhanced_brightness_limit = enhanced_brightness_limit
        self.enhanced_contrast_limit = enhanced_contrast_limit
        self.enhanced_blur_sigma = enhanced_blur_sigma
        self.enhanced_jpeg_quality = enhanced_jpeg_quality
        self.enhanced_deblock_d = enhanced_deblock_d
        self.enhanced_deblock_sigma_color = enhanced_deblock_sigma_color
        self.enhanced_deblock_sigma_space = enhanced_deblock_sigma_space
        self.chroma_blur_ksize = chroma_blur_ksize

    def _choose_mode(self, family_key: str | None) -> str:
        ordinary_p = (
            self.ordinary_mode_probability_enhanced
            if family_key in self.enhanced_families
            else self.ordinary_mode_probability_non_enhanced
        )
        return "ordinary" if np.random.random() < ordinary_p else "enhanced"

    def _apply_ordinary_mode(self, img: np.ndarray) -> np.ndarray:
        result = _apply_brightness_contrast(
            img,
            self.ordinary_brightness_limit,
            self.ordinary_contrast_limit,
        )
        result = _apply_jpeg_roundtrip(result, self.ordinary_jpeg_quality)
        result = _apply_bilateral(
            result,
            self.ordinary_deblock_d,
            self.ordinary_deblock_sigma_color,
            self.ordinary_deblock_sigma_space,
        )
        return _apply_unsharp(
            result,
            self.ordinary_unsharp_sigma,
            self.ordinary_unsharp_amount,
        )

    def _apply_enhanced_mode(self, img: np.ndarray) -> np.ndarray:
        result = _apply_brightness_contrast(
            img,
            self.enhanced_brightness_limit,
            self.enhanced_contrast_limit,
        )
        sigma = np.random.uniform(*self.enhanced_blur_sigma)
        result = cv2.GaussianBlur(result, (0, 0), sigmaX=sigma, sigmaY=sigma)
        result = _apply_chroma_blur(result, self.chroma_blur_ksize)
        result = _apply_jpeg_roundtrip(result, self.enhanced_jpeg_quality)
        return _apply_bilateral(
            result,
            self.enhanced_deblock_d,
            self.enhanced_deblock_sigma_color,
            self.enhanced_deblock_sigma_space,
        )

    def apply_for_family(
        self,
        img: np.ndarray,
        family_key: str | None = None,
        **params,
    ) -> np.ndarray:
        mode = self._choose_mode(family_key)
        if mode == "ordinary":
            return self._apply_ordinary_mode(img)
        return self._apply_enhanced_mode(img)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply_for_family(img, family_key=None, **params)

    def get_transform_init_args_names(self):
        return (
            "ordinary_mode_probability_non_enhanced",
            "ordinary_mode_probability_enhanced",
            "enhanced_families",
            "ordinary_brightness_limit",
            "ordinary_contrast_limit",
            "ordinary_jpeg_quality",
            "ordinary_deblock_d",
            "ordinary_deblock_sigma_color",
            "ordinary_deblock_sigma_space",
            "ordinary_unsharp_sigma",
            "ordinary_unsharp_amount",
            "enhanced_brightness_limit",
            "enhanced_contrast_limit",
            "enhanced_blur_sigma",
            "enhanced_jpeg_quality",
            "enhanced_deblock_d",
            "enhanced_deblock_sigma_color",
            "enhanced_deblock_sigma_space",
            "chroma_blur_ksize",
        )


class TeamsHybridCodecSimulation(ImageOnlyTransform):
    """
    Sidecar Track B prototype: split policy by family.

    - enhanced fake families keep the legacy blur-heavy simulator
    - ordinary families use the adaptive mixture, which fits the current
      ordinary Teams buckets better than the legacy single-mode preset
    """

    def __init__(
        self,
        enhanced_families: tuple[str, ...] = (
            "visomaster_enhanced_fake",
            "deeplive_enhanced_fake",
        ),
        ordinary_mode_probability_non_enhanced: float = 0.75,
        ordinary_mode_probability_enhanced: float = 0.25,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.enhanced_families = tuple(enhanced_families)
        self.ordinary_mode_probability_non_enhanced = float(
            np.clip(ordinary_mode_probability_non_enhanced, 0.0, 1.0)
        )
        self.ordinary_mode_probability_enhanced = float(
            np.clip(ordinary_mode_probability_enhanced, 0.0, 1.0)
        )
        self._legacy = TeamsCodecSimulation(always_apply=True, p=1.0)
        self._adaptive = TeamsAdaptiveCodecSimulation(
            always_apply=True,
            p=1.0,
            enhanced_families=self.enhanced_families,
            ordinary_mode_probability_non_enhanced=self.ordinary_mode_probability_non_enhanced,
            ordinary_mode_probability_enhanced=self.ordinary_mode_probability_enhanced,
        )

    def apply_for_family(
        self,
        img: np.ndarray,
        family_key: str | None = None,
        **params,
    ) -> np.ndarray:
        if family_key in self.enhanced_families:
            return self._legacy.apply(img)
        return self._adaptive.apply_for_family(img, family_key=family_key)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply_for_family(img, family_key=None, **params)

    def get_transform_init_args_names(self):
        return (
            "enhanced_families",
            "ordinary_mode_probability_non_enhanced",
            "ordinary_mode_probability_enhanced",
        )
