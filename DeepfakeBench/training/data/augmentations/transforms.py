"""
Custom transforms for augmentation pipelines.

These transforms extend Albumentations to provide functionality needed
for deepfake detection training that isn't available in albumentations==0.4.6.
"""

import cv2
import numpy as np
import random
from albumentations.core.transforms_interface import ImageOnlyTransform, BasicTransform


# ---------------------------------------------------------------------------
# Color Temperature (CCT) Simulation  — R12
# ---------------------------------------------------------------------------

def _kelvin_to_rgb(kelvin: int) -> tuple[float, float, float]:
    """Return (R, G, B) gains for a given colour temperature (Tanner Helland)."""
    temp = kelvin / 100.0
    # Red
    if temp <= 66:
        r = 255.0
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = max(0.0, min(255.0, r))
    # Green
    if temp <= 66:
        g = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = max(0.0, min(255.0, g))
    # Blue
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(temp - 10) - 305.0447927307
        b = max(0.0, min(255.0, b))
    return r, g, b

# Pre-compute daylight reference once.
_REF_R, _REF_G, _REF_B = _kelvin_to_rgb(6500)


class ColorTemperatureShift(ImageOnlyTransform):
    """Simulate colour-temperature (CCT) shift via per-channel RGB scaling.

    Uses Tanner Helland's RGB-from-Kelvin approximation.  Gains are
    relative to a 6500 K daylight reference so that 6500 K ≈ identity.

    Lower CCT → warmer (more red/yellow, less blue).
    Higher CCT → cooler (more blue, less red).

    Closes the R/B ratio gap documented in LIGHTING_ROBUSTNESS_REPORT.md:
    training real R/B ~1.6, production webcam ~1.2.

    Args:
        cct_range: (min_K, max_K) — uniform sample range.
                   Default (2700, 8000) covers warm incandescent to overcast.
        always_apply: Whether to always apply.
        p: Probability of applying.
    """

    def __init__(
        self,
        cct_range: tuple[int, int] = (2700, 8000),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.cct_range = cct_range

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        cct = random.randint(self.cct_range[0], self.cct_range[1])
        tgt_r, tgt_g, tgt_b = _kelvin_to_rgb(cct)

        gain_r = tgt_r / _REF_R
        gain_g = tgt_g / _REF_G
        gain_b = tgt_b / max(_REF_B, 1e-6)

        # Albumentations passes images as RGB uint8
        out = image.astype(np.float32)
        out[:, :, 0] *= gain_r  # R
        out[:, :, 1] *= gain_g  # G
        out[:, :, 2] *= gain_b  # B
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple:
        return ("cct_range",)


# ---------------------------------------------------------------------------
# Directional Shadow Simulation
# ---------------------------------------------------------------------------

# Supported shadow directions.  Cardinals cover axis-aligned lighting and
# diagonals cover the far more common real-world case (desk lamp at an
# angle, window to one side, etc.).
_SHADOW_DIRECTIONS = (
    "left", "right", "top", "bottom",
    "top_left", "top_right", "bottom_left", "bottom_right",
)


class DirectionalShadow(ImageOnlyTransform):
    """Simulate a directional shadow gradient across a face crop.

    Production false-positives often come from uneven indoor lighting
    (desk lamp from one side, window behind, etc.).  This transform
    creates a smooth brightness attenuation from one side/corner, forcing
    the model to be invariant to directional illumination.

    The shadow mask is built entirely with vectorised NumPy (no Python
    loops) using ``np.linspace`` + broadcasting.

    Ported from the reference ``apply_shadow()`` in
    ``tools/lighting_showcase.py`` with the following improvements:

    * 8 directions (4 cardinal + 4 diagonal) instead of 4
    * Fully vectorised — no per-pixel Python loop
    * Per-call random sampling of direction, intensity, and softness

    Compatible with **albumentations==0.4.6** ``ImageOnlyTransform``.

    Args:
        intensity_range: (min, max) shadow attenuation.  0 = no shadow,
                         1 = fully black in the shadow region.
        softness_range:  (min, max) fraction of the relevant image
                         dimension used for the gradient transition zone.
        directions:      Tuple of allowed directions to sample from.
        always_apply:    Whether to always apply.
        p:               Probability of applying.
    """

    def __init__(
        self,
        intensity_range: tuple = (0.15, 0.50),
        softness_range: tuple = (0.20, 0.50),
        directions: tuple = _SHADOW_DIRECTIONS,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.intensity_range = intensity_range
        self.softness_range = softness_range
        self.directions = directions

    @staticmethod
    def _make_1d_shadow(length: int, softness_frac: float, intensity: float,
                        invert: bool) -> np.ndarray:
        """Build a 1-D shadow profile of shape ``(length,)``.

        The profile has three zones:
        * **Shadow zone** — constant attenuation at ``(1 - intensity)``
        * **Transition zone** — linear ramp from shadow to full brightness
        * **Lit zone** — constant brightness at 1.0

        When *invert* is True the shadow is on the high-index end instead
        of the low-index end.
        """
        transition = max(1, int(length * softness_frac))
        shadow_len = length - transition

        shadow_val = 1.0 - intensity
        shadow_zone = np.full(shadow_len, shadow_val, dtype=np.float32)
        transition_zone = np.linspace(shadow_val, 1.0, transition, dtype=np.float32)
        profile = np.concatenate([shadow_zone, transition_zone])

        if invert:
            profile = profile[::-1].copy()  # copy to keep C-contiguous
        return profile

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        h, w = image.shape[:2]
        direction = random.choice(self.directions)
        intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
        softness = random.uniform(self.softness_range[0], self.softness_range[1])

        # Build 2-D mask via outer product / broadcasting of 1-D profiles.
        if direction in ("left", "right"):
            # Shadow on left → low-index columns dark, right → high-index dark
            profile_h = self._make_1d_shadow(w, softness, intensity,
                                             invert=(direction == "right"))
            mask = np.ones((h, 1), dtype=np.float32) * profile_h[np.newaxis, :]  # (h, w)

        elif direction in ("top", "bottom"):
            profile_v = self._make_1d_shadow(h, softness, intensity,
                                             invert=(direction == "bottom"))
            mask = profile_v[:, np.newaxis] * np.ones((1, w), dtype=np.float32)  # (h, w)

        else:
            # Diagonal: combine horizontal + vertical profiles, take the
            # element-wise minimum (darkest wins) to simulate corner lighting.
            h_invert = direction.endswith("right")
            v_invert = direction.startswith("bottom")
            profile_h = self._make_1d_shadow(w, softness, intensity, invert=h_invert)
            profile_v = self._make_1d_shadow(h, softness, intensity, invert=v_invert)
            mask_h = np.ones((h, 1), dtype=np.float32) * profile_h[np.newaxis, :]
            mask_v = profile_v[:, np.newaxis] * np.ones((1, w), dtype=np.float32)
            mask = np.minimum(mask_h, mask_v)

        out = image.astype(np.float32) * mask[:, :, np.newaxis]
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple:
        return ("intensity_range", "softness_range", "directions")


# ---------------------------------------------------------------------------
# Gamma-Up (always brighten)
# ---------------------------------------------------------------------------

class GammaUp(ImageOnlyTransform):
    """Apply a gamma correction that **always brightens** the image.

    ``RandomGamma`` from albumentations samples symmetrically around
    gamma=100 (i.e. identity).  We need a dedicated "push brighter"
    transform because the training distribution has mean brightness ~104
    while production (webcam / Teams) reaches 160–205.

    Uses gamma values **< 1** (in the 0-1 scale), which lifts midtones
    and shadows toward white.  Specifically, ``gamma`` is sampled
    uniformly from ``gamma_range`` and applied as::

        out = 255 * (image / 255) ** gamma

    where ``gamma < 1`` → brighter.

    In the albumentations convention (used by ``RandomGamma``),
    gamma=100 is identity.  Here we use the raw 0-1 float convention
    for clarity and precision.

    Compatible with **albumentations==0.4.6** ``ImageOnlyTransform``.

    Args:
        gamma_range: (min, max) gamma values.  Must satisfy
                     0 < min <= max < 1 for the "always brighten"
                     guarantee.
        always_apply: Whether to always apply.
        p:            Probability of applying.
    """

    def __init__(
        self,
        gamma_range: tuple = (0.45, 0.85),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        if gamma_range[0] <= 0 or gamma_range[1] >= 1.0:
            raise ValueError(
                f"gamma_range must be (0, 1) exclusive for brightening; got {gamma_range}"
            )
        self.gamma_range = gamma_range

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        # Build a look-up table for speed (256 entries for uint8).
        lut = np.array(
            [255.0 * ((i / 255.0) ** gamma) for i in range(256)],
            dtype=np.uint8,
        )
        # Apply LUT per channel (cv2.LUT handles 3-channel images).
        return cv2.LUT(image, lut)

    def get_transform_init_args_names(self) -> tuple:
        return ("gamma_range",)


class CustomUnsharpMask(ImageOnlyTransform):
    """
    A custom implementation of UnsharpMask compatible with Albumentations 0.4.6.
    
    This replicates the core logic of the modern UnsharpMask transform by creating
    a blurred version of the image and subtracting it to create a sharpening mask.
    
    Args:
        blur_limit: Range of kernel sizes for Gaussian blur (must be odd integers)
        alpha: Range of sharpening strength multipliers
        threshold: Minimum difference to apply sharpening (reduces noise amplification)
        always_apply: Whether to always apply this transform
        p: Probability of applying this transform
        
    Example:
        >>> transform = CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.5, 1.0), p=0.7)
        >>> result = transform(image=img)['image']
    """

    def __init__(
        self, 
        blur_limit: tuple[int, int] = (3, 9), 
        alpha: tuple[float, float] = (0.5, 1.0), 
        threshold: int = 10, 
        always_apply: bool = False, 
        p: float = 0.5
    ):
        super(CustomUnsharpMask, self).__init__(always_apply, p)
        if blur_limit[0] % 2 == 0 or blur_limit[1] % 2 == 0:
            raise ValueError("blur_limit values must be odd integers.")
        self.blur_limit = blur_limit
        self.alpha = alpha
        self.threshold = threshold

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        # Select random parameters for this specific application
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 2, 2)
        current_alpha = random.uniform(self.alpha[0], self.alpha[1])

        # Create the blurred version of the image using OpenCV's GaussianBlur
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Calculate the high-pass mask (the difference)
        # Convert to float to prevent clipping during subtraction
        image_float = image.astype(np.float32)
        blurred_float = blurred.astype(np.float32)
        mask = image_float - blurred_float

        # Apply the sharpening mask, respecting the threshold to avoid amplifying noise
        if self.threshold > 0:
            apply_condition = np.abs(mask) >= self.threshold
            sharpened_mask = mask * current_alpha
            image_float[apply_condition] += sharpened_mask[apply_condition]
        else:
            image_float += mask * current_alpha

        # Clip values to the valid [0, 255] range and convert back to uint8
        return np.clip(image_float, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> tuple:
        return ("blur_limit", "alpha", "threshold")


class NoOp(BasicTransform):
    """
    A transform that does nothing - returns the image unmodified.
    
    Useful in OneOf blocks where you want a probability of applying no transform.
    
    Example:
        >>> # 70% chance of sharpening, 30% chance of nothing
        >>> A.OneOf([
        ...     CustomUnsharpMask(p=0.7),
        ...     NoOp(p=0.3)
        ... ], p=1.0)
    """

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super(NoOp, self).__init__(always_apply, p)

    @property
    def targets(self) -> dict:
        # This defines what data types (e.g., 'image', 'mask') this transform can handle.
        return {"image": self.apply}

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # The core function: just return the image unmodified.
        return img

    def get_transform_init_args_names(self) -> tuple:
        # Required for serialization, just return an empty tuple.
        return ()


# ==============================================================================
# --- Landmark-Based Occlusion Transforms (Task B) ---
# ==============================================================================

# MediaPipe/DLIB landmark indices for facial regions
# These indices map to the 68-point DLIB model or 478-point MediaPipe model
LANDMARK_REGIONS = {
    # 68-point DLIB landmark indices
    'dlib68': {
        'left_eye': list(range(36, 42)),       # Points 36-41
        'right_eye': list(range(42, 48)),      # Points 42-47
        'nose': list(range(27, 36)),           # Points 27-35
        'mouth': list(range(48, 68)),          # Points 48-67
        'left_eyebrow': list(range(17, 22)),   # Points 17-21
        'right_eyebrow': list(range(22, 27)),  # Points 22-26
        'jawline': list(range(0, 17)),         # Points 0-16
        'forehead': [],  # No direct DLIB points - estimated from eyebrows
    },
    # Key regions for MediaPipe 478-point model
    'mediapipe': {
        'left_eye': [33, 133, 160, 144, 153, 154, 155, 157, 158, 159, 173, 246],
        'right_eye': [362, 263, 387, 373, 380, 381, 382, 384, 385, 386, 398, 466],
        'nose': [1, 2, 3, 4, 5, 195, 197, 168, 6, 122, 196, 351],
        'mouth': [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
                  178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 402, 405, 409, 415],
        'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        'right_eyebrow': [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
    }
}

# Default occlusion targets with their visual significance weights
OCCLUSION_REGIONS = {
    'left_eye': {'weight': 0.25, 'padding_factor': 1.5},
    'right_eye': {'weight': 0.25, 'padding_factor': 1.5},
    'nose': {'weight': 0.15, 'padding_factor': 1.3},
    'mouth': {'weight': 0.25, 'padding_factor': 1.4},
    'left_eyebrow': {'weight': 0.05, 'padding_factor': 1.2},
    'right_eyebrow': {'weight': 0.05, 'padding_factor': 1.2},
}


class LandmarkOcclusion(ImageOnlyTransform):
    """
    Occludes facial regions based on landmark coordinates.
    
    This transform uses pre-computed facial landmarks to intelligently
    occlude specific facial regions (eyes, nose, mouth, eyebrows).
    
    Args:
        regions: List of region names to potentially occlude
            Options: 'left_eye', 'right_eye', 'nose', 'mouth', 'left_eyebrow', 'right_eyebrow'
        num_regions: Number of regions to occlude per image (tuple for random range)
        occlusion_color: RGB color for occlusion (None for random)
        use_ellipse: If True, use ellipse occlusion; if False, use rectangle
        padding_factor: Multiplier to expand occlusion region beyond landmark bounds
        landmark_format: 'dlib68' or 'mediapipe' for landmark index mapping
        always_apply: Whether to always apply this transform
        p: Probability of applying this transform
        
    Example:
        >>> transform = LandmarkOcclusion(
        ...     regions=['left_eye', 'right_eye', 'mouth'],
        ...     num_regions=(1, 2),
        ...     p=0.5
        ... )
        >>> result = transform(image=img, landmarks=landmarks)['image']
    """
    
    def __init__(
        self,
        regions: list[str] = None,
        num_regions: tuple[int, int] = (1, 2),
        occlusion_color: tuple[int, int, int] = None,  # None = random color
        use_ellipse: bool = True,
        padding_factor: float = 1.3,
        landmark_format: str = 'dlib68',
        always_apply: bool = False,
        p: float = 0.5
    ):
        super(LandmarkOcclusion, self).__init__(always_apply, p)
        self.regions = regions or list(OCCLUSION_REGIONS.keys())
        self.num_regions = num_regions
        self.occlusion_color = occlusion_color
        self.use_ellipse = use_ellipse
        self.padding_factor = padding_factor
        self.landmark_format = landmark_format
        
    def apply(self, image: np.ndarray, landmarks: np.ndarray = None, **params) -> np.ndarray:
        """
        Apply landmark-based occlusion to the image.
        
        Args:
            image: Input image (H, W, C)
            landmarks: Landmark coordinates as numpy array of shape (N, 2)
                       where N is number of landmarks (68 for DLIB, 478 for MediaPipe)
        
        Returns:
            Occluded image
        """
        if landmarks is None or len(landmarks) == 0:
            # No landmarks provided - return unmodified
            return image
            
        image = image.copy()
        h, w = image.shape[:2]
        
        # Determine number of regions to occlude
        n_regions = random.randint(self.num_regions[0], self.num_regions[1])
        
        # Sample regions based on weights
        available_regions = [r for r in self.regions if r in OCCLUSION_REGIONS]
        if not available_regions:
            return image
            
        weights = [OCCLUSION_REGIONS[r]['weight'] for r in available_regions]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample without replacement
        n_to_sample = min(n_regions, len(available_regions))
        selected_regions = np.random.choice(
            available_regions, 
            size=n_to_sample, 
            replace=False, 
            p=probs
        )
        
        # Get landmark indices for the format
        landmark_indices = LANDMARK_REGIONS.get(self.landmark_format, LANDMARK_REGIONS['dlib68'])
        
        for region in selected_regions:
            indices = landmark_indices.get(region, [])
            if not indices:
                continue
                
            # Get region bounds from landmarks
            try:
                region_landmarks = landmarks[indices]
                if len(region_landmarks) == 0:
                    continue
                    
                # Calculate bounding box with padding
                x_min, y_min = region_landmarks.min(axis=0)
                x_max, y_max = region_landmarks.max(axis=0)
                
                # Get region-specific padding
                region_padding = OCCLUSION_REGIONS.get(region, {}).get('padding_factor', self.padding_factor)
                
                # Apply padding
                width = x_max - x_min
                height = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                x_min = int(max(0, x_center - width * region_padding / 2))
                x_max = int(min(w, x_center + width * region_padding / 2))
                y_min = int(max(0, y_center - height * region_padding / 2))
                y_max = int(min(h, y_center + height * region_padding / 2))
                
                # Determine occlusion color
                if self.occlusion_color is not None:
                    color = self.occlusion_color
                else:
                    # Random color
                    color = tuple(random.randint(0, 255) for _ in range(3))
                
                # Apply occlusion
                if self.use_ellipse:
                    center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                    axes = (int((x_max - x_min) / 2), int((y_max - y_min) / 2))
                    if axes[0] > 0 and axes[1] > 0:
                        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)
                else:
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, -1)
                    
            except (IndexError, ValueError):
                # Skip if landmarks are invalid
                continue
                
        return image

    def get_params_dependent_on_targets(self, params: dict) -> dict:
        """Pass landmarks through to apply method."""
        landmarks = params.get('landmarks', None)
        return {'landmarks': landmarks}
    
    @property
    def targets_as_params(self) -> list:
        """Declare that we need landmarks as a parameter."""
        return ['landmarks']

    def get_transform_init_args_names(self) -> tuple:
        return ('regions', 'num_regions', 'occlusion_color', 'use_ellipse', 
                'padding_factor', 'landmark_format')


class GaussianBlurOcclusion(ImageOnlyTransform):
    """
    Applies Gaussian blur to facial regions based on landmarks.
    
    Instead of solid color occlusion, this blurs the target region
    to mask details while maintaining some visual continuity.
    
    Args:
        regions: List of region names to potentially blur
        num_regions: Number of regions to blur per image
        blur_strength: Kernel size for Gaussian blur (odd number)
        padding_factor: Multiplier to expand blur region
        landmark_format: 'dlib68' or 'mediapipe'
        always_apply: Whether to always apply
        p: Probability of applying
        
    Example:
        >>> transform = GaussianBlurOcclusion(
        ...     regions=['left_eye', 'right_eye'],
        ...     blur_strength=31,
        ...     p=0.3
        ... )
    """
    
    def __init__(
        self,
        regions: list[str] = None,
        num_regions: tuple[int, int] = (1, 2),
        blur_strength: tuple[int, int] = (21, 51),
        padding_factor: float = 1.3,
        landmark_format: str = 'dlib68',
        always_apply: bool = False,
        p: float = 0.5
    ):
        super(GaussianBlurOcclusion, self).__init__(always_apply, p)
        self.regions = regions or list(OCCLUSION_REGIONS.keys())
        self.num_regions = num_regions
        self.blur_strength = blur_strength
        self.padding_factor = padding_factor
        self.landmark_format = landmark_format
        
    def apply(self, image: np.ndarray, landmarks: np.ndarray = None, **params) -> np.ndarray:
        if landmarks is None or len(landmarks) == 0:
            return image
            
        image = image.copy()
        h, w = image.shape[:2]
        
        n_regions = random.randint(self.num_regions[0], self.num_regions[1])
        available_regions = [r for r in self.regions if r in OCCLUSION_REGIONS]
        if not available_regions:
            return image
            
        weights = [OCCLUSION_REGIONS[r]['weight'] for r in available_regions]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        n_to_sample = min(n_regions, len(available_regions))
        selected_regions = np.random.choice(
            available_regions, size=n_to_sample, replace=False, p=probs
        )
        
        landmark_indices = LANDMARK_REGIONS.get(self.landmark_format, LANDMARK_REGIONS['dlib68'])
        
        for region in selected_regions:
            indices = landmark_indices.get(region, [])
            if not indices:
                continue
                
            try:
                region_landmarks = landmarks[indices]
                if len(region_landmarks) == 0:
                    continue
                    
                x_min, y_min = region_landmarks.min(axis=0)
                x_max, y_max = region_landmarks.max(axis=0)
                
                region_padding = OCCLUSION_REGIONS.get(region, {}).get('padding_factor', self.padding_factor)
                
                width = x_max - x_min
                height = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                x_min = int(max(0, x_center - width * region_padding / 2))
                x_max = int(min(w, x_center + width * region_padding / 2))
                y_min = int(max(0, y_center - height * region_padding / 2))
                y_max = int(min(h, y_center + height * region_padding / 2))
                
                # Get blur kernel size
                ksize = random.randrange(self.blur_strength[0], self.blur_strength[1] + 2, 2)
                if ksize % 2 == 0:
                    ksize += 1
                    
                # Extract region, blur it, and paste back
                region_img = image[y_min:y_max, x_min:x_max]
                if region_img.size > 0:
                    blurred_region = cv2.GaussianBlur(region_img, (ksize, ksize), 0)
                    image[y_min:y_max, x_min:x_max] = blurred_region
                    
            except (IndexError, ValueError):
                continue
                
        return image
    
    def get_params_dependent_on_targets(self, params: dict) -> dict:
        landmarks = params.get('landmarks', None)
        return {'landmarks': landmarks}
    
    @property
    def targets_as_params(self) -> list:
        return ['landmarks']

    def get_transform_init_args_names(self) -> tuple:
        return ('regions', 'num_regions', 'blur_strength', 'padding_factor', 'landmark_format')


class PixelateOcclusion(ImageOnlyTransform):
    """
    Pixelates facial regions based on landmarks.
    
    Creates a mosaic/pixelation effect over target regions
    to obscure details while maintaining structure hints.
    
    Args:
        regions: List of region names to potentially pixelate
        num_regions: Number of regions to pixelate
        pixel_size: Range of pixelation block sizes
        padding_factor: Multiplier to expand region
        landmark_format: 'dlib68' or 'mediapipe'
        always_apply: Whether to always apply
        p: Probability
        
    Example:
        >>> transform = PixelateOcclusion(
        ...     regions=['mouth'],
        ...     pixel_size=(8, 16),
        ...     p=0.2
        ... )
    """
    
    def __init__(
        self,
        regions: list[str] = None,
        num_regions: tuple[int, int] = (1, 2),
        pixel_size: tuple[int, int] = (8, 20),
        padding_factor: float = 1.3,
        landmark_format: str = 'dlib68',
        always_apply: bool = False,
        p: float = 0.5
    ):
        super(PixelateOcclusion, self).__init__(always_apply, p)
        self.regions = regions or list(OCCLUSION_REGIONS.keys())
        self.num_regions = num_regions
        self.pixel_size = pixel_size
        self.padding_factor = padding_factor
        self.landmark_format = landmark_format
        
    def apply(self, image: np.ndarray, landmarks: np.ndarray = None, **params) -> np.ndarray:
        if landmarks is None or len(landmarks) == 0:
            return image
            
        image = image.copy()
        h, w = image.shape[:2]
        
        n_regions = random.randint(self.num_regions[0], self.num_regions[1])
        available_regions = [r for r in self.regions if r in OCCLUSION_REGIONS]
        if not available_regions:
            return image
            
        weights = [OCCLUSION_REGIONS[r]['weight'] for r in available_regions]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        n_to_sample = min(n_regions, len(available_regions))
        selected_regions = np.random.choice(
            available_regions, size=n_to_sample, replace=False, p=probs
        )
        
        landmark_indices = LANDMARK_REGIONS.get(self.landmark_format, LANDMARK_REGIONS['dlib68'])
        
        for region in selected_regions:
            indices = landmark_indices.get(region, [])
            if not indices:
                continue
                
            try:
                region_landmarks = landmarks[indices]
                if len(region_landmarks) == 0:
                    continue
                    
                x_min, y_min = region_landmarks.min(axis=0)
                x_max, y_max = region_landmarks.max(axis=0)
                
                region_padding = OCCLUSION_REGIONS.get(region, {}).get('padding_factor', self.padding_factor)
                
                width = x_max - x_min
                height = y_max - y_min
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                
                x_min = int(max(0, x_center - width * region_padding / 2))
                x_max = int(min(w, x_center + width * region_padding / 2))
                y_min = int(max(0, y_center - height * region_padding / 2))
                y_max = int(min(h, y_center + height * region_padding / 2))
                
                # Get pixelation size
                pix_size = random.randint(self.pixel_size[0], self.pixel_size[1])
                
                # Extract region
                region_img = image[y_min:y_max, x_min:x_max]
                if region_img.size > 0:
                    rh, rw = region_img.shape[:2]
                    if rh > pix_size and rw > pix_size:
                        # Downscale then upscale to create pixelation effect
                        small = cv2.resize(region_img, (rw // pix_size, rh // pix_size), 
                                          interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(small, (rw, rh), 
                                              interpolation=cv2.INTER_NEAREST)
                        image[y_min:y_max, x_min:x_max] = pixelated
                    
            except (IndexError, ValueError):
                continue
                
        return image
    
    def get_params_dependent_on_targets(self, params: dict) -> dict:
        landmarks = params.get('landmarks', None)
        return {'landmarks': landmarks}
    
    @property
    def targets_as_params(self) -> list:
        return ['landmarks']

    def get_transform_init_args_names(self) -> tuple:
        return ('regions', 'num_regions', 'pixel_size', 'padding_factor', 'landmark_format')


# ==============================================================================
# --- Region BBox-Based Occlusion (for DeepLive GCS landmark format) ---
# ==============================================================================

# MediaPipe 478-point landmark indices for facial regions
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
MEDIAPIPE_REGION_INDICES = {
    'left_eye': [
        # Left eye contour
        33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        # Left iris
        468, 469, 470, 471, 472,
    ],
    'right_eye': [
        # Right eye contour  
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
        # Right iris
        473, 474, 475, 476, 477,
    ],
    'nose': [
        # Nose bridge and tip
        1, 2, 3, 4, 5, 6, 168, 197, 195, 5,
        # Nose bottom
        94, 19, 1, 274, 457,
        # Nostrils
        97, 98, 64, 294, 327, 326,
    ],
    'mouth': [
        # Outer lip
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
        # Inner lip
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
    ],
    'left_eyebrow': [
        # Left eyebrow
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    ],
    'right_eyebrow': [
        # Right eyebrow
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    ],
    'forehead': [
        # Forehead area (approximate using top face landmarks)
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
        103, 67, 109, 10,
    ],
    'chin': [
        # Chin/jaw area
        152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    ],
}


class RegionBBoxOcclusion(ImageOnlyTransform):
    """
    Occludes facial regions using MediaPipe 478-point landmarks.
    
    This transform computes bounding boxes dynamically from the raw MediaPipe
    landmark coordinates (478 points with normalized x, y, z).
    
    Expected landmark formats:
    
    1. DeepLive GCS format (dict with 'landmarks' array):
        {
            "landmarks": [{"x": 0.43, "y": 0.45, "z": -0.02}, ...],  # 478 points
            "face_detected": true,
            ...
        }
    
    2. DeepLiveLandmark object with 'landmarks' list attribute
    
    3. Numpy array of shape (478, 2) or (478, 3) with normalized coordinates
    
    4. List of dicts with 'x', 'y' keys
    
    Coordinates are normalized [0, 1] relative to image dimensions.
    
    Args:
        regions: List of region names to potentially occlude.
            Options: 'left_eye', 'right_eye', 'nose', 'mouth', 
                     'left_eyebrow', 'right_eyebrow', 'forehead', 'chin'
        num_regions: Range of regions to occlude per image (min, max)
        occlusion_type: Type of occlusion to apply
            - 'solid': Solid color fill (random or specified)
            - 'blur': Gaussian blur
            - 'pixelate': Pixelation effect
            - 'noise': Random noise fill
        occlusion_color: RGB color for solid occlusion (None = random)
        blur_strength: Kernel size range for blur (must be odd)
        pixel_size: Block size range for pixelation
        padding_factor: Multiplier to expand occlusion region
        use_ellipse: Use ellipse shape (True) or rectangle (False)
        blend_edges: Blend occlusion edges for smoother transition
        always_apply: Whether to always apply this transform
        p: Probability of applying this transform
        
    Example:
        >>> transform = RegionBBoxOcclusion(
        ...     regions=['left_eye', 'right_eye', 'mouth'],
        ...     num_regions=(1, 2),
        ...     occlusion_type='blur',
        ...     p=0.5
        ... )
        >>> # Pass landmark data from DeepLive GCS
        >>> result = transform(image=img, landmarks=frame_landmark_data)['image']
    """
    
    def __init__(
        self,
        regions: list[str] = None,
        num_regions: tuple[int, int] = (1, 2),
        occlusion_type: str = 'solid',  # 'solid', 'blur', 'pixelate', 'noise'
        occlusion_color: tuple[int, int, int] = None,
        blur_strength: tuple[int, int] = (21, 51),
        pixel_size: tuple[int, int] = (8, 16),
        padding_factor: float = 1.3,
        use_ellipse: bool = True,
        blend_edges: bool = False,
        always_apply: bool = False,
        p: float = 0.5
    ):
        super(RegionBBoxOcclusion, self).__init__(always_apply, p)
        self.regions = regions or ['left_eye', 'right_eye', 'mouth', 'nose']
        self.num_regions = num_regions
        self.occlusion_type = occlusion_type
        self.occlusion_color = occlusion_color
        self.blur_strength = blur_strength
        self.pixel_size = pixel_size
        self.padding_factor = padding_factor
        self.use_ellipse = use_ellipse
        self.blend_edges = blend_edges
    
    def _extract_landmarks_array(self, landmarks) -> np.ndarray:
        """
        Extract normalized (x, y) coordinates from various landmark formats.
        
        Returns:
            numpy array of shape (N, 2) with normalized x, y coordinates
        """
        # Already numpy array
        if isinstance(landmarks, np.ndarray):
            if landmarks.shape[-1] >= 2:
                return landmarks[:, :2]
            return landmarks
        
        # DeepLiveLandmark object or dict with frame data
        if hasattr(landmarks, 'landmarks'):
            lm_data = landmarks.landmarks
        elif isinstance(landmarks, dict):
            if 'landmarks' in landmarks:
                lm_data = landmarks['landmarks']
            else:
                return None
        elif isinstance(landmarks, list):
            lm_data = landmarks
        else:
            return None
        
        # Convert list of dicts to array
        if lm_data and isinstance(lm_data[0], dict):
            coords = [(p.get('x', 0), p.get('y', 0)) for p in lm_data]
            return np.array(coords, dtype=np.float32)
        
        return np.array(lm_data, dtype=np.float32)[:, :2] if lm_data else None
    
    def _get_region_bbox(
        self, 
        landmarks_array: np.ndarray, 
        region: str, 
        img_shape: tuple
    ) -> tuple:
        """
        Compute bounding box for a region from landmark coordinates.
        
        Args:
            landmarks_array: (N, 2) array of normalized x, y coordinates
            region: Region name
            img_shape: (height, width) of image
            
        Returns:
            (x_min, y_min, x_max, y_max) in pixel coordinates, or None
        """
        h, w = img_shape[:2]
        
        # Get landmark indices for this region
        indices = MEDIAPIPE_REGION_INDICES.get(region)
        if indices is None:
            return None
        
        # Filter valid indices
        valid_indices = [i for i in indices if i < len(landmarks_array)]
        if not valid_indices:
            return None
        
        # Get coordinates for region landmarks
        region_coords = landmarks_array[valid_indices]
        
        # Compute bounding box (normalized)
        x_min, y_min = region_coords.min(axis=0)
        x_max, y_max = region_coords.max(axis=0)
        
        # Apply padding
        width = x_max - x_min
        height = y_max - y_min
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        x_min = max(0, x_center - width * self.padding_factor / 2)
        x_max = min(1, x_center + width * self.padding_factor / 2)
        y_min = max(0, y_center - height * self.padding_factor / 2)
        y_max = min(1, y_center + height * self.padding_factor / 2)
        
        # Convert to pixel coordinates
        return (
            int(x_min * w),
            int(y_min * h),
            int(x_max * w),
            int(y_max * h)
        )
    
    def _apply_solid_occlusion(
        self, 
        image: np.ndarray, 
        x_min: int, y_min: int, 
        x_max: int, y_max: int
    ) -> np.ndarray:
        """Apply solid color occlusion."""
        if self.occlusion_color is not None:
            color = self.occlusion_color
        else:
            color = tuple(random.randint(0, 255) for _ in range(3))
        
        if self.use_ellipse:
            center = ((x_min + x_max) // 2, (y_min + y_max) // 2)
            axes = ((x_max - x_min) // 2, (y_max - y_min) // 2)
            if axes[0] > 0 and axes[1] > 0:
                cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, -1)
        
        return image
    
    def _apply_blur_occlusion(
        self, 
        image: np.ndarray, 
        x_min: int, y_min: int, 
        x_max: int, y_max: int
    ) -> np.ndarray:
        """Apply Gaussian blur occlusion."""
        ksize = random.randrange(self.blur_strength[0], self.blur_strength[1] + 2, 2)
        
        # Ensure kernel size is odd
        if ksize % 2 == 0:
            ksize += 1
        
        region = image[y_min:y_max, x_min:x_max]
        if region.size > 0:
            blurred = cv2.GaussianBlur(region, (ksize, ksize), 0)
            
            if self.use_ellipse and self.blend_edges:
                # Create ellipse mask for blending
                mask = np.zeros(region.shape[:2], dtype=np.float32)
                center = ((x_max - x_min) // 2, (y_max - y_min) // 2)
                axes = ((x_max - x_min) // 2, (y_max - y_min) // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                mask = mask[:, :, np.newaxis]
                blended = (blurred * mask + region * (1 - mask)).astype(np.uint8)
                image[y_min:y_max, x_min:x_max] = blended
            else:
                image[y_min:y_max, x_min:x_max] = blurred
        
        return image
    
    def _apply_pixelate_occlusion(
        self, 
        image: np.ndarray, 
        x_min: int, y_min: int, 
        x_max: int, y_max: int
    ) -> np.ndarray:
        """Apply pixelation occlusion."""
        pix_size = random.randint(self.pixel_size[0], self.pixel_size[1])
        
        region = image[y_min:y_max, x_min:x_max]
        if region.size > 0:
            rh, rw = region.shape[:2]
            if rh > pix_size and rw > pix_size:
                small = cv2.resize(
                    region, 
                    (max(1, rw // pix_size), max(1, rh // pix_size)),
                    interpolation=cv2.INTER_LINEAR
                )
                pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
                image[y_min:y_max, x_min:x_max] = pixelated
        
        return image
    
    def _apply_noise_occlusion(
        self, 
        image: np.ndarray, 
        x_min: int, y_min: int, 
        x_max: int, y_max: int
    ) -> np.ndarray:
        """Apply random noise occlusion."""
        region = image[y_min:y_max, x_min:x_max]
        if region.size > 0:
            noise = np.random.randint(0, 256, region.shape, dtype=np.uint8)
            
            if self.use_ellipse and self.blend_edges:
                mask = np.zeros(region.shape[:2], dtype=np.float32)
                center = ((x_max - x_min) // 2, (y_max - y_min) // 2)
                axes = ((x_max - x_min) // 2, (y_max - y_min) // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
                mask = mask[:, :, np.newaxis]
                blended = (noise * mask + region * (1 - mask)).astype(np.uint8)
                image[y_min:y_max, x_min:x_max] = blended
            else:
                image[y_min:y_max, x_min:x_max] = noise
        
        return image
        
    def apply(self, image: np.ndarray, landmarks=None, **params) -> np.ndarray:
        """
        Apply region-based occlusion to the image.
        
        Args:
            image: Input image (H, W, C)
            landmarks: MediaPipe 478-point landmarks in various formats:
                - DeepLiveLandmark object with 'landmarks' attribute
                - Dict with 'landmarks' key containing [{x, y, z}, ...]
                - List of dicts [{x, y, z}, ...]
                - Numpy array of shape (478, 2) or (478, 3)
        
        Returns:
            Occluded image
        """
        if landmarks is None:
            return image
        
        # Check if face was detected
        if hasattr(landmarks, 'face_detected') and not landmarks.face_detected:
            return image
        if isinstance(landmarks, dict) and not landmarks.get('face_detected', True):
            return image
        
        # Extract landmarks array from various formats
        landmarks_array = self._extract_landmarks_array(landmarks)
        if landmarks_array is None or len(landmarks_array) == 0:
            return image
        
        image = image.copy()
        h, w = image.shape[:2]
        
        # Determine number of regions to occlude
        n_regions = random.randint(self.num_regions[0], self.num_regions[1])
        
        # Filter available regions (those we can compute bboxes for)
        available_regions = []
        for region in self.regions:
            bbox = self._get_region_bbox(landmarks_array, region, (h, w))
            if bbox is not None:
                available_regions.append((region, bbox))
        
        if not available_regions:
            return image
        
        # Sample regions (with weights)
        weights = [OCCLUSION_REGIONS.get(r, {}).get('weight', 0.1) for r, _ in available_regions]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        n_to_sample = min(n_regions, len(available_regions))
        indices = np.random.choice(
            len(available_regions),
            size=n_to_sample,
            replace=False,
            p=probs
        )
        
        # Apply occlusion to selected regions
        for idx in indices:
            region_name, (x_min, y_min, x_max, y_max) = available_regions[idx]
            
            # Apply the selected occlusion type
            if self.occlusion_type == 'blur':
                image = self._apply_blur_occlusion(image, x_min, y_min, x_max, y_max)
            elif self.occlusion_type == 'pixelate':
                image = self._apply_pixelate_occlusion(image, x_min, y_min, x_max, y_max)
            elif self.occlusion_type == 'noise':
                image = self._apply_noise_occlusion(image, x_min, y_min, x_max, y_max)
            else:  # 'solid' or default
                image = self._apply_solid_occlusion(image, x_min, y_min, x_max, y_max)
        
        return image
    
    def get_params_dependent_on_targets(self, params: dict) -> dict:
        """Pass landmarks through to apply method."""
        landmarks = params.get('landmarks', None)
        return {'landmarks': landmarks}
    
    @property
    def targets_as_params(self) -> list:
        """Declare that we need landmarks as a parameter."""
        return ['landmarks']

    def get_transform_init_args_names(self) -> tuple:
        return ('regions', 'num_regions', 'occlusion_type', 'occlusion_color',
                'blur_strength', 'pixel_size', 'padding_factor', 'use_ellipse', 
                'blend_edges')


# ==============================================================================
# --- Video Codec Simulation (Phase 1 → Augmentation Design) ---
# ==============================================================================

class VideoCodecSimulation(ImageOnlyTransform):
    """
    Simulate video-call / webcam codec artifacts as a single coherent transform.

    Motivation (Phase 1 quality fingerprint analysis):
        VCD Zoom reals show a characteristic "codec fingerprint": edges are
        softened (low Tenengrad/edge density) but high-frequency FFT energy is
        ELEVATED due to codec noise (ringing, mosquito noise, quantization grain).
        This creates a frequency profile (flat PSD slope, high freq_high_ratio)
        that the model confuses with deepfake artifacts.

        Existing augmentations (GaussianBlur, GaussNoise, ImageCompression)
        applied independently don't reproduce this *coherent* pattern. This
        transform chains the effects in the order a real video codec produces
        them, creating the characteristic webcam fingerprint.

    Pipeline (mirrors real codec behavior):
        1. Resolution reduction  — lower capture resolution (downscale → upscale)
        2. Bilateral filter      — deblocking/loop filter (smooths flat regions,
                                   partially preserves edges)
        3. Block quantization    — subtle 8×8/16×16 block boundary discontinuities
        4. Codec noise injection — frequency-shaped noise (more energy in mid/high
                                   bands, mimicking ringing and mosquito noise)
        5. JPEG compression      — I-frame coding simulation

    All parameters are randomly sampled per-image within configured ranges
    to produce diverse but realistic codec degradation levels.

    Compatible with albumentations 0.4.6.

    Args:
        codec_quality: (min, max) overall codec quality level [0=worst, 100=best].
            Controls the intensity of ALL sub-effects proportionally.
        downscale_range: (min, max) scale factor for resolution reduction step.
        bilateral_d: (min, max) diameter for bilateral filter.
        bilateral_sigma_color: (min, max) sigma for color space filtering.
        bilateral_sigma_space: (min, max) sigma for coordinate space filtering.
        block_size: Block size for quantization artifacts (8 or 16).
        block_strength: (min, max) strength of block boundary discontinuities.
        codec_noise_std: (min, max) std-dev of frequency-shaped codec noise.
        jpeg_quality: (min, max) JPEG quality for I-frame simulation.
            If None, derived from codec_quality.
        always_apply: Force application regardless of probability.
        p: Probability of applying this transform.
    """

    def __init__(
        self,
        codec_quality=(30, 80),
        downscale_range=(0.5, 0.85),
        bilateral_d=(5, 11),
        bilateral_sigma_color=(40, 90),
        bilateral_sigma_space=(40, 90),
        block_size=8,
        block_strength=(0.3, 1.5),
        codec_noise_std=(3.0, 12.0),
        jpeg_quality=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.codec_quality = codec_quality
        self.downscale_range = downscale_range
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.block_size = block_size
        self.block_strength = block_strength
        self.codec_noise_std = codec_noise_std
        self.jpeg_quality = jpeg_quality

    def apply(self, img, **params):
        """Apply the full codec simulation chain."""
        h, w = img.shape[:2]
        result = img.copy()

        # Sample a global quality level that drives all sub-effects
        q = random.uniform(self.codec_quality[0], self.codec_quality[1])
        # Normalized severity: 1.0 = worst quality, 0.0 = best quality
        severity = 1.0 - (q / 100.0)

        # ── Step 1: Resolution reduction ──
        # Lower codec_quality → more aggressive downscale
        scale = random.uniform(
            self.downscale_range[0] + (1.0 - severity) * 0.1,
            self.downscale_range[1],
        )
        scale = min(scale, 1.0)
        if scale < 0.95:
            new_h, new_w = max(16, int(h * scale)), max(16, int(w * scale))
            # Downscale with AREA (anti-aliased), upscale with CUBIC or LINEAR
            # (simulates webcam sensor → display pipeline)
            small = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)
            interp = random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])
            result = cv2.resize(small, (w, h), interpolation=interp)

        # ── Step 2: Bilateral filter (deblocking) ──
        # Simulates H.264/H.265 in-loop deblocking filter: smooths flat
        # regions while partially preserving edges
        d = random.randint(self.bilateral_d[0], self.bilateral_d[1])
        if d % 2 == 0:
            d += 1  # must be odd
        sigma_c = random.uniform(
            self.bilateral_sigma_color[0],
            self.bilateral_sigma_color[1],
        )
        sigma_s = random.uniform(
            self.bilateral_sigma_space[0],
            self.bilateral_sigma_space[1],
        )
        # Scale filter strength by severity
        sigma_c *= (0.5 + 0.5 * severity)
        sigma_s *= (0.5 + 0.5 * severity)
        result = cv2.bilateralFilter(result, d, sigma_c, sigma_s)

        # ── Step 3: Block quantization artifacts ──
        # Add subtle discontinuities at block boundaries (8×8 for H.264,
        # 16×16 for H.265). Strength proportional to severity.
        bs = self.block_size
        strength = random.uniform(self.block_strength[0], self.block_strength[1])
        strength *= severity
        if strength > 0.05 and h > bs * 2 and w > bs * 2:
            result_f = result.astype(np.float32)
            # Horizontal block boundaries
            for col in range(bs, w - 1, bs):
                noise_col = np.random.normal(0, strength, (h, 1, result.shape[2] if result.ndim == 3 else 1))
                if result.ndim == 2:
                    noise_col = noise_col[:, :, 0]
                result_f[:, col:col+1] += noise_col.astype(np.float32)
            # Vertical block boundaries
            for row in range(bs, h - 1, bs):
                noise_row = np.random.normal(0, strength, (1, w, result.shape[2] if result.ndim == 3 else 1))
                if result.ndim == 2:
                    noise_row = noise_row[:, :, 0]
                result_f[row:row+1, :] += noise_row.astype(np.float32)
            result = np.clip(result_f, 0, 255).astype(np.uint8)

        # ── Step 4: Codec noise (frequency-shaped) ──
        # Real codecs add noise concentrated in mid/high frequencies
        # (ringing around edges, mosquito noise in textured regions).
        # We simulate this with Gaussian noise filtered to emphasize
        # higher frequencies (no DC/low component).
        noise_std = random.uniform(self.codec_noise_std[0], self.codec_noise_std[1])
        noise_std *= severity
        if noise_std > 0.5:
            noise = np.random.normal(0, noise_std, result.shape).astype(np.float32)
            # High-pass filter the noise: subtract blurred version
            # This removes DC/low-freq component, keeping mid+high freq
            blur_k = random.choice([3, 5, 7])
            noise_lf = cv2.GaussianBlur(noise, (blur_k, blur_k), 0)
            noise_hf = noise - noise_lf
            # Scale to desired std after filtering
            current_std = noise_hf.std()
            if current_std > 0.1:
                noise_hf = noise_hf * (noise_std / current_std)
            result = np.clip(result.astype(np.float32) + noise_hf, 0, 255).astype(np.uint8)

        # ── Step 5: JPEG compression (I-frame simulation) ──
        if self.jpeg_quality is not None:
            jpeg_q = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
        else:
            # Derive from codec_quality: lower codec_q → lower JPEG quality
            jpeg_q = int(q * 0.8 + 10)  # maps [30,80] → [34,74]
            jpeg_q = max(25, min(95, jpeg_q))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]
        _, enc = cv2.imencode('.jpg', result, encode_param)
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR if result.ndim == 3 else cv2.IMREAD_GRAYSCALE)
        # imencode/imdecode uses BGR, but albumentations passes RGB
        # The encode→decode roundtrip preserves channel order since we
        # don't convert, we just compress and decompress the channel bytes.

        return result

    def get_transform_init_args_names(self) -> tuple:
        return (
            'codec_quality', 'downscale_range',
            'bilateral_d', 'bilateral_sigma_color', 'bilateral_sigma_space',
            'block_size', 'block_strength',
            'codec_noise_std', 'jpeg_quality',
        )


