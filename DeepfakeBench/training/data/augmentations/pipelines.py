"""
Pre-defined augmentation pipelines for deepfake detection training.

Pipeline Versions:
- V3: Legacy, conservative augmentations (original production)
- V4: Moderately aggressive (step-up from V3)
- V5: Hybrid (V4 + occasional heavy degradation)
- V6: Source-dependent portfolio (different augs for different data sources)
- V7: Unified portfolio (simplified V6 without source checking)
- Surgical: Dynamic property-based augmentations
- quality_targeted_family: Family-aware quality routing for Phase 4

All pipelines are compatible with albumentations==0.4.6
"""

import cv2
import random
import numpy as np
import albumentations as A

from .transforms import CustomUnsharpMask, NoOp, VideoCodecSimulation


# ==============================================================================
# --- Augmentation Pipeline V3 (Legacy, Compatible with albumentations==0.4.6) ---
# ==============================================================================

def create_v3_pipeline() -> A.Compose:
    """
    Original production pipeline - conservative augmentations.
    
    Characteristics:
    - Mild sharpening (UnsharpMask p=0.7)
    - Light compression/noise (p=0.6)
    - Gentle color shifts
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        # Step 1: CALIBRATED Quality Transformation
        CustomUnsharpMask(
            blur_limit=(3, 9),
            alpha=(0.5, 1.0),
            threshold=10,
            p=0.7
        ),

        # Step 2: Realistic Compression & Noise
        A.OneOf([
            A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
            A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ], p=0.6),

        # Step 3: GENTLE Color Augmentation
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
    ])


# Pre-instantiated for backward compatibility
revised_augmentation_pipeline_legacy = create_v3_pipeline()


# ==============================================================================
# --- Augmentation Pipeline V4 (Moderately Aggressive) ---
# ==============================================================================

def create_v4_pipeline() -> A.Compose:
    """
    Moderately aggressive pipeline - step-up from V3.
    
    Characteristics:
    - Stronger sharpening (alpha up to 1.2, p=0.75)
    - Higher compression/noise probability (p=0.7)
    - Wider color shift ranges
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),

        # Step 1: CALIBRATED Quality Transformation
        CustomUnsharpMask(
            blur_limit=(3, 9),
            alpha=(0.6, 1.2),  # Increased sharpening strength
            threshold=10,
            p=0.75  # Increased from 0.7
        ),

        # Step 2: CALIBRATED Compression & Noise
        A.OneOf([
            A.ImageCompression(quality_lower=45, quality_upper=90, p=0.5),
            A.GaussNoise(var_limit=(10.0, 65.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        ], p=0.7),  # Increased from 0.6

        # Step 3: CALIBRATED Color Augmentation
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
        A.HueSaturationValue(
            hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=12, p=0.45
        ),
    ])


# Pre-instantiated for backward compatibility
augmentation_pipeline_v4 = create_v4_pipeline()


# ==============================================================================
# --- Augmentation Pipeline V5 (Hybrid: Moderate + Heavy Degradation) ---
# ==============================================================================

def _create_degradation_block() -> A.Compose:
    """Heavy degradation block used in V5 pipeline."""
    return A.Compose([
        A.OneOf([
            A.Compose([
                A.Downscale(scale_min=0.3, scale_max=0.6, interpolation=cv2.INTER_AREA, p=0.8),
                A.Resize(height=224, width=224, interpolation=cv2.INTER_LINEAR, always_apply=True)
            ], p=0.7),
            NoOp(p=0.3)
        ], p=1.0),
        A.ImageCompression(quality_lower=25, quality_upper=70, p=0.7),
        A.GaussianBlur(blur_limit=(3, 11), p=0.4),
    ])


def create_v5_pipeline() -> A.Compose:
    """
    Hybrid pipeline - V4 augmentations + occasional heavy degradation.
    
    Characteristics:
    - Applies V4 augmentations to every image
    - 50% chance of additional heavy degradation (downscale + strong compression)
    """
    degradation_block = _create_degradation_block()
    v4_pipeline = create_v4_pipeline()
    
    return A.Compose([
        # First, apply the moderate augmentations
        v4_pipeline,

        # THEN, apply heavy degradation with 50% probability
        A.OneOf([
            A.Compose([degradation_block], p=0.5),
            NoOp(p=0.5)
        ], p=1.0)
    ])


# Pre-instantiated for backward compatibility
augmentation_pipeline_v5 = create_v5_pipeline()


# ==============================================================================
# --- Augmentation Strategy V6 Components ---
# ==============================================================================

def create_v6_simulator_pipeline() -> A.Compose:
    """Social media simulator - heavy compression and processing."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.7, 1.5), threshold=10, p=0.9),
        A.ImageCompression(quality_lower=40, quality_upper=85, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    ])


def create_v6_generalist_pipeline() -> A.Compose:
    """Generalist 'kitchen sink' augmentations."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.6, 1.2), threshold=10, p=0.75),
        A.OneOf([
            A.ImageCompression(quality_lower=45, quality_upper=90, p=0.5),
            A.GaussNoise(var_limit=(10.0, 65.0), p=0.3),
        ], p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    ])


def create_v6_purist_pipeline() -> A.Compose:
    """Minimal augmentations - preserve original quality."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    ])


def create_v6_mild_pipeline() -> A.Compose:
    """Mild augmentations for Train-Effort data."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        CustomUnsharpMask(blur_limit=(3, 7), alpha=(0.2, 0.7), threshold=10, p=0.5),
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.5)
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ])


# Pre-instantiated V6 component pipelines
AUG_PIPELINE_V6_SIMULATOR = create_v6_simulator_pipeline()
AUG_PIPELINE_V4_GENERALIST = create_v6_generalist_pipeline()
AUG_PIPELINE_PURIST = create_v6_purist_pipeline()
AUG_PIPELINE_V3_MILD = create_v6_mild_pipeline()


def apply_augmentation_v6(img_np: np.ndarray, frame_dict: dict) -> np.ndarray:
    """
    Applies source-dependent, probabilistic augmentation strategy (V6).
    
    This function determines the data source from the frame path and applies
    the corresponding augmentation portfolio:
    - Train-Primary (df40-frames-recropped-rfa85): Portfolio selection
    - Train-Effort (other sources): Mild augmentations
    
    Args:
        img_np: Input image as numpy array (H, W, C), uint8
        frame_dict: Dictionary containing 'path' key with frame source path
        
    Returns:
        Augmented image as numpy array
    """
    frame_path = frame_dict.get('path', '')

    # Check if the source is "Train-Primary"
    if "df40-frames-recropped-rfa85" in frame_path:
        # Apply the "Portfolio" Augmentation for Train-Primary data
        pipelines = [AUG_PIPELINE_V6_SIMULATOR, AUG_PIPELINE_V4_GENERALIST, AUG_PIPELINE_PURIST]
        weights = [0.4, 0.3, 0.3]
        chosen_pipeline = random.choices(pipelines, weights=weights, k=1)[0]
        return chosen_pipeline(image=img_np)['image']
    else:
        # Apply the "Mild" Augmentation for "Train-Effort" data
        if random.random() < 0.5:
            return AUG_PIPELINE_V3_MILD(image=img_np)['image']
        else:
            return A.HorizontalFlip(p=0.5)(image=img_np)['image']


def apply_augmentation_v7(img_np: np.ndarray) -> np.ndarray:
    """
    Unified portfolio augmentation (V7).
    
    Simplified version of V6 without source-dependent logic.
    Randomly selects from portfolio of pipelines.
    
    Args:
        img_np: Input image as numpy array (H, W, C), uint8
        
    Returns:
        Augmented image as numpy array
    """
    pipelines = [AUG_PIPELINE_V6_SIMULATOR, AUG_PIPELINE_V4_GENERALIST, AUG_PIPELINE_PURIST]
    weights = [0.4, 0.45, 0.15]
    chosen_pipeline = random.choices(pipelines, weights=weights, k=1)[0]
    return chosen_pipeline(image=img_np)['image']


# ==============================================================================
# --- Helper Pipelines (for surgical augmentation) ---
# ==============================================================================

def create_degrade_quality_pipeline() -> A.Compose:
    """Pipeline to aggressively degrade image quality."""
    return A.Compose([
        A.OneOf([
            A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8),
            A.GaussianBlur(blur_limit=(5, 11), p=0.6),
            A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
        ], p=1.0)
    ])


def create_enhance_quality_pipeline() -> A.Compose:
    """Pipeline to enhance/sharpen image quality."""
    return A.Compose([
        A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
    ])


def create_social_media_pipeline() -> A.Compose:
    """Pipeline to simulate social media compression artifacts."""
    return A.Compose([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_AREA, p=0.8),
        A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
    ], p=1.0)


# Pre-instantiated helper pipelines
degrade_quality_pipeline = create_degrade_quality_pipeline()
enhance_quality_pipeline = create_enhance_quality_pipeline()
social_media_pipeline = create_social_media_pipeline()


# ==============================================================================
# --- Surgical Augmentation Pipeline ---
# ==============================================================================

def create_surgical_augmentation_pipeline(
    config: dict,
    frame_properties: dict
) -> A.Compose:
    """
    Dynamically constructs an Albumentations pipeline based on configuration
    and specific frame properties.
    
    This allows property-aware augmentation where transforms are selected
    based on the characteristics of each individual frame (e.g., sharpness bucket).
    
    Args:
        config: Dictionary with augmentation configuration:
            - use_geometric: bool - Add rotation/scale transforms
            - use_color_jitter: bool - Add color augmentations
            - sharpness_adjust_prob: float - Probability of sharpness-based adjustment
            - use_advanced_noise: bool - Add noise/artifact simulation
            - advanced_noise_prob: float - Probability of advanced noise
            - use_occlusion: bool - Add cutout augmentation
            - occlusion_prob: float - Probability of cutout
        frame_properties: Dictionary with frame-level properties:
            - sharpness_bucket: str - One of 'q1', 'q2', 'q3', 'q4'
            
    Returns:
        A.Compose pipeline tailored to the frame
    """
    transforms_list = []

    # 1. Base & Geometric
    transforms_list.append(A.HorizontalFlip(p=0.5))
    if config.get('use_geometric', False):
        transforms_list.append(A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.12, rotate_limit=7,
            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.7
        ))

    # Color jitter to maintain robustness
    if config.get('use_color_jitter', False):
        transforms_list.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
            )
        ])

    # 2. Surgical Sharpness Adjustment
    sharpness_bucket = frame_properties.get('sharpness_bucket')
    chance_for_sharpness_adjustment = config.get('sharpness_adjust_prob', 0.5)
    if random.random() < chance_for_sharpness_adjustment:
        if sharpness_bucket == 'q4':
            # Sharp images: degrade to teach robustness
            transforms_list.append(degrade_quality_pipeline)
        elif sharpness_bucket == 'q1':
            # Blurry images: enhance to diversify
            transforms_list.append(enhance_quality_pipeline)

    # 3. Advanced Noise & Artifact Simulation
    if config.get('use_advanced_noise', False):
        transforms_list.append(A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            social_media_pipeline
        ], p=config.get('advanced_noise_prob', 0.6)))

    # 4. Occlusion
    if config.get('use_occlusion', False):
        transforms_list.append(A.Cutout(
            num_holes=8, max_h_size=24, max_w_size=24, fill_value=0,
            p=config.get('occlusion_prob', 0.5)
        ))

    return A.Compose(transforms_list)


def create_general_augmentation_pipeline(config: dict) -> A.Compose:
    """
    Creates a robust, general-purpose augmentation pipeline for strategies
    that do not have access to frame-level properties.
    
    Supports selecting different augmentation versions via config.
    
    Args:
        config: Dictionary with augmentation configuration:
            - augmentation_params.version: int (3, 4, or 5) for pre-defined pipelines
            - augmentation_params.use_geometric: bool
            - augmentation_params.use_color_jitter: bool
            - augmentation_params.use_occlusion: bool
            - augmentation_params.occlusion_prob: float
            
    Returns:
        A.Compose pipeline
    """
    # Check for the augmentation version from the dedicated params dictionary
    aug_params = config.get('augmentation_params', {})
    aug_version = aug_params.get('version')
    
    if aug_version == 3:
        return create_v3_pipeline()
    elif aug_version == 4:
        return create_v4_pipeline()
    elif aug_version == 5:
        return create_v5_pipeline()

    # Fallback to configurable "general" logic if version is not specified
    transforms_list = [
        A.HorizontalFlip(p=0.5),
    ]
    
    if aug_params.get('use_geometric', False):
        transforms_list.append(A.ShiftScaleRotate(
            shift_limit=0.0625, scale_limit=0.12, rotate_limit=7,
            interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.7
        ))
    
    if aug_params.get('use_color_jitter', False):
        transforms_list.extend([
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
            )
        ])
    
    transforms_list.append(A.OneOf([
        A.ImageCompression(quality_lower=50, quality_upper=80, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    ], p=0.8))
    
    if aug_params.get('use_occlusion', False):
        transforms_list.append(A.Cutout(
            num_holes=8, max_h_size=24, max_w_size=24, fill_value=0,
            p=aug_params.get('occlusion_prob', 0.5)
        ))
    
    return A.Compose(transforms_list)


# ==============================================================================
# --- Landmark Occlusion Pipeline (Task B: DeepLive) ---
# ==============================================================================

def create_landmark_occlusion_pipeline(
    occlusion_type: str = 'solid',
    regions: list[str] = None,
    num_regions: tuple[int, int] = (1, 2),
    occlusion_prob: float = 0.5,
    landmark_format: str = 'dlib68',
    additional_augs: bool = False
) -> A.Compose:
    """
    Creates a pipeline with landmark-based intelligent occlusions.
    
    This is the primary augmentation pipeline for Task B (DeepLive dataset).
    It uses facial landmarks to occlude specific facial regions (eyes, nose, mouth)
    rather than random rectangular cutouts.
    
    Args:
        occlusion_type: Type of occlusion to apply
            - 'solid': Solid color fill (default)
            - 'blur': Gaussian blur occlusion
            - 'pixelate': Pixelation effect
            - 'mixed': Random mix of all types
        regions: List of regions to occlude (default: all regions)
            Options: 'left_eye', 'right_eye', 'nose', 'mouth', 'left_eyebrow', 'right_eyebrow'
        num_regions: Tuple (min, max) for number of regions to occlude per image
        occlusion_prob: Probability of applying occlusion (0.0 to 1.0)
        landmark_format: 'dlib68' or 'mediapipe' landmark format
        additional_augs: If True, add light compression/noise augmentations
        
    Returns:
        Albumentations Compose pipeline
        
    Example:
        >>> pipeline = create_landmark_occlusion_pipeline(
        ...     occlusion_type='mixed',
        ...     regions=['left_eye', 'right_eye', 'mouth'],
        ...     num_regions=(1, 2),
        ...     occlusion_prob=0.5
        ... )
        >>> result = pipeline(image=img, landmarks=landmarks)
        
    Note:
        The pipeline expects 'landmarks' to be passed as additional data:
        >>> result = pipeline(image=img, landmarks=np.array([[x1,y1], [x2,y2], ...]))
    """
    from .transforms import LandmarkOcclusion, GaussianBlurOcclusion, PixelateOcclusion
    
    transforms_list = []
    
    # Horizontal flip (always included)
    transforms_list.append(A.HorizontalFlip(p=0.5))
    
    # Select occlusion transform based on type
    if occlusion_type == 'solid':
        transforms_list.append(
            LandmarkOcclusion(
                regions=regions,
                num_regions=num_regions,
                use_ellipse=True,
                landmark_format=landmark_format,
                p=occlusion_prob
            )
        )
    elif occlusion_type == 'blur':
        transforms_list.append(
            GaussianBlurOcclusion(
                regions=regions,
                num_regions=num_regions,
                blur_strength=(21, 51),
                landmark_format=landmark_format,
                p=occlusion_prob
            )
        )
    elif occlusion_type == 'pixelate':
        transforms_list.append(
            PixelateOcclusion(
                regions=regions,
                num_regions=num_regions,
                pixel_size=(8, 20),
                landmark_format=landmark_format,
                p=occlusion_prob
            )
        )
    elif occlusion_type == 'mixed':
        # Random choice between all occlusion types
        transforms_list.append(
            A.OneOf([
                LandmarkOcclusion(
                    regions=regions,
                    num_regions=num_regions,
                    use_ellipse=True,
                    landmark_format=landmark_format,
                    p=1.0
                ),
                GaussianBlurOcclusion(
                    regions=regions,
                    num_regions=num_regions,
                    blur_strength=(21, 51),
                    landmark_format=landmark_format,
                    p=1.0
                ),
                PixelateOcclusion(
                    regions=regions,
                    num_regions=num_regions,
                    pixel_size=(8, 20),
                    landmark_format=landmark_format,
                    p=1.0
                ),
            ], p=occlusion_prob)
        )
    
    # Optional: Add light image augmentations
    if additional_augs:
        transforms_list.extend([
            A.OneOf([
                A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
                A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
            ], p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.3),
        ])
    
    # Create compose with additional_targets for landmarks
    return A.Compose(
        transforms_list,
        additional_targets={'landmarks': 'keypoints'}
    )


# Pre-instantiated pipelines for common configurations
landmark_occlusion_solid = create_landmark_occlusion_pipeline(occlusion_type='solid')
landmark_occlusion_blur = create_landmark_occlusion_pipeline(occlusion_type='blur')
landmark_occlusion_mixed = create_landmark_occlusion_pipeline(occlusion_type='mixed')


# ==============================================================================
# --- Quality-Robust Augmentation Pipeline ---
# ==============================================================================
# Designed to break the "low quality = fake / high quality = real" shortcut
# by aggressively augmenting image quality in BOTH directions (degrade + enhance).
#
# All transforms are compatible with albumentations==0.4.6.
#
# Three strength tiers:
#   - 'light'    : Mild quality variation. Good for fine-tuning or early curriculum.
#   - 'moderate' : Balanced. Recommended for fine-tuning from a converged checkpoint.
#   - 'strong'   : Aggressive. Suitable for from-scratch training or late curriculum.
# ==============================================================================

# Per-tier parameter presets
_QUALITY_ROBUST_PRESETS = {
    'light': {
        # --- Quality degradation ---
        'jpeg_lower': 55,           # less aggressive than moderate
        'jpeg_upper': 95,
        'jpeg_p': 0.4,
        'blur_limit': (3, 7),
        'blur_p': 0.3,
        'noise_var': (5.0, 35.0),
        'noise_p': 0.2,
        'quality_group_p': 0.35,    # probability that ANY quality aug fires
        # --- Quality enhancement (sharpening) ---
        'sharpen_alpha': (0.3, 0.8),
        'sharpen_light': (0.5, 1.0),
        'sharpen_p': 0.25,
        # --- Downscale + upscale ---
        'downscale_min': 0.6,
        'downscale_max': 0.85,
        'downscale_p': 0.15,
        # --- Color ---
        'brightness_limit': 0.2,
        'contrast_limit': 0.2,
        'hue_shift': 10,
        'sat_shift': 20,
        'val_shift': 20,
        'color_p': 0.5,
    },
    'moderate': {
        'jpeg_lower': 40,
        'jpeg_upper': 90,
        'jpeg_p': 0.5,
        'blur_limit': (3, 9),
        'blur_p': 0.4,
        'noise_var': (10.0, 60.0),
        'noise_p': 0.3,
        'quality_group_p': 0.50,
        'sharpen_alpha': (0.4, 1.0),
        'sharpen_light': (0.5, 1.0),
        'sharpen_p': 0.35,
        'downscale_min': 0.45,
        'downscale_max': 0.80,
        'downscale_p': 0.25,
        'brightness_limit': 0.25,
        'contrast_limit': 0.25,
        'hue_shift': 12,
        'sat_shift': 25,
        'val_shift': 25,
        'color_p': 0.55,
    },
    'strong': {
        'jpeg_lower': 30,
        'jpeg_upper': 85,
        'jpeg_p': 0.6,
        'blur_limit': (3, 11),
        'blur_p': 0.5,
        'noise_var': (15.0, 80.0),
        'noise_p': 0.4,
        'quality_group_p': 0.65,
        'sharpen_alpha': (0.5, 1.2),
        'sharpen_light': (0.5, 1.0),
        'sharpen_p': 0.45,
        'downscale_min': 0.35,
        'downscale_max': 0.75,
        'downscale_p': 0.35,
        'brightness_limit': 0.3,
        'contrast_limit': 0.3,
        'hue_shift': 15,
        'sat_shift': 30,
        'val_shift': 30,
        'color_p': 0.60,
    },
}


def create_quality_robust_pipeline(strength: str = 'moderate', **kwargs) -> A.Compose:
    """
    Create a quality-robust augmentation pipeline that breaks quality-based
    shortcuts by augmenting images in BOTH directions:
      - Degrade: JPEG compression, blur, noise, downscale+upscale
      - Enhance: Sharpening (IAASharpen — 0.4.6 compatible)

    The pipeline ensures the model cannot rely on "blurry = fake" or
    "sharp = real" by making both real and fake images span the full
    quality spectrum during training.

    Args:
        strength: One of 'light', 'moderate', 'strong'.
                  Defaults to 'moderate'.

    Returns:
        A.Compose pipeline (albumentations 0.4.6 compatible)
    """
    if strength not in _QUALITY_ROBUST_PRESETS:
        raise ValueError(
            f"Unknown quality_robust strength '{strength}'. "
            f"Choose from: {list(_QUALITY_ROBUST_PRESETS.keys())}"
        )
    p = _QUALITY_ROBUST_PRESETS[strength]

    return A.Compose([
        # 1) Horizontal flip — always present
        A.HorizontalFlip(p=0.5),

        # 2) Quality DEGRADATION — one-of (blur / compression / noise)
        A.OneOf([
            A.ImageCompression(quality_lower=p['jpeg_lower'],
                               quality_upper=p['jpeg_upper'], p=p['jpeg_p']),
            A.GaussianBlur(blur_limit=p['blur_limit'], p=p['blur_p']),
            A.GaussNoise(var_limit=p['noise_var'], p=p['noise_p']),
        ], p=p['quality_group_p']),

        # 3) Quality ENHANCEMENT — sharpening (simulates enhancers / post-processing)
        #    IAASharpen is the albumentations 0.4.6 name for imgaug-backed sharpen
        A.IAASharpen(alpha=p['sharpen_alpha'],
                     lightness=p['sharpen_light'], p=p['sharpen_p']),

        # 4) Downscale + implicit upscale — simulates low-res capture / webcam
        #    A.Downscale already handles resize-back in albumentations 0.4.6
        A.Downscale(scale_min=p['downscale_min'],
                    scale_max=p['downscale_max'],
                    interpolation=cv2.INTER_AREA, p=p['downscale_p']),

        # 5) Color augmentation — prevents color/lighting shortcuts
        A.RandomBrightnessContrast(brightness_limit=p['brightness_limit'],
                                   contrast_limit=p['contrast_limit'], p=p['color_p']),
        A.HueSaturationValue(hue_shift_limit=p['hue_shift'],
                             sat_shift_limit=p['sat_shift'],
                             val_shift_limit=p['val_shift'], p=p['color_p'] * 0.8),
    ])


# Pre-instantiated quality-robust pipelines
quality_robust_light = create_quality_robust_pipeline('light')
quality_robust_moderate = create_quality_robust_pipeline('moderate')
quality_robust_strong = create_quality_robust_pipeline('strong')


# ==============================================================================
# --- Webcam / Video-Call Codec Simulation Pipeline ---
# ==============================================================================

def create_webcam_codec_pipeline(
    codec_quality=(25, 75),
    downscale_range=(0.5, 0.85),
    p=0.5,
) -> A.Compose:
    """
    Standalone pipeline that simulates webcam / video-call codec artifacts.

    Applies the full VideoCodecSimulation chain: resolution reduction →
    bilateral deblocking → block quantization → frequency-shaped noise →
    JPEG I-frame compression.

    Useful for:
      - Dedicated codec-degradation augmentation experiments
      - Adding to A.OneOf blocks alongside other degradation options
      - Testing the effect of codec simulation in isolation

    Args:
        codec_quality: (min, max) overall quality [0=worst, 100=best].
        downscale_range: (min, max) scale factor for resolution reduction.
        p: Probability of applying.

    Returns:
        A.Compose pipeline (albumentations 0.4.6 compatible)
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        VideoCodecSimulation(
            codec_quality=codec_quality,
            downscale_range=downscale_range,
            p=p,
        ),
    ])


# ==============================================================================
# --- Family-Aware Quality-Targeted Router (Phase 4) ---
# ==============================================================================
# Metadata-routed augmentations that target quality shortcut risks by family:
#   - df40_fake
#   - deeplive_non_enhanced_fake
#   - deeplive_enhanced_fake
#   - visomaster_fake
#   - visomaster_enhanced_fake
#   - df40_real
#   - realpool_real
#   - external_real
#
# Router signature:
#   router(image, landmarks=None, meta={label, source, method}) -> image
# ==============================================================================

_TEAMS_PASSTHROUGH_DEFAULTS = {
    # Base Teams passthrough stays intentionally conservative by default.
    "teams_passthrough_flip_p": 0.50,
    "teams_passthrough_brightness_contrast_p": 0.30,
    "teams_passthrough_brightness_limit": 0.08,
    "teams_passthrough_contrast_limit": 0.08,
    # Special Teams passthrough block: fully opt-in, off by default.
    "teams_passthrough_special_aug_enabled": False,
    "teams_passthrough_special_shift_p": 0.0,
    "teams_passthrough_special_shift": 0.02,
    "teams_passthrough_special_scale": 0.05,
    "teams_passthrough_special_rotate": 3,
    "teams_passthrough_special_cct_p": 0.0,
    "teams_passthrough_special_cct_range": (2700, 8000),
    "teams_passthrough_special_shadow_p": 0.0,
    "teams_passthrough_special_shadow_intensity": (0.15, 0.45),
    "teams_passthrough_special_shadow_softness": (0.20, 0.50),
    "teams_passthrough_special_gamma_up_p": 0.0,
    "teams_passthrough_special_gamma_up_range": (0.45, 0.85),
}

_QUALITY_TARGETED_PRESETS = {
    "light": {
        "jpeg_lower": 62,
        "jpeg_upper": 95,
        "blur_limit": (3, 5),
        "noise_var": (4.0, 20.0),
        "downscale_min": 0.72,
        "downscale_max": 0.90,
        "quality_p": 0.38,
        "color_p": 0.35,
        "color_brightness": 0.12,
        "color_contrast": 0.12,
        "hue_shift": 8,
        "sat_shift": 14,
        # Webcam codec simulation probability (applied to ALL families)
        "webcam_codec_p": 0.10,
        "webcam_codec_quality": (40, 80),
        "val_shift": 14,
        "sharpen_alpha_balanced": (0.15, 0.35),
        "sharpen_alpha_real": (0.18, 0.45),
        # Context variation block (lighting/exposure/framing robustness)
        "context_variation_enabled": False,
        "context_variation_gamma_limit": (90, 110),
        "context_variation_brightness": 0.15,
        "context_variation_contrast": 0.15,
        "context_variation_shift": 0.03,
        "context_variation_scale": 0.10,
        "context_variation_rotate": 6,
        "context_variation_oneof_p": 0.18,
        "context_variation_individual_p": 0.10,
        # Keep optional sidecar keys aligned across presets. The current
        # combined-source allowlist derives valid YAML override keys from the
        # first preset dict, so missing keys here become silent no-ops.
        "context_variation_cct_p": 0.0,
        "context_variation_cct_range": (2700, 8000),
        "context_variation_shadow_p": 0.0,
        "context_variation_shadow_intensity": (0.15, 0.45),
        "context_variation_shadow_softness": (0.20, 0.50),
        "context_variation_gamma_up_p": 0.0,
        "context_variation_gamma_up_range": (0.45, 0.85),
        "real_noise_p": 0.0,
        "real_noise_var": (5.0, 20.0),
        "real_sharpen_p": 0.50,
        "fake_extra_degrade_p": 0.0,
        **_TEAMS_PASSTHROUGH_DEFAULTS,
    },
    "moderate": {
        "jpeg_lower": 48,
        "jpeg_upper": 92,
        "blur_limit": (3, 7),
        "noise_var": (8.0, 35.0),
        "downscale_min": 0.58,
        "downscale_max": 0.85,
        "quality_p": 0.52,
        "color_p": 0.45,
        "color_brightness": 0.18,
        "color_contrast": 0.18,
        "hue_shift": 10,
        "sat_shift": 20,
        "val_shift": 20,
        "webcam_codec_p": 0.15,
        "webcam_codec_quality": (30, 75),
        "sharpen_alpha_balanced": (0.20, 0.50),
        "sharpen_alpha_real": (0.24, 0.55),
        # Context variation block (lighting/exposure/framing robustness)
        "context_variation_enabled": False,
        "context_variation_gamma_limit": (90, 110),
        "context_variation_brightness": 0.15,
        "context_variation_contrast": 0.15,
        "context_variation_shift": 0.03,
        "context_variation_scale": 0.10,
        "context_variation_rotate": 6,
        "context_variation_oneof_p": 0.18,
        "context_variation_individual_p": 0.10,
        "context_variation_cct_p": 0.0,
        "context_variation_cct_range": (2700, 8000),
        "context_variation_shadow_p": 0.0,
        "context_variation_shadow_intensity": (0.15, 0.45),
        "context_variation_shadow_softness": (0.20, 0.50),
        "context_variation_gamma_up_p": 0.0,
        "context_variation_gamma_up_range": (0.45, 0.85),
        "real_noise_p": 0.0,
        "real_noise_var": (5.0, 20.0),
        "real_sharpen_p": 0.50,
        "fake_extra_degrade_p": 0.0,
        **_TEAMS_PASSTHROUGH_DEFAULTS,
    },
    "strong": {
        "jpeg_lower": 40,
        "jpeg_upper": 90,
        "blur_limit": (3, 9),
        "noise_var": (10.0, 45.0),
        "downscale_min": 0.50,
        "downscale_max": 0.80,
        "quality_p": 0.60,
        "color_p": 0.52,
        "color_brightness": 0.22,
        "color_contrast": 0.22,
        "hue_shift": 12,
        "sat_shift": 24,
        "val_shift": 24,
        "webcam_codec_p": 0.22,
        "webcam_codec_quality": (25, 70),
        "sharpen_alpha_balanced": (0.24, 0.60),
        "sharpen_alpha_real": (0.26, 0.62),
        # Context variation block (lighting/exposure/framing robustness)
        "context_variation_enabled": False,
        "context_variation_gamma_limit": (85, 115),
        "context_variation_brightness": 0.20,
        "context_variation_contrast": 0.20,
        "context_variation_shift": 0.04,
        "context_variation_scale": 0.12,
        "context_variation_rotate": 8,
        "context_variation_oneof_p": 0.22,
        "context_variation_individual_p": 0.12,
        "context_variation_cct_p": 0.0,
        "context_variation_cct_range": (2700, 8000),
        "context_variation_shadow_p": 0.0,
        "context_variation_shadow_intensity": (0.15, 0.45),
        "context_variation_shadow_softness": (0.20, 0.50),
        "context_variation_gamma_up_p": 0.0,
        "context_variation_gamma_up_range": (0.45, 0.85),
        "real_noise_p": 0.0,
        "real_noise_var": (5.0, 20.0),
        "real_sharpen_p": 0.50,
        "fake_extra_degrade_p": 0.0,
        **_TEAMS_PASSTHROUGH_DEFAULTS,
    },
    # -----------------------------------------------------------------------
    # VCD-targeted preset: designed to break quality-label shortcuts.
    # Sharpens reals toward VCD's quality profile, degrades fakes more,
    # keeps codec sim modest (wrong direction for VCD sharpness gap).
    # -----------------------------------------------------------------------
    "vcd_targeted": {
        "jpeg_lower": 40,
        "jpeg_upper": 90,
        "blur_limit": (3, 9),
        "noise_var": (10.0, 45.0),
        "downscale_min": 0.50,
        "downscale_max": 0.80,
        "quality_p": 0.60,
        "color_p": 0.52,
        "color_brightness": 0.22,
        "color_contrast": 0.22,
        "hue_shift": 12,
        "sat_shift": 24,
        "val_shift": 24,
        # Codec sim stays modest — wrong direction for VCD sharpness gap
        "webcam_codec_p": 0.12,
        "webcam_codec_quality": (35, 80),
        # Increased sharpening for reals (pushing toward VCD tenengrad 40.9)
        "sharpen_alpha_balanced": (0.24, 0.60),
        "sharpen_alpha_real": (0.30, 0.70),
        # Real-side noise injection (VCD noise_estimate 1.08 vs DF40 0.38)
        "real_noise_p": 0.25,
        "real_noise_var": (5.0, 20.0),
        # Extra sharpening probability for reals
        "real_sharpen_p": 0.60,
        # Fake degradation bump (break quality-label correlation from fake side)
        "fake_extra_degrade_p": 0.15,
        # Context variation block (lighting/exposure/framing robustness)
        # Enabled by default for vcd_targeted — addresses lighting FP gap.
        # R12: transforms are independent (compound), asymmetric brightness
        # biased upward (+0.60) to cover bright webcam captures (mean ~160
        # vs training ~104), wider hue shift (20), and CCT simulation.
        "context_variation_enabled": True,
        "context_variation_gamma_limit": (70, 130),
        "context_variation_brightness": (-0.20, 0.60),
        "context_variation_contrast": 0.25,
        "context_variation_shift": 0.04,
        "context_variation_scale": 0.12,
        "context_variation_rotate": 8,
        "context_variation_oneof_p": 0.18,  # kept for compat, ignored in R12
        "context_variation_individual_p": 0.15,
        "context_variation_cct_p": 0.15,
        "context_variation_cct_range": (2700, 8000),
        # Directional shadow simulation — off by default, enable when ready.
        # Addresses directional-lighting FP gap (LIGHTING_ROBUSTNESS_REPORT §6).
        "context_variation_shadow_p": 0.0,
        "context_variation_shadow_intensity": (0.15, 0.45),
        "context_variation_shadow_softness": (0.20, 0.50),
        # Gamma-up (always-brighten) — off by default, enable when ready.
        # Addresses upward brightness gap (training ~104, production ~160-205).
        "context_variation_gamma_up_p": 0.0,
        "context_variation_gamma_up_range": (0.45, 0.85),
        "hue_shift": 20,
        **_TEAMS_PASSTHROUGH_DEFAULTS,
    },
}

_DEFAULT_ENHANCED_STRATEGIES = (
    "quality_enhancement",
    "edge_cases_enhanced",
    "minimal_processing_enhanced",
)


def _build_context_variation_block(p: dict) -> list:
    """Build context variation transforms for lighting/exposure/framing robustness.

    Returns a list of albumentations transforms (may be empty if disabled).
    All ops are compatible with albumentations==0.4.6.

    R12 change: transforms are now independent (no OneOf wrapper) so that
    gamma, brightness/contrast, shift/scale/rotate, and CCT can all
    compound per-image.  ``context_variation_oneof_p`` is kept in presets
    for backward compat but ignored (individual_p governs each transform).
    """
    if not p.get("context_variation_enabled", False):
        return []

    ind_p = p.get("context_variation_individual_p", 0.10)

    transforms = [
        A.RandomGamma(
            gamma_limit=p.get("context_variation_gamma_limit", (90, 110)),
            p=ind_p,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=p.get("context_variation_brightness", (-0.20, 0.60)),
            contrast_limit=p.get("context_variation_contrast", 0.15),
            p=ind_p,
        ),
        A.ShiftScaleRotate(
            shift_limit=p.get("context_variation_shift", 0.03),
            scale_limit=p.get("context_variation_scale", 0.10),
            rotate_limit=p.get("context_variation_rotate", 6),
            border_mode=cv2.BORDER_REFLECT_101,
            p=ind_p,
        ),
    ]

    # CCT (colour temperature) simulation — closes the R/B ratio gap
    # between training data (~1.4-1.6) and production webcam (~1.2).
    cct_p = p.get("context_variation_cct_p", 0.0)
    if cct_p > 0:
        from .transforms import ColorTemperatureShift
        transforms.append(
            ColorTemperatureShift(
                cct_range=p.get("context_variation_cct_range", (3500, 8000)),
                p=cct_p,
            )
        )

    # Directional shadow — simulates uneven indoor lighting (desk lamp,
    # window to one side). Addresses LIGHTING_ROBUSTNESS_REPORT §6.
    shadow_p = p.get("context_variation_shadow_p", 0.0)
    if shadow_p > 0:
        from .transforms import DirectionalShadow
        transforms.append(
            DirectionalShadow(
                intensity_range=p.get("context_variation_shadow_intensity", (0.15, 0.45)),
                softness_range=p.get("context_variation_shadow_softness", (0.20, 0.50)),
                p=shadow_p,
            )
        )

    # Gamma-up (always-brighten) — dedicated upward brightness push to
    # cover the training→production brightness gap (mean 104 → 160-205).
    gamma_up_p = p.get("context_variation_gamma_up_p", 0.0)
    if gamma_up_p > 0:
        from .transforms import GammaUp
        transforms.append(
            GammaUp(
                gamma_range=p.get("context_variation_gamma_up_range", (0.45, 0.85)),
                p=gamma_up_p,
            )
        )

    return transforms


def _build_teams_passthrough_special_block(p: dict | None) -> list:
    """Build the fully opt-in special Teams passthrough augmentation block."""
    if not p or not p.get("teams_passthrough_special_aug_enabled", False):
        return []

    transforms = []

    shift_p = p.get("teams_passthrough_special_shift_p", 0.0)
    if shift_p > 0:
        transforms.append(
            A.ShiftScaleRotate(
                shift_limit=p.get("teams_passthrough_special_shift", 0.02),
                scale_limit=p.get("teams_passthrough_special_scale", 0.05),
                rotate_limit=p.get("teams_passthrough_special_rotate", 3),
                border_mode=cv2.BORDER_REFLECT_101,
                p=shift_p,
            )
        )

    cct_p = p.get("teams_passthrough_special_cct_p", 0.0)
    if cct_p > 0:
        from .transforms import ColorTemperatureShift
        transforms.append(
            ColorTemperatureShift(
                cct_range=p.get("teams_passthrough_special_cct_range", (2700, 8000)),
                p=cct_p,
            )
        )

    shadow_p = p.get("teams_passthrough_special_shadow_p", 0.0)
    if shadow_p > 0:
        from .transforms import DirectionalShadow
        transforms.append(
            DirectionalShadow(
                intensity_range=p.get(
                    "teams_passthrough_special_shadow_intensity",
                    (0.15, 0.45),
                ),
                softness_range=p.get(
                    "teams_passthrough_special_shadow_softness",
                    (0.20, 0.50),
                ),
                p=shadow_p,
            )
        )

    gamma_up_p = p.get("teams_passthrough_special_gamma_up_p", 0.0)
    if gamma_up_p > 0:
        from .transforms import GammaUp
        transforms.append(
            GammaUp(
                gamma_range=p.get(
                    "teams_passthrough_special_gamma_up_range",
                    (0.45, 0.85),
                ),
                p=gamma_up_p,
            )
        )

    return transforms


def _build_family_quality_pipeline(family_key: str, p: dict) -> A.Compose:
    """Build a family-specific quality pipeline using 0.4.6-compatible ops."""
    # Shared transform blocks (kept inline for readability/traceability).
    balanced_degrade = [
        A.ImageCompression(quality_lower=p["jpeg_lower"], quality_upper=p["jpeg_upper"], p=1.0),
        A.GaussianBlur(blur_limit=p["blur_limit"], p=1.0),
        A.GaussNoise(var_limit=p["noise_var"], p=1.0),
        A.Downscale(
            scale_min=p["downscale_min"],
            scale_max=p["downscale_max"],
            interpolation=cv2.INTER_AREA,
            p=1.0,
        ),
    ]

    # Webcam / video-call codec simulation — coherent pipeline that
    # produces the characteristic flat-PSD + high-freq-noise fingerprint
    # seen in Zoom VCD and webcam captures.  Applied to ALL families so
    # both reals and fakes sometimes look like webcam video.
    webcam_codec_step = VideoCodecSimulation(
        codec_quality=p.get("webcam_codec_quality", (30, 80)),
        p=p.get("webcam_codec_p", 0.0),
    )

    color_block = A.OneOf(
        [
            A.RandomBrightnessContrast(
                brightness_limit=p["color_brightness"],
                contrast_limit=p["color_contrast"],
                p=1.0,
            ),
            A.HueSaturationValue(
                hue_shift_limit=p["hue_shift"],
                sat_shift_limit=p["sat_shift"],
                val_shift_limit=p["val_shift"],
                p=1.0,
            ),
        ],
        p=p["color_p"],
    )

    # Context variation block: lighting/exposure/framing robustness.
    # Applied to ALL families after existing quality/color transforms.
    context_variation = _build_context_variation_block(p)

    # Optional extra degradation for fakes (vcd_targeted: break quality shortcut
    # from the fake side — ensure low-quality fakes exist in training).
    fake_extra_degrade_p = p.get("fake_extra_degrade_p", 0.0)

    # Fake families.
    if family_key == "df40_fake":
        df40_fake_steps = [
            A.HorizontalFlip(p=0.5),
            A.OneOf(balanced_degrade, p=p["quality_p"]),
            A.IAASharpen(alpha=p["sharpen_alpha_balanced"], lightness=(0.6, 1.0), p=0.26),
            color_block,
            *context_variation,
            webcam_codec_step,
        ]
        # Inject extra heavy degradation for a fraction of fakes
        if fake_extra_degrade_p > 0:
            df40_fake_steps.insert(2, A.OneOf([
                A.Downscale(scale_min=0.35, scale_max=0.55, interpolation=cv2.INTER_AREA, p=1.0),
                A.GaussianBlur(blur_limit=(5, 11), p=1.0),
            ], p=fake_extra_degrade_p))
        return A.Compose(df40_fake_steps)

    if family_key == "deeplive_non_enhanced_fake":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(balanced_degrade, p=min(0.9, p["quality_p"] + 0.08)),
                A.IAASharpen(alpha=p["sharpen_alpha_balanced"], lightness=(0.6, 1.0), p=0.22),
                color_block,
                *context_variation,
                webcam_codec_step,
            ]
        )

    if family_key == "deeplive_enhanced_fake":
        # WMA-style degradation emphasis: downscale/compress/blur weighted heavier.
        enhanced_degrade = [
            A.Downscale(
                scale_min=max(0.35, p["downscale_min"] - 0.10),
                scale_max=max(0.72, p["downscale_max"] - 0.06),
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
            A.ImageCompression(
                quality_lower=max(30, p["jpeg_lower"] - 10),
                quality_upper=p["jpeg_upper"],
                p=1.0,
            ),
            A.ImageCompression(
                quality_lower=max(26, p["jpeg_lower"] - 14),
                quality_upper=max(70, p["jpeg_upper"] - 8),
                p=1.0,
            ),
            A.GaussianBlur(blur_limit=(3, max(7, p["blur_limit"][1] + 2)), p=1.0),
        ]
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(enhanced_degrade, p=min(0.95, p["quality_p"] + 0.22)),
                A.GaussNoise(var_limit=(p["noise_var"][0], p["noise_var"][1] * 1.15), p=0.28),
                color_block,
                *context_variation,
                A.IAASharpen(alpha=(0.08, 0.22), lightness=(0.6, 1.0), p=0.08),
                webcam_codec_step,
            ]
        )

    if family_key == "visomaster_fake":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(
                    [
                        A.ImageCompression(
                            quality_lower=max(36, p["jpeg_lower"] - 8),
                            quality_upper=p["jpeg_upper"],
                            p=1.0,
                        ),
                        A.GaussianBlur(blur_limit=p["blur_limit"], p=1.0),
                        A.Downscale(
                            scale_min=max(0.45, p["downscale_min"] - 0.08),
                            scale_max=max(0.74, p["downscale_max"] - 0.04),
                            interpolation=cv2.INTER_AREA,
                            p=1.0,
                        ),
                        A.GaussNoise(var_limit=p["noise_var"], p=1.0),
                    ],
                    p=min(0.92, p["quality_p"] + 0.12),
                ),
                color_block,
                *context_variation,
                A.IAASharpen(alpha=(0.10, 0.30), lightness=(0.6, 1.0), p=0.14),
                webcam_codec_step,
            ]
        )

    if family_key == "visomaster_enhanced_fake":
        # Post-hoc face-enhanced fakes (GFPGAN, CodeFormer, GPEN-BFR, etc.).
        # Enhancement smooths out GAN artifacts and increases perceived quality,
        # making fakes look closer to real images.  Apply aggressive degradation
        # (heavier compression, downscale, blur) so the model learns to detect
        # underlying fake structure despite enhancement.
        # Similar treatment to deeplive_enhanced_fake but tuned for multi-enhancer variety.
        enhanced_degrade = [
            A.Downscale(
                scale_min=max(0.35, p["downscale_min"] - 0.10),
                scale_max=max(0.70, p["downscale_max"] - 0.08),
                interpolation=cv2.INTER_AREA,
                p=1.0,
            ),
            A.ImageCompression(
                quality_lower=max(28, p["jpeg_lower"] - 12),
                quality_upper=p["jpeg_upper"],
                p=1.0,
            ),
            A.ImageCompression(
                quality_lower=max(24, p["jpeg_lower"] - 16),
                quality_upper=max(68, p["jpeg_upper"] - 10),
                p=1.0,
            ),
            A.GaussianBlur(blur_limit=(3, max(9, p["blur_limit"][1] + 4)), p=1.0),
        ]
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.OneOf(enhanced_degrade, p=min(0.96, p["quality_p"] + 0.24)),
                A.GaussNoise(var_limit=(p["noise_var"][0], p["noise_var"][1] * 1.2), p=0.30),
                color_block,
                *context_variation,
                A.IAASharpen(alpha=(0.06, 0.18), lightness=(0.6, 1.0), p=0.06),
                webcam_codec_step,
            ]
        )

    # Real families.
    # vcd_targeted presets add dedicated noise + sharpen params to bridge the
    # gap between soft DF40 reals and sharper/noisier webcam-captured VCD reals.
    real_sharpen_p = p.get("real_sharpen_p", 0.50)
    real_noise_p = p.get("real_noise_p", 0.0)
    real_noise_var = p.get("real_noise_var", (5.0, 20.0))

    if family_key == "df40_real":
        df40_real_steps = [
            A.HorizontalFlip(p=0.5),
            A.IAASharpen(alpha=p["sharpen_alpha_real"], lightness=(0.7, 1.0), p=real_sharpen_p),
            color_block,
            *context_variation,
            A.OneOf(
                [
                    A.ImageCompression(quality_lower=max(58, p["jpeg_lower"]), quality_upper=95, p=1.0),
                    A.GaussNoise(var_limit=(3.0, max(12.0, p["noise_var"][1] * 0.6)), p=1.0),
                    # Dedicated real-noise option (bridges DF40→VCD noise gap)
                    A.GaussNoise(var_limit=real_noise_var, p=1.0),
                ],
                p=max(0.20, real_noise_p + 0.05),
            ),
            webcam_codec_step,
        ]
        return A.Compose(df40_real_steps)

    if family_key in {"realpool_real", "external_real"}:
        ext_real_steps = [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.ImageCompression(quality_lower=max(52, p["jpeg_lower"]), quality_upper=95, p=1.0),
                    A.GaussianBlur(blur_limit=(3, min(7, p["blur_limit"][1])), p=1.0),
                    A.Downscale(
                        scale_min=max(0.62, p["downscale_min"]),
                        scale_max=max(0.90, p["downscale_max"]),
                        interpolation=cv2.INTER_AREA,
                        p=1.0,
                    ),
                    A.GaussNoise(var_limit=(4.0, max(18.0, p["noise_var"][1] * 0.8)), p=1.0),
                ],
                p=min(0.85, p["quality_p"] + 0.04),
            ),
            color_block,
            *context_variation,
            A.IAASharpen(alpha=(0.15, 0.38), lightness=(0.7, 1.0), p=max(0.18, real_sharpen_p * 0.4)),
            webcam_codec_step,
        ]
        # Add dedicated noise injection for external/pool reals if vcd_targeted
        if real_noise_p > 0:
            ext_real_steps.insert(
                -1,  # Before webcam_codec_step
                A.GaussNoise(var_limit=real_noise_var, p=real_noise_p),
            )
        return A.Compose(ext_real_steps)

    return create_quality_robust_pipeline("moderate")


def _build_teams_passthrough_pipeline(p: dict | None = None) -> A.Compose:
    """
    Build a minimal augmentation pipeline for Teams-passthrough data.

    Teams data has **already been through the full codec pipeline** (H.264,
    auto-exposure, sharpening, denoising, YUV 4:2:0 subsampling).  Applying
    further degradation (blur, downscale, codec sim) would destroy the
    authentic codec fingerprint the model should learn.

    By default, only spatial flip and very light colour jitter are applied.
    An explicit opt-in special block exists for sidecar experiments that want
    extra Teams-native nuisance robustness without affecting the default path.
    """
    p = p or {}

    return A.Compose([
        A.HorizontalFlip(p=p.get("teams_passthrough_flip_p", 0.5)),
        A.RandomBrightnessContrast(
            brightness_limit=p.get("teams_passthrough_brightness_limit", 0.08),
            contrast_limit=p.get("teams_passthrough_contrast_limit", 0.08),
            p=p.get("teams_passthrough_brightness_contrast_p", 0.3),
        ),
        *_build_teams_passthrough_special_block(p),
    ])


class QualityTargetedFamilyRouter:
    """Callable router that applies family-aware quality augmentation."""

    def __init__(
        self,
        strength: str = "moderate",
        routing_mode: str = "family_aware",
        enhanced_strategy_names: tuple[str, ...] = _DEFAULT_ENHANCED_STRATEGIES,
        preset_overrides: dict | None = None,
        teams_codec_simulation: dict | None = None,
    ):
        if strength not in _QUALITY_TARGETED_PRESETS:
            raise ValueError(
                f"Unknown quality_targeted_family strength '{strength}'. "
                f"Choose from: {list(_QUALITY_TARGETED_PRESETS.keys())}"
            )
        self.strength = strength
        self.routing_mode = routing_mode
        self.enhanced_strategy_names = enhanced_strategy_names
        # Merge any YAML-level overrides into the hardcoded preset dict.
        self._preset = {**_QUALITY_TARGETED_PRESETS[strength], **(preset_overrides or {})}
        self._fallback = create_quality_robust_pipeline(strength if strength in {"light", "moderate", "strong"} else "moderate")
        self._pipelines = {
            "df40_fake": _build_family_quality_pipeline("df40_fake", self._preset),
            "deeplive_non_enhanced_fake": _build_family_quality_pipeline("deeplive_non_enhanced_fake", self._preset),
            "deeplive_enhanced_fake": _build_family_quality_pipeline("deeplive_enhanced_fake", self._preset),
            "visomaster_fake": _build_family_quality_pipeline("visomaster_fake", self._preset),
            "visomaster_enhanced_fake": _build_family_quality_pipeline("visomaster_enhanced_fake", self._preset),
            "df40_real": _build_family_quality_pipeline("df40_real", self._preset),
            "realpool_real": _build_family_quality_pipeline("realpool_real", self._preset),
            "external_real": _build_family_quality_pipeline("external_real", self._preset),
            # Teams passthrough data has already been through the codec pipeline.
            # Default path stays minimal; extra Teams nuisance knobs are opt-in.
            "deeplive_teams_fake": _build_teams_passthrough_pipeline(self._preset),
            "deeplive_teams_real": _build_teams_passthrough_pipeline(self._preset),
        }

        # ── Teams codec simulation (optional post-pipeline step) ──────
        # When enabled, apply TeamsCodecSimulation after the per-family
        # pipeline with the given probability — but skip families in
        # exclude_families (e.g. actual Teams data that already has the
        # real codec fingerprint).
        tcs_cfg = teams_codec_simulation or {}
        self._teams_sim_enabled = tcs_cfg.get("enabled", False)
        self._teams_sim_p = tcs_cfg.get("probability", 0.15)
        self._teams_sim_exclude = set(tcs_cfg.get("exclude_families", []))
        self._teams_sim_policy = str(tcs_cfg.get("policy", "legacy_single") or "legacy_single")
        self._teams_sim = None
        if self._teams_sim_enabled:
            from .teams_simulation import (
                TeamsAdaptiveCodecSimulation,
                TeamsCodecSimulation,
                TeamsHybridCodecSimulation,
            )

            if self._teams_sim_policy in {"legacy", "legacy_single", "single"}:
                self._teams_sim = TeamsCodecSimulation(always_apply=True, p=1.0)
            elif self._teams_sim_policy in {"adaptive", "adaptive_mixture", "mixture"}:
                self._teams_sim = TeamsAdaptiveCodecSimulation(
                    always_apply=True,
                    p=1.0,
                    enhanced_families=tuple(
                        tcs_cfg.get("enhanced_families")
                        or ("visomaster_enhanced_fake", "deeplive_enhanced_fake")
                    ),
                    ordinary_mode_probability_non_enhanced=float(
                        tcs_cfg.get("ordinary_mode_probability_non_enhanced", 0.75)
                    ),
                    ordinary_mode_probability_enhanced=float(
                        tcs_cfg.get("ordinary_mode_probability_enhanced", 0.25)
                    ),
                )
            elif self._teams_sim_policy in {"family_split", "hybrid"}:
                self._teams_sim = TeamsHybridCodecSimulation(
                    always_apply=True,
                    p=1.0,
                    enhanced_families=tuple(
                        tcs_cfg.get("enhanced_families")
                        or ("visomaster_enhanced_fake", "deeplive_enhanced_fake")
                    ),
                    ordinary_mode_probability_non_enhanced=float(
                        tcs_cfg.get("ordinary_mode_probability_non_enhanced", 0.75)
                    ),
                    ordinary_mode_probability_enhanced=float(
                        tcs_cfg.get("ordinary_mode_probability_enhanced", 0.25)
                    ),
                )
            else:
                raise ValueError(
                    "Unknown teams_codec_simulation policy "
                    f"'{self._teams_sim_policy}'. "
                    "Choose from: legacy_single, adaptive_mixture, family_split"
                )

    def __call__(self, image: np.ndarray, landmarks=None, meta: dict | None = None) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            return image

        if self.routing_mode != "family_aware":
            result = self._fallback(image=image)["image"]
            return self._maybe_apply_teams_sim(result, family_key=None)

        meta = meta or {}
        from utils.grouping import infer_family_key  # Local import avoids tight coupling.

        family_key = infer_family_key(
            label=meta.get("label"),
            method=meta.get("method"),
            source=meta.get("source"),
            enhanced_strategy_names=self.enhanced_strategy_names,
        )
        pipeline = self._pipelines.get(family_key, self._fallback)
        result = pipeline(image=image)["image"]
        return self._maybe_apply_teams_sim(result, family_key)

    def _maybe_apply_teams_sim(self, image: np.ndarray, family_key: str | None) -> np.ndarray:
        """Optionally apply TeamsCodecSimulation as a post-pipeline step."""
        if not self._teams_sim_enabled or self._teams_sim is None:
            return image
        if family_key and family_key in self._teams_sim_exclude:
            return image
        if np.random.random() < self._teams_sim_p:
            family_apply = getattr(self._teams_sim, "apply_for_family", None)
            if callable(family_apply):
                return family_apply(image, family_key=family_key)
            return self._teams_sim.apply(image)
        return image


def create_quality_targeted_family_router(
    strength: str = "moderate",
    routing_mode: str = "family_aware",
    enhanced_strategy_names: tuple[str, ...] = _DEFAULT_ENHANCED_STRATEGIES,
    preset_overrides: dict | None = None,
    teams_codec_simulation: dict | None = None,
) -> QualityTargetedFamilyRouter:
    """
    Create a family-aware quality-targeted augmentation router.

    The returned callable accepts:
        router(image, landmarks=None, meta={label, source, method})

    Args:
        preset_overrides: Dict of keys to override in the preset (e.g.,
            ``{"context_variation_enabled": True}`` from YAML config).
        teams_codec_simulation: Dict with keys ``enabled`` (bool),
            ``probability`` (float, default 0.15), optional ``policy``
            (``legacy_single``, ``adaptive_mixture``, or ``family_split``), and optional
            ``exclude_families`` (list of family keys to skip).
    """
    return QualityTargetedFamilyRouter(
        strength=strength,
        routing_mode=routing_mode,
        enhanced_strategy_names=enhanced_strategy_names,
        preset_overrides=preset_overrides,
        teams_codec_simulation=teams_codec_simulation,
    )
