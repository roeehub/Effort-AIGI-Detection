"""
Augmentation module for training pipeline.

This module provides:
- Custom transforms (CustomUnsharpMask, NoOp)
- Landmark-based occlusion transforms (LandmarkOcclusion, GaussianBlurOcclusion, PixelateOcclusion)
- Region bbox occlusion for DeepLive GCS data (RegionBBoxOcclusion) - computes bboxes from MediaPipe 478-point landmarks
- Pre-defined pipelines (V3-V7, surgical, landmark_occlusion)
- A registry for easy pipeline selection

Usage:
    from data.augmentations import get_pipeline
    
    # Get a pre-defined pipeline by version
    pipeline = get_pipeline(version=3)
    result = pipeline(image=img_np)['image']
    
    # Get surgical pipeline (requires config and frame properties)
    pipeline = get_pipeline(version='surgical', config=cfg, frame_properties=props)
    
    # Get V6/V7 (returns a callable that handles pipeline selection internally)
    augment_fn = get_pipeline(version=6)
    result = augment_fn(img_np, frame_dict)
    
    # Get landmark occlusion pipeline (Task B)
    from data.augmentations import LandmarkOcclusion
    pipeline = get_pipeline(version='landmark_occlusion', occlusion_type='mixed')
    result = pipeline(image=img_np, landmarks=landmark_array)['image']
    
    # For DeepLive GCS data with MediaPipe 478-point landmarks:
    from data.augmentations import RegionBBoxOcclusion
    transform = RegionBBoxOcclusion(
        regions=['left_eye', 'right_eye', 'mouth'],  # Uses MEDIAPIPE_REGION_INDICES
        occlusion_type='blur',
        p=0.5
    )
    # landmarks can be DeepLiveLandmark object, dict with 'landmarks' key, or raw list
    result = transform(image=img_np, landmarks=deeplive_landmark_obj)['image']
"""

from .registry import get_pipeline, PIPELINE_REGISTRY, register_pipeline
from .transforms import (
    CustomUnsharpMask, 
    NoOp,
    LandmarkOcclusion,
    GaussianBlurOcclusion,
    PixelateOcclusion,
    RegionBBoxOcclusion,
    VideoCodecSimulation,
    ColorTemperatureShift,
    DirectionalShadow,
    GammaUp,
    LANDMARK_REGIONS,
    OCCLUSION_REGIONS,
    MEDIAPIPE_REGION_INDICES,
)
from .teams_simulation import (
    TeamsCodecSimulation,
    TeamsAdaptiveCodecSimulation,
    TeamsHybridCodecSimulation,
)

__all__ = [
    'get_pipeline',
    'PIPELINE_REGISTRY', 
    'register_pipeline',
    'CustomUnsharpMask',
    'NoOp',
    # Landmark-based transforms (raw coordinate format)
    'LandmarkOcclusion',
    'GaussianBlurOcclusion', 
    'PixelateOcclusion',
    # DeepLive GCS MediaPipe 478-point format
    'RegionBBoxOcclusion',
    'LANDMARK_REGIONS',
    'OCCLUSION_REGIONS',
    'MEDIAPIPE_REGION_INDICES',
    # Video codec simulation
    'VideoCodecSimulation',
    # Lighting robustness transforms
    'ColorTemperatureShift',
    'DirectionalShadow',
    'GammaUp',
    # Teams-specific codec simulation
    'TeamsCodecSimulation',
    'TeamsAdaptiveCodecSimulation',
    'TeamsHybridCodecSimulation',
]
