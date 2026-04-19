"""
Augmentation pipeline registry.

Provides a unified interface to access all augmentation pipelines by version number
or name. This allows the rest of the codebase to get pipelines without knowing
the implementation details.

Usage:
    from data.augmentations import get_pipeline
    
    # Get static pipeline (returns A.Compose)
    pipeline = get_pipeline(version=3)
    result = pipeline(image=img_np)['image']
    
    # Get callable augmentation function (V6/V7)
    augment_fn = get_pipeline(version=6)
    result = augment_fn(img_np, frame_dict)  # V6 needs frame_dict
    result = augment_fn(img_np)              # V7 doesn't need it
    
    # Get surgical pipeline (requires config and frame_properties)
    pipeline = get_pipeline(
        version='surgical', 
        config=surgical_config, 
        frame_properties={'sharpness_bucket': 'q4'}
    )
"""

from typing import Callable, Union
import albumentations as A

from .pipelines import (
    # Factory functions
    create_v3_pipeline,
    create_v4_pipeline,
    create_v5_pipeline,
    create_v6_simulator_pipeline,
    create_v6_generalist_pipeline,
    create_v6_purist_pipeline,
    create_v6_mild_pipeline,
    create_surgical_augmentation_pipeline,
    create_general_augmentation_pipeline,
    create_degrade_quality_pipeline,
    create_enhance_quality_pipeline,
    create_social_media_pipeline,
    create_landmark_occlusion_pipeline,
    create_quality_robust_pipeline,
    create_quality_targeted_family_router,
    create_webcam_codec_pipeline,
    # Callable functions
    apply_augmentation_v6,
    apply_augmentation_v7,
    # Pre-instantiated pipelines (for backward compatibility)
    revised_augmentation_pipeline_legacy,
    augmentation_pipeline_v4,
    augmentation_pipeline_v5,
    AUG_PIPELINE_V6_SIMULATOR,
    AUG_PIPELINE_V4_GENERALIST,
    AUG_PIPELINE_PURIST,
    AUG_PIPELINE_V3_MILD,
    degrade_quality_pipeline,
    enhance_quality_pipeline,
    social_media_pipeline,
    landmark_occlusion_solid,
    landmark_occlusion_blur,
    landmark_occlusion_mixed,
    quality_robust_light,
    quality_robust_moderate,
    quality_robust_strong,
)


# Type alias for pipeline or callable
PipelineType = Union[A.Compose, Callable]


# Registry mapping version identifiers to factory functions or callables
PIPELINE_REGISTRY: dict[Union[int, str], Callable] = {
    # Static pipelines (return A.Compose)
    3: create_v3_pipeline,
    'v3': create_v3_pipeline,
    'legacy': create_v3_pipeline,
    
    4: create_v4_pipeline,
    'v4': create_v4_pipeline,
    'moderate': create_v4_pipeline,
    
    5: create_v5_pipeline,
    'v5': create_v5_pipeline,
    'hybrid': create_v5_pipeline,
    
    # Callable functions (V6/V7)
    6: lambda **kwargs: apply_augmentation_v6,
    'v6': lambda **kwargs: apply_augmentation_v6,
    'source_dependent': lambda **kwargs: apply_augmentation_v6,
    
    7: lambda **kwargs: apply_augmentation_v7,
    'v7': lambda **kwargs: apply_augmentation_v7,
    'portfolio': lambda **kwargs: apply_augmentation_v7,
    
    # Special pipelines
    'surgical': create_surgical_augmentation_pipeline,
    'general': create_general_augmentation_pipeline,
    
    # Landmark-based occlusion pipelines (Task B)
    'landmark_occlusion': create_landmark_occlusion_pipeline,
    'landmark': create_landmark_occlusion_pipeline,
    'occlusion': create_landmark_occlusion_pipeline,
    
    # Quality-robust pipelines (break quality shortcuts)
    'quality_robust': lambda **kw: create_quality_robust_pipeline(kw.get('strength', 'moderate')),
    'quality_robust_light': lambda **kw: create_quality_robust_pipeline('light'),
    'quality_robust_moderate': lambda **kw: create_quality_robust_pipeline('moderate'),
    'quality_robust_strong': lambda **kw: create_quality_robust_pipeline('strong'),
    'quality_targeted_family': lambda **kw: create_quality_targeted_family_router(
        strength=kw.get('strength', 'moderate'),
        routing_mode=kw.get('routing_mode', 'family_aware'),
        enhanced_strategy_names=tuple(kw.get('enhanced_strategy_names') or ()),
    ),
    
    # Component pipelines
    'degrade': create_degrade_quality_pipeline,
    'enhance': create_enhance_quality_pipeline,
    'social_media': create_social_media_pipeline,
    'simulator': create_v6_simulator_pipeline,
    'generalist': create_v6_generalist_pipeline,
    'purist': create_v6_purist_pipeline,
    'mild': create_v6_mild_pipeline,

    # Webcam / video-call codec simulation
    'webcam_codec': lambda **kw: create_webcam_codec_pipeline(
        codec_quality=kw.get('codec_quality', (25, 75)),
        downscale_range=kw.get('downscale_range', (0.5, 0.85)),
        p=kw.get('p', 0.5),
    ),
    'codec': lambda **kw: create_webcam_codec_pipeline(
        codec_quality=kw.get('codec_quality', (25, 75)),
        downscale_range=kw.get('downscale_range', (0.5, 0.85)),
        p=kw.get('p', 0.5),
    ),
}


def register_pipeline(version: Union[int, str]):
    """
    Decorator to register a new pipeline factory.
    
    Example:
        @register_pipeline(version=8)
        def create_v8_pipeline(config=None):
            return A.Compose([...])
    """
    def decorator(fn: Callable) -> Callable:
        PIPELINE_REGISTRY[version] = fn
        return fn
    return decorator


def get_pipeline(
    version: Union[int, str],
    config: dict = None,
    frame_properties: dict = None,
    **kwargs
) -> PipelineType:
    """
    Get an augmentation pipeline by version.
    
    Args:
        version: Pipeline identifier (int like 3,4,5,6,7 or str like 'surgical', 'general')
        config: Configuration dict (required for 'surgical' and 'general' pipelines)
        frame_properties: Frame properties dict (required for 'surgical' pipeline)
        **kwargs: Additional arguments passed to the factory function
        
    Returns:
        - For versions 3-5: A.Compose pipeline
        - For versions 6-7: Callable function (img_np, [frame_dict]) -> img_np
        - For 'surgical': A.Compose pipeline tailored to frame_properties
        - For 'general': A.Compose pipeline based on config
        
    Raises:
        ValueError: If version is not found in registry
        ValueError: If required arguments are missing for special pipelines
        
    Example:
        # Simple version-based
        pipeline = get_pipeline(3)
        result = pipeline(image=img)['image']
        
        # V6 with frame context
        augment = get_pipeline(6)
        result = augment(img, {'path': 'gs://bucket/frame.png'})
        
        # Surgical with properties
        pipeline = get_pipeline('surgical', config=cfg, frame_properties=props)
    """
    if version not in PIPELINE_REGISTRY:
        available = sorted([k for k in PIPELINE_REGISTRY.keys() if isinstance(k, int)])
        raise ValueError(
            f"Unknown augmentation version: {version}. "
            f"Available versions: {available} and aliases like 'surgical', 'general', etc."
        )
    
    factory = PIPELINE_REGISTRY[version]
    
    # Handle special cases that need arguments
    if version == 'surgical':
        if config is None:
            raise ValueError("'surgical' pipeline requires 'config' argument")
        if frame_properties is None:
            raise ValueError("'surgical' pipeline requires 'frame_properties' argument")
        return factory(config=config, frame_properties=frame_properties, **kwargs)
    
    if version == 'general':
        if config is None:
            raise ValueError("'general' pipeline requires 'config' argument")
        return factory(config=config, **kwargs)
    
    # For V6/V7, the factory returns the callable directly
    if version in (6, 7, 'v6', 'v7', 'source_dependent', 'portfolio'):
        return factory(**kwargs)
    
    # For static pipelines, call factory with no args (or pass kwargs if needed)
    return factory(**kwargs) if kwargs else factory()


def get_pipeline_info(version: Union[int, str] = None) -> dict:
    """
    Get information about available pipelines.
    
    Args:
        version: Specific version to get info for, or None for all
        
    Returns:
        Dictionary with pipeline information
    """
    info = {
        3: {
            'name': 'V3 Legacy',
            'type': 'static',
            'description': 'Conservative augmentations, original production pipeline',
            'aliases': ['v3', 'legacy'],
        },
        4: {
            'name': 'V4 Moderate',
            'type': 'static', 
            'description': 'Moderately aggressive, step-up from V3',
            'aliases': ['v4', 'moderate'],
        },
        5: {
            'name': 'V5 Hybrid',
            'type': 'static',
            'description': 'V4 + occasional heavy degradation',
            'aliases': ['v5', 'hybrid'],
        },
        6: {
            'name': 'V6 Source-Dependent',
            'type': 'callable',
            'description': 'Different augmentation portfolios based on data source',
            'aliases': ['v6', 'source_dependent'],
            'signature': 'apply_augmentation_v6(img_np, frame_dict) -> img_np',
        },
        7: {
            'name': 'V7 Portfolio',
            'type': 'callable',
            'description': 'Unified portfolio without source checking',
            'aliases': ['v7', 'portfolio'],
            'signature': 'apply_augmentation_v7(img_np) -> img_np',
        },
        'surgical': {
            'name': 'Surgical',
            'type': 'dynamic',
            'description': 'Property-aware augmentation based on frame characteristics',
            'requires': ['config', 'frame_properties'],
        },
        'general': {
            'name': 'General',
            'type': 'configurable',
            'description': 'Configurable pipeline for non-property-based strategies',
            'requires': ['config'],
        },
    }
    
    if version is not None:
        # Normalize version to canonical form
        canonical = version
        if version in ('v3', 'legacy'):
            canonical = 3
        elif version in ('v4', 'moderate'):
            canonical = 4
        elif version in ('v5', 'hybrid'):
            canonical = 5
        elif version in ('v6', 'source_dependent'):
            canonical = 6
        elif version in ('v7', 'portfolio'):
            canonical = 7
        return info.get(canonical, {'error': f'Unknown version: {version}'})
    
    return info


# Expose pre-instantiated pipelines for backward compatibility
__all__ = [
    # Main API
    'get_pipeline',
    'register_pipeline',
    'PIPELINE_REGISTRY',
    'get_pipeline_info',
    # Pre-instantiated (backward compat)
    'revised_augmentation_pipeline_legacy',
    'augmentation_pipeline_v4',
    'augmentation_pipeline_v5',
    'AUG_PIPELINE_V6_SIMULATOR',
    'AUG_PIPELINE_V4_GENERALIST',
    'AUG_PIPELINE_PURIST',
    'AUG_PIPELINE_V3_MILD',
    'degrade_quality_pipeline',
    'enhance_quality_pipeline',
    'social_media_pipeline',
    # Callable functions
    'apply_augmentation_v6',
    'apply_augmentation_v7',
    # Factory functions
    'create_surgical_augmentation_pipeline',
    'create_general_augmentation_pipeline',
]
