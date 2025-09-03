import cv2
import numpy as np
import random
import albumentations as A
import matplotlib.pyplot as plt
import logging
import os
from typing import Dict, List

# --- Setup basic logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger(__name__)

# ==============================================================================
# --- AUGMENTATION PIPELINES (albumentations==0.4.6 compatible) ---
# ==============================================================================

# --- Pipeline to aggressively degrade image quality ---
degrade_quality_pipeline = A.Compose([
    A.OneOf([
        # Simulates re-encoding artifacts
        A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8),
        # Simulates motion or out-of-focus blur
        A.GaussianBlur(blur_limit=(5, 11), p=0.6),
        # Simulates generic noise
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
    ], p=1.0)
])

# --- Pipeline to enhance image quality ---
enhance_quality_pipeline = A.Compose([
    # This is an effective "unsharp mask" that sharpens details.
    # Note: `IAASharpen` comes from the 'imgaug' library integration in Albumentations.
    A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
])

# --- Pipeline to simulate social media compression ---
social_media_pipeline = A.Compose([
    # First, slightly blur the image to mimic initial quality loss
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    # Downscale and then upscale to simulate resolution changes during upload
    A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=cv2.INTER_AREA, p=0.8),
    # Apply strong JPEG compression, a hallmark of social media platforms
    A.ImageCompression(quality_lower=30, quality_upper=60, p=1.0),
], p=1.0)


def create_surgical_augmentation_pipeline(
        config: Dict,
        frame_properties: Dict
) -> A.Compose:
    """
    Dynamically constructs an Albumentations augmentation pipeline based on a
    configuration dictionary and specific frame properties.

    Args:
        config (Dict): A dictionary controlling which augmentation categories are active.
                       Example: {'use_geometric': True, 'use_occlusion': False, ...}
        frame_properties (Dict): A dictionary containing metadata about the frame,
                                 notably 'sharpness_bucket' (e.g., 'q1', 'q4').

    Returns:
        A.Compose: The configured Albumentations pipeline.
    """
    transforms_list = []

    # --- 1. Base & Geometric Augmentations ---
    # HorizontalFlip is almost always beneficial for face detectors.
    transforms_list.append(A.HorizontalFlip(p=0.5))

    if config.get('use_geometric', False):
        log.info("... Adding Geometric augmentations (ShiftScaleRotate)")
        # Combines shifting, scaling, and rotation in one efficient operation.
        # Parameters are kept subtle to avoid unrealistic distortion of faces.
        transforms_list.append(A.ShiftScaleRotate(
            shift_limit=0.0625,  # Max 6.25% shift in any direction
            scale_limit=0.1,  # Scale between 90% and 110%
            rotate_limit=7,  # Rotate between -7 and +7 degrees
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.7
        ))

    # --- 2. Surgical Sharpness Adjustment ---
    # This is the core of your property-based strategy.
    sharpness_bucket = frame_properties.get('sharpness_bucket')
    chance_for_sharpness_adjustment = config.get('sharpness_adjust_prob', 0.5)

    if random.random() < chance_for_sharpness_adjustment:
        if sharpness_bucket == 'q4':  # Very sharp image
            log.info(f"... Frame is sharp (q4), applying DEGRADE pipeline to create counter-example.")
            transforms_list.append(degrade_quality_pipeline)
        elif sharpness_bucket == 'q1':  # Very blurry image
            log.info(f"... Frame is blurry (q1), applying ENHANCE pipeline to create counter-example.")
            transforms_list.append(enhance_quality_pipeline)

    # --- 3. Advanced Noise & Artifact Simulation ---
    if config.get('use_advanced_noise', False):
        log.info("... Adding Advanced Noise/Artifact augmentations")
        # Probabilistically applies one of several realistic artifact simulations.
        transforms_list.append(A.OneOf([
            # Simulates noise from a digital camera sensor.
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            # Simulates the harsh compression of social media platforms.
            social_media_pipeline
        ], p=config.get('advanced_noise_prob', 0.6)))

    # --- 4. Occlusion Augmentations ---
    if config.get('use_occlusion', False):
        log.info("... Adding Occlusion augmentation (Cutout)")
        # In albumentations v0.4.6, 'Cutout' is the equivalent of 'RandomErasing'.
        # It removes rectangular patches from the image, forcing the model
        # to learn from distributed features rather than single local cues.
        transforms_list.append(A.Cutout(
            num_holes=8,  # Number of patches to remove
            max_h_size=24,  # Max height of a patch
            max_w_size=24,  # Max width of a patch
            fill_value=0,  # Fill with black
            p=config.get('occlusion_prob', 0.5)
        ))

    return A.Compose(transforms_list)


def main():
    """
    Main function to load a test image, apply various augmentation strategies,
    and display the results in a grid for visual comparison.
    """
    test_image_path = "/Users/roeedar/Desktop/faces_yolo/face_n_4.jpg"
    log.info(f"Loading test image from: {test_image_path}")

    if not os.path.exists(test_image_path):
        log.error(f"Test image not found at the specified path. Please update the path.")
        return

    # Load image with OpenCV and convert from BGR (OpenCV default) to RGB (matplotlib default)
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- Define different test scenarios ---
    # 1. Mock frame properties to test the "surgical" logic
    sharp_frame_props = {'sharpness_bucket': 'q4'}
    blurry_frame_props = {'sharpness_bucket': 'q1'}

    # 2. Define augmentation configurations to test
    configs_to_test = {
        "Base (No Aug)": {},
        "Surgical Only": {'sharpness_adjust_prob': 1.0},
        "Geometric": {'use_geometric': True},
        "Adv. Noise": {'use_advanced_noise': True},
        "Occlusion (Cutout)": {'use_occlusion': True},
        "All On": {
            'use_geometric': True,
            'use_advanced_noise': True,
            'use_occlusion': True,
            'sharpness_adjust_prob': 1.0
        }
    }

    # --- Create visualization grid ---
    # Rows: Original, Sharp Frame (Degraded), Blurry Frame (Enhanced)
    # Columns: Configuration Names
    num_rows = 3
    num_cols = len(configs_to_test)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 12))
    fig.suptitle("Flexible Augmentation Pipeline Visualizer (albumentations==0.4.6)", fontsize=20)

    for col_idx, (config_name, config) in enumerate(configs_to_test.items()):
        log.info(f"--- Testing Configuration: '{config_name}' ---")

        # --- Row 0: Original Image with this config ---
        # (This row doesn't use surgical props, just the general augmentations)
        ax = axes[0, col_idx]
        if col_idx == 0:
            ax.imshow(image)
            ax.set_title("Original Image", fontsize=12)
        else:
            pipeline_orig = create_surgical_augmentation_pipeline(config, {})
            augmented_orig = pipeline_orig(image=image)['image']
            ax.imshow(augmented_orig)
            ax.set_title(config_name, fontsize=12)
        ax.set_ylabel("Base", fontsize=14, rotation=0, labelpad=40)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Row 1: Simulating a SHARP frame (should be degraded) ---
        ax = axes[1, col_idx]
        pipeline_sharp = create_surgical_augmentation_pipeline(config, sharp_frame_props)
        augmented_sharp = pipeline_sharp(image=image)['image']
        ax.imshow(augmented_sharp)
        ax.set_ylabel("Sharp Frame\n(Degraded)", fontsize=14, rotation=0, labelpad=40)
        ax.set_xticks([])
        ax.set_yticks([])

        # --- Row 2: Simulating a BLURRY frame (should be enhanced) ---
        ax = axes[2, col_idx]
        pipeline_blurry = create_surgical_augmentation_pipeline(config, blurry_frame_props)
        augmented_blurry = pipeline_blurry(image=image)['image']
        ax.imshow(augmented_blurry)
        ax.set_ylabel("Blurry Frame\n(Enhanced)", fontsize=14, rotation=0, labelpad=40)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    log.info("Displaying augmentation grid. Close the window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
