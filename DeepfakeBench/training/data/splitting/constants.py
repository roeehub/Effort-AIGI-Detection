"""
Constants for method categorization.

These sets define which methods belong to which category (EFS, face-swap, etc.)
and are used for identity extraction and splitting logic.
"""

import re
from typing import Set


# ---------------------------------------------------------------------------
# Method Categories
# ---------------------------------------------------------------------------

# Entire Face Synthesis (EFS) - methods that generate a new face from scratch
EFS_METHODS: Set[str] = {
    "DiT", "SiT", "ddim", "RDDM",
    "VQGAN", "StyleGAN2", "StyleGAN3", "StyleGANXL",
}

# Regular methods - source_target (target = 2nd token in video name)
REG_METHODS: Set[str] = {
    "simswap", "fsgan", "faceswap", "fomm", "facedancer", "inswap",
    "one_shot_free", "blendface", "lia", "mobileswap", "mcnet",
    "uniface", "MRAA", "facevid2vid", "wav2lip", "sadtalker", "danet",
    "e4s", "pirender", "tpsm",
}

# Reversed methods - target_source (target = 1st token in video name)
REV_METHODS: Set[str] = set()

# Methods to exclude from all splits
EXCLUDE_METHODS: Set[str] = {
    "hyperreenact",
}

# Regex for extracting 3-digit identity numbers
RE_3DIGIT = re.compile(r"\b(\d{3})\b")


# ---------------------------------------------------------------------------
# Method Category Mapping
# ---------------------------------------------------------------------------

def get_method_category(method: str) -> str:
    """
    Get the category for a given method.
    
    Returns:
        'efs' for Entire Face Synthesis
        'face_swap' for face swapping methods
        'face_reenact' for face reenactment methods
        'real' for real video sources
        'unknown' otherwise
    """
    if method in EFS_METHODS:
        return 'efs'
    elif method in REG_METHODS or method in REV_METHODS:
        # Could further distinguish between swap and reenact based on method
        return 'face_manipulation'
    elif method in {'FaceForensics++', 'Celeb-real', 'YouTube-real', 'external_youtube_avspeech'}:
        return 'real'
    return 'unknown'
