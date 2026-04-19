"""
VideoInfo dataclass for representing video metadata.

This is the core data structure used throughout the pipeline to represent
a single video with its frames and metadata.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class VideoInfo:
    """
    Represents a single video with its metadata and frame paths.
    
    Attributes:
        label: 'real' or 'fake'
        method: Generation/source method name
        video_id: Folder/identifier name for the video
        frame_paths: List of GCS paths to frame images
        identity: Target identity (numeric), used for identity-based splitting
        label_id: Numeric label (0 for real, 1 for fake)
    """
    label: str  # 'real' | 'fake'
    method: str  # generation/source method
    video_id: str  # folder name
    frame_paths: List[str]  # gs:// paths
    identity: int  # target identity (numeric)
    label_id: int = 0  # a numeric label ID

    def __post_init__(self):
        # 1) strict label check
        if self.label not in {"real", "fake"}:
            raise ValueError(
                f"[VideoInfo] Invalid label '{self.label}' "
                f"for video '{self.method}/{self.video_id}'. "
                "Allowed: 'real' or 'fake'."
            )
        self.label_id = 0 if self.label == 'real' else 1

        # 2) basic sanity on frame list
        if not self.frame_paths:
            raise ValueError(
                f"[VideoInfo] Empty frame list for video "
                f"'{self.method}/{self.video_id}'."
            )
    
    @property
    def num_frames(self) -> int:
        """Number of frames in this video."""
        return len(self.frame_paths)
    
    @property
    def is_real(self) -> bool:
        """Whether this is a real video."""
        return self.label == 'real'
    
    @property
    def is_fake(self) -> bool:
        """Whether this is a fake video."""
        return self.label == 'fake'
