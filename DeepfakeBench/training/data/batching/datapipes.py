"""
Custom DataPipes for torchdata-based data loading.

These DataPipes handle custom sampling strategies for the property-balanced
and round-robin data loading approaches.
"""
import random
from typing import Any, Callable, Dict, Iterator, List, Optional

from torchdata.datapipes.iter import IterDataPipe


class CustomRoundRobinDataPipe(IterDataPipe):
    """A DataPipe that yields items from multiple source DataPipes in round-robin order.
    
    This is used for the per_method strategy to cycle through different generation
    methods evenly during training.
    """
    
    def __init__(self, datapipes: List[IterDataPipe]):
        self.datapipes = datapipes
        self.num_sources = len(datapipes)
    
    def __iter__(self) -> Iterator:
        iterators = [iter(dp) for dp in self.datapipes]
        active = list(range(self.num_sources))
        
        while active:
            next_active = []
            for idx in active:
                try:
                    yield next(iterators[idx])
                    next_active.append(idx)
                except StopIteration:
                    pass
            active = next_active


class CustomSampleMultiplexerDataPipe(IterDataPipe):
    """A DataPipe that samples from multiple sources based on weights.
    
    This is the core component of the property_balancing strategy, enabling
    hierarchical sampling where each source (e.g., method category) is sampled
    according to its configured weight.
    
    Usage:
        >>> pipes = [pipe_for_cat_A, pipe_for_cat_B]
        >>> weights = [0.6, 0.4]  # 60% from A, 40% from B
        >>> multiplexer = CustomSampleMultiplexerDataPipe(pipes, weights)
    """
    
    def __init__(self, datapipes: List[IterDataPipe], weights: List[float]):
        """Initialize the multiplexer.
        
        Args:
            datapipes: List of source DataPipes to sample from
            weights: Sampling probabilities for each source (should sum to 1.0)
        """
        if not datapipes:
            raise ValueError("CustomSampleMultiplexerDataPipe requires at least one datapipe")
        if len(datapipes) != len(weights):
            raise ValueError("Number of datapipes must match number of weights")
        
        self.datapipes = datapipes
        self.weights = weights
    
    def __iter__(self) -> Iterator:
        # Use itertools.cycle for infinite iteration over each source
        import itertools
        iterators = [itertools.cycle(dp) for dp in self.datapipes]
        
        while True:
            # random.choices returns a list, so we take [0]
            chosen_idx = random.choices(range(len(iterators)), weights=self.weights, k=1)[0]
            yield next(iterators[chosen_idx])


class MateFinderDataPipe(IterDataPipe):
    """A DataPipe that finds 'mate' frames from the same video clip.
    
    Given an anchor frame, this pipe finds additional frames from the same clip_id
    to create temporal consistency in training batches. This is crucial for the
    property_balancing strategy's anchor-mate approach.
    
    The pipe yields `frames_per_video` frames for each anchor frame received.
    """
    
    def __init__(
        self,
        source_dp: IterDataPipe,
        clip_lookup: Dict[str, List[Dict]],
        frames_per_video: int = 2
    ):
        """Initialize the mate finder.
        
        Args:
            source_dp: Source DataPipe providing anchor frames
            clip_lookup: Dictionary mapping clip_id to list of frame dictionaries
            frames_per_video: Number of frames to yield per video (including anchor)
        """
        self.source_dp = source_dp
        self.clip_lookup = clip_lookup
        self.frames_per_video = frames_per_video
    
    def __iter__(self) -> Iterator[Dict]:
        for anchor_frame in self.source_dp:
            clip_id = anchor_frame.get('clip_id')
            
            if clip_id is None:
                # No clip_id - just yield the anchor
                yield anchor_frame
                continue
            
            # Get all frames from this clip
            clip_frames = self.clip_lookup.get(clip_id, [anchor_frame])
            
            # Select frames_per_video frames (including anchor)
            if len(clip_frames) <= self.frames_per_video:
                # Not enough frames - yield all we have
                for frame in clip_frames:
                    yield frame
            else:
                # Sample frames_per_video - 1 mates (anchor is already selected)
                # Always include the anchor
                other_frames = [f for f in clip_frames if f is not anchor_frame]
                if other_frames:
                    mates = random.sample(
                        other_frames, 
                        min(self.frames_per_video - 1, len(other_frames))
                    )
                else:
                    mates = []
                
                yield anchor_frame
                for mate in mates:
                    yield mate


def build_clip_to_frames_lookup(frames: List[Dict]) -> Dict[str, List[Dict]]:
    """Build a lookup table from clip_id to frame dictionaries.
    
    This is used by MateFinderDataPipe to efficiently find all frames
    belonging to the same video clip.
    
    Args:
        frames: List of frame dictionaries, each with a 'clip_id' key
        
    Returns:
        Dictionary mapping clip_id to list of frame dictionaries
    """
    from collections import defaultdict
    
    lookup = defaultdict(list)
    for frame in frames:
        clip_id = frame.get('clip_id')
        if clip_id:
            lookup[clip_id].append(frame)
    
    return dict(lookup)
