"""
Property-balanced batching strategy implementation.

This is the most sophisticated strategy, using hierarchical sampling based on:
1. Method categories (real vs fake, with subcategories)
2. Property buckets (e.g., sharpness quartiles)
3. Anchor-mate pairing for temporal consistency

This ensures balanced exposure across both generation methods AND quality properties.
"""
import math
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper

from .base import BatchingStrategy
from .datapipes import CustomSampleMultiplexerDataPipe, MateFinderDataPipe, build_clip_to_frames_lookup
from .loaders import load_and_process_property_batch, collate_fn


class PropertyBalancedStrategy(BatchingStrategy):
    """
    Batching strategy that balances by both method categories and property buckets.
    
    This strategy implements the Anchor-Mate approach:
    1. First samples by method category (with configurable weights)
    2. Within each category, balances by property bucket (e.g., sharpness quartile)
    3. For each anchor frame, finds mate frames from the same video clip
    4. Loads frames in parallel batches for efficiency
    
    Configuration (via data_config['dataloader_params']):
    - frames_per_batch: GPU batch size
    - frames_per_video: Number of frames to sample per video/clip
    - real_label_ratio: Ratio of real samples (default 0.5)
    - real_category_weights: Dict of category -> weight for real samples
    - fake_category_weights: Dict of category -> weight for fake samples
    
    Or use lesson_data_control for dynamic method grouping:
    - lesson_data_control.enabled: True to use dynamic groups
    - lesson_data_control.real_method_groups: Dict of group_name -> {methods, weight}
    - lesson_data_control.fake_method_groups: Dict of group_name -> {methods, weight}
    
    Best for:
    - Training where property balance is crucial (preventing shortcut learning)
    - When you want to control the real/fake ratio explicitly
    - When you need fine-grained control over method exposure
    """
    
    @property
    def name(self) -> str:
        return "property_balancing"
    
    def create_train_loader(self, train_data: List[Dict]) -> DataLoader:
        """Create a property-balanced training dataloader.
        
        Args:
            train_data: List of frame dictionaries from Parquet manifest
            
        Returns:
            DataLoader with hierarchical property-balanced sampling
        """
        dl_params = self.data_config.get('dataloader_params', {})
        BATCH_SIZE = dl_params.get('frames_per_batch', 64)
        NUM_WORKERS = dl_params.get('num_workers', 8)
        FRAMES_PER_VIDEO = dl_params.get('frames_per_video', 2)
        
        print(f"Configuring property-balanced loader with {FRAMES_PER_VIDEO} frames per video.")
        
        # Get real/fake balance ratio
        real_label_ratio = self._get_real_label_ratio(dl_params)
        fake_label_ratio = 1.0 - real_label_ratio
        print(f"Using real/fake label ratio: {real_label_ratio:.2f} / {fake_label_ratio:.2f}")
        
        # Minimum bucket size for efficiency
        MIN_BUCKET_SIZE = BATCH_SIZE
        
        # Ensure all frames have label_id
        self._ensure_label_ids(train_data)
        
        # Build clip lookup for mate finding
        clip_lookup = build_clip_to_frames_lookup(train_data)
        print(f"Built clip_id lookup table with {len(clip_lookup)} unique clips.")
        
        # Check if using lesson data control
        lesson_data_control = self.config.get('lesson_data_control', {})
        use_lesson_control = lesson_data_control.get('enabled', False)
        
        # Create master streams for real and fake data
        if use_lesson_control:
            master_real_stream = self._create_dynamic_real_stream(
                train_data, lesson_data_control, MIN_BUCKET_SIZE
            )
            master_fake_stream = self._create_dynamic_fake_stream(
                train_data, lesson_data_control, MIN_BUCKET_SIZE
            )
        else:
            master_real_stream = self._create_static_real_stream(
                train_data, dl_params, MIN_BUCKET_SIZE
            )
            master_fake_stream = self._create_static_fake_stream(
                train_data, dl_params, MIN_BUCKET_SIZE
            )
        
        # Final multiplexer with real/fake balance
        anchor_pipe = CustomSampleMultiplexerDataPipe(
            [master_real_stream, master_fake_stream],
            [real_label_ratio, fake_label_ratio]
        )
        
        # Sharding for multi-worker support
        if NUM_WORKERS > 0:
            anchor_pipe = anchor_pipe.sharding_filter()
        anchor_pipe = anchor_pipe.shuffle(buffer_size=10000)
        
        # Find mates for each anchor
        combined_pipe = MateFinderDataPipe(anchor_pipe, clip_lookup, frames_per_video=FRAMES_PER_VIDEO)
        
        # Batch for I/O efficiency, then parallel load
        io_batch_size = BATCH_SIZE * 4
        combined_pipe = combined_pipe.batch(io_batch_size).flatmap(
            partial(load_and_process_property_batch, config=self.config, mode='train')
        )
        
        return DataLoader(
            combined_pipe,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn=collate_fn,
            persistent_workers=True if NUM_WORKERS > 0 else False,
            prefetch_factor=dl_params.get('prefetch_factor', 4)
        )
    
    def _get_real_label_ratio(self, dl_params: Dict) -> float:
        """Get and validate the real/fake label ratio."""
        real_label_ratio = dl_params.get('real_label_ratio')
        if real_label_ratio is None:
            return 0.5
        if not (0.0 < real_label_ratio < 1.0):
            print(f"real_label_ratio must be between 0 and 1. Got: {real_label_ratio}")
            print("Defaulting to 0.5 (50/50 balance).")
            return 0.5
        return real_label_ratio
    
    def _ensure_label_ids(self, frames: List[Dict]) -> None:
        """Ensure all frames have label_id field."""
        for frame_info in frames:
            if 'label_id' not in frame_info:
                if 'label' not in frame_info:
                    raise KeyError(f"Frame dictionary missing 'label' and 'label_id': {frame_info}")
                frame_info['label_id'] = 0 if frame_info['label'] == 'real' else 1
    
    def _create_dynamic_real_stream(
        self,
        all_frames: List[Dict],
        lesson_control: Dict,
        min_bucket_size: int
    ):
        """Create real data stream using dynamic method grouping."""
        print("--- Using DYNAMIC method grouping for REAL data stream. ---")
        
        real_method_groups = lesson_control.get('real_method_groups', {})
        all_real_frames = [f for f in all_frames if f['label_id'] == 0]
        
        streams = []
        weights = []
        group_info_log = []
        
        for group_name, group_info in real_method_groups.items():
            group_methods = set(group_info.get('methods', []))
            group_weight = group_info.get('weight')
            
            frames_for_group = [f for f in all_real_frames if f.get('method') in group_methods]
            
            if not frames_for_group:
                print(f"WARNING: No frames found for real group '{group_name}', skipping.")
                continue
            
            print(f"  - Group '{group_name}': Found {len(frames_for_group)} frames from {len(group_methods)} method(s).")
            
            # Group by property bucket
            frames_by_bucket = defaultdict(list)
            for frame in frames_for_group:
                frames_by_bucket[frame['sharpness_bucket']].append(frame)
            
            # Create property-balanced stream for this group
            consolidated = self._consolidate_small_buckets(frames_by_bucket, min_bucket_size, group_name)
            group_stream = self._create_master_stream_for_label(consolidated)
            
            if group_stream:
                streams.append(group_stream)
                weights.append(group_weight if group_weight is not None else 1.0)
                group_info_log.append({
                    'name': group_name,
                    'methods': list(group_methods),
                    'frame_count': len(frames_for_group),
                    'specified_weight': group_weight
                })
        
        if not streams:
            raise ValueError("Cannot create dataloader: no valid REAL data streams created.")
        
        # Normalize weights
        weights = self._normalize_weights(weights)
        self._log_sampling_summary("REAL", group_info_log, weights)
        
        return CustomSampleMultiplexerDataPipe(streams, weights)
    
    def _create_dynamic_fake_stream(
        self,
        all_frames: List[Dict],
        lesson_control: Dict,
        min_bucket_size: int
    ):
        """Create fake data stream using dynamic method grouping."""
        print("--- Using DYNAMIC method grouping for FAKE data stream. ---")
        
        fake_method_groups = lesson_control.get('fake_method_groups', {})
        if not fake_method_groups:
            raise ValueError("`lesson_data_control` is enabled, but `fake_method_groups` is empty.")
        
        all_fake_frames = [f for f in all_frames if f['label_id'] == 1]
        
        streams = []
        weights = []
        group_info_log = []
        
        for group_name, group_info in fake_method_groups.items():
            group_methods = set(group_info.get('methods', []))
            group_weight = group_info.get('weight')
            
            if not group_methods:
                print(f"WARNING: Dynamic group '{group_name}' has no methods defined, skipping.")
                continue
            
            frames_for_group = [f for f in all_fake_frames if f['method'] in group_methods]
            
            if not frames_for_group:
                print(f"WARNING: No frames found for dynamic group '{group_name}', skipping.")
                continue
            
            print(f"  - Group '{group_name}': Found {len(frames_for_group)} frames across {len(group_methods)} methods.")
            
            # Group by property bucket
            frames_by_bucket = defaultdict(list)
            for frame in frames_for_group:
                frames_by_bucket[frame['sharpness_bucket']].append(frame)
            
            # Create property-balanced stream
            consolidated = self._consolidate_small_buckets(frames_by_bucket, min_bucket_size, group_name)
            group_stream = self._create_master_stream_for_label(consolidated)
            
            if group_stream:
                streams.append(group_stream)
                weights.append(group_weight if group_weight is not None else 1.0)
                group_info_log.append({
                    'name': group_name,
                    'methods': list(group_methods),
                    'frame_count': len(frames_for_group),
                    'specified_weight': group_weight
                })
        
        if not streams:
            raise ValueError("Cannot create dataloader: no valid FAKE data streams created.")
        
        # Normalize weights
        weights = self._normalize_weights(weights)
        self._log_sampling_summary("FAKE", group_info_log, weights)
        
        return CustomSampleMultiplexerDataPipe(streams, weights)
    
    def _create_static_real_stream(
        self,
        all_frames: List[Dict],
        dl_params: Dict,
        min_bucket_size: int
    ):
        """Create real data stream using unified pooling (no per-method weighting)."""
        print("--- Using UNIFIED real data stream (no per-method weighting). ---")
        
        allowed_real_sources = self.data_config.get('dataset_methods', {}).get('use_real_sources', [])
        if not allowed_real_sources:
            raise ValueError("`dataset_methods.use_real_sources` is empty or not defined.")
        
        allowed_set = set(allowed_real_sources)
        all_real_frames = [
            f for f in all_frames
            if f['label_id'] == 0 and f.get('method') in allowed_set
        ]
        
        if not all_real_frames:
            raise ValueError("No REAL frames found for the sources specified in `use_real_sources`.")
        
        # Group by property bucket
        frames_by_bucket = defaultdict(list)
        for frame in all_real_frames:
            frames_by_bucket[frame['sharpness_bucket']].append(frame)
        
        consolidated = self._consolidate_small_buckets(frames_by_bucket, min_bucket_size, "real_master")
        master_stream = self._create_master_stream_for_label(consolidated)
        
        if not master_stream:
            raise ValueError("Cannot create dataloader: the master REAL data stream could not be created.")
        
        return master_stream
    
    def _create_static_fake_stream(
        self,
        all_frames: List[Dict],
        dl_params: Dict,
        min_bucket_size: int
    ):
        """Create fake data stream using static category weighting."""
        print("--- Using STATIC method_category grouping for FAKE data stream. ---")
        
        fake_category_weights = dl_params.get('fake_category_weights', {})
        
        if not fake_category_weights or abs(sum(fake_category_weights.values()) - 1.0) > 1e-6:
            raise ValueError(f"Fake category weights do not sum to 1.0! Got: {sum(fake_category_weights.values())}")
        
        print(f"Using FAKE category weights: {fake_category_weights}")
        
        # Segregate fake frames by category
        fake_frames_by_category = defaultdict(list)
        for frame in all_frames:
            if frame['label_id'] == 1:
                category = frame.get('method_category')
                if category and category in fake_category_weights:
                    fake_frames_by_category[category].append(frame)
        
        streams = []
        weights = []
        
        for category, weight in fake_category_weights.items():
            frames = fake_frames_by_category.get(category)
            if not frames:
                print(f"WARNING: No frames found for FAKE category '{category}', skipping.")
                continue
            
            frames_by_bucket = defaultdict(list)
            for frame in frames:
                frames_by_bucket[frame['sharpness_bucket']].append(frame)
            
            consolidated = self._consolidate_small_buckets(frames_by_bucket, min_bucket_size, category)
            category_stream = self._create_master_stream_for_label(consolidated)
            
            if category_stream:
                streams.append(category_stream)
                weights.append(weight)
        
        if not streams:
            raise ValueError("Cannot create dataloader: no valid FAKE data streams created.")
        
        # Normalize weights if needed
        weight_sum = sum(weights)
        if not math.isclose(weight_sum, 1.0):
            print(f"WARNING: Fake weights sum to {weight_sum:.3f}, renormalizing to 1.0")
            weights = [w/weight_sum for w in weights]
        
        return CustomSampleMultiplexerDataPipe(streams, weights)
    
    def _consolidate_small_buckets(
        self,
        frames_by_bucket: Dict[str, List],
        min_size: int,
        source_name: str
    ) -> Dict[str, List]:
        """Merge small buckets to avoid tiny, inefficient cycled iterators."""
        consolidated = {}
        overflow = []
        
        for bucket_name, frames in frames_by_bucket.items():
            if len(frames) >= min_size:
                consolidated[bucket_name] = frames
            else:
                overflow.extend(frames)
        
        if overflow:
            if 'consolidated_overflow' in consolidated:
                consolidated['consolidated_overflow'].extend(overflow)
            else:
                consolidated['consolidated_overflow'] = overflow
        
        return consolidated
    
    def _create_master_stream_for_label(
        self,
        frames_by_bucket: Dict[str, List]
    ) -> Optional[CustomSampleMultiplexerDataPipe]:
        """Create a property-balanced stream from bucket-grouped frames."""
        if not frames_by_bucket:
            return None
        
        import itertools
        
        streams = []
        weights = []
        total_frames = sum(len(frames) for frames in frames_by_bucket.values())
        
        for bucket_name, frames in frames_by_bucket.items():
            if frames:
                # Create infinite cycled iterator wrapped in IterableWrapper
                cycled = itertools.cycle(frames)
                pipe = IterableWrapper(cycled)
                streams.append(pipe)
                # Weight by bucket size for even sampling
                weights.append(len(frames) / total_frames)
        
        if not streams:
            return None
        
        return CustomSampleMultiplexerDataPipe(streams, weights)
    
    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """Normalize weights to sum to 1.0."""
        weight_sum = sum(weights)
        if not math.isclose(weight_sum, 1.0):
            return [w/weight_sum for w in weights]
        return weights
    
    def _log_sampling_summary(
        self,
        label_type: str,
        group_info_log: List[Dict],
        final_weights: List[float]
    ) -> None:
        """Log a summary of the sampling configuration."""
        print(f"\n=== {label_type} METHOD SAMPLING SUMMARY ===")
        for i, group_info in enumerate(group_info_log):
            print(f"  Group '{group_info['name']}':")
            print(f"    Methods: {', '.join(group_info['methods'])}")
            print(f"    Frame count: {group_info['frame_count']:,}")
            print(f"    Specified weight: {group_info['specified_weight']}")
            print(f"    Final sampling weight: {final_weights[i]:.4f} ({final_weights[i]*100:.2f}%)")
        print(f"Total {label_type.lower()} streams: {len(group_info_log)}")
        print(f"Total {label_type.lower()} weight: {sum(final_weights):.6f}")
        print("=" * 40 + "\n")
