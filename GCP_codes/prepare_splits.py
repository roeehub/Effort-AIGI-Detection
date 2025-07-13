# prepare_splits.py
import yaml
import random
from collections import defaultdict
from fsspec.core import get_fs_for_uri
from pathlib import Path
from dataclasses import dataclass

@dataclass
class VideoInfo:
    label: str          # 'real' or 'fake'
    method: str         # The source or generation method
    video_id: str       # The unique video folder name
    frame_paths: list   # List of full gs:// paths to frames

def prepare_video_splits(config_path: str):
    """
    Lists files in GCS, groups them by video, and creates stratified,
    reproducible train/validation splits at the video level.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cfg_gcp = config['gcp']
    cfg_data = config['data_params']
    cfg_methods = config['methods']

    BUCKET_URI = f"gs://{cfg_gcp['bucket_name']}"
    SEED = cfg_data['seed']
    VAL_RATIO = cfg_data['val_split_ratio']
    SUBSET_PERCENT = cfg_data['data_subset_percentage']  
    
    # Use fsspec to interact with GCS
    fs = get_fs_for_uri(BUCKET_URI)[0]

    print("Listing all files in bucket (this may take a moment)...")
    all_files = fs.glob(f"{BUCKET_URI}/**")
    print(f"Found {len(all_files)} total file paths.")

    # Group frame paths by video
    video_groups = defaultdict(lambda: defaultdict(list))
    allowed_methods = set(cfg_methods['use_real_sources'] + cfg_methods['use_fake_methods'])

    for file_path in all_files:
        # Using pathlib for robust path manipulation
        p = Path(file_path)
        if p.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
            continue # Skip non-image files like .DS_Store

        try:
            label = p.parts[-4]  # 'real' or 'fake'
            method = p.parts[-3] # 'simswap', 'FaceForensics++', etc.
            video_id = p.parts[-2] # The video folder name
        except IndexError:
            continue # Skip files not in the expected directory structure

        if method in allowed_methods:
            video_groups[(label, method)][video_id].append(f"gs://{file_path}")
            
    # Convert groups to a flat list of VideoInfo objects
    all_videos = []
    for (label, method), videos in video_groups.items():
        for video_id, frames in videos.items():
            # Sort frames for consistent ordering
            frames.sort()
            all_videos.append(VideoInfo(label, method, video_id, frames))

    print(f"Discovered {len(all_videos)} videos across {len(video_groups)} methods/sources.")

    # --- Stratified Splitting Logic ---
    random.seed(SEED)
    train_videos = []
    val_videos = []

    # Group by method for stratification
    videos_by_method = defaultdict(list)
    for video in all_videos:
        videos_by_method[video.method].append(video)

    for method, video_list in videos_by_method.items():
        # 1. Apply subset percentage if not 100%
        if SUBSET_PERCENT < 1.0:
            random.shuffle(video_list)
            subset_size = int(len(video_list) * SUBSET_PERCENT)
            video_list = video_list[:subset_size]

        if not video_list:
            continue

        # 2. Shuffle and split into train/val for this method
        random.shuffle(video_list)
        val_size = int(len(video_list) * VAL_RATIO)
        
        # Ensure at least one validation sample if possible, and one training sample
        val_size = max(1, val_size) if len(video_list) > 1 else 0
        
        val_videos.extend(video_list[:val_size])
        train_videos.extend(video_list[val_size:])

    print(f"Split complete. Train videos: {len(train_videos)}, Val videos: {len(val_videos)}")
    
    # Final shuffle of the entire train/val sets
    random.shuffle(train_videos)
    random.shuffle(val_videos)
    
    return train_videos, val_videos, config