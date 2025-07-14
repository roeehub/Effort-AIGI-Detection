# dataloaders.py
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, Zipper, Demultiplexer
from torchdata.datapipes.map import Mapper
from fsspec.core import open_files
from collections import defaultdict

from prepare_splits import VideoInfo, prepare_video_splits

def load_video_frames(video_info: VideoInfo, num_frames: int, augmentation: callable = None):
    """
    A 'map' function that takes a VideoInfo object, samples frames,
    downloads them from GCS, and processes them into a tensor.
    """
    # 1. Sample `num_frames` from the available frames
    if len(video_info.frame_paths) > num_frames:
        frame_paths_to_load = random.sample(video_info.frame_paths, num_frames)
    else:
        # If fewer frames than required, sample with replacement
        frame_paths_to_load = random.choices(video_info.frame_paths, k=num_frames)

    # 2. Open the selected files from GCS in a single batch request
    # open_files is efficient as it can use asyncio underneath
    files = open_files(frame_paths_to_load, mode='rb')
    
    # 3. Load images and stack them into a tensor
    frames = []
    for f in files:
        with f as stream:
            img = Image.open(stream).convert('RGB')
            frames.append(transforms.ToTensor()(img))
    
    # Stack frames into a (C, T, H, W) tensor
    video_tensor = torch.stack(frames, dim=1)

    # 4. Apply augmentations if provided
    if augmentation:
        video_tensor = augmentation(video_tensor)

    label_tensor = torch.tensor(1.0 if video_info.label == 'real' else 0.0)

    return video_tensor, label_tensor, video_info.method

def create_base_videopipe(videos, num_frames, augmentation):
    return (
        IterableWrapper(videos)
        .shuffle()
        .sharding_filter()
        .map(lambda x: load_video_frames(x, num_frames, augmentation))
        .prefetch(10) # Prefetch items for smoother training
    )

### Dataloader Implementations ###

# 1. Naive Dataloader (50% fake, 50% real)
def create_naive_dataloader(train_videos, config):
    cfg_data = config['data_params']
    cfg_dl = config['dataloader_params']
    
    real_videos = [v for v in train_videos if v.label == 'real']
    fake_videos = [v for v in train_videos if v.label == 'fake']

    print(f"Naive Dataloader: {len(real_videos)} real videos, {len(fake_videos)} fake videos.")
    
    real_pipe = create_base_videopipe(real_videos, cfg_data['num_frames_per_video'], None)
    fake_pipe = create_base_videopipe(fake_videos, cfg_data['num_frames_per_video'], None)

    # Zip pairs a sample from each pipe together, ensuring a 1:1 ratio
    # We use a long-running pipe and a shorter one to cycle through the data
    # whichever is shorter will be the length of the dataset
    combined_pipe = Zipper(real_pipe, fake_pipe).flatmap(lambda x: x)
    
    return DataLoader(
        combined_pipe,
        batch_size=cfg_dl['batch_size'],
        num_workers=cfg_dl['num_workers'],
    )

# 2. Method-Aware Dataloader
def create_method_aware_dataloaders(train_videos, config):
    cfg_data = config['data_params']
    cfg_dl = config['dataloader_params']
    
    # Group videos by method
    videos_by_method = defaultdict(list)
    for v in train_videos:
        videos_by_method[v.method].append(v)
        
    dataloaders = {}
    for method, videos in videos_by_method.items():
        print(f"Creating dataloader for method '{method}' with {len(videos)} videos.")
        pipe = create_base_videopipe(videos, cfg_data['num_frames_per_video'], None)
        dataloaders[method] = DataLoader(
            pipe, batch_size=cfg_dl['batch_size'], num_workers=cfg_dl['num_workers']
        )
    return dataloaders

# 3. For Method-Aware Balanced, you control it in the training loop (see train.py)

# 4. Your Suggestion: Simple Shuffled Dataloader (Good for baselines)
def create_simple_dataloader(train_videos, config):
    cfg_data = config['data_params']
    cfg_dl = config['dataloader_params']
    
    pipe = create_base_videopipe(train_videos, cfg_data['num_frames_per_video'], None)
    return DataLoader(
        pipe,
        batch_size=cfg_dl['batch_size'],
        num_workers=cfg_dl['num_workers'],
    )