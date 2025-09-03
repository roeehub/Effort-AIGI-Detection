# --- verify_property_dataloader.py ---

import os
import sys
import warnings
from collections import Counter, defaultdict
import random
import io  # <-- Add this import for byte streams

# --- Setup: Silence deprecation warnings and add project root to path ---
warnings.filterwarnings("ignore", category=UserWarning, message="The 'datapipes', 'dataloader2' modules are deprecated")
if '.' not in sys.path:
    sys.path.append('.')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from sklearn.model_selection import GroupShuffleSplit
import logging
import fsspec  # <-- Add fsspec import

# --- Imports needed for self-contained augmentation logic ---
from PIL import Image
from torchvision import transforms as T
import albumentations as A
from torch.utils.data import IterDataPipe

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# =================================================================================
# --- Configuration (Mimicking train_sweep.py and config files) ---
# =================================================================================
CONFIG = {
    'resolution': 224,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'use_data_augmentation': True,
}

DATA_CONFIG = {
    'dataloader_params': {
        'strategy': 'property_balancing',
        'frames_per_batch': 64,
        'num_workers': 4,
        'prefetch_factor': 2,
    },
    'data_params': {
        'seed': 42,
        'val_split_ratio': 0.1,  # 10% of real videos for validation
    },
    'methods': {
        'use_real_sources': ["FaceForensics++", "Celeb-real", "youtube-real", "external_youtube_avspeech"],
        'use_fake_methods_for_training': [
            "simswap", "mobileswap", "faceswap", "inswap", "blendface", "fsgan", "uniface",
            "pirender", "facevid2vid", "lia", "fomm", "MRAA", "wav2lip", "mcnet", "danet",
            "VQGAN", "StyleGAN3", "StyleGAN2", "SiT", "RDDM", "ddim"
        ],
    }
}
MANIFEST_PATH = "frame_properties.parquet"


# =================================================================================
# --- PICKLE-SAFE DATAPIPE & DATALOADER HELPERS ---
# =================================================================================

class CustomRoundRobinDataPipe(IterDataPipe):
    """
    A pickle-safe replacement for RoundRobinMultiplexer. It takes multiple
    input datapipes and yields one item from each in a continuous cycle.
    """

    def __init__(self, *datapipes):
        super().__init__()
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs for CustomRoundRobinDataPipe must be IterDataPipe objects.")
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        while True:
            for it in iterators:
                try:
                    yield next(it)
                except StopIteration:
                    # This should not be reached if input datapipes are .cycle()'d
                    return


class CustomSampleMultiplexerDataPipe(IterDataPipe):
    """
    A pickle-safe replacement for SampleMultiplexer. It probabilistically yields
    an item from one of the given datapipes based on the provided weights.
    """

    def __init__(self, datapipes, weights):
        super().__init__()
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs for CustomSampleMultiplexerDataPipe must be IterDataPipe objects.")
        self.datapipes = datapipes
        self.weights = weights
        if len(self.datapipes) != len(self.weights):
            raise ValueError("Number of datapipes must match the number of weights.")

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        indices = np.arange(len(self.datapipes))
        while True:
            # Choose the index of the next datapipe to pull from
            chosen_index = np.random.choice(indices, p=self.weights)
            try:
                yield next(iterators[chosen_index])
            except StopIteration:
                # This should also not be reached if inputs are infinite
                return


# --- NEW (SOLUTION): Worker initialization function for cloud access ---
def worker_init_fn(worker_id):
    """Initializes an fsspec filesystem object for each worker process."""
    global fs
    fs = fsspec.filesystem("gcs")


class FrameTransformer:
    """
    A pickle-safe callable class that applies transformations to a frame dictionary.
    Replaces the nested function `_transform_item` which caused pickling errors.
    """

    def __init__(self, resolution, mean, std, is_train):
        self.resolution = resolution
        self.is_train = is_train
        self.normalize_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __call__(self, frame_dict):
        """The actual transform logic for one data item."""
        try:
            # --- MODIFIED (SOLUTION): Use fsspec to read bytes from GCS first ---
            with fs.open(frame_dict['path'], 'rb') as f:
                image_bytes = f.read()

            # Now open the in-memory bytes with Pillow
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # --- END MODIFICATION ---

            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
            if self.is_train:
                img = surgical_data_aug(img, frame_dict)
            image_tensor = self.normalize_transform(img)
            return {**frame_dict, 'image': image_tensor}
        except Exception as e:
            # This will now catch genuine processing errors, not FileNotFound
            log.warning(f"Could not process file {frame_dict.get('path', 'N/A')}: {e}")
            return None


def get_video_id(item: dict) -> str:
    """Returns the video_id from a data item dictionary for grouping."""
    return item['video_id']


def get_group_elements(group: list) -> list:
    """Returns the list of frames from a group tuple created by .groupby()."""
    return group[1]


def is_not_none(item) -> bool:
    """Filter function to remove None items from the pipeline."""
    return item is not None


# =================================================================================
# --- NEW DATALOADER IMPLEMENTATION ---
# =================================================================================

def _consolidate_small_buckets(frames_by_bucket: dict, min_size: int, label_name: str) -> dict:
    """
    Helper to consolidate buckets smaller than min_size into a single 'misc' bucket.
    """
    consolidated_buckets = {}
    misc_frames = []
    small_bucket_keys = []

    for bucket_name, frames in frames_by_bucket.items():
        if len(frames) < min_size:
            misc_frames.extend(frames)
            small_bucket_keys.append(bucket_name)
        else:
            consolidated_buckets[bucket_name] = frames

    if misc_frames:
        consolidated_buckets[f'misc_{label_name}'] = misc_frames
        log.info(f"Consolidated {len(small_bucket_keys)} small {label_name} buckets "
                 f"(e.g., {small_bucket_keys[:3]}) into one 'misc' bucket of size {len(misc_frames)}.")

    return consolidated_buckets


def _create_master_stream_for_label(consolidated_buckets: dict) -> IterDataPipe:
    """
    Creates a master stream for a single label by round-robin sampling.
    """
    if not consolidated_buckets:
        return None

    bucket_datapipes = [
        IterableWrapper(frames).cycle().shuffle()
        for frames in consolidated_buckets.values()
    ]
    return CustomRoundRobinDataPipe(*bucket_datapipes)


def _create_property_balanced_loader_new(all_train_frames: list[dict], config: dict, data_config: dict) -> DataLoader:
    """
    The NEW and IMPROVED dataloader implementation.
    """
    BATCH_SIZE = data_config['dataloader_params']['frames_per_batch']
    MIN_BUCKET_SIZE = BATCH_SIZE
    NUM_WORKERS = data_config['dataloader_params']['num_workers']

    # --- Step 1 & 2: Segregate and Consolidate ---
    log.info("Step 1-2: Segregating frames and consolidating small buckets...")
    real_frames_by_bucket = defaultdict(list)
    fake_frames_by_bucket = defaultdict(list)
    for frame_info in all_train_frames:
        bucket = frame_info['property_bucket']
        if frame_info['label_id'] == 0:
            real_frames_by_bucket[bucket].append(frame_info)
        else:
            fake_frames_by_bucket[bucket].append(frame_info)
    consolidated_real_buckets = _consolidate_small_buckets(real_frames_by_bucket, MIN_BUCKET_SIZE, 'real')
    consolidated_fake_buckets = _consolidate_small_buckets(fake_frames_by_bucket, MIN_BUCKET_SIZE, 'fake')

    # --- Step 3: Create master streams ---
    log.info("Step 3: Creating master data streams for real and fake labels...")
    real_master_stream = _create_master_stream_for_label(consolidated_real_buckets)
    fake_master_stream = _create_master_stream_for_label(consolidated_fake_buckets)
    if real_master_stream is None or fake_master_stream is None:
        raise ValueError("Cannot create dataloader: one or both label types have no data.")

    # --- Step 4: Probabilistically balance labels ---
    log.info("Step 4: Combining streams with 50/50 probabilistic sampler...")
    combined_pipe = CustomSampleMultiplexerDataPipe(
        datapipes=[real_master_stream, fake_master_stream], weights=[0.5, 0.5]
    )

    # --- Step 5: Enforce Anchor-Mate Pairing ---
    log.info("Step 5: Grouping by video_id to create anchor-mate pairs...")
    combined_pipe = combined_pipe.groupby(
        group_key_fn=get_video_id, group_size=2, guaranteed_group_size=True
    )
    combined_pipe = combined_pipe.map(get_group_elements)
    combined_pipe = combined_pipe.flatten()

    # --- Step 6: Applying transforms, batching, and Dataloader Creation ---
    log.info("Step 6: Applying transforms, batching, and creating DataLoader...")
    transform_fn = FrameTransformer(
        resolution=config['resolution'], mean=config['mean'], std=config['std'],
        is_train=config['use_data_augmentation']
    )
    combined_pipe = combined_pipe.map(transform_fn)
    combined_pipe = combined_pipe.filter(is_not_none)
    combined_pipe = combined_pipe.batch(BATCH_SIZE)
    combined_pipe = combined_pipe.collate()

    if NUM_WORKERS > 0:
        combined_pipe = combined_pipe.prefetch(data_config['dataloader_params']['prefetch_factor'])

    # --- MODIFIED (SOLUTION): Add worker_init_fn to the DataLoader ---
    # Also handle the case for num_workers=0 (main process)
    worker_init = worker_init_fn if NUM_WORKERS > 0 else None
    if NUM_WORKERS == 0:
        # If not using workers, initialize filesystem in the main process
        global fs
        fs = fsspec.filesystem("gcs")

    return DataLoader(
        combined_pipe,
        num_workers=NUM_WORKERS,
        batch_size=None,
        pin_memory=True,
        worker_init_fn=worker_init
    )


# ... (The rest of your file, including surgical_data_aug, helper functions, and the main execution block, remains unchanged) ...
# =================================================================================
# --- Helper & Verification Functions ---
# =================================================================================

# --- ROBUST, SELF-CONTAINED MULTIPLEXER REPLACEMENTS ---

def custom_round_robin(datapipes: list) -> iter:
    """
    A custom Python generator to replicate RoundRobinMultiplexer functionality.
    It takes a list of datapipes, assumes they are infinite, and yields one
    item from each in a continuous cycle.
    """
    iterators = [iter(dp) for dp in datapipes]
    while True:
        for it in iterators:
            try:
                yield next(it)
            except StopIteration:
                return


def custom_sample_multiplexer(datapipes: list, weights: list) -> iter:
    """
    A custom Python generator to replicate SampleMultiplexer functionality.
    It probabilistically yields an item from one of the given datapipes
    based on the provided weights.
    """
    iterators = [iter(dp) for dp in datapipes]
    indices = np.arange(len(datapipes))
    while True:
        chosen_index = np.random.choice(indices, p=weights)
        try:
            yield next(iterators[chosen_index])
        except StopIteration:
            return


# --- SELF-CONTAINED AUGMENTATION LOGIC ---

transform_base = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.HueSaturationValue(p=0.1),
])

degrade_quality_pipeline = A.Compose([
    A.OneOf([
        A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8),
        A.GaussianBlur(blur_limit=(5, 11), p=0.6),
        A.GaussNoise(var_limit=(20.0, 80.0), p=0.4),
    ], p=1.0)
])

enhance_quality_pipeline = A.Compose([
    A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9),
])


def surgical_data_aug(img: Image.Image, frame_properties: dict) -> Image.Image:
    """
    Applies augmentations dynamically based on frame properties to create "counter-examples"
    and break trivial correlations like "blurry = fake" or "sharp = real".
    """
    img_np = np.array(img)
    chance_for_sharpness_adjustment = 0.5

    img_np = transform_base(image=img_np)['image']
    sharpness_bucket = frame_properties.get('sharpness_bucket')

    if sharpness_bucket == 'q4':
        if random.random() < chance_for_sharpness_adjustment:
            img_np = degrade_quality_pipeline(image=img_np)['image']
    elif sharpness_bucket == 'q1':
        if random.random() < chance_for_sharpness_adjustment:
            img_np = enhance_quality_pipeline(image=img_np)['image']
    elif sharpness_bucket in ['q2', 'q3']:
        if random.random() < chance_for_sharpness_adjustment / 2:
            if random.random() < 0.5:
                img_np = degrade_quality_pipeline(image=img_np)['image']
            else:
                img_np = enhance_quality_pipeline(image=img_np)['image']

    return Image.fromarray(img_np)


def add_property_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the crucial 'property_bucket' column to the DataFrame."""
    log.info("Adding property buckets to manifest data...")
    df['sharpness_bucket'] = pd.qcut(df['sharpness'], 4, labels=[f's_q{i}' for i in range(1, 5)], duplicates='drop')
    df['size_bucket'] = pd.qcut(df['file_size_kb'], 4, labels=[f'f_q{i}' for i in range(1, 5)], duplicates='drop')
    df['property_bucket'] = df['sharpness_bucket'].astype(str) + '_' + df['size_bucket'].astype(str)
    log.info(f"Created {df['property_bucket'].nunique()} unique property buckets.")
    return df


def prepare_unbalanced_train_frames(df: pd.DataFrame, data_cfg: dict) -> list[dict]:
    """
    Replicates the NEW splitting logic: isolates the full, UNBALANCED training pool.
    """
    log.info("Preparing the full, unbalanced training frame pool...")
    real_methods = set(data_cfg['methods']['use_real_sources'])
    train_fake_methods = set(data_cfg['methods']['use_fake_methods_for_training'])

    real_df = df[df['method'].isin(real_methods)]
    train_fake_df = df[df['method'].isin(train_fake_methods)]

    gss = GroupShuffleSplit(n_splits=1, test_size=data_cfg['data_params']['val_split_ratio'],
                            random_state=data_cfg['data_params']['seed'])
    train_idx, _ = next(gss.split(real_df, groups=real_df['video_id']))
    train_real_df = real_df.iloc[train_idx]

    train_df = pd.concat([train_real_df, train_fake_df])
    log.info(
        f"Final training pool size: {len(train_df):,} frames ({train_real_df.shape[0]:,} real, {train_fake_df.shape[0]:,} fake).")
    return train_df.to_dict('records')


def unnormalize_image(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.permute(1, 2, 0).numpy()


def validate_and_visualize_batch(batch: dict, original_df: pd.DataFrame, batch_history: list):
    """Performs all checks and visualizations for a single batch."""
    print("\n" + "=" * 80)
    print("||" + " " * 28 + "ANALYZING NEW BATCH" + " " * 29 + "||")
    print("=" * 80)

    batch_paths = batch['path']
    batch_df = original_df[original_df['path'].isin(batch_paths)].copy()
    batch_df['path_cat'] = pd.Categorical(batch_df['path'], categories=batch_paths, ordered=True)
    batch_df.sort_values('path_cat', inplace=True)
    labels = batch['label_id'].numpy()

    # --- 1. Label Balance Verification (Probabilistic) ---
    print("\n--- [Check 1/3] Label Balance (Probabilistic) ---")
    label_counts = Counter(labels)
    real_count = label_counts.get(0, 0)
    fake_count = label_counts.get(1, 0)
    batch_history.append({'real': real_count, 'fake': fake_count})

    total_real = sum(h['real'] for h in batch_history)
    total_fake = sum(h['fake'] for h in batch_history)
    total_samples = total_real + total_fake

    print(f"Current Batch -> Real (0): {real_count} | Fake (1): {fake_count}")
    print(
        f"Cumulative Avg -> Real: {total_real / total_samples:.2%} | Fake: {total_fake / total_samples:.2%} (from {len(batch_history)} batches)")
    if abs(real_count - fake_count) > BATCH_SIZE * 0.25:  # Allow up to 25% deviation in a single batch
        print("⚠️  STATUS: Batch has notable deviation, but this is expected occasionally.")
    else:
        print("✅ STATUS: Within expected probabilistic range.")

    # --- 2. Property Distribution Observation ---
    print("\n--- [Check 2/3] Property Distribution ---")
    real_batch_df = batch_df[batch_df['label_id'] == 0]
    fake_batch_df = batch_df[batch_df['label_id'] == 1]

    print("Real Frame Properties in Batch:")
    print(real_batch_df['property_bucket'].value_counts().to_string())
    print("\nFake Frame Properties in Batch:")
    print(fake_batch_df['property_bucket'].value_counts().to_string())
    print("\n✅ STATUS: Diverse properties are being sampled for both labels.")

    # --- 3. Anchor-Mate Pairing Verification ---
    print("\n--- [Check 3/3] Anchor-Mate Pairing ---")
    video_id_counts = batch_df['video_id'].value_counts()
    if all(video_id_counts == 2):
        print(f"✅ STATUS: Correct. Found {len(video_id_counts)} unique videos, each with exactly 2 frames.")
    else:
        print("❌ STATUS: INCORRECT PAIRING DETECTED!")
        print("Videos with counts other than 2:\n", video_id_counts[video_id_counts != 2])

    # --- 4. Image Visualization ---
    if input("\nDisplay image grid for this batch? (y/n): ").lower() == 'y':
        num_images_to_show = min(len(labels), 16)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f"Batch Visualization (First {num_images_to_show} Images)", fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i >= num_images_to_show:
                ax.axis('off')
                continue

            img_np = unnormalize_image(batch['image'][i], CONFIG['mean'], CONFIG['std'])
            label_text = "Real" if labels[i] == 0 else "Fake"
            video_id = batch_df.iloc[i]['video_id']
            prop_bucket = batch_df.iloc[i]['property_bucket']

            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(f"{video_id}\n{label_text} | {prop_bucket}", fontsize=8)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# =================================================================================
# --- Main Execution Block ---
# =================================================================================

def main():
    """Main execution function."""
    if not os.path.exists(MANIFEST_PATH):
        log.error(f"Manifest file not found at '{MANIFEST_PATH}'. Please run data preparation first.")
        sys.exit(1)

    df = pd.read_parquet(MANIFEST_PATH)
    df['label_id'] = np.where(df['label'] == 'real', 0, 1)
    df_bucketed = add_property_buckets(df)

    train_frames = prepare_unbalanced_train_frames(df_bucketed, DATA_CONFIG)

    loader = _create_property_balanced_loader_new(train_frames, CONFIG, DATA_CONFIG)
    data_iterator = iter(loader)

    batch_num = 1
    batch_history = []

    global BATCH_SIZE
    BATCH_SIZE = DATA_CONFIG['dataloader_params']['frames_per_batch']

    while True:
        action = input(f"\nPress ENTER to load and verify batch #{batch_num} or type 'q' to quit: ").lower()
        if action == 'q':
            break

        try:
            log.info(f"Loading batch #{batch_num}...")
            batch = next(data_iterator)
            validate_and_visualize_batch(batch, df_bucketed, batch_history)
            batch_num += 1
        except StopIteration:
            log.info("Dataloader exhausted. All batches have been processed.")
            break
        except Exception:
            log.exception("An error occurred while loading or validating the batch.")
            break

    log.info("Verification script finished.")


if __name__ == "__main__":
    main()
