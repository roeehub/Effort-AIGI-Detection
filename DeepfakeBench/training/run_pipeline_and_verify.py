# --- run_and_verify_pipeline.py ---
# This script combines the logic from analyze_split_strategies.py and
# verify_property_dataloader.py into a single, sequential workflow.

import os
import sys
import warnings
from collections import Counter, defaultdict
import random
import io
import fsspec
import logging

# --- Setup: Silence deprecation warnings and add project root to path ---
warnings.filterwarnings("ignore", category=UserWarning, message="The 'datapipes', 'dataloader2' modules are deprecated")
if '.' not in sys.path:
    sys.path.append('.')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
from torchvision import transforms as T
import albumentations as A

# Configure logging for clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# =================================================================================
# --- Configuration (Unified from both files) ---
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
        'use_fake_methods_for_validation': [
            "facedancer", "sadtalker", "DiT", "StyleGANXL", "e4s", "one_shot_free"
        ],
    }
}
MANIFEST_PATH = "frame_properties.parquet"


# =================================================================================
# --- STAGE 1: Data Splitting and Preparation (from analyze_split_strategies.py)
# =================================================================================

def report_counts(df: pd.DataFrame, stage_name: str):
    """Prints detailed counts of frames and unique videos for a given DataFrame."""
    if df.empty:
        print(f"\n--- {stage_name} ---\n  DataFrame is empty.")
        return

    real_df = df[df['label'] == 'real']
    fake_df = df[df['label'] == 'fake']

    real_true_videos = real_df[['method', 'video_id']].drop_duplicates().shape[0]
    fake_true_videos = fake_df[['method', 'video_id']].drop_duplicates().shape[0]

    print(f"\n{'=' * 25} {stage_name} {'=' * 25}")
    print(f"  Real: {real_df.shape[0]:>9,} frames | {real_true_videos:>6,} unique videos")
    print(f"  Fake: {fake_df.shape[0]:>9,} frames | {fake_true_videos:>6,} unique videos")
    print("-" * 75)
    print(f"  TOTAL: {df.shape[0]:>8,} frames | {real_true_videos + fake_true_videos:>6,} unique videos")
    print("=" * 75)


def prepare_data_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Performs the full data filtering and splitting logic to produce the final,
    unbalanced training data pool.
    """
    log.info("Starting data preparation and splitting process...")

    # --- STAGE 1A: FILTER BY ALL CONFIGURED METHODS ---
    real_methods = set(cfg['methods']['use_real_sources'])
    train_fake_methods = set(cfg['methods']['use_fake_methods_for_training'])
    val_fake_methods = set(cfg['methods']['use_fake_methods_for_validation'])
    all_methods_in_use = real_methods.union(train_fake_methods).union(val_fake_methods)

    df_filtered = df[df['method'].isin(all_methods_in_use)]
    report_counts(df_filtered, "STAGE 1: After Filtering by ALL Config Methods")

    # --- STAGE 1B: ISOLATE TRAINING POOL (Remove hold-out validation methods) ---
    train_pool_methods = real_methods.union(train_fake_methods)
    df_train_pool_all = df_filtered[df_filtered['method'].isin(train_pool_methods)]
    report_counts(df_train_pool_all, "STAGE 2: Pool of Methods for Training")

    # --- STAGE 1C: SPLIT REAL VIDEOS (Reserve some for validation) ---
    real_df = df_train_pool_all[df_train_pool_all['method'].isin(real_methods)]
    train_fake_df = df_train_pool_all[~df_train_pool_all['method'].isin(real_methods)]

    gss = GroupShuffleSplit(
        n_splits=1, test_size=cfg['data_params']['val_split_ratio'], random_state=cfg['data_params']['seed']
    )
    # The groups are the video_id strings to ensure all frames from one video stay together
    train_idx, val_idx = next(gss.split(real_df, groups=real_df['video_id']))

    train_real_df = real_df.iloc[train_idx]
    val_real_df = real_df.iloc[val_idx]  # We create this but won't use it, as requested

    log.info(
        f"Splitting real videos: {train_real_df[['method', 'video_id']].drop_duplicates().shape[0]} for training, "
        f"{val_real_df[['method', 'video_id']].drop_duplicates().shape[0]} for validation."
    )

    # This is the final, complete pool of data available for the training dataloader
    df_unbalanced_train_pool = pd.concat([train_real_df, train_fake_df])

    # We add the property buckets here, after splitting, to the final training data
    log.info("Adding property buckets to the final training data manifest...")
    df_unbalanced_train_pool['sharpness_bucket'] = pd.qcut(df_unbalanced_train_pool['sharpness'], 4,
                                                           labels=[f's_q{i}' for i in range(1, 5)], duplicates='drop')
    df_unbalanced_train_pool['size_bucket'] = pd.qcut(df_unbalanced_train_pool['file_size_kb'], 4,
                                                      labels=[f'f_q{i}' for i in range(1, 5)], duplicates='drop')
    df_unbalanced_train_pool['property_bucket'] = df_unbalanced_train_pool['sharpness_bucket'].astype(str) + '_' + \
                                                  df_unbalanced_train_pool['size_bucket'].astype(str)
    log.info(f"Created {df_unbalanced_train_pool['property_bucket'].nunique()} unique property buckets.")

    return df_unbalanced_train_pool


# =================================================================================
# --- STAGE 2: Dataloader Creation & Verification (from verify_property_dataloader.py)
# =================================================================================

# --- All helper functions for the dataloader are placed here ---
# (Custom DataPipes, FrameTransformer, worker_init_fn, augmentations, etc.)

class CustomRoundRobinDataPipe(IterDataPipe):
    def __init__(self, *datapipes):
        super().__init__()
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        while True:
            for it in iterators:
                try:
                    yield next(it)
                except StopIteration:
                    return


class CustomSampleMultiplexerDataPipe(IterDataPipe):
    def __init__(self, datapipes, weights):
        super().__init__()
        self.datapipes = datapipes
        self.weights = weights

    def __iter__(self):
        iterators = [iter(dp) for dp in self.datapipes]
        indices = np.arange(len(self.datapipes))
        while True:
            chosen_index = np.random.choice(indices, p=self.weights)
            try:
                yield next(iterators[chosen_index])
            except StopIteration:
                return


def worker_init_fn(worker_id):
    """Initializes fsspec and sets a unique random seed for each worker."""
    global fs
    fs = fsspec.filesystem("gcs")

    initial_seed = torch.initial_seed()

    # Use the modulo operator to constrain the 64-bit seed to a 32-bit range
    seed = (initial_seed + worker_id) % (2 ** 32)

    random.seed(seed)
    np.random.seed(seed)


class FrameTransformer:
    def __init__(self, resolution, mean, std, is_train):
        self.resolution = resolution
        self.is_train = is_train
        self.normalize_transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __call__(self, frame_dict):
        try:
            with fs.open(frame_dict['path'], 'rb') as f:
                image_bytes = f.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img = img.resize((self.resolution, self.resolution), Image.BICUBIC)
            if self.is_train: img = surgical_data_aug(img, frame_dict)
            image_tensor = self.normalize_transform(img)
            return {**frame_dict, 'image': image_tensor}
        except Exception as e:
            log.warning(f"Could not process file {frame_dict.get('path', 'N/A')}: {e}")
            return None


def get_video_id(item: dict) -> str: return item['video_id']


def get_group_elements(group: list) -> list: return group[1]


def is_not_none(item) -> bool: return item is not None


def _consolidate_small_buckets(frames_by_bucket: dict, min_size: int, label_name: str) -> dict:
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
    return consolidated_buckets


def _create_master_stream_for_label(consolidated_buckets: dict) -> IterDataPipe:
    if not consolidated_buckets: return None
    bucket_datapipes = [IterableWrapper(frames).cycle() for frames in consolidated_buckets.values()]
    return CustomRoundRobinDataPipe(*bucket_datapipes)


# +++ NEW HELPER FUNCTION AND DATAPIPE FOR ANCHOR-MATE LOGIC +++
def _build_video_to_frames_lookup(all_frames: list[dict]) -> dict[str, list[dict]]:
    """Builds a lookup table mapping clip_id to a list of its frame dictionaries."""
    lookup = defaultdict(list)
    for frame in all_frames:
        lookup[frame['clip_id']].append(frame)  # MODIFIED: Use clip_id
    return lookup


class MateFinderDataPipe(IterDataPipe):
    """
    Takes a stream of 'anchor' frames and for each one, finds and yields a random
    'mate' frame from the same video.
    """

    def __init__(self, source_datapipe: IterDataPipe, lookup: dict):
        super().__init__()
        self.source_datapipe = source_datapipe
        self.lookup = lookup

    def __iter__(self):
        for anchor_frame in self.source_datapipe:
            # First, yield the anchor frame itself, which was selected by our balancing logic
            yield anchor_frame

            clip_id = anchor_frame['clip_id']
            possible_mates = self.lookup.get(clip_id, [])

            # Filter out the anchor frame itself to ensure the mate is different
            mates_pool = [m for m in possible_mates if m['path'] != anchor_frame['path']]

            if not mates_pool:
                # Edge case: video has only 1 frame. Yield the anchor again to maintain
                # batch size parity.
                log.warning(f"Could not find a different mate for {anchor_frame['path']}. Duplicating anchor.")
                yield anchor_frame
            else:
                # Randomly select a mate and yield it
                mate_frame = random.choice(mates_pool)
                yield mate_frame


# --- The core dataloader creation function (MODIFIED) ---
def create_property_balanced_loader(all_train_frames: list[dict], config: dict, data_config: dict) -> DataLoader:
    BATCH_SIZE = data_config['dataloader_params']['frames_per_batch']
    MIN_BUCKET_SIZE = BATCH_SIZE
    NUM_WORKERS = data_config['dataloader_params']['num_workers']

    # +++ ADDED: Build the video_id -> frames lookup table ONCE. +++
    video_lookup = _build_video_to_frames_lookup(all_train_frames)
    log.info(f"Built video_id lookup table with {len(video_lookup)} unique videos.")

    # 1. Segregate by label and property
    real_frames_by_bucket = defaultdict(list)
    fake_frames_by_bucket = defaultdict(list)
    for frame_info in all_train_frames:
        bucket = frame_info['property_bucket']
        if frame_info['label_id'] == 0:
            real_frames_by_bucket[bucket].append(frame_info)
        else:
            fake_frames_by_bucket[bucket].append(frame_info)

    # 2. Consolidate small buckets
    consolidated_real_buckets = _consolidate_small_buckets(real_frames_by_bucket, MIN_BUCKET_SIZE, 'real')
    consolidated_fake_buckets = _consolidate_small_buckets(fake_frames_by_bucket, MIN_BUCKET_SIZE, 'fake')

    # 3. Create master streams
    real_master_stream = _create_master_stream_for_label(consolidated_real_buckets)
    fake_master_stream = _create_master_stream_for_label(consolidated_fake_buckets)
    if real_master_stream is None or fake_master_stream is None:
        raise ValueError("Cannot create dataloader: one or both label types have no data.")

    # 4. Probabilistically balance labels to create a stream of "anchors"
    anchor_pipe = CustomSampleMultiplexerDataPipe([real_master_stream, fake_master_stream], [0.5, 0.5])

    # 5. SHARDING AND SHUFFLING
    if NUM_WORKERS > 0:
        anchor_pipe = anchor_pipe.sharding_filter()
    anchor_pipe = anchor_pipe.shuffle(buffer_size=10000)

    # --- REPLACED: The old coincidental groupby logic is removed ---
    # combined_pipe = combined_pipe.groupby(group_key_fn=get_video_id, group_size=2, guaranteed_group_size=True)
    # combined_pipe = combined_pipe.map(get_group_elements).flatten()

    # +++ ADDED: Insert the intentional MateFinderDataPipe +++
    # This pipe takes the stream of anchors and doubles its length by inserting a mate after each anchor.
    # A batch of 64 will now contain 32 property-balanced anchors and their 32 corresponding mates.
    combined_pipe = MateFinderDataPipe(anchor_pipe, video_lookup)

    # 6. Apply transforms, batching, and create DataLoader
    transform_fn = FrameTransformer(config['resolution'], config['mean'], config['std'],
                                    config['use_data_augmentation'])
    combined_pipe = combined_pipe.map(transform_fn).filter(is_not_none).batch(BATCH_SIZE).collate()

    if NUM_WORKERS > 0:
        combined_pipe = combined_pipe.prefetch(data_config['dataloader_params']['prefetch_factor'])

    worker_init = worker_init_fn if NUM_WORKERS > 0 else None
    if NUM_WORKERS == 0:
        global fs;
        fs = fsspec.filesystem("gcs")

    return DataLoader(combined_pipe, num_workers=NUM_WORKERS, batch_size=None, pin_memory=True,
                      worker_init_fn=worker_init)


# --- All visualization and helper functions ---
def unnormalize_image(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.permute(1, 2, 0).numpy()


def validate_and_visualize_batch(batch: dict, original_df: pd.DataFrame, batch_history: list):
    print("\n" + "=" * 80 + "\n||" + " " * 28 + "ANALYZING NEW BATCH" + " " * 29 + "||\n" + "=" * 80)
    batch_paths = batch['path']
    batch_df = original_df[original_df['path'].isin(batch_paths)].copy()

    # Reorder the dataframe to match the batch order for correct visualization
    # Note: With mates, paths can be duplicated if a frame is an anchor and also chosen as a mate.
    # We'll handle this by creating a temporary mapping.
    temp_df = pd.DataFrame({'path': batch_paths})
    temp_df = temp_df.merge(original_df.drop_duplicates(subset=['path']), on='path', how='left')
    batch_df = temp_df

    labels = batch['label_id'].numpy()
    BATCH_SIZE = len(labels)

    # 1. Label Balance
    print("\n--- [Check 1/3] Label Balance (Probabilistic) ---")
    label_counts = Counter(labels);
    real_count = label_counts.get(0, 0);
    fake_count = label_counts.get(1, 0)
    batch_history.append({'real': real_count, 'fake': fake_count})
    total_real = sum(h['real'] for h in batch_history);
    total_fake = sum(h['fake'] for h in batch_history)
    total_samples = total_real + total_fake
    print(f"Current Batch -> Real (0): {real_count} | Fake (1): {fake_count}")
    print(
        f"Cumulative Avg -> Real: {total_real / total_samples:.2%} | Fake: {total_fake / total_samples:.2%} (from {len(batch_history)} batches)")

    # 2. Property Distribution
    print("\n--- [Check 2/3] Property Distribution ---")
    print("Real Frame Properties:\n", batch_df[batch_df['label_id'] == 0]['property_bucket'].value_counts().to_string())
    print("\nFake Frame Properties:\n",
          batch_df[batch_df['label_id'] == 1]['property_bucket'].value_counts().to_string())

    # 3. Anchor-Mate Pairing
    print("\n--- [Check 3/3] Anchor-Mate Pairing ---")
    # MODIFIED: Check pairing by the unique clip_id
    clip_id_counts = batch_df['clip_id'].value_counts()
    if all(clip_id_counts % 2 == 0):
        print(f"✅ STATUS: Correct. All clip IDs appear in pairs (or multiples of 2).")
    else:
        print("❌ STATUS: INCORRECT PAIRING DETECTED!\n", clip_id_counts[clip_id_counts % 2 != 0])

    # 4. Visualization
    if input("\nDisplay image grid for this batch? (y/n): ").lower() == 'y':
        # (Visualization code is unchanged)
        num_images_to_show = min(len(labels), 16)
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle(f"Batch Visualization (First {num_images_to_show} Images)", fontsize=16)

        for i, ax in enumerate(axes.flat):
            if i >= num_images_to_show:
                ax.axis('off')
                continue

            # The first item is the Anchor, the second is its Mate, etc.
            pair_status = "Anchor" if i % 2 == 0 else "Mate"

            img_np = unnormalize_image(batch['image'][i], CONFIG['mean'], CONFIG['std'])
            label_text = "Real" if labels[i] == 0 else "Fake"
            # MODIFIED: Display the clip_id for easier verification
            clip_id = batch_df.iloc[i]['clip_id']
            prop_bucket = batch_df.iloc[i]['property_bucket']

            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(f"({pair_status}) ClipID: {clip_id}\n{label_text} | {prop_bucket}", fontsize=8)
            ax.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


# (Surgical data aug logic is unchanged)
def surgical_data_aug(img: Image.Image, frame_properties: dict) -> Image.Image:
    img_np = np.array(img)
    # Augmentation pipelines (transform_base, degrade_quality_pipeline, etc.)
    return Image.fromarray(img_np)


# =================================================================================
# --- MAIN EXECUTION BLOCK ---
# =================================================================================

def main():
    """Main execution function."""
    if not os.path.exists(MANIFEST_PATH):
        log.error(f"Manifest file not found at '{MANIFEST_PATH}'. Please run data preparation first.")
        sys.exit(1)

    # --- Load the single source of truth ---
    df_full = pd.read_parquet(MANIFEST_PATH)
    df_full['label_id'] = np.where(df_full['label'] == 'real', 0, 1)

    # --- STAGE 1: Get the final training data pool using the analysis logic ---
    df_train_final = prepare_data_splits(df_full, DATA_CONFIG)

    print("\n" + "#" * 30 + " DATALOADER SUMMARY " + "#" * 30)
    report_counts(df_train_final, "FINAL UNBALANCED TRAINING POOL")

    # --- Answering your specific questions ---
    BATCH_SIZE = DATA_CONFIG['dataloader_params']['frames_per_batch']
    num_train_frames = len(df_train_final)

    print("\nQ: How many batches do we have?")
    print(f"A: The dataloader uses infinite streams (.cycle()), so it can produce endless batches.")
    print(f"   One 'epoch' is defined by the number of unique ANCHORS sampled, which is based on original frames.")
    print(f"   One epoch would consist of approximately: "
          f"{int(num_train_frames / (BATCH_SIZE / 2))} batches ({num_train_frames:,} anchors / {int(BATCH_SIZE / 2)} anchors per batch).")
    print("#" * 80)

    # Convert dataframe to the list of dictionaries the loader expects
    train_frames_list = df_train_final.to_dict('records')

    # --- STAGE 2: Create the dataloader from the final training pool ---
    log.info("Creating the property-balanced dataloader from the final training pool...")
    train_loader = create_property_balanced_loader(train_frames_list, CONFIG, DATA_CONFIG)
    data_iterator = iter(train_loader)

    # --- STAGE 3: Interactively verify batches ---
    batch_num = 1
    batch_history = []
    while True:
        action = input(f"\nPress ENTER to load and verify batch #{batch_num} or type 'q' to quit: ").lower()
        if action == 'q': break
        try:
            log.info(f"Loading batch #{batch_num}...")
            batch = next(data_iterator)
            validate_and_visualize_batch(batch, df_train_final, batch_history)
            batch_num += 1
        except StopIteration:
            log.info("Dataloader exhausted. This shouldn't happen with .cycle().")
            break
        except Exception:
            log.exception("An error occurred while loading or validating the batch.")
            break

    log.info("Verification script finished.")


if __name__ == "__main__":
    # NOTE: Surgical augmentation logic needs to be defined for this to run.
    # It has been stubbed out above for brevity but should be copied from the original file.
    transform_base = A.Compose(
        [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2), A.HueSaturationValue(p=0.1)])
    degrade_quality_pipeline = A.Compose([A.OneOf(
        [A.ImageCompression(quality_lower=40, quality_upper=70, p=0.8), A.GaussianBlur(blur_limit=(5, 11), p=0.6),
         A.GaussNoise(var_limit=(20.0, 80.0), p=0.4)], p=1.0)])
    enhance_quality_pipeline = A.Compose([A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.9)])


    def surgical_data_aug(img: Image.Image, frame_properties: dict) -> Image.Image:
        img_np = np.array(img)
        chance_for_sharpness_adjustment = 0.5
        img_np = transform_base(image=img_np)['image']
        sharpness_bucket = frame_properties.get('sharpness_bucket')
        if sharpness_bucket == 'q4':
            if random.random() < chance_for_sharpness_adjustment: img_np = degrade_quality_pipeline(image=img_np)[
                'image']
        elif sharpness_bucket == 'q1':
            if random.random() < chance_for_sharpness_adjustment: img_np = enhance_quality_pipeline(image=img_np)[
                'image']
        elif sharpness_bucket in ['q2', 'q3']:
            if random.random() < chance_for_sharpness_adjustment / 2:
                if random.random() < 0.5:
                    img_np = degrade_quality_pipeline(image=img_np)['image']
                else:
                    img_np = enhance_quality_pipeline(image=img_np)['image']
        return Image.fromarray(img_np)


    main()
