# analyze_split_strategies.py
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import logging

# --- Setup: Add project root to path and configure logging ---
if '.' not in sys.path:
    sys.path.append('.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration (Updated to be complete and accurate) ---
DATA_CONFIG = {
    'data_params': {
        'seed': 42,
        'val_split_ratio': 0.1,  # 10% for validation split of real videos
    },
    'methods': {
        'use_real_sources': ["FaceForensics++", "Celeb-real", "youtube-real", "external_youtube_avspeech"],
        'use_fake_methods_for_training': [
            # Face-swapping (FS)
            "simswap", "mobileswap", "faceswap", "inswap", "blendface", "fsgan", "uniface",
            # Face-reenactment (FR)
            "pirender", "facevid2vid", "lia", "fomm", "MRAA", "wav2lip", "mcnet", "danet",
            # Entire Face Synthesis (EFS)
            "VQGAN", "StyleGAN3", "StyleGAN2", "SiT", "RDDM", "ddim"
        ],
        'use_fake_methods_for_validation': [
            # This is the hold-out set for testing generalization.
            "facedancer", "sadtalker", "DiT", "StyleGANXL", "e4s", "one_shot_free"
        ],
    }
}
MANIFEST_PATH = "frame_properties.parquet"


def _balance_df_by_label_old(df: pd.DataFrame, real_methods: set, seed: int) -> pd.DataFrame:
    """The OLD balancing function from prepare_splits.py, which undersamples based on unique `video_id` strings."""
    if df.empty: return pd.DataFrame()
    is_real = df['method'].isin(real_methods)
    real_df, fake_df = df[is_real], df[~is_real]

    # OLD logic: counts unique video_id strings, not true unique videos
    real_video_ids = real_df['video_id'].unique()
    fake_video_ids = fake_df['video_id'].unique()

    num_real_videos = len(real_video_ids)
    num_fake_videos = len(fake_video_ids)

    if num_real_videos == 0 or num_fake_videos == 0: return df

    # The target size is based on the count of unique IDs, which is the source of the data loss
    target_size = min(num_real_videos, num_fake_videos)
    log.info(f"[OLD_BALANCE] Target unique video IDs per class: {target_size}")

    rng = np.random.default_rng(seed=seed)
    if num_real_videos > target_size:
        sampled_ids = rng.choice(real_video_ids, size=target_size, replace=False)
        real_df = real_df[real_df['video_id'].isin(sampled_ids)]
    elif num_fake_videos > target_size:
        # This is where the undersampling happens
        sampled_ids = rng.choice(fake_video_ids, size=target_size, replace=False)
        fake_df = fake_df[fake_df['video_id'].isin(sampled_ids)]

    return pd.concat([real_df, fake_df])


def report_counts(df: pd.DataFrame, stage_name: str):
    """
    Prints detailed counts, now correctly identifying a unique video as a
    (method, video_id) pair, while also reporting the number of unique
    video_id strings for context.
    """
    if df.empty:
        print(f"\n--- {stage_name} ---\n  DataFrame is empty.")
        return

    real_df = df[df['label'] == 'real']
    fake_df = df[df['label'] == 'fake']

    # --- Correctly count TRUE unique videos using both method and video_id ---
    real_true_videos = real_df[['method', 'video_id']].drop_duplicates().shape[0]
    fake_true_videos = fake_df[['method', 'video_id']].drop_duplicates().shape[0]

    # Also count the unique ID strings for context
    real_unique_ids = real_df['video_id'].nunique()
    fake_unique_ids = fake_df['video_id'].nunique()

    print(f"\n{'=' * 25} {stage_name} {'=' * 25}")
    print(
        f"  Real: {real_df.shape[0]:>9,} frames | {real_true_videos:>6,} videos (from {real_unique_ids:>6,} unique IDs)")
    print(
        f"  Fake: {fake_df.shape[0]:>9,} frames | {fake_true_videos:>6,} videos (from {fake_unique_ids:>6,} unique IDs)")
    print("-" * 75)
    print(f"  TOTAL: {df.shape[0]:>8,} frames | {real_true_videos + fake_true_videos:>6,} videos")
    print("=" * 75)


def main():
    """Walk through the data splitting process and report counts at each step."""
    log.info(f"Loading property manifest from '{MANIFEST_PATH}'...")
    df = pd.read_parquet(MANIFEST_PATH)

    # --- STAGE 0: RAW MANIFEST ---
    report_counts(df, "STAGE 0: Full Raw Manifest")

    # --- STAGE 1: FILTER BY ALL CONFIGURED METHODS ---
    cfg = DATA_CONFIG
    real_methods = set(cfg['methods']['use_real_sources'])
    train_fake_methods = set(cfg['methods']['use_fake_methods_for_training'])
    val_fake_methods = set(cfg['methods']['use_fake_methods_for_validation'])
    all_methods_in_use = real_methods.union(train_fake_methods).union(val_fake_methods)

    df_filtered = df[df['method'].isin(all_methods_in_use)]
    report_counts(df_filtered, "STAGE 1: After Filtering by ALL Config Methods")

    # --- STAGE 2: ISOLATE TRAINING POOL (Remove hold-out validation methods) ---
    train_pool_methods = real_methods.union(train_fake_methods)
    df_train_pool_all = df_filtered[df_filtered['method'].isin(train_pool_methods)]
    report_counts(df_train_pool_all, "STAGE 2: Pool of Methods for Training")

    # --- STAGE 3: SPLIT REAL VIDEOS (Reserve some for validation) ---
    real_df = df_train_pool_all[df_train_pool_all['method'].isin(real_methods)]
    train_fake_df = df_train_pool_all[~df_train_pool_all['method'].isin(real_methods)]

    gss = GroupShuffleSplit(
        n_splits=1, test_size=cfg['data_params']['val_split_ratio'], random_state=cfg['data_params']['seed']
    )
    train_idx, val_idx = next(gss.split(real_df, groups=real_df['video_id']))

    train_real_df = real_df.iloc[train_idx]
    val_real_df = real_df.iloc[val_idx]

    log.info(
        f"Splitting real videos: {train_real_df[['method', 'video_id']].drop_duplicates().shape[0]} for training, {val_real_df[['method', 'video_id']].drop_duplicates().shape[0]} for validation.")

    # This is the final, complete pool of data available for the training dataloader
    df_unbalanced_train_pool = pd.concat([train_real_df, train_fake_df])
    report_counts(df_unbalanced_train_pool, "STAGE 3: FINAL Unbalanced Training Pool")

    print("\n" + "#" * 80)
    print("### NEW STRATEGY: The dataloader receives this entire pool and balances batches on-the-fly.")
    print("#" * 80)

    # --- STAGE 4 (FOR COMPARISON): Apply the OLD undersampling strategy ---
    print("\n" + "#" * 80)
    print("### OLD STRATEGY (FOR COMPARISON): Undersample the pool before creating the dataloader.")
    print("#" * 80)
    df_balanced_old = _balance_df_by_label_old(df_unbalanced_train_pool, real_methods, cfg['data_params']['seed'])
    report_counts(df_balanced_old, "STAGE 4: OLD Balanced Pool (via undersampling)")

    # --- FINAL ANALYSIS (Using correct counting) ---
    frames_discarded = len(df_unbalanced_train_pool) - len(df_balanced_old)

    # Correctly count true unique videos in and out
    fake_videos_in = df_unbalanced_train_pool[df_unbalanced_train_pool['label'] == 'fake'][
        ['method', 'video_id']].drop_duplicates().shape[0]
    fake_videos_out = \
        df_balanced_old[df_balanced_old['label'] == 'fake'][['method', 'video_id']].drop_duplicates().shape[0]
    videos_discarded = fake_videos_in - fake_videos_out

    print("\n" + "=" * 75)
    print(" " * 27 + ">>> ANALYSIS <<<")
    print("-" * 75)
    print(f"By removing upfront balancing, the new strategy will utilize:")
    if frames_discarded > 0 and videos_discarded > 0:
        print(f"  - {frames_discarded:>9,} MORE frames (+{frames_discarded / len(df_balanced_old):.1%})")
        print(f"  - {videos_discarded:>9,} MORE fake videos (+{videos_discarded / fake_videos_out:.1%})")
    else:
        print("  - All available data. No frames or videos were being discarded.")
    print("=" * 75)


if __name__ == "__main__":
    main()
