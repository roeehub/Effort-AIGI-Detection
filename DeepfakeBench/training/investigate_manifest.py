import sys
import pandas as pd
import numpy as np

# --- Add project root to path to allow imports ---
if '.' not in sys.path:
    sys.path.append('.')

# --- Imports from your project files ---
# We use the exact same functions to replicate the dataloader's input
from verify_property_dataloader import (
    DATA_CONFIG,
    MANIFEST_PATH,
    add_property_buckets,
    prepare_train_frames_from_df
)


def analyze_training_data(train_frames: list[dict]):
    """
    Performs a deep analysis of the frame distribution that is fed
    into the property-balanced dataloader.
    """
    if not train_frames:
        print("ERROR: The list of training frames is empty. Cannot analyze.")
        return

    print("\n" + "=" * 80)
    print("||" + " " * 18 + "PROPERTY MANIFEST & DATALOADER INPUT ANALYSIS" + " " * 18 + "||")
    print("=" * 80)

    # Convert the list of dicts back to a DataFrame for easy analysis
    df = pd.DataFrame(train_frames)
    df['label_str'] = np.where(df['label_id'] == 0, 'real', 'fake')

    # --- 1. Overall Balance Analysis ---
    print("\n--- [1/3] Overall Balance of the Dataloader's Input Pool ---")
    frame_counts = df['label_str'].value_counts()
    video_counts = df.groupby('label_str')['video_id'].nunique()

    print(f"Total Frames: {len(df):,}")
    print(f"  - Real Frames: {frame_counts.get('real', 0):,}")
    print(f"  - Fake Frames: {frame_counts.get('fake', 0):,}")
    print("\n" + "-" * 40)
    print(f"Total Unique Videos: {df['video_id'].nunique():,}")
    print(f"  - Real Videos: {video_counts.get('real', 0):,}")
    print(f"  - Fake Videos: {video_counts.get('fake', 0):,}")
    print("\n" + "-" * 40)
    if video_counts.get('real', 0) != video_counts.get('fake', 0):
        print("⚠️  WARNING: The number of unique REAL and FAKE videos is NOT balanced.")
        print("   The property dataloader is designed to fix frame-level imbalance,")
        print("   but a large video-level imbalance in its input can still cause issues.")
        print("   Consider balancing the video counts *before* creating the dataloader.")
    else:
        print("✅ INFO: The number of unique REAL and FAKE videos is balanced.")

    # --- 2. Property Bucket Deep Dive ---
    print("\n--- [2/3] Property Bucket Distribution Analysis ---")

    # Create a pivot table to see frame counts per bucket/label
    pivot_frames = pd.pivot_table(
        df,
        index='property_bucket',
        columns='label_str',
        values='path',
        aggfunc='count',
        fill_value=0
    )

    # Create a pivot table for unique video counts
    pivot_videos = pd.pivot_table(
        df,
        index='property_bucket',
        columns='label_str',
        values='video_id',
        aggfunc=pd.Series.nunique,
        fill_value=0
    )

    # Combine for a comprehensive view
    analysis_df = pivot_frames.join(pivot_videos, lsuffix='_frames', rsuffix='_videos')
    analysis_df.columns = ['fake_frames', 'real_frames', 'fake_videos', 'real_videos']
    analysis_df = analysis_df[['real_frames', 'fake_frames', 'real_videos', 'fake_videos']]  # Reorder

    print("Distribution of frames and unique videos across property buckets:")
    print(analysis_df)

    # Identify "Orphan" buckets - the most likely cause of imbalance
    orphan_buckets_real = analysis_df[analysis_df['fake_frames'] == 0].index.tolist()
    orphan_buckets_fake = analysis_df[analysis_df['real_frames'] == 0].index.tolist()

    if orphan_buckets_real or orphan_buckets_fake:
        print("\n" + "-" * 40)
        print("❌ CRITICAL: Orphan buckets found! These buckets lack a corresponding")
        print("   class, which breaks the dataloader's Zipper mechanism and leads")
        print("   to severe label imbalance in the batches.")
        if orphan_buckets_real:
            print(f"  - Buckets with REAL frames but no FAKE frames: {orphan_buckets_real}")
        if orphan_buckets_fake:
            print(f"  - Buckets with FAKE frames but no REAL frames: {orphan_buckets_fake}")
    else:
        print("\n✅ INFO: No orphan buckets found. All property buckets have both real and fake frames.")

    # --- 3. Video Integrity Check (for Mate Pairing) ---
    print("\n--- [3/3] Video Integrity for Anchor-Mate Pairing ---")
    video_frame_counts = df['video_id'].value_counts()
    single_frame_videos = video_frame_counts[video_frame_counts == 1].count()

    print(f"Found {single_frame_videos:,} videos with only a single frame in the training set.")
    if single_frame_videos > 0:
        print("   (For these videos, the anchor frame will be paired with itself, which is expected behavior).")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE.")
    print("=" * 80)


def main():
    """Main execution function."""
    print(f"Loading manifest from: '{MANIFEST_PATH}'...")
    try:
        df = pd.read_parquet(MANIFEST_PATH)
    except FileNotFoundError:
        print(f"ERROR: Manifest file not found at '{MANIFEST_PATH}'")
        print("Please ensure the manifest exists before running this analysis.")
        sys.exit(1)

    # Replicate the exact pre-processing from the verification script
    df['label_id'] = np.where(df['label'] == 'real', 0, 1)
    df_bucketed = add_property_buckets(df)

    # Get the exact list of frames the dataloader would receive
    print("\nReplicating the data splitting process...")
    train_frames = prepare_train_frames_from_df(df_bucketed, DATA_CONFIG)

    # Run the analysis
    analyze_training_data(train_frames)


if __name__ == "__main__":
    main()
