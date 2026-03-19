#!/usr/bin/env python3
"""
analyze_buckets.py

A diagnostic script to analyze the distribution of frames across property buckets
for the training set. It replicates the data preparation logic to ensure the
analysis is performed on the exact data pool the dataloader would see.

This helps us understand data sparsity and decide on dataloader strategies.
"""
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from rich.console import Console
from rich.table import Table

# --- Reuse configuration from your verification script ---
# Ensure this script is run from the same directory.
if '.' not in sys.path:
    sys.path.append('.')
try:
    from verify_property_dataloader import DATA_CONFIG, MANIFEST_PATH
except ImportError:
    print("ERROR: Could not import from 'verify_property_dataloader.py'.")
    print("Please ensure this script is in the same directory.")
    sys.exit(1)

# --- Analysis Parameters ---
MIN_BUCKET_SIZE_THRESHOLD = 64  # Buckets with fewer frames than this are considered "small"


def create_training_pool_df(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Replicates the logic from prepare_splits.py to create the final,
    unbalanced training data pool that our new dataloader would receive.
    """
    print("--- Replicating the training set creation logic (without final balancing) ---")
    SEED = cfg['data_params']['seed']

    # 1. Partition DataFrame by method type from config
    real_methods = set(cfg['methods']['use_real_sources'])
    train_fake_methods = set(cfg['methods']['use_fake_methods_for_training'])

    real_df = df[df['method'].isin(real_methods)]
    train_fake_df = df[df['method'].isin(train_fake_methods)]
    print(f"Found {len(real_df):,} real frames and {len(train_fake_df):,} training-fake frames.")

    # 2. Create a mutually exclusive split of REAL videos for train/val
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=cfg['data_params']['val_split_ratio'],
        random_state=SEED
    )
    train_real_idx, _ = next(gss.split(real_df, groups=real_df['video_id']))
    train_real_df = real_df.iloc[train_real_idx]
    print(f"Split real videos -> {len(train_real_df):,} frames reserved for training.")

    # 3. Assemble the final training set POOL (unbalanced by label)
    train_pool_df = pd.concat([train_real_df, train_fake_df])
    print(f"Total frames in the final training pool: {len(train_pool_df):,}\n")
    return train_pool_df


def add_property_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the crucial 'property_bucket' column to the DataFrame.
    """
    print("--- Creating property buckets using 4 quantiles for sharpness and file size ---")

    # Create 4 quantile-based buckets for sharpness and file size
    try:
        df['sharpness_bucket'] = pd.qcut(df['sharpness'], 4, labels=['s_q1', 's_q2', 's_q3', 's_q4'], duplicates='drop')
        df['size_bucket'] = pd.qcut(df['file_size_kb'], 4, labels=['f_q1', 'f_q2', 'f_q3', 'f_q4'], duplicates='drop')

        # Combine them into a single 'property_bucket' column
        df['property_bucket'] = df['sharpness_bucket'].astype(str) + '_' + df['size_bucket'].astype(str)

        print(f"Successfully created {df['property_bucket'].nunique()} unique property buckets.\n")
    except Exception as e:
        print(f"ERROR creating buckets: {e}")
        print("This can happen if a column has too little variance to form quantiles.")
        sys.exit(1)

    return df


def print_bucket_analysis_table(console: Console, title: str, df_counts: pd.DataFrame):
    """Prints a formatted table and summary stats for bucket counts."""
    if df_counts.empty:
        console.print(f"[bold yellow]No buckets found for '{title}'.[/]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Property Bucket", style="cyan", no_wrap=True)
    table.add_column("Frame Count", justify="right", style="white")

    small_bucket_count = 0
    for _, row in df_counts.sort_values('frame_count').iterrows():
        count_str = f"[bold red]{row['frame_count']:,}[/]" if row[
                                                                  'frame_count'] < MIN_BUCKET_SIZE_THRESHOLD else f"{row['frame_count']:,}"
        if row['frame_count'] < MIN_BUCKET_SIZE_THRESHOLD:
            small_bucket_count += 1
        table.add_row(row['property_bucket'], count_str)

    console.print(table)

    # Summary Statistics
    console.print(f"[bold]Summary for '{title}':[/]")
    console.print(f"  - Total Unique Buckets: [bold cyan]{len(df_counts)}[/]")
    console.print(f"  - Buckets with < {MIN_BUCKET_SIZE_THRESHOLD} frames: [bold red]{small_bucket_count}[/]")
    console.print(f"  - Min Frames in a Bucket: [bold yellow]{df_counts['frame_count'].min():,}[/]")
    console.print(f"  - Max Frames in a Bucket: [bold green]{df_counts['frame_count'].max():,}[/]")
    console.print(f"  - Median Frames per Bucket: [bold blue]{df_counts['frame_count'].median():,.0f}[/]")
    console.print(f"  - Mean Frames per Bucket:   [bold blue]{df_counts['frame_count'].mean():,.0f}[/]\n")


def main():
    """Main execution function."""
    console = Console()
    manifest_path = Path(MANIFEST_PATH)
    if not manifest_path.exists():
        console.print(f"[bold red]ERROR:[/] Manifest file not found at '{manifest_path}'")
        return

    console.print(f"Loading manifest from '{manifest_path}'...")
    df = pd.read_parquet(manifest_path)

    # 1. Create the training data pool we care about
    train_pool_df = create_training_pool_df(df, DATA_CONFIG)

    # 2. Add property buckets to this specific pool
    train_pool_df = add_property_buckets(train_pool_df)

    # 3. Analyze the bucket distribution
    bucket_counts = train_pool_df.groupby(['label', 'property_bucket']).size().reset_index(name='frame_count')

    real_bucket_counts = bucket_counts[bucket_counts['label'] == 'real']
    fake_bucket_counts = bucket_counts[bucket_counts['label'] == 'fake']

    # 4. Print the reports
    print_bucket_analysis_table(console, "REAL Frame Buckets", real_bucket_counts)
    print_bucket_analysis_table(console, "FAKE Frame Buckets", fake_bucket_counts)

    console.print("[bold yellow]CONCLUSION:[/]")
    console.print("Review the tables above. Red numbers indicate 'small' buckets that are candidates")
    console.print("for consolidation into a 'misc' bucket to prevent over-sampling and improve stability.")


if __name__ == "__main__":
    main()
