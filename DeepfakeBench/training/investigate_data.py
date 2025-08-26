# investigate_data.py

import os
import random
import io
import argparse
from pathlib import Path

import cv2
import fsspec
import numpy as np
import pandas as pd
import yaml
from google.cloud import storage
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
GCS_BUCKET_NAME = "df40-frames"
CONFIG_PATH = './config/dataloader_config.yml'
OUTPUT_CSV = "data_analysis_results.csv"
OUTPUT_PLOT_DIR = "data_analysis_plots"
SAMPLE_VIDEOS_PER_METHOD = 30
SAMPLE_FRAMES_PER_VIDEO = 3


# --- Helper Functions ---


def get_laplacian_variance(image_bytes: bytes, gcs_path: str) -> float:
    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if img is None:
            tqdm.write(f"\n[OpenCV Warning] cv2.imdecode failed for {gcs_path}. Returning 0.0 sharpness.")
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance
    except Exception as e:
        tqdm.write(f"\n[OpenCV Error] Exception in get_laplacian_variance for {gcs_path}: {e}. Returning 0.0.")
        return 0.0


def get_frame_stats(gcs_path: str, fs: fsspec.AbstractFileSystem) -> dict | None:
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()
        file_size = len(image_bytes)
        with Image.open(io.BytesIO(image_bytes)) as img:
            width, height = img.size
        sharpness = get_laplacian_variance(image_bytes, gcs_path)
        path_relative_to_bucket = gcs_path.split(f"gs://{GCS_BUCKET_NAME}/", 1)[1]
        path_parts = Path(path_relative_to_bucket).parts
        if len(path_parts) < 3: return None
        label, method = path_parts[0], path_parts[1]
        if label not in ['real', 'fake']: return None
        return {
            "label": label, "method": method, "width": width, "height": height,
            "file_size_kb": file_size / 1024, "sharpness": sharpness,
        }
    except Exception as e:
        tqdm.write(f"\n[Pillow Error] Could not process {gcs_path}: {e}")
        return None


def sample_and_save_data():
    """ Connects to GCS, runs the stratified sampling, and saves results to a CSV."""
    print("--- Mode: Sampling Data from GCS ---")
    print(f"Loading methods from '{CONFIG_PATH}'...")
    with open(CONFIG_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    real_sources = data_config['all_methods']['use_real_sources']
    fake_methods = data_config['all_methods']['use_fake_methods']
    all_methods = real_sources + fake_methods
    print(f"Found {len(real_sources)} real sources and {len(fake_methods)} fake methods.")

    try:
        print(f"Connecting to GCS bucket: gs://{GCS_BUCKET_NAME}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        fs = fsspec.filesystem('gcs')
    except Exception as e:
        print(f"\n[Error] Could not connect to GCS. Details: {e}")
        return

    all_stats = []
    print("\n--- Starting Sampling Process ---")
    for method in tqdm(all_methods, desc="Overall Progress", unit="method"):
        label = "real" if method in real_sources else "fake"
        prefix = f"{label}/{method}/"
        all_blobs_under_method = list(bucket.list_blobs(prefix=prefix))
        if not all_blobs_under_method:
            tqdm.write(f"\n[Warning] No files found for prefix '{prefix}'. Skipping method '{method}'.")
            continue

        video_paths = set()
        for blob in all_blobs_under_method:
            path_parts = Path(blob.name).parts
            if len(path_parts) > 2: video_paths.add('/'.join(path_parts[:-1]) + '/')

        video_prefixes = list(video_paths)
        if not video_prefixes:
            tqdm.write(f"\n[Warning] No video directories derived for '{prefix}'. Skipping method '{method}'.")
            continue

        num_to_sample_videos = min(len(video_prefixes), SAMPLE_VIDEOS_PER_METHOD)
        selected_videos = random.sample(video_prefixes, num_to_sample_videos)

        for video_prefix in selected_videos:
            frame_blobs = list(bucket.list_blobs(prefix=video_prefix))
            valid_frame_blobs = [b for b in frame_blobs if not b.name.endswith('/')]
            if not valid_frame_blobs: continue

            num_to_sample_frames = min(len(valid_frame_blobs), SAMPLE_FRAMES_PER_VIDEO)
            selected_frames = random.sample(valid_frame_blobs, num_to_sample_frames)

            for frame_blob in selected_frames:
                gcs_path = f"gs://{GCS_BUCKET_NAME}/{frame_blob.name}"
                stats = get_frame_stats(gcs_path, fs)
                if stats: all_stats.append(stats)

    if not all_stats:
        print("\n[Error] No data was collected. Review warnings.")
        return

    print(f"\n--- Sampling Complete ---")
    print(f"Collected stats for a total of {len(all_stats)} frames.")
    df = pd.DataFrame(all_stats)
    df['image_area'] = df['width'] * df['height']
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Raw data saved to '{OUTPUT_CSV}'")


def plot_from_csv(csv_path: str):
    """ Loads the data from the CSV and generates analysis plots with corrected plotting logic. """
    print(f"--- Mode: Plotting from '{csv_path}' ---")
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found at '{csv_path}'. Please run the 'sample' mode first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch
        sns.set_theme(style="whitegrid")

        plot_metrics = {
            "file_size_kb": "File Size (KB)",
            "sharpness": "Sharpness (Laplacian Variance)",
            "image_area": "Image Area (pixels, Width * Height)"
        }

        # Create the method-to-color mapping, which remains the correct approach.
        method_to_label = df[['method', 'label']].drop_duplicates().set_index('method')['label']
        color_map = {'real': 'g', 'fake': 'r'}
        custom_palette = method_to_label.map(color_map).to_dict()
        order = sorted(df['method'].unique())

        for col, title in plot_metrics.items():
            print(f"  - Generating plot for {title}...")
            plt.figure(figsize=(20, 12))

            # --- START: FINAL, ROBUST PLOTTING LOGIC ---
            # Follow the FutureWarning's advice precisely to eliminate warnings.
            # 1. Assign 'method' to `hue`.
            # 2. Provide the custom_palette that maps each method to a color.
            # 3. Set `legend=False` because we are creating our own custom legend.
            ax = sns.boxplot(
                data=df,
                x='method',
                y=col,
                order=order,
                hue='method',  # Explicitly use method for hue
                palette=custom_palette,
                legend=False  # Disable the default legend
            )
            # --- END ---

            plt.title(f'Distribution of {title} by Method', fontsize=20)
            plt.xlabel("Method", fontsize=14)
            plt.ylabel(title, fontsize=14)

            # This is the correct way to set rotated tick labels for a categorical axis.
            # The UserWarning is a known, minor issue in some matplotlib/seaborn versions
            # when modifying labels this way, but the output is correct.
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            # Create the clean, manual legend with the corrected 'facecolor' parameter.
            legend_elements = [Patch(facecolor='g', label='Real'), Patch(facecolor='r', label='Fake')]
            ax.legend(handles=legend_elements, title="Label")

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_PLOT_DIR, f"{col}_distribution.png")
            plt.savefig(plot_path)
            plt.close()

        print(f"\n✅ All plots saved in '{OUTPUT_PLOT_DIR}'.")

    except ImportError:
        print("\n[Warning] `matplotlib` or `seaborn` not installed. Skipping plot generation.")
    except Exception as e:
        print(f"\n[Plotting Error] An unexpected error occurred: {e}")


def main():
    """ Main entry point to control script behavior. """
    parser = argparse.ArgumentParser(description="Investigate dataset properties for trivial cues.")
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'sample', 'plot'],
        help="Script mode: 'sample' to fetch data from GCS, 'plot' to generate plots from CSV, 'all' to do both."
    )
    args = parser.parse_args()

    if args.mode in ['all', 'sample']:
        sample_and_save_data()

    if args.mode in ['all', 'plot']:
        plot_from_csv(OUTPUT_CSV)


if __name__ == "__main__":
    main()
