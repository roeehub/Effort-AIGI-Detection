# investigate_data.py

import os
import random
import argparse
from pathlib import Path
import concurrent.futures
from functools import partial

import cv2
import fsspec
import numpy as np
import pandas as pd
import yaml
from google.cloud import storage
from tqdm import tqdm
from scipy import fft

# --- Configuration ---
# Train Config
DF40_BUCKET_NAME = "df40-frames"
CONFIG_PATH = './config/dataloader_config.yml'
SAMPLE_VIDEOS_PER_METHOD_TRAIN = 30
SAMPLE_FRAMES_PER_VIDEO_TRAIN = 3

# Test Config
TEST_BUCKET_PREFIX = "deep-fake-test-10-08-25-frames"
SAMPLE_FRAMES_PER_VIDEO_TEST = 3

# Global Config
OUTPUT_CSV = "data_analysis_results.csv"
OUTPUT_PLOT_DIR = "data_analysis_plots"
DEBUG_IMAGE_PATH = "/Users/roeedar/Desktop/debug/wang2.png"
# NEW: Default number of parallel workers for processing
DEFAULT_WORKERS = os.cpu_count() * 2


# --- Tier 1 & Tier 2 Analysis Functions (Unchanged) ---
def get_laplacian_variance(image_np: np.ndarray, gcs_path: str = "local") -> float:
    """Calculates sharpness using Laplacian variance."""
    try:
        if image_np is None or image_np.size == 0:
            tqdm.write(f"\n[OpenCV Warning] Invalid image array for {gcs_path}. Returning 0.0 sharpness.")
            return 0.0
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    except Exception as e:
        tqdm.write(f"\n[OpenCV Error] Exception in get_laplacian_variance for {gcs_path}: {e}. Returning 0.0.")
        return 0.0


def get_color_stats(image_np: np.ndarray) -> dict:
    """Calculates mean R, G, B channel values."""
    if image_np is None or image_np.ndim < 3:
        return {"mean_r": 0, "mean_g": 0, "mean_b": 0}
    # Assuming BGR from OpenCV, so we reverse for RGB convention
    mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))
    return {"mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b}


def get_freq_domain_ratio(image_np: np.ndarray, radius_ratio: float = 0.2) -> float:
    """Calculates the ratio of high-frequency to low-frequency energy in the FFT spectrum."""
    try:
        if image_np is None or image_np.size == 0: return 0.0
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        f = fft.fft2(gray)
        fshift = fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2

        # Create a circular mask for low frequencies
        radius = int(min(crow, ccol) * radius_ratio)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)

        low_freq_energy = np.sum(magnitude_spectrum[mask == 1])
        high_freq_energy = np.sum(magnitude_spectrum[mask == 0])

        if low_freq_energy == 0: return np.inf
        return high_freq_energy / low_freq_energy
    except Exception:
        return 0.0


def get_frame_stats(gcs_path: str, fs: fsspec.AbstractFileSystem) -> dict | None:
    """Computes all stats for a single frame from GCS."""
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()

        file_size = len(image_bytes)
        image_np_raw = np.frombuffer(image_bytes, np.uint8)
        image_np_bgr = cv2.imdecode(image_np_raw, cv2.IMREAD_COLOR)

        if image_np_bgr is None:
            tqdm.write(f"\n[OpenCV Warning] cv2.imdecode failed for {gcs_path}.")
            return None

        height, width, _ = image_np_bgr.shape

        # --- Run all analysis ---
        sharpness = get_laplacian_variance(image_np_bgr, gcs_path)
        color_stats = get_color_stats(image_np_bgr)
        freq_ratio = get_freq_domain_ratio(image_np_bgr)

        return {
            "width": width, "height": height, "file_size_kb": file_size / 1024,
            "sharpness": sharpness, "freq_ratio": freq_ratio, **color_stats,
        }
    except Exception as e:
        tqdm.write(f"\n[Processing Error] Could not process {gcs_path}: {e}")
        return None


# --- NEW: Parallel Processing Function ---
def _process_frames_in_parallel(frame_paths_with_metadata, fs, num_workers):
    """
    Takes a list of tuples (gcs_path, metadata) and processes them in parallel.
    """
    stats_list = []

    # Use partial to pre-fill the 'fs' argument for our processing function
    processing_function = partial(get_frame_stats, fs=fs)

    # Create a list of GCS paths to feed into the executor
    gcs_paths = [item[0] for item in frame_paths_with_metadata]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to apply the function to all paths and wrap with tqdm for progress
        results = list(
            tqdm(executor.map(processing_function, gcs_paths), total=len(gcs_paths), desc="Processing Frames"))

    # Now, combine the results with the original metadata
    for i, stats in enumerate(results):
        if stats:
            # Original metadata was stored at frame_paths_with_metadata[i][1]
            metadata = frame_paths_with_metadata[i][1]
            stats.update(metadata)
            stats_list.append(stats)

    return stats_list


# --- MODIFIED: Sampling Logic ---

def _sample_train_data(bucket, fs, num_workers):
    """Specific sampling logic for the DF40 (train) dataset."""
    print("\n--- Sampling from TRAIN dataset (DF40) ---")
    with open(CONFIG_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    real_sources = data_config['all_methods']['use_real_sources']
    fake_methods = data_config['all_methods']['use_fake_methods']
    all_methods = real_sources + fake_methods

    frame_paths_to_process = []
    for method in tqdm(all_methods, desc="Discovering Train Videos", unit="method"):
        label = "real" if method in real_sources else "fake"
        prefix = f"{label}/{method}/"
        video_paths = set('/'.join(Path(b.name).parts[:-1]) + '/' for b in bucket.list_blobs(prefix=prefix))
        if not video_paths: continue

        num_videos_to_sample = min(len(video_paths), SAMPLE_VIDEOS_PER_METHOD_TRAIN)
        selected_videos = random.sample(list(video_paths), num_videos_to_sample)

        for video_prefix in selected_videos:
            frame_blobs = [b for b in bucket.list_blobs(prefix=video_prefix) if not b.name.endswith('/')]
            if not frame_blobs: continue

            num_frames_to_sample = min(len(frame_blobs), SAMPLE_FRAMES_PER_VIDEO_TRAIN)
            selected_frames = random.sample(frame_blobs, num_frames_to_sample)

            for frame_blob in selected_frames:
                gcs_path = f"gs://{bucket.name}/{frame_blob.name}"
                metadata = {"label": label, "method": method, "source": "train"}
                frame_paths_to_process.append((gcs_path, metadata))

    print(f"Discovered {len(frame_paths_to_process)} train frames to analyze.")
    return _process_frames_in_parallel(frame_paths_to_process, fs, num_workers)


def _sample_test_data(bucket, fs, num_workers):
    """Specific sampling logic for the OOD (test) dataset with corrected method discovery."""
    print(f"\n--- Sampling from TEST dataset ({bucket.name}) ---")
    methods_and_labels = []
    for label in ['real', 'fake']:
        iterator = bucket.list_blobs(prefix=f'{label}/', delimiter='/')
        for page in iterator.pages:
            for prefix in page.prefixes:
                method_name = prefix.strip('/').split('/')[-1]
                methods_and_labels.append((method_name, label))

    frame_paths_to_process = []
    for method, label in tqdm(methods_and_labels, desc="Discovering Test Videos", unit="method"):
        prefix = f"{label}/{method}/"
        video_paths = set('/'.join(Path(b.name).parts[:-1]) + '/' for b in bucket.list_blobs(prefix=prefix))
        if not video_paths: continue

        for video_prefix in list(video_paths):
            frame_blobs = [b for b in bucket.list_blobs(prefix=video_prefix) if not b.name.endswith('/')]
            if not frame_blobs: continue

            num_frames_to_sample = min(len(frame_blobs), SAMPLE_FRAMES_PER_VIDEO_TEST)
            selected_frames = random.sample(frame_blobs, num_frames_to_sample)

            for frame_blob in selected_frames:
                gcs_path = f"gs://{bucket.name}/{frame_blob.name}"
                metadata = {"label": label, "method": method, "source": "test"}
                frame_paths_to_process.append((gcs_path, metadata))

    print(f"Discovered {len(frame_paths_to_process)} test frames to analyze.")
    return _process_frames_in_parallel(frame_paths_to_process, fs, num_workers)


# --- MODIFIED: Main Controller Functions ---

def run_sampling(datasets, test_bucket_suffix, num_workers):
    """Connects to GCS and runs sampling, saving intermediate results for robustness."""
    print("--- Mode: Sampling Data from GCS ---")
    print(f"Using {num_workers} parallel workers for processing.")

    try:
        print("Connecting to GCS...")
        storage_client = storage.Client()
        fs = fsspec.filesystem('gcs')
    except Exception as e:
        print(f"\n[Error] Could not connect to GCS. Details: {e}")
        return

    all_stats = []
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing data file '{OUTPUT_CSV}'. Removing to start fresh.")
        os.remove(OUTPUT_CSV)

    if 'train' in datasets:
        train_bucket = storage_client.bucket(DF40_BUCKET_NAME)
        train_stats = _sample_train_data(train_bucket, fs, num_workers)
        if train_stats:
            all_stats.extend(train_stats)
            print("\nSaving intermediate results for TRAIN dataset...")
            pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Saved {len(train_stats)} train frame stats to '{OUTPUT_CSV}'")

    if 'test' in datasets:
        if not test_bucket_suffix:
            print("[Error] --test_bucket_suffix is required when sampling 'test' or 'both' datasets.")
            return
        test_bucket_name = f"{TEST_BUCKET_PREFIX}-{test_bucket_suffix}"
        test_bucket = storage_client.bucket(test_bucket_name)
        test_stats = _sample_test_data(test_bucket, fs, num_workers)
        if test_stats:
            all_stats.extend(test_stats)
            print("\nSaving final results for ALL sampled datasets...")
            pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Saved {len(test_stats)} test frame stats. Total stats saved: {len(all_stats)}.")

    if not all_stats:
        print("\n[Error] No data was collected. Review logs for warnings.")
        return

    print(f"\n--- Sampling Complete ---")


def run_plotting(csv_path):
    """
    Loads data from CSV and generates analysis plots.
    - Groups test methods on the left.
    - Correctly handles methods that contain both real and fake videos.
    """
    print(f"--- Mode: Plotting from '{csv_path}' ---")
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: '{csv_path}'. Run sampling first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid")

        if 'image_area' not in df.columns:
            df['image_area'] = df['width'] * df['height']

        plot_metrics = {
            "file_size_kb": "File Size (KB)",
            "sharpness": "Sharpness (Laplacian Variance)",
            "image_area": "Image Area (pixels)",
            "freq_ratio": "High/Low Freq. Energy Ratio",
            "mean_r": "Mean Red Channel",
            "mean_g": "Mean Green Channel",
            "mean_b": "Mean Blue Channel"
        }

        # --- REVISED: Custom sorting logic for the X-axis 'method' column ---
        all_methods = df['method'].unique()
        # Identify which methods appear in the test set vs. train set
        methods_in_test = df[df['source'] == 'test']['method'].unique()
        methods_in_train = df[df['source'] == 'train']['method'].unique()

        # Prioritize methods that appear in the test set, then the rest
        test_methods_sorted = sorted(list(methods_in_test))
        # Ensure train methods are unique and sorted
        train_methods_sorted = sorted(list(set(methods_in_train) - set(methods_in_test)))

        order = test_methods_sorted + train_methods_sorted
        # --- END OF REVISION ---

        for col, title in plot_metrics.items():
            print(f"  - Generating plot for {title}...")
            plt.figure(figsize=(28, 14))

            # --- MAJOR CHANGE: Using `hue` for correct color mapping ---
            # We now use 'method' for x, and 'source' combined with 'label' for hue.
            # This allows seaborn to handle cases where a method has real/fake in test/train.
            ax = sns.boxplot(data=df, x='method', y=col, hue='label', order=order,
                             palette={'real': 'g', 'fake': 'r'})

            # This part is a bit more complex to add the (test)/(train) indicators back
            # We will modify the xticklabels directly for clarity.
            new_labels = []
            for tick_label in ax.get_xticklabels():
                method_name = tick_label.get_text()
                sources = []
                if method_name in methods_in_test:
                    sources.append('test')
                if method_name in methods_in_train:
                    sources.append('train')
                new_labels.append(f"{method_name} ({'/'.join(sources)})")
            ax.set_xticklabels(new_labels)
            # --- END OF MAJOR CHANGE ---

            plt.title(f'Distribution of {title} by Method and Label', fontsize=20)
            plt.xlabel("Method (Source)", fontsize=14)
            plt.ylabel(title, fontsize=14)
            ax.tick_params(axis='x', rotation=45, labelsize=12)
            # Ensure the legend is correctly labeled
            ax.legend(title='Label')

            # --- Add a visual separator between test and train data ---
            if test_methods_sorted and train_methods_sorted:
                # The line goes after the last method that appears in the test set
                separator_pos = len(test_methods_sorted) - 0.5
                ax.axvline(separator_pos, color='k', linestyle='--', linewidth=2, alpha=0.7)

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_PLOT_DIR, f"{col}_distribution.png")
            plt.savefig(plot_path)
            plt.close()

        print(f"\n✅ All plots saved in '{OUTPUT_PLOT_DIR}'.")

    except ImportError:
        print("\n[Warning] `matplotlib` or `seaborn` not installed. Skipping plot generation.")
    except Exception as e:
        print(f"\n[Plotting Error] An unexpected error occurred: {e}")


def _generate_quantile_table(df: pd.DataFrame, metric: str, title: str):
    """Helper function to create and print a formatted quantile table."""
    print(f"\n--- {title} ---")
    if metric not in df.columns:
        print(f"[Warning] Metric '{metric}' not found in the DataFrame. Skipping table.")
        return

    # Calculate quantiles using modern pandas aggregation
    quantile_df = df.groupby(['method', 'label']).agg(
        p25=(metric, lambda x: x.quantile(0.25)),
        median=(metric, 'median'),
        p75=(metric, lambda x: x.quantile(0.75)),
        count=(metric, 'size')  # Also get the count of samples for context
    ).sort_index()

    # Format for better readability
    for col in ['p25', 'median', 'p75']:
        quantile_df[col] = quantile_df[col].map('{:,.2f}'.format)

    print(quantile_df)
    print("-" * (len(title) + 6))


def _calculate_overlap_percentage(df, group1_method, group1_label, group2_method, group2_label, metric='sharpness'):
    """Helper to calculate the 'Killer Stat' overlap."""
    group1_df = df[(df['method'] == group1_method) & (df['label'] == group1_label)]
    group2_df = df[(df['method'] == group2_method) & (df['label'] == group2_label)]

    if group1_df.empty or group2_df.empty:
        return None, None  # One of the groups doesn't exist in the data

    median_g2 = group2_df[metric].median()
    median_g1 = group1_df[metric].median()

    # % of Group 1 sharper than Group 2's median
    overlap_1_vs_2 = (group1_df[metric] > median_g2).mean() * 100

    # % of Group 2 blurrier than Group 1's median
    overlap_2_vs_1 = (group2_df[metric] < median_g1).mean() * 100

    return overlap_1_vs_2, overlap_2_vs_1


def run_numerical_report(csv_path):
    """
    Loads data from the analysis CSV and generates numerical summary tables
    to quantitatively assess data imbalance.
    """
    print(f"--- Mode: Generating Numerical Report from '{csv_path}' ---")
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: '{csv_path}'. Run sampling first.")
        return

    df = pd.read_csv(csv_path)
    # Ensure we only look at training data for this analysis
    df = df[df['source'] == 'train'].copy()

    if df.empty:
        print("[Error] No training data found in the CSV. Cannot generate report.")
        return

    # --- 1. Generate Quantile Tables for all methods ---
    _generate_quantile_table(df, 'sharpness', 'Quantile Breakdown for Sharpness (Training Set)')
    _generate_quantile_table(df, 'mean_r', 'Quantile Breakdown for Mean Red Channel (Training Set)')
    _generate_quantile_table(df, 'file_size_kb', 'Quantile Breakdown for File Size (KB) (Training Set)')

    # --- 2. Calculate the "Killer Stat" for Key Comparisons ---
    print("\n\n--- Overlap Analysis ('The Killer Stat') ---")
    print("This analysis reveals how easily classes can be separated by a single metric.")

    # Comparison 1: Blurry Real vs. Sharp Fake
    print("\n[Comparison 1]: Blurry Reals (Celeb-real) vs. A Sharp Fake (SIT)")
    overlap1, overlap2 = _calculate_overlap_percentage(df, 'Celeb-real', 'real', 'SIT', 'fake')
    if overlap1 is not None:
        print(f"  - Percentage of 'Celeb-real' frames SHARPER than the median 'SIT' frame: {overlap1:.2f}%")
        print(f"  - Percentage of 'SIT' frames BLURRIER than the median 'Celeb-real' frame: {overlap2:.2f}%")
    else:
        print("  - Could not perform comparison; one or both methods not found in training data.")

    # Comparison 2: Sharp Real vs. Blurry Fake
    print("\n[Comparison 2]: Sharp Reals (external_youtube_avspeech) vs. A Blurry Fake (FaceShifter)")
    overlap3, overlap4 = _calculate_overlap_percentage(df, 'external_youtube_avspeech', 'real', 'FaceShifter', 'fake')
    if overlap3 is not None:
        print(
            f"  - Percentage of 'youtube_avspeech' frames SHARPER than the median 'FaceShifter' frame: {overlap3:.2f}%")
        print(
            f"  - Percentage of 'FaceShifter' frames BLURRIER than the median 'youtube_avspeech' frame: {overlap4:.2f}%")
    else:
        print("  - Could not perform comparison; one or both methods not found in training data.")
    print("-" * 42)

    print("\n✅ Numerical report finished.")


def run_debug(image_path):
    """Runs all analysis on a single local image for debugging."""
    print(f"--- Mode: Debugging with image '{image_path}' ---")
    if not os.path.exists(image_path):
        print(f"[Error] Debug image not found at '{image_path}'.")
        # Create a dummy image if it doesn't exist
        print("Creating a dummy 128x128 black image for debugging.")
        dummy_img = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.imwrite(image_path, dummy_img)

    image_np_bgr = cv2.imread(image_path)
    if image_np_bgr is None:
        print("[Error] Could not read the debug image with OpenCV.")
        return

    print("\n--- Running Analysis Functions ---")
    # Tier 1
    sharpness = get_laplacian_variance(image_np_bgr)
    height, width, _ = image_np_bgr.shape
    file_size_kb = os.path.getsize(image_path) / 1024

    # Tier 2
    color_stats = get_color_stats(image_np_bgr)
    freq_ratio = get_freq_domain_ratio(image_np_bgr)

    print("\n--- Results ---")
    print(f"  Tier 1 Metrics:")
    print(f"    - Dimensions    : {width}x{height}")
    print(f"    - File Size     : {file_size_kb:.2f} KB")
    print(f"    - Sharpness     : {sharpness:.2f}")
    print(f"\n  Tier 2 Metrics:")
    print(f"    - Mean Red      : {color_stats['mean_r']:.2f}")
    print(f"    - Mean Green    : {color_stats['mean_g']:.2f}")
    print(f"    - Mean Blue     : {color_stats['mean_b']:.2f}")
    print(f"    - Freq. Ratio   : {freq_ratio:.4f}")
    print("\n✅ Debug mode finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Investigate dataset properties for trivial cues across train and test sets.")
    parser.add_argument(
        '--mode', type=str, default='all', choices=['all', 'sample', 'plot', 'report'],
        help="Script mode: 'sample' to fetch data, 'plot' to generate plots, 'report' for numerical summary, 'all' for sample+plot."
    )
    parser.add_argument(
        '--dataset', type=str, default='both', choices=['train', 'test', 'both'],
        help="Which dataset(s) to analyze."
    )
    parser.add_argument(
        '--test_bucket_suffix', type=str, choices=['yolo', 'yolo-haar'],
        help="Suffix for the test bucket name (e.g., 'yolo'). Required if '--dataset' includes 'test'."
    )
    # --- NEW ARGUMENT ---
    parser.add_argument(
        '--workers', type=int, default=DEFAULT_WORKERS,
        help=f"Number of parallel workers for downloading and processing frames. Default: {DEFAULT_WORKERS}"
    )
    parser.add_argument(
        '--debug', type=str, nargs='?', const=DEBUG_IMAGE_PATH, default=None,
        help="Run in debug mode on a single local image. Optionally provide a path to the image."
    )
    args = parser.parse_args()

    if args.debug:
        run_debug(args.debug)
        return

    datasets_to_run = []
    if args.dataset == 'train':
        datasets_to_run = ['train']
    elif args.dataset == 'test':
        datasets_to_run = ['test']
    elif args.dataset == 'both':
        datasets_to_run = ['train', 'test']

    # --- MODIFIED CALLS ---
    if args.mode in ['all', 'sample']:
        run_sampling(datasets_to_run, args.test_bucket_suffix, args.workers)

    # Check if CSV exists before plotting or reporting
    csv_exists = os.path.exists(OUTPUT_CSV)

    if args.mode in ['all', 'plot']:
        if not csv_exists:
            print(f"[Error] CSV file '{OUTPUT_CSV}' not found. Cannot plot.")
        else:
            run_plotting(OUTPUT_CSV)

    if args.mode == 'report':
        if not csv_exists:
            print(f"[Error] CSV file '{OUTPUT_CSV}' not found. Run sampling first to create it.")
        else:
            run_numerical_report(OUTPUT_CSV)


if __name__ == "__main__":
    main()
