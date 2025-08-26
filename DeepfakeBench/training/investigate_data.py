# investigate_data.py

import os
import random
import io
import argparse
from pathlib import Path

import cv2  # noqa
import fsspec  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import yaml  # noqa
from google.cloud import storage  # noqa
from PIL import Image  # noqa
from tqdm import tqdm  # noqa
from scipy import fft  # noqa

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


# --- Tier 1 & Tier 2 Analysis Functions ---

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


# --- Sampling Logic ---

def _sample_train_data(bucket, fs):
    """Specific sampling logic for the DF40 (train) dataset."""
    print("\n--- Sampling from TRAIN dataset (DF40) ---")
    with open(CONFIG_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    real_sources = data_config['all_methods']['use_real_sources']
    fake_methods = data_config['all_methods']['use_fake_methods']
    all_methods = real_sources + fake_methods

    stats_list = []
    for method in tqdm(all_methods, desc="Train Methods", unit="method"):
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
                stats = get_frame_stats(gcs_path, fs)
                if stats:
                    stats.update({"label": label, "method": method, "source": "train"})
                    stats_list.append(stats)
    return stats_list


def _sample_test_data(bucket, fs):
    """Specific sampling logic for the OOD (test) dataset with corrected method discovery."""
    print(f"\n--- Sampling from TEST dataset ({bucket.name}) ---")

    # --- START OF BUG FIX ---
    # Use a list of tuples to correctly handle methods with the same name but different labels.
    methods_and_labels = []
    for label in ['real', 'fake']:
        iterator = bucket.list_blobs(prefix=f'{label}/', delimiter='/')
        for page in iterator.pages:
            for prefix in page.prefixes:
                # prefix will be like 'fake/tiktok/'
                method_name = prefix.strip('/').split('/')[-1]
                methods_and_labels.append((method_name, label))
    # --- END OF BUG FIX ---

    stats_list = []
    # Iterate over the list of tuples
    for method, label in tqdm(methods_and_labels, desc="Test Methods", unit="method"):
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
                stats = get_frame_stats(gcs_path, fs)
                if stats:
                    stats.update({"label": label, "method": method, "source": "test"})
                    stats_list.append(stats)
    return stats_list


# --- Main Controller Functions ---

def run_sampling(datasets, test_bucket_suffix):
    """Connects to GCS and runs sampling, saving intermediate results for robustness."""
    print("--- Mode: Sampling Data from GCS ---")

    try:
        print("Connecting to GCS...")
        storage_client = storage.Client()
        fs = fsspec.filesystem('gcs')
    except Exception as e:
        print(f"\n[Error] Could not connect to GCS. Details: {e}")
        return

    all_stats = []
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing data file '{OUTPUT_CSV}'. Will append new data.")
        # To avoid re-sampling, you might want to load existing data or clear the list.
        # For simplicity, we'll start fresh. A more complex system could avoid re-sampling.
        os.remove(OUTPUT_CSV)  # Start fresh to avoid duplicates if script is re-run

    if 'train' in datasets:
        train_bucket = storage_client.bucket(DF40_BUCKET_NAME)
        train_stats = _sample_train_data(train_bucket, fs)
        if train_stats:
            all_stats.extend(train_stats)
            # --- ROBUSTNESS FIX: Save after train is done ---
            print("\nSaving intermediate results for TRAIN dataset...")
            pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Saved {len(train_stats)} train frame stats to '{OUTPUT_CSV}'")
            # --- END OF FIX ---

    if 'test' in datasets:
        if not test_bucket_suffix:
            print("[Error] --test_bucket_suffix is required when sampling 'test' or 'both' datasets.")
            return
        test_bucket_name = f"{TEST_BUCKET_PREFIX}-{test_bucket_suffix}"
        test_bucket = storage_client.bucket(test_bucket_name)
        test_stats = _sample_test_data(test_bucket, fs)
        if test_stats:
            all_stats.extend(test_stats)
            # --- ROBUSTNESS FIX: Save after test is done (overwriting the intermediate file) ---
            print("\nSaving final results for ALL sampled datasets...")
            pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
            print(f"✅ Saved {len(test_stats)} test frame stats. Total stats saved: {len(all_stats)}.")
            # --- END OF FIX ---

    if not all_stats:
        print("\n[Error] No data was collected. Review logs for warnings.")
        return

    print(f"\n--- Sampling Complete ---")


def run_plotting(csv_path):
    """Loads data from the CSV and generates analysis plots."""
    print(f"--- Mode: Plotting from '{csv_path}' ---")
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: '{csv_path}'. Run sampling first.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Patch
        sns.set_theme(style="whitegrid")

        # Add the 'source' to the method name for clear plotting
        df['method_source'] = df['method'] + ' (' + df['source'] + ')'

        plot_metrics = {
            "file_size_kb": "File Size (KB)",
            "sharpness": "Sharpness (Laplacian Variance)",
            "image_area": "Image Area (pixels)",
            "freq_ratio": "High/Low Freq. Energy Ratio",
            "mean_r": "Mean Red Channel",
            "mean_g": "Mean Green Channel",
            "mean_b": "Mean Blue Channel"
        }

        color_map = {'real': 'g', 'fake': 'r'}
        # Map the combined 'method_source' to the original label's color
        method_source_to_label = df[['method_source', 'label']].drop_duplicates().set_index('method_source')['label']
        custom_palette = method_source_to_label.map(color_map).to_dict()
        order = sorted(df['method_source'].unique())

        for col, title in plot_metrics.items():
            print(f"  - Generating plot for {title}...")
            plt.figure(figsize=(24, 12))

            ax = sns.boxplot(data=df, x='method_source', y=col, order=order, palette=custom_palette)

            plt.title(f'Distribution of {title} by Method and Source', fontsize=20)
            plt.xlabel("Method (Source)", fontsize=14)
            plt.ylabel(title, fontsize=14)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            legend_elements = [Patch(facecolor='g', label='Real'), Patch(facecolor='r', label='Fake')]
            ax.legend(handles=legend_elements, title="Label")

            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_PLOT_DIR, f"{col}_distribution.png")
            plt.savefig(plot_path)
            plt.close()

        print(f"\n✅ All plots saved in '{OUTPUT_PLOT_DIR}'.")

    except ImportError:
        print("\n[Warning] `matplotlib`, `seaborn`, or `scipy` not installed. Skipping plot generation.")
    except Exception as e:
        print(f"\n[Plotting Error] An unexpected error occurred: {e}")


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
        '--mode', type=str, default='all', choices=['all', 'sample', 'plot'],
        help="Script mode: 'sample' to fetch data, 'plot' to generate plots, 'all' to do both."
    )
    parser.add_argument(
        '--dataset', type=str, default='both', choices=['train', 'test', 'both'],
        help="Which dataset(s) to analyze."
    )
    parser.add_argument(
        '--test_bucket_suffix', type=str, choices=['yolo', 'yolo-haar'],
        help="Suffix for the test bucket name (e.g., 'yolo'). Required if '--dataset' includes 'test'."
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

    if args.mode in ['all', 'sample']:
        run_sampling(datasets_to_run, args.test_bucket_suffix)

    if args.mode in ['all', 'plot']:
        run_plotting(OUTPUT_CSV)


if __name__ == "__main__":
    main()
