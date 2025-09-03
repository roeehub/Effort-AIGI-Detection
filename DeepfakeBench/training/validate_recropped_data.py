"""
validate_recropped_data.py
================================================
The definitive script to validate the final, re-cropped dataset.

Purpose:
To quantitatively verify that the re-cropping process was successful by:
1. Confirming there is no longer significant variance in Relative Face Area (RFA)
   *within* the training set sources.
2. Confirming there is no longer a systematic "domain shift" in RFA *between*
   the training set and the OOD test set.

This script merges the GCS sampling logic with a thread-safe model loader for
stable, high-performance analysis.
"""
import os
import random
import argparse
import threading
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

try:
    from video_preprocessor import initialize_yolo_model, _get_yolo_face_box
except ImportError:
    print("\n[ERROR] Could not import from 'video_preprocessor.py'. Please ensure it's in the same directory.")
    exit(1)

# --- Configuration ---
CONFIG_PATH = './config/dataloader_config.yml'
OUTPUT_CSV = "final_validation_results.csv"
OUTPUT_PLOT_DIR = "final_validation_plots"
DEFAULT_WORKERS = os.cpu_count() or 4
SAMPLE_VIDEOS_PER_METHOD_TRAIN = 30
SAMPLE_FRAMES_PER_VIDEO_TRAIN = 3
SAMPLE_FRAMES_PER_VIDEO_TEST = 3

# --- THREAD-SAFE YOLO INITIALIZER ---
thread_local = threading.local()


def get_yolo_for_thread():
    """Ensures each thread gets its own instance of the YOLO model."""
    if not hasattr(thread_local, "yolo_model"):
        thread_local.yolo_model = initialize_yolo_model()
    return thread_local.yolo_model


# --- Core Analysis Logic ---
def get_face_crop_stats(gcs_path: str, fs: fsspec.AbstractFileSystem) -> dict | None:
    """Worker function to analyze a single image from GCS."""
    yolo_model = get_yolo_for_thread()
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), np.uint8)
        image_np_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_np_bgr is None: return None
        image_area = image_np_bgr.shape[0] * image_np_bgr.shape[1]
        bbox = _get_yolo_face_box(image_np_bgr, yolo_model)
        if bbox is None: return {"relative_face_area": 0.0}
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return {"relative_face_area": face_area / image_area if image_area > 0 else 0.0}
    except Exception as e:
        tqdm.write(f"\n[Error] Processing {gcs_path}: {e}")
        return None


def _process_in_parallel(paths_with_metadata, fs, num_workers):
    stats_list = []
    processing_func = partial(get_face_crop_stats, fs=fs)
    paths = [item[0] for item in paths_with_metadata]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(processing_func, paths), total=len(paths), desc="Analyzing Face Crops"))
    for i, stats in enumerate(results):
        if stats:
            stats.update(paths_with_metadata[i][1])
            stats_list.append(stats)
    return stats_list


# --- GCS Data Sampling Logic ---
def _sample_train_data(train_bucket_name, fs, num_workers):
    print(f"\n--- Sampling from TRAIN bucket: {train_bucket_name} ---")
    storage_client = storage.Client()
    bucket = storage_client.bucket(train_bucket_name)
    with open(CONFIG_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    methods = data_config['all_methods']['use_real_sources'] + data_config['all_methods']['use_fake_methods']

    paths_to_process = []
    for method in tqdm(methods, desc="Discovering Train Videos"):
        label = "real" if method in data_config['all_methods']['use_real_sources'] else "fake"
        video_paths = set(
            '/'.join(Path(b.name).parts[:-1]) + '/' for b in bucket.list_blobs(prefix=f"{label}/{method}/"))
        if not video_paths: continue
        selected_videos = random.sample(list(video_paths), min(len(video_paths), SAMPLE_VIDEOS_PER_METHOD_TRAIN))
        for video_prefix in selected_videos:
            frame_blobs = [b for b in bucket.list_blobs(prefix=video_prefix) if not b.name.endswith('/')]
            if not frame_blobs: continue
            selected_frames = random.sample(frame_blobs, min(len(frame_blobs), SAMPLE_FRAMES_PER_VIDEO_TRAIN))
            for frame in selected_frames:
                metadata = {"label": label, "method": method, "source": "train"}
                paths_to_process.append((f"gs://{bucket.name}/{frame.name}", metadata))
    print(f"Discovered {len(paths_to_process)} train frames to analyze.")
    return _process_in_parallel(paths_to_process, fs, num_workers)


def _sample_test_data(test_bucket_name, fs, num_workers):
    print(f"\n--- Sampling from TEST bucket: {test_bucket_name} ---")
    storage_client = storage.Client()
    bucket = storage_client.bucket(test_bucket_name)
    methods_and_labels = []
    for label in ['real', 'fake']:
        for page in bucket.list_blobs(prefix=f'{label}/', delimiter='/').pages:
            for prefix in page.prefixes:
                methods_and_labels.append((prefix.strip('/').split('/')[-1], label))

    paths_to_process = []
    for method, label in tqdm(methods_and_labels, desc="Discovering Test Videos"):
        video_paths = set(
            '/'.join(Path(b.name).parts[:-1]) + '/' for b in bucket.list_blobs(prefix=f"{label}/{method}/"))
        for video_prefix in list(video_paths):
            frame_blobs = [b for b in bucket.list_blobs(prefix=video_prefix) if not b.name.endswith('/')]
            if not frame_blobs: continue
            selected_frames = random.sample(frame_blobs, min(len(frame_blobs), SAMPLE_FRAMES_PER_VIDEO_TEST))
            for frame in selected_frames:
                metadata = {"label": label, "method": method, "source": "test"}
                paths_to_process.append((f"gs://{bucket.name}/{frame.name}", metadata))
    print(f"Discovered {len(paths_to_process)} test frames to analyze.")
    return _process_in_parallel(paths_to_process, fs, num_workers)


# --- Plotting and Reporting (Unchanged) ---
def run_plotting(csv_path):
    # This function remains the same as your provided script
    print(f"--- Mode: Plotting from '{csv_path}' ---")
    if not os.path.exists(csv_path): print(f"[Error] CSV not found: '{csv_path}'"); return
    df = pd.read_csv(csv_path)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    try:
        import matplotlib.pyplot as plt;
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        if 'method' in df.columns and not df.empty:
            test_methods = sorted(df[df['source'] == 'test']['method'].unique())
            train_methods = sorted(list(set(df[df['source'] == 'train']['method'].unique()) - set(test_methods)))
            order = test_methods + train_methods
            plt.figure(figsize=(28, 14))
            ax = sns.boxplot(data=df, x='method', y='relative_face_area', hue='source', order=order,
                             palette={'train': 'skyblue', 'test': 'salmon'})
            if test_methods and train_methods:
                ax.axvline(len(test_methods) - 0.5, color='k', linestyle='--', linewidth=2, alpha=0.7)
            plt.title('Distribution of Relative Face Area by Method (Post-Processing)', fontsize=20)
            plt.xlabel("Method", fontsize=14);
            plt.ylabel("Relative Face Area", fontsize=14)
            plt.xticks(rotation=45, ha="right");
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_PLOT_DIR, "final_validation_rfa_distribution.png")
            plt.savefig(plot_path);
            plt.close()
            print(f"\n✅ Plot saved to '{plot_path}'.")
        else:
            print("[Warning] No data to plot.")
    except ImportError:
        print("\n[Warning] `matplotlib` or `seaborn` not installed.")
    except Exception as e:
        print(f"\n[Plotting Error] {e}")


def run_numerical_report(csv_path):
    # This function remains the same as your provided script
    print(f"--- Mode: Generating Numerical Report from '{csv_path}' ---")
    if not os.path.exists(csv_path): print(f"[Error] CSV not found: '{csv_path}'"); return
    df = pd.read_csv(csv_path)
    if df.empty: print("[Warning] CSV is empty."); return
    print("\n--- Final Quantile Breakdown for Relative Face Area ---")
    quantile_df = df.groupby(['source', 'method', 'label']).agg(
        p25=('relative_face_area', lambda x: x.quantile(0.25)),
        median=('relative_face_area', 'median'),
        p75=('relative_face_area', lambda x: x.quantile(0.75)),
        mean=('relative_face_area', 'mean'),
        std=('relative_face_area', 'std'),
        count=('relative_face_area', 'size')).sort_index()
    for col in ['p25', 'median', 'p75', 'mean', 'std']:
        quantile_df[col] = quantile_df[col].map('{:.3f}'.format)
    pd.set_option('display.max_rows', None);
    print(quantile_df);
    print("-" * 47)
    print("\n✅ Numerical report finished.")


def main():
    parser = argparse.ArgumentParser(description="Validate re-cropped GCS dataset consistency.")
    parser.add_argument('--mode', default='all', choices=['all', 'analyze', 'plot', 'report'])
    parser.add_argument('--dataset', default='both', choices=['train', 'test', 'both'])
    parser.add_argument('--train_bucket', required=True, help="Name of the re-cropped TRAIN GCS bucket.")
    parser.add_argument('--test_bucket', required=True, help="Name of the re-cropped TEST GCS bucket.")
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if os.path.exists(OUTPUT_CSV):
        print(f"Removing old results file: {OUTPUT_CSV}")
        os.remove(OUTPUT_CSV)

    if args.mode in ['all', 'analyze']:
        fs = fsspec.filesystem('gcs')
        all_stats = []
        if args.dataset in ['train', 'both']:
            all_stats.extend(_sample_train_data(args.train_bucket, fs, args.workers))
        if args.dataset in ['test', 'both']:
            all_stats.extend(_sample_test_data(args.test_bucket, fs, args.workers))

        if not all_stats: print("\nNo data was collected."); return
        pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Saved {len(all_stats)} frame stats to '{OUTPUT_CSV}'.")

    if not os.path.exists(OUTPUT_CSV):
        print(f"CSV '{OUTPUT_CSV}' not found. Run analysis first.");
        return

    if args.mode in ['all', 'plot']: run_plotting(OUTPUT_CSV)
    if args.mode in ['all', 'report']: run_numerical_report(OUTPUT_CSV)


if __name__ == "__main__":
    main()
