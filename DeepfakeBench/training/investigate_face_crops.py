"""
investigate_face_crops.py (v2 with Prefix Analysis)
================================================
A diagnostic script to quantitatively analyze face crop sizes across different
data sources in the training and OOD test sets.

Purpose:
To verify the hypothesis that some data sources have systematically different
face crops. This version includes a '--gcs_prefix' flag to allow analysis of
specific subdirectories, such as sample outputs from a processing pipeline.
"""
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

try:
    from video_preprocessor import initialize_yolo_model, _get_yolo_face_box
except ImportError:
    print("\n[ERROR] Could not import from 'video_preprocessor.py'.")
    print("Please ensure the file is in the same directory or your PYTHONPATH is set correctly.")
    exit(1)

# --- Configuration ---
DF40_BUCKET_NAME = "df40-frames-recropped-rfa85"
CONFIG_PATH = './config/dataloader_config.yml'
SAMPLE_VIDEOS_PER_METHOD_TRAIN = 30
SAMPLE_FRAMES_PER_VIDEO_TRAIN = 3

TEST_BUCKET_PREFIX = "deep-fake-test-recropped-rfa85"
SAMPLE_FRAMES_PER_VIDEO_TEST = 3

OUTPUT_CSV = "face_crop_analysis_results.csv"
OUTPUT_PLOT_DIR = "face_crop_analysis_plots"
DEFAULT_WORKERS = os.cpu_count() or 4


# --- Core Logic (Unchanged) ---
def get_face_crop_stats(gcs_path: str, fs: fsspec.AbstractFileSystem, yolo_model) -> dict | None:
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), np.uint8)
        image_np_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_np_bgr is None:
            tqdm.write(f"\n[OpenCV Warning] cv2.imdecode failed for {gcs_path}.")
            return None
        img_height, img_width, _ = image_np_bgr.shape
        image_area = img_height * img_width
        bbox = _get_yolo_face_box(image_np_bgr, yolo_model)
        if bbox is None:
            return {"relative_face_area": 0.0}
        x0, y0, x1, y1 = bbox
        face_area = (x1 - x0) * (y1 - y0)
        relative_face_area = face_area / image_area if image_area > 0 else 0.0
        return {"relative_face_area": relative_face_area}
    except Exception as e:
        tqdm.write(f"\n[Processing Error] Could not process {gcs_path}: {e}")
        return None


def _process_frames_in_parallel(frame_paths_with_metadata, fs, yolo_model, num_workers):
    stats_list = []
    processing_function = partial(get_face_crop_stats, fs=fs, yolo_model=yolo_model)
    gcs_paths = [item[0] for item in frame_paths_with_metadata]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(executor.map(processing_function, gcs_paths), total=len(gcs_paths), desc="Analyzing Face Crops"))
    for i, stats in enumerate(results):
        if stats:
            metadata = frame_paths_with_metadata[i][1]
            stats.update(metadata)
            stats_list.append(stats)
    return stats_list


# --- Data Sampling Logic (MODIFIED FOR GCS PREFIX) ---
def _get_full_prefix(base_prefix, gcs_prefix_arg):
    if gcs_prefix_arg:
        return os.path.join(gcs_prefix_arg, base_prefix.lstrip('/'))
    return base_prefix


def _sample_train_data(bucket, fs, yolo_model, num_workers, gcs_prefix):
    print("\n--- Sampling from TRAIN dataset (DF40) ---")
    if gcs_prefix: print(f"Analyzing under prefix: '{gcs_prefix}'")
    with open(CONFIG_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    all_methods = data_config['all_methods']['use_real_sources'] + data_config['all_methods']['use_fake_methods']
    frame_paths_to_process = []
    for method in tqdm(all_methods, desc="Discovering Train Videos", unit="method"):
        label = "real" if method in data_config['all_methods']['use_real_sources'] else "fake"
        full_method_prefix = _get_full_prefix(f"{label}/{method}/", gcs_prefix)
        video_paths = set('/'.join(Path(b.name).parts[:-1]) + '/' for b in bucket.list_blobs(prefix=full_method_prefix))
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
    return _process_frames_in_parallel(frame_paths_to_process, fs, yolo_model, num_workers)


def _sample_test_data(bucket, fs, yolo_model, num_workers, gcs_prefix):
    print(f"\n--- Sampling from TEST dataset ({bucket.name}) ---")
    if gcs_prefix: print(f"Analyzing under prefix: '{gcs_prefix}'")
    methods_and_labels = []
    for label in ['real', 'fake']:
        full_label_prefix = _get_full_prefix(f'{label}/', gcs_prefix)
        iterator = bucket.list_blobs(prefix=full_label_prefix, delimiter='/')
        for page in iterator.pages:
            for prefix in page.prefixes:
                method_name = prefix.strip('/').split('/')[-1]
                methods_and_labels.append((method_name, label))
    frame_paths_to_process = []
    for method, label in tqdm(methods_and_labels, desc="Discovering Test Videos", unit="method"):
        base_prefix = f"{label}/{method}/"
        full_prefix = _get_full_prefix(base_prefix, gcs_prefix)
        video_paths = set('/'.join(Path(b.name).parts[:-1]) + '/' for b in bucket.list_blobs(prefix=full_prefix))
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
    return _process_frames_in_parallel(frame_paths_to_process, fs, yolo_model, num_workers)


# --- Main Controller Functions ---
def run_sampling(args):
    print("--- Mode: Sampling & Analyzing Data from GCS ---")
    print(f"Using {args.workers} parallel workers.")
    try:
        print("Initializing models and connecting to GCS...")
        yolo_model = initialize_yolo_model()
        storage_client = storage.Client()
        fs = fsspec.filesystem('gcs')
    except Exception as e:
        print(f"\n[Error] Could not connect to GCS or initialize models. Details: {e}")
        return
    all_stats = []
    if os.path.exists(OUTPUT_CSV):
        print(f"Found existing data file '{OUTPUT_CSV}'. Removing to start fresh.")
        os.remove(OUTPUT_CSV)

    datasets_to_run = ['train', 'test'] if args.dataset == 'both' else [args.dataset]
    if 'train' in datasets_to_run:
        train_bucket = storage_client.bucket(DF40_BUCKET_NAME)
        train_stats = _sample_train_data(train_bucket, fs, yolo_model, args.workers, args.gcs_prefix)
        if train_stats: all_stats.extend(train_stats)
    if 'test' in datasets_to_run:
        if not args.test_bucket_suffix:
            print("[Error] --test_bucket_suffix is required when sampling 'test' or 'both' datasets.")
            return
        test_bucket_name = f"{TEST_BUCKET_PREFIX}-{args.test_bucket_suffix}"
        test_bucket = storage_client.bucket(test_bucket_name)
        test_stats = _sample_test_data(test_bucket, fs, yolo_model, args.workers, args.gcs_prefix)
        if test_stats: all_stats.extend(test_stats)
    if not all_stats:
        print("\n[Error] No data was collected. Review logs for warnings.")
        return
    print("\nSaving final results...")
    pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved {len(all_stats)} frame stats to '{OUTPUT_CSV}'.")
    print(f"\n--- Analysis Complete ---")


# --- Plotting and Reporting Functions (Unchanged) ---
def run_plotting(csv_path):
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
        methods_in_test = df[df['source'] == 'test']['method'].unique()
        methods_in_train = df[df['source'] == 'train']['method'].unique()
        order = sorted(list(set(methods_in_test) | set(methods_in_train)))
        plt.figure(figsize=(28, 14))
        ax = sns.boxplot(data=df, x='method', y='relative_face_area', hue='label', order=order,
                         palette={'real': 'g', 'fake': 'r'})
        new_labels = [f"{tick.get_text()} ({'/'.join(df[df['method'] == tick.get_text()]['source'].unique())})" for tick
                      in ax.get_xticklabels()]
        ax.set_xticklabels(new_labels)
        plt.title('Distribution of Relative Face Area by Method', fontsize=20)
        plt.xlabel("Method (Source)", fontsize=14)
        plt.ylabel("Relative Face Area (Face BBox / Image Area)", fontsize=14)
        ax.tick_params(axis='x', rotation=45, labelsize=12)
        ax.legend(title='Label')
        if methods_in_test.any() and set(methods_in_train) - set(methods_in_test):
            separator_pos = len(sorted(list(methods_in_test))) - 0.5
            ax.axvline(separator_pos, color='k', linestyle='--', linewidth=2, alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_PLOT_DIR, "relative_face_area_distribution.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\n✅ Plot saved to '{plot_path}'.")
    except ImportError:
        print("\n[Warning] `matplotlib` or `seaborn` not installed. Skipping plot generation.")
    except Exception as e:
        print(f"\n[Plotting Error] An unexpected error occurred: {e}")


def run_numerical_report(csv_path):
    print(f"--- Mode: Generating Numerical Report from '{csv_path}' ---")
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: '{csv_path}'.")
        return
    df = pd.read_csv(csv_path)
    metric = 'relative_face_area'
    print(f"\n--- Quantile Breakdown for {metric} ---")
    quantile_df = df.groupby(['source', 'method', 'label']).agg(p25=(metric, lambda x: x.quantile(0.25)),
                                                                median=(metric, 'median'),
                                                                p75=(metric, lambda x: x.quantile(0.75)),
                                                                mean=(metric, 'mean'), std=(metric, 'std'),
                                                                count=(metric, 'size')).sort_index()
    for col in ['p25', 'median', 'p75', 'mean', 'std']:
        quantile_df[col] = quantile_df[col].map('{:.3f}'.format)
    pd.set_option('display.max_rows', None)
    print(quantile_df)
    print("-" * 47)
    print("\n✅ Numerical report finished.")


def main():
    parser = argparse.ArgumentParser(description="Investigate face crop consistency across dataset sources.")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'sample', 'plot', 'report'],
                        help="Script mode.")
    parser.add_argument('--dataset', type=str, default='both', choices=['train', 'test', 'both'],
                        help="Which dataset(s) to analyze.")
    parser.add_argument('--test_bucket_suffix', type=str, choices=['yolo', 'yolo-haar'],
                        help="Suffix for the test bucket name.")
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel workers. Default: {DEFAULT_WORKERS}")
    parser.add_argument('--gcs_prefix', type=str, default=None,
                        help="Optional GCS prefix to analyze a specific subdirectory.")
    args = parser.parse_args()

    if args.mode in ['all', 'sample']:
        run_sampling(args)
    csv_exists = os.path.exists(OUTPUT_CSV)
    if not csv_exists and args.mode in ['plot', 'report']:
        print(f"[Error] CSV file '{OUTPUT_CSV}' not found. Run sampling first.")
        return
    if args.mode in ['all', 'plot']:
        run_plotting(OUTPUT_CSV)
    if args.mode in ['all', 'report']:
        run_numerical_report(OUTPUT_CSV)


if __name__ == "__main__":
    main()
