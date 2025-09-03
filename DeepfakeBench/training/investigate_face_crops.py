"""
investigate_face_crops.py (v4 - Production Ready, Thread-Safe)
================================================
A diagnostic script to quantitatively analyze face crop sizes.

New in this version:
- Fix: Solves multi-threading errors (e.g., 'Conv' object has no attribute 'bn')
  by implementing a thread-safe YOLO model initializer. Each worker thread
  now gets its own independent model instance, preventing state corruption.
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

# --- Config ---
CONFIG_PATH = './config/dataloader_config.yml'
OUTPUT_CSV = "face_crop_analysis_results.csv"
OUTPUT_PLOT_DIR = "face_crop_analysis_plots"
DEFAULT_WORKERS = os.cpu_count() or 4
IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}

# --- THREAD-SAFE YOLO INITIALIZER (THE FIX) ---
thread_local = threading.local()


def get_yolo_for_thread():
    """Ensures each thread gets its own instance of the YOLO model."""
    if not hasattr(thread_local, "yolo_model"):
        # This function is called once per thread.
        thread_local.yolo_model = initialize_yolo_model()
    return thread_local.yolo_model


# --- Core Logic (Adapted for Thread Safety) ---
def get_face_crop_stats(path: str, fs: fsspec.AbstractFileSystem) -> dict | None:
    """Worker function to analyze a single image."""
    yolo_model = get_yolo_for_thread()  # Each thread gets its model here
    try:
        with fs.open(path, 'rb') as f:
            image_bytes = np.frombuffer(f.read(), np.uint8)
        image_np_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image_np_bgr is None: return None
        image_area = image_np_bgr.shape[0] * image_np_bgr.shape[1]
        bbox = _get_yolo_face_box(image_np_bgr, yolo_model)
        if bbox is None: return {"relative_face_area": 0.0}
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return {"relative_face_area": face_area / image_area if image_area > 0 else 0.0}
    except Exception as e:
        tqdm.write(f"\n[Error] Processing {path}: {e}");
        return None


def _process_in_parallel(paths_with_metadata, fs, num_workers):
    stats_list = []
    # Note: We no longer pass the yolo_model from the main thread
    processing_func = partial(get_face_crop_stats, fs=fs)
    paths = [item[0] for item in paths_with_metadata]

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(processing_func, paths), total=len(paths), desc="Analyzing Face Crops"))

    for i, stats in enumerate(results):
        if stats:
            stats.update(paths_with_metadata[i][1]);
            stats_list.append(stats)
    return stats_list


# --- Local Directory Analysis Logic (Adapted for Thread Safety) ---
def analyze_local_directory(args):
    print(f"--- Mode: Analyzing Local Directory: {args.local_dir} ---")
    local_path = Path(args.local_dir)
    if not local_path.is_dir():
        print(f"[ERROR] Local directory not found: {args.local_dir}");
        return

    print("Discovering image files...")
    all_files = [p for p in local_path.rglob('*') if p.suffix.lower() in IMG_EXTENSIONS]

    paths_to_process = []
    for f in all_files:
        try:
            relative_parts = f.relative_to(local_path).parts
            label = relative_parts[0]
            method = relative_parts[1]
            metadata = {"label": label, "method": method, "source": "local_sample"}
            paths_to_process.append((str(f), metadata))
        except IndexError:
            tqdm.write(f"\n[Warning] Could not parse metadata from path: {f}. Skipping.")

    print(f"Found {len(paths_to_process)} frames to analyze.")
    if not paths_to_process: return

    fs = fsspec.filesystem('file')
    # Note: We no longer initialize the model here in the main thread
    return _process_in_parallel(paths_to_process, fs, args.workers)


# --- Plotting and Reporting (Unchanged) ---
def run_plotting(csv_path):
    print(f"--- Mode: Plotting from '{csv_path}' ---")
    if not os.path.exists(csv_path): print(f"[Error] CSV not found: '{csv_path}'"); return
    df = pd.read_csv(csv_path)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    try:
        import matplotlib.pyplot as plt;
        import seaborn as sns
        sns.set_theme(style="whitegrid")
        # Ensure 'method' column exists and handle potential empty DataFrame
        if 'method' in df.columns and not df.empty:
            order = sorted(df['method'].unique())
            plt.figure(figsize=(28, 14))
            sns.boxplot(data=df, x='method', y='relative_face_area', hue='label', order=order,
                        palette={'real': 'g', 'fake': 'r'})
            plt.title('Distribution of Relative Face Area by Method', fontsize=20)
            plt.xlabel("Method", fontsize=14);
            plt.ylabel("Relative Face Area", fontsize=14)
            plt.xticks(rotation=45, ha="right");
            plt.tight_layout()
            plot_path = os.path.join(OUTPUT_PLOT_DIR, "relative_face_area_distribution.png")
            plt.savefig(plot_path);
            plt.close()
            print(f"\n✅ Plot saved to '{plot_path}'.")
        else:
            print("[Warning] No data to plot. 'method' column might be missing or DataFrame is empty.")
    except ImportError:
        print("\n[Warning] `matplotlib` or `seaborn` not installed.")
    except Exception as e:
        print(f"\n[Plotting Error] {e}")


def run_numerical_report(csv_path):
    print(f"--- Mode: Generating Numerical Report from '{csv_path}' ---")
    if not os.path.exists(csv_path): print(f"[Error] CSV not found: '{csv_path}'"); return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("[Warning] CSV file is empty. No report to generate.")
        return
    print("\n--- Quantile Breakdown for Relative Face Area ---")
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
    parser = argparse.ArgumentParser(description="Investigate face crop consistency.")
    parser.add_argument('--mode', default='all', choices=['all', 'analyze', 'plot', 'report'])
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--local_dir', type=str, help="Analyze a local directory of images.")
    source_group.add_argument('--csv_file', type=str, help="Run plotting/reporting on an existing CSV file.")

    args = parser.parse_args()

    # Determine the CSV path to use
    csv_path = args.csv_file if args.csv_file else OUTPUT_CSV

    if args.mode in ['all', 'analyze']:
        if args.local_dir:
            all_stats = analyze_local_directory(args)
            if all_stats:
                if os.path.exists(OUTPUT_CSV): os.remove(OUTPUT_CSV)
                pd.DataFrame(all_stats).to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Saved {len(all_stats)} frame stats to '{OUTPUT_CSV}'.")
            else:
                print("\nNo data was collected.");
                return
        elif not args.csv_file:
            print("[Error] Must provide --local_dir for analysis mode.")
            return

    if not os.path.exists(csv_path) and args.mode in ['plot', 'report', 'all']:
        print(f"[Error] CSV file '{csv_path}' not found. Run analysis first.")
        return

    if args.mode in ['all', 'plot']: run_plotting(csv_path)
    if args.mode in ['all', 'report']: run_numerical_report(csv_path)


if __name__ == "__main__":
    main()
