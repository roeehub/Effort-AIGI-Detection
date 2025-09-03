"""
process_synthetic_videos_HQ.py
=================================
This script implements a three-stage, resumable, and parallelized pipeline to
download the HIGHEST QUALITY MP4s from the Rapidata/text-2-video-human-preferences-veo2
dataset and extract high-quality, high-confidence cropped faces.

**Key Feature:**
- Bypasses the dataset's default low-quality GIF URLs by programmatically
  constructing direct download links to the original MP4 files.

**Features:**
- Metadata Logging: Clear summaries after each stage.
- Multiprocessing: Stages 2 and 3 are parallelized for speed.
- Resumable: The pipeline can be stopped and restarted.
- Quality Control: Stage 3 uses high confidence and minimum size filters.
"""
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from multiprocessing import Pool, cpu_count
from urllib.request import urlretrieve

import cv2
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

try:
    from video_preprocessor import _get_yolo_face_box, extract_yolo_face, initialize_yolo_model
except ImportError:
    print("[ERROR] Could not import 'video_preprocessor.py'. Make sure it's in the same directory.")
    sys.exit(1)

# ──────────────────────────
# 📍 Configuration
# ──────────────────────────
CONFIG = {
    "NUM_WORKERS": max(1, cpu_count() - 1),
    "DATASET_NAME": "Rapidata/text-2-video-human-preferences-veo2",
    "STAGE_1_OUTPUT": "intermediate_step1_text_filtered.csv",
    "STAGE_2_OUTPUT": "intermediate_step2_face_verified.csv",
    "STAGE_3_LOG": "intermediate_step3_extraction_log.csv",
    "VIDEO_CACHE_DIR": "video_cache_mp4",  # Use a new cache for MP4s
    "FACES_OUTPUT_DIR": "output_faces",
    "FACE_KEYWORDS": [
        "face", "person", "man", "woman", "boy", "girl", "human", "people", "portrait",
        "selfie", "actor", "actress", "child", "baby", "crowd", "headshot",
        "close-up of a person", "smiling", "laughing", "talking"
    ],
    "SPARSE_CHECK_FRAMES": 10,
    "SPARSE_CONF_THRESHOLD": 0.15,
    "DENSE_CONF_THRESHOLD": 0.85,
    "MIN_FACE_PIXEL_SIZE": 96,
    "CROP_METHOD": "yolo"
}


# ──────────────────────────
# 🛠️ URL Transformation Helper
# ──────────────────────────

def construct_mp4_url(gif_url: str) -> str:
    """
    Transforms a dataset's GIF URL into a direct download URL for the original MP4.
    Example Input:  https://.../some_hash/0000_sora_1.gif
    Example Output: https://huggingface.co/datasets/Rapidata/text-2-video-human-preferences-veo2/resolve/main/videos/0000_sora_1.mp4
    """
    base_filename = Path(gif_url).name
    mp4_filename = Path(base_filename).with_suffix('.mp4').name

    # This is the direct download template for files in the repo
    return f"https://huggingface.co/datasets/{CONFIG['DATASET_NAME']}/resolve/main/videos/{mp4_filename}"


# ──────────────────────────
# 🚀 Pipeline Stage 1
# ──────────────────────────

def run_stage_1_text_filter():
    """Filters dataset, then transforms GIF URLs to high-quality MP4 URLs."""
    print("--- Starting Stage 1: Text-based Filtering & URL Transformation ---")

    keyword_pattern = "|".join(CONFIG["FACE_KEYWORDS"])

    print(f"[*] Loading dataset: {CONFIG['DATASET_NAME']}...")
    try:
        dataset = load_dataset(CONFIG["DATASET_NAME"], split="train", verification_mode="no_checks")
    except Exception as e:
        print(f"\n[ERROR] Failed to load the dataset. Details: {e}")
        sys.exit(1)

    df = dataset.to_pandas()
    initial_records = len(df)

    print(f"[*] Successfully loaded {initial_records} records.")
    print(f"[*] Filtering records with keyword pattern...")
    filtered_df = df[df["prompt"].str.contains(keyword_pattern, case=False, regex=True)].copy()

    def extract_model_name(url):
        try:
            return Path(url).name.split('_')[1]
        except IndexError:
            return "unknown"

    v1 = filtered_df[['prompt', 'video1']].rename(columns={'video1': 'video_url'})
    v1['model_name'] = v1['video_url'].apply(extract_model_name)
    v2 = filtered_df[['prompt', 'video2']].rename(columns={'video2': 'video_url'})
    v2['model_name'] = v2['video_url'].apply(extract_model_name)

    all_videos_df = pd.concat([v1, v2], ignore_index=True).drop_duplicates(subset=['video_url']).reset_index(drop=True)

    # --- IMPORTANT: URL TRANSFORMATION STEP ---
    print("[*] Constructing direct MP4 URLs to ensure highest quality...")
    all_videos_df['mp4_url'] = all_videos_df['video_url'].apply(construct_mp4_url)
    # ---

    unique_videos_count = len(all_videos_df)
    print(f"[*] Saved text-filtered candidates to '{CONFIG['STAGE_1_OUTPUT']}'")
    all_videos_df.to_csv(CONFIG['STAGE_1_OUTPUT'], index=False)

    print("\n--- Stage 1 Summary ---")
    print(f"[*] Started with:            {initial_records * 2} video files (approx)")
    print(f"[*] Remaining after filter:  {unique_videos_count} unique videos")
    print("[*] Status:                  High-quality MP4 URLs are ready for download.")
    print("-------------------------")

    return all_videos_df


# ──────────────────────────
# 🚀 Pipeline Stage 2 (Parallel)
# ──────────────────────────

def verify_video_worker(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker function for sparse face verification using MP4 URLs."""
    yolo_model = initialize_yolo_model()
    yolo_model.conf = CONFIG["SPARSE_CONF_THRESHOLD"]

    # --- Use the new mp4_url for downloading ---
    video_url = args["mp4_url"]

    local_path = Path(CONFIG["VIDEO_CACHE_DIR"]) / Path(video_url).name

    try:
        if not local_path.exists(): urlretrieve(video_url, local_path)
        cap = cv2.VideoCapture(str(local_path))
        if not cap.isOpened(): return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < CONFIG["SPARSE_CHECK_FRAMES"]:
            cap.release()
            return None
        frame_indices = random.sample(range(total_frames), CONFIG["SPARSE_CHECK_FRAMES"])
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret and _get_yolo_face_box(frame, model=yolo_model) is not None:
                cap.release()
                return args
        cap.release()
        return None
    except Exception:
        return None


# NOTE: run_stage_2 function remains the same as it just orchestrates the workers.
def run_stage_2_sparse_face_check(input_df: pd.DataFrame):
    """Orchestrates the parallel sparse face verification."""
    videos_to_verify_count = len(input_df)
    print(f"\n--- Starting Stage 2: Sparse Face Verification (using {CONFIG['NUM_WORKERS']} workers) ---")
    os.makedirs(CONFIG["VIDEO_CACHE_DIR"], exist_ok=True)

    tasks = input_df.to_dict('records')
    verified_videos = []

    with Pool(processes=CONFIG['NUM_WORKERS']) as pool:
        with tqdm(total=len(tasks), desc="Verifying videos") as pbar:
            for result in pool.imap_unordered(verify_video_worker, tasks):
                if result is not None: verified_videos.append(result)
                pbar.update()

    verified_df = pd.DataFrame(verified_videos)
    videos_passed_count = len(verified_df)
    print(f"\n[*] Saved verified video list to '{CONFIG['STAGE_2_OUTPUT']}'")
    verified_df.to_csv(CONFIG["STAGE_2_OUTPUT"], index=False)

    print("\n--- Stage 2 Summary ---")
    print(f"[*] Started with:          {videos_to_verify_count} video candidates")
    print(f"[*] Filtered out:          {videos_to_verify_count - videos_passed_count} (no face detected)")
    print(f"[*] Remaining for Stage 3: {videos_passed_count} videos")
    print("-------------------------")

    return verified_df


# ──────────────────────────
# 🚀 Pipeline Stage 3 (Parallel)
# ──────────────────────────

def extract_faces_worker(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Worker function for dense, high-quality face extraction from MP4s."""
    yolo_model = initialize_yolo_model()
    yolo_model.conf = CONFIG["DENSE_CONF_THRESHOLD"]

    # --- Use the new mp4_url ---
    video_url = args["mp4_url"]
    model_name = args["model_name"]

    video_filename = Path(video_url).name
    local_path = Path(CONFIG["VIDEO_CACHE_DIR"]) / video_filename
    video_face_dir = Path(CONFIG["FACES_OUTPUT_DIR"]) / Path(video_filename).stem
    os.makedirs(video_face_dir, exist_ok=True)

    if not local_path.exists(): return []

    extraction_log = []
    try:
        cap = cv2.VideoCapture(str(local_path))
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            box = _get_yolo_face_box(frame, model=yolo_model)
            if box is not None:
                face_w, face_h = box[2] - box[0], box[3] - box[1]
                if face_w >= CONFIG["MIN_FACE_PIXEL_SIZE"] and face_h >= CONFIG["MIN_FACE_PIXEL_SIZE"]:
                    cropped_face = extract_yolo_face(frame)
                    if cropped_face is not None:
                        save_path = video_face_dir / f"frame_{frame_count:04d}_face.jpg"
                        cv2.imwrite(str(save_path), cropped_face)
                        extraction_log.append({
                            "source_video": video_filename, "model_name": model_name,
                            "frame": frame_count, "face_path": str(save_path),
                            "original_face_width": int(face_w), "original_face_height": int(face_h)
                        })
        cap.release()
    except Exception:
        return []
    return extraction_log


# NOTE: run_stage_3 function remains the same as it just orchestrates the workers.
def run_stage_3_dense_face_extraction(input_df: pd.DataFrame):
    """Orchestrates the parallel dense face extraction."""
    videos_to_process_count = len(input_df)
    print(f"\n--- Starting Stage 3: Dense Face Extraction (using {CONFIG['NUM_WORKERS']} workers) ---")
    os.makedirs(CONFIG["FACES_OUTPUT_DIR"], exist_ok=True)

    tasks = input_df.to_dict('records')
    all_logs = []

    with Pool(processes=CONFIG['NUM_WORKERS']) as pool:
        with tqdm(total=len(tasks), desc="Extracting faces") as pbar:
            for result_list in pool.imap_unordered(extract_faces_worker, tasks):
                if result_list: all_logs.extend(result_list)
                pbar.update()

    log_df = pd.DataFrame(all_logs)
    total_faces_extracted = len(log_df)
    videos_with_faces = log_df['source_video'].nunique() if not log_df.empty else 0

    if not log_df.empty:
        print(f"\n[*] Saved extraction log to '{CONFIG['STAGE_3_LOG']}'")
        log_df.to_csv(CONFIG["STAGE_3_LOG"], index=False)

    print("\n--- Stage 3 Summary ---")
    print(f"[*] Started with:            {videos_to_process_count} verified videos")
    print(f"[*] Videos yielding faces:   {videos_with_faces}")
    print(
        f"[*] Total faces extracted:   {total_faces_extracted} (High-Confidence, >{CONFIG['MIN_FACE_PIXEL_SIZE']}px from MP4s)")
    print("-------------------------")


# ──────────────────────────
# 🚀 Main Orchestrator
# ──────────────────────────
# NOTE: The main function remains the same as it just calls the stages in order.
def main():
    """Main function to orchestrate the pipeline."""
    start_time = time.time()

    if Path(CONFIG["STAGE_3_LOG"]).exists():
        print("✅ Pipeline has already completed. Final log exists.")
        print(f"   -> Find cropped faces in '{CONFIG['FACES_OUTPUT_DIR']}'")
        return

    if Path(CONFIG["STAGE_2_OUTPUT"]).exists():
        print("✅ Stage 1 and 2 already complete. Loading results and starting Stage 3.")
        verified_videos_df = pd.read_csv(CONFIG["STAGE_2_OUTPUT"])
        run_stage_3_dense_face_extraction(verified_videos_df)

    elif Path(CONFIG["STAGE_1_OUTPUT"]).exists():
        print("✅ Stage 1 already complete. Loading results and starting Stage 2.")
        text_filtered_df = pd.read_csv(CONFIG["STAGE_1_OUTPUT"])
        verified_videos_df = run_stage_2_sparse_face_check(text_filtered_df)
        if not verified_videos_df.empty:
            run_stage_3_dense_face_extraction(verified_videos_df)

    else:
        print("🚀 Starting pipeline from scratch.")
        text_filtered_df = run_stage_1_text_filter()
        if not text_filtered_df.empty:
            verified_videos_df = run_stage_2_sparse_face_check(text_filtered_df)
            if not verified_videos_df.empty:
                run_stage_3_dense_face_extraction(verified_videos_df)

    end_time = time.time()
    print(f"\n🎉 Pipeline finished successfully in {end_time - start_time:.2f} seconds!")


if __name__ == "__main__":
    with torch.no_grad():
        main()
