"""
process_poc_phase_1.py
======================
A parallelized pipeline to process local videos and upload cropped face frames to GCS.

This script processes videos from a local folder structure:
- real/
  - video1.mp4, video2.mp4, ...
- fake/
  - method1/
    - video1.mp4, ...
  - method2/
    - video1.mp4, ...

And uploads 32 uniformly sampled, face-cropped frames to GCS bucket "poc-phase-1-test":
- real/
  - video_name/
    - 0000.png, 0001.png, ..., 0031.png
- fake/
  - method_video_name/
    - 0000.png, 0001.png, ..., 0031.png
"""
import argparse
import logging
import sys
import tempfile
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import numpy as np
from google.cloud import storage
from tqdm import tqdm

try:
    import video_preprocessor as vp
except ImportError:
    print("ERROR: Could not import 'video_preprocessor.py'. Make sure it's in the same directory.")
    sys.exit(1)

# ──────────────────────────
# 📍 Global Configuration
# ──────────────────────────
NUM_CORES = 8
FRAMES_TO_EXTRACT = 32  # Exactly 32 frames per video
FAILURE_TOLERANCE = 16  # Allow up to 16 frames to fail detection (minimum 16 frames required)

# ──────────────────────────
# ⚙️ Logging & Initialization
# ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)

_model_cache = {}
gcs_client_worker = None


def get_yolo_model():
    """Cached YOLO model getter for worker processes."""
    if "yolo" not in _model_cache:
        _model_cache["yolo"] = vp.initialize_yolo_model()
    return _model_cache["yolo"]


def worker_initializer():
    """Initialize worker process with GCS client and YOLO model."""
    global gcs_client_worker
    gcs_client_worker = storage.Client()
    get_yolo_model()
    logging.info("Worker initialized with GCS client and YOLO model.")


# ──────────────────────────
# ✨ Core Processing Logic
# ──────────────────────────

def crop_and_resize_face(frame_bgr: np.ndarray, box: tuple) -> Optional[np.ndarray]:
    """
    Crops a face from a frame using bounding box coordinates, converts it to a
    square, and resizes it to MODEL_IMG_SIZE. This logic matches video_preprocessor.py.
    """
    x0, y0, x1, y1 = [int(coord) for coord in box]
    h, w = frame_bgr.shape[:2]

    width, height = x1 - x0, y1 - y0
    center_x, center_y = x0 + width / 2, y0 + height / 2
    side_length = max(width, height)
    
    sq_x0 = max(0, int(center_x - side_length / 2))
    sq_y0 = max(0, int(center_y - side_length / 2))
    sq_x1 = min(w, int(center_x + side_length / 2))
    sq_y1 = min(h, int(center_y + side_length / 2))

    cropped_face = frame_bgr[sq_y0:sq_y1, sq_x0:sq_x1]
    if cropped_face.size == 0:
        return None

    return cv2.resize(
        cropped_face,
        (vp.MODEL_IMG_SIZE, vp.MODEL_IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )


def sample_and_crop_faces_uniformly(video_path: Path, num_frames: int = FRAMES_TO_EXTRACT) -> Optional[List[np.ndarray]]:
    """
    Samples exactly num_frames uniformly from a video and crops faces using YOLO.
    Uses robust sequential reading to avoid frame seeking issues.
    Will accept videos with at least (num_frames - FAILURE_TOLERANCE) valid face frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_required_frames = num_frames - FAILURE_TOLERANCE
    
    # Adjust sampling strategy based on video length
    if total_frames < min_required_frames:
        logging.warning(f"Video {video_path.name} has only {total_frames} frames, need at least {min_required_frames}")
        cap.release()
        return None
    
    # Sample fewer frames if video is shorter than target
    frames_to_sample = min(total_frames, num_frames)
    
    # Generate frame indices to sample uniformly
    frame_indices_to_sample = set(np.linspace(0, total_frames - 1, frames_to_sample, dtype=int))
    
    frames_to_process = []
    current_frame_num = 0

    # Sequentially read the video to grab target frames
    while len(frames_to_process) < len(frame_indices_to_sample):
        ret, frame = cap.read()
        if not ret:
            break
        
        if current_frame_num in frame_indices_to_sample:
            frames_to_process.append(frame)
        
        current_frame_num += 1

    cap.release()

    if len(frames_to_process) < min_required_frames:
        logging.warning(
            f"Could not read enough frames from {video_path}. Got {len(frames_to_process)}, needed at least {min_required_frames}."
        )
        return None

    # Batch process all frames with YOLO
    model = get_yolo_model()
    results = model.predict(frames_to_process, conf=vp.YOLO_CONF_THRESHOLD, verbose=False)

    successful_crops = []
    for frame_bgr, result in zip(frames_to_process, results):
        if result.boxes.shape[0] > 0:
            # Get the largest face
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(areas)]

            cropped_face = crop_and_resize_face(frame_bgr, tuple(best_box))
            if cropped_face is not None:
                successful_crops.append(cropped_face)

    # Check if we have enough valid faces after detection
    if len(successful_crops) >= min_required_frames:
        return successful_crops  # Return all successful crops (may be fewer than num_frames)
    else:
        logging.warning(
            f"Found only {len(successful_crops)} faces in {video_path} after sampling {len(frames_to_process)} frames. Required at least {min_required_frames}."
        )
        return None


# ──────────────────────────
# ☁️ GCS Worker
# ──────────────────────────

def process_task_worker(task: Dict, progress_data: dict, target_bucket_name: str):
    """
    Worker function that processes a single video task and uploads frames to GCS.
    """
    global gcs_client_worker
    target_bucket = gcs_client_worker.bucket(target_bucket_name)

    video_path = Path(task['video_path'])
    label = task['label']  # 'real' or 'fake'
    folder_name = task['folder_name']  # video name or method_video_name
    
    try:
        # Check if this video is already processed (check for last frame)
        gcs_prefix = f"{label}/{folder_name}/"
        last_frame_blob_name = f"{gcs_prefix}{FRAMES_TO_EXTRACT - 1:04d}.png"
        
        if target_bucket.blob(last_frame_blob_name).exists():
            logging.info(f"Skipping {video_path.name} - already processed")
            progress_data['skipped'] += 1
            return

        # Process the video
        cropped_faces = sample_and_crop_faces_uniformly(video_path, FRAMES_TO_EXTRACT)
        
        if not cropped_faces:
            logging.warning(f"Failed to extract faces from '{video_path.name}'.")
            progress_data['failed'] += 1
            return

        # Upload all frames to GCS
        for i, face_img in enumerate(cropped_faces):
            _, buffer = cv2.imencode(".png", face_img)
            blob_name = f"{gcs_prefix}{i:04d}.png"
            blob = target_bucket.blob(blob_name)
            blob.upload_from_string(buffer.tobytes(), content_type="image/png")

        logging.info(f"Successfully processed {video_path.name} -> gs://{target_bucket_name}/{gcs_prefix} ({len(cropped_faces)} frames)")
        progress_data['processed'] += 1

    except Exception as e:
        logging.error(f"FATAL EXCEPTION processing '{video_path.name}': {e}", exc_info=True)
        progress_data['failed'] += 1


# ──────────────────────────
# 📥 Task Generation
# ──────────────────────────

def generate_tasks_from_local(source_path: Path) -> List[Dict]:
    """
    Generates processing tasks from the local folder structure.
    
    Expected structure:
    source_path/
      real/
        video1.mp4
        video2.mp4
      fake/
        method1/
          video1.mp4
        method2/
          video1.mp4
    """
    tasks = []
    
    if not source_path.exists() or not source_path.is_dir():
        logging.error(f"Source path does not exist or is not a directory: {source_path}")
        sys.exit(1)
    
    # Process real videos
    real_dir = source_path / "real"
    if real_dir.exists() and real_dir.is_dir():
        logging.info(f"Scanning real videos in {real_dir}...")
        real_videos = sorted(list(real_dir.glob("*.mp4")) + list(real_dir.glob("*.MP4")) + 
                           list(real_dir.glob("*.mov")) + list(real_dir.glob("*.MOV")) +
                           list(real_dir.glob("*.avi")) + list(real_dir.glob("*.AVI")))
        
        for video_path in real_videos:
            folder_name = video_path.stem  # Just the video name without extension
            tasks.append({
                'video_path': str(video_path),
                'label': 'real',
                'folder_name': folder_name
            })
        logging.info(f"Found {len(real_videos)} real videos.")
    else:
        logging.warning(f"Real directory not found: {real_dir}")
    
    # Process fake videos
    fake_dir = source_path / "fake"
    if fake_dir.exists() and fake_dir.is_dir():
        logging.info(f"Scanning fake videos in {fake_dir}...")
        
        # Iterate through method subdirectories
        method_dirs = [d for d in fake_dir.iterdir() if d.is_dir()]
        
        for method_dir in sorted(method_dirs):
            method_name = method_dir.name
            fake_videos = sorted(list(method_dir.glob("*.mp4")) + list(method_dir.glob("*.MP4")) +
                               list(method_dir.glob("*.mov")) + list(method_dir.glob("*.MOV")) +
                               list(method_dir.glob("*.avi")) + list(method_dir.glob("*.AVI")))
            
            for video_path in fake_videos:
                video_name = video_path.stem
                # Prepend method name to video folder name
                folder_name = f"{method_name}_{video_name}"
                tasks.append({
                    'video_path': str(video_path),
                    'label': 'fake',
                    'folder_name': folder_name
                })
            
            logging.info(f"Found {len(fake_videos)} videos in method '{method_name}'.")
    else:
        logging.warning(f"Fake directory not found: {fake_dir}")
    
    return tasks


# ──────────────────────────
# 🚀 Main Execution
# ──────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Process local videos and upload 32 cropped face frames to GCS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--source-path',
        type=str,
        required=True,
        help='Path to the root directory containing real/ and fake/ subdirectories.'
    )
    parser.add_argument(
        '--target-bucket',
        type=str,
        default='poc-phase-1-test',
        help='Destination GCS bucket name (default: poc-phase-1-test).'
    )
    parser.add_argument(
        '--cores',
        type=int,
        default=NUM_CORES,
        help=f'Number of CPU cores to use (default: {NUM_CORES}).'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='List tasks without processing or uploading.'
    )
    
    args = parser.parse_args()
    
    source_path = Path(args.source_path)
    
    # Generate tasks
    logging.info("=" * 60)
    logging.info("Starting POC Phase 1 Video Processing Pipeline")
    logging.info("=" * 60)
    logging.info(f"Source Path: {source_path}")
    logging.info(f"Target Bucket: gs://{args.target_bucket}/")
    logging.info(f"Frames per video: {FRAMES_TO_EXTRACT}")
    logging.info("=" * 60)
    
    tasks = generate_tasks_from_local(source_path)
    
    if not tasks:
        logging.error("No videos found to process. Check your source path structure.")
        sys.exit(1)
    
    logging.info(f"\nGenerated {len(tasks)} tasks for processing.")
    
    if args.dry_run:
        logging.info("\n--- DRY RUN MODE: Tasks Preview ---")
        for i, task in enumerate(tasks[:10], 1):  # Show first 10
            logging.info(f"{i}. {task['label']}/{task['folder_name']} <- {Path(task['video_path']).name}")
        if len(tasks) > 10:
            logging.info(f"... and {len(tasks) - 10} more tasks.")
        logging.info("\nDry run complete. No processing performed.")
        return
    
    # Process videos in parallel
    logging.info(f"\nInitializing {args.cores} worker processes (this may take a moment)...")
    
    with Manager() as manager:
        progress_data = manager.dict({'processed': 0, 'skipped': 0, 'failed': 0})
        
        worker_func = partial(
            process_task_worker,
            progress_data=progress_data,
            target_bucket_name=args.target_bucket
        )
        
        with Pool(processes=args.cores, initializer=worker_initializer) as pool:
            list(tqdm(
                pool.imap_unordered(worker_func, tasks),
                total=len(tasks),
                desc="Processing Videos",
                unit="video"
            ))
        
        # Final summary
        logging.info("\n" + "=" * 60)
        logging.info("           PROCESSING COMPLETE")
        logging.info("=" * 60)
        logging.info(f"Successfully processed: {progress_data['processed']}")
        logging.info(f"Skipped (already exist): {progress_data['skipped']}")
        logging.info(f"Failed (could not process): {progress_data['failed']}")
        logging.info("=" * 60)
        
        total = progress_data['processed'] + progress_data['skipped'] + progress_data['failed']
        if total > 0:
            success_rate = (progress_data['processed'] + progress_data['skipped']) / total * 100
            logging.info(f"Success rate: {success_rate:.1f}%")
        
        logging.info(f"\nResults uploaded to: gs://{args.target_bucket}/")
        logging.info("  Structure:")
        logging.info("    real/")
        logging.info("      <video_name>/")
        logging.info("        0000.png ... 0031.png")
        logging.info("    fake/")
        logging.info("      <method>_<video_name>/")
        logging.info("        0000.png ... 0031.png")


if __name__ == "__main__":
    main()
