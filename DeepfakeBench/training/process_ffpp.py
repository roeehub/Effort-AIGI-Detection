import argparse
import logging
import os
import sys
import tempfile
import time
from functools import partial
from multiprocessing import Pool, Manager
from pathlib import Path
from typing import List, Optional, Dict, Tuple

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
# --- Processing Parameters ---
NUM_CORES = 8
FRAMES_TO_SAMPLE = 35
FRAMES_TO_KEEP = 32
FAILURE_TOLERANCE = FRAMES_TO_SAMPLE - FRAMES_TO_KEEP

# ──────────────────────────
# ⚙️ Logging & Initialization
# ──────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(processName)s] [%(levelname)s] - %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
_model_cache = {}
gcs_client_worker = None


def get_yolo_model():
    if "yolo" not in _model_cache:
        _model_cache["yolo"] = vp.initialize_yolo_model()
    return _model_cache["yolo"]


def worker_initializer():
    global gcs_client_worker
    gcs_client_worker = storage.Client()
    get_yolo_model()


# ──────────────────────────
# ✨ Core Processing Logic
# ──────────────────────────

def crop_and_resize_face(frame_bgr: np.ndarray, box: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """
    Crops a face from a frame using bounding box coordinates, converts it to a
    square, and resizes it. This logic is adapted from video_preprocessor.py.
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

    # Use the image size from the imported vp module
    return cv2.resize(cropped_face, (vp.MODEL_IMG_SIZE, vp.MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)


def sample_and_crop_faces_batched(video_path: Path) -> Optional[List[np.ndarray]]:
    """
    More robustly samples frames from a video, detects faces in a batch,
    and returns a list of cropped faces. The function succeeds if at least
    FRAMES_TO_KEEP faces are found out of the FRAMES_TO_SAMPLE attempts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < FRAMES_TO_SAMPLE:
        cap.release()
        return None

    frame_indices_to_sample = np.linspace(0, total_frames - 1, FRAMES_TO_SAMPLE, dtype=int)
    frames_for_yolo = []
    original_bgr_frames_read = []

    for frame_idx in frame_indices_to_sample:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Keep both the original BGR for cropping and the RGB version for YOLO
            original_bgr_frames_read.append(frame)
            frames_for_yolo.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames_for_yolo) < FRAMES_TO_KEEP:
        logging.warning(
            f"Could not read enough frames from {video_path}. Needed {FRAMES_TO_KEEP}, got {len(frames_for_yolo)}.")
        return None

    model = get_yolo_model()
    # Process all collected frames in a single batch for efficiency
    results = model.predict(frames_for_yolo, conf=vp.YOLO_CONF_THRESHOLD, verbose=False)

    successful_crops = []
    # Loop through all results to gather as many valid faces as possible
    for frame_bgr, result in zip(original_bgr_frames_read, results):
        if result.boxes.shape[0] > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(areas)]

            cropped_face = crop_and_resize_face(frame_bgr, tuple(best_box))
            if cropped_face is not None:
                successful_crops.append(cropped_face)

    # Only after trying all frames, check if we have enough
    if len(successful_crops) >= FRAMES_TO_KEEP:
        return successful_crops[:FRAMES_TO_KEEP]
    else:
        logging.warning(
            f"Found only {len(successful_crops)} faces in {video_path} after sampling {len(frames_for_yolo)} frames. Required {FRAMES_TO_KEEP}.")
        return None


# ──────────────────────────
# ☁️ Unified GCS Worker (Handles Local & GCS sources)
# ──────────────────────────
def process_task_worker(task: Dict, progress_data: dict, target_bucket_name: str, gcs_root_prefix: str):
    """
    A generic worker that can process a task from either a local file or GCS.
    """
    global gcs_client_worker
    target_bucket = gcs_client_worker.bucket(target_bucket_name)

    video_id, method = task['gcs_path_parts']
    source_ref = task['source_ref']
    source_name_for_logs = source_ref.get('local_path') or source_ref.get('gcs_blob')

    try:
        gcs_prefix = f"{gcs_root_prefix}/{method}_{gcs_root_prefix}/{video_id}/"
        last_frame_blob_name = f"{gcs_prefix}{FRAMES_TO_KEEP - 1:04d}.png"
        if target_bucket.blob(last_frame_blob_name).exists():
            progress_data['skipped'] += 1
            return

        # --- Generic Source Handling ---
        if 'local_path' in source_ref:
            # Process a local file directly
            local_video_path = Path(source_ref['local_path'])
            cropped_faces = sample_and_crop_faces_batched(local_video_path)
        elif 'gcs_bucket' in source_ref:
            # Download GCS file to a temporary location for processing
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp_f:
                source_bucket = gcs_client_worker.bucket(source_ref['gcs_bucket'])
                source_blob = source_bucket.blob(source_ref['gcs_blob'])
                source_blob.download_to_filename(tmp_f.name)
                local_video_path = Path(tmp_f.name)
                cropped_faces = sample_and_crop_faces_batched(local_video_path)
        else:
            raise ValueError(f"Invalid task source_ref: {source_ref}")

        if not cropped_faces:
            logging.warning(f"Failed to extract sufficient faces from '{source_name_for_logs}'.")
            progress_data['failed'] += 1
            return

        # Upload results to GCS
        for i, face_img in enumerate(cropped_faces):
            _, buffer = cv2.imencode(".png", face_img)
            blob_name = f"{gcs_prefix}{i:04d}.png"
            blob = target_bucket.blob(blob_name)
            blob.upload_from_string(buffer.tobytes(), content_type="image/png")

        progress_data['processed'] += 1

    except Exception as e:
        logging.error(f"FATAL EXCEPTION on '{source_name_for_logs}': {e}", exc_info=True)
        progress_data['failed'] += 1


# ──────────────────────────
# 📥 Task Generation Profiles
# ──────────────────────────

def generate_tasks_gcs_default(args: argparse.Namespace) -> List[Dict]:
    """Task generation for the original GCS processing mode."""
    logging.info(f"Using 'gcs-default' profile. Method name is inferred from source bucket.")
    tasks = []
    client = storage.Client()
    blobs = client.list_blobs(args.gcs_bucket, prefix=args.gcs_prefix)
    method_name = args.gcs_bucket
    video_blobs = [b for b in blobs if b.name.lower().endswith((".mp4", ".mov", ".avi")) and not b.name.endswith('/')]

    for blob in video_blobs:
        video_id = Path(blob.name).parent.name
        tasks.append({
            "source_ref": {"gcs_bucket": blob.bucket.name, "gcs_blob": blob.name},
            "gcs_path_parts": (video_id, method_name),
        })
    return tasks


def generate_tasks_local_vcd(args: argparse.Namespace) -> List[Dict]:
    """Task generation for the local VCD dataset."""
    logging.info(f"Using 'local-vcd' profile. video_id is inferred from filename.")
    tasks = []
    source_path = Path(args.source_path)
    if not source_path.is_dir():
        logging.error(f"Provided source path is not a directory: {source_path}")
        sys.exit(1)

    video_paths = [p for p in source_path.rglob('*') if p.suffix.lower() in ['.mp4', '.mov', '.avi']]
    for path in video_paths:
        video_id = path.stem  # e.g., "0380a333cb0fef69001fb4260e2a705f_1920x1080_30"
        tasks.append({
            "source_ref": {"local_path": str(path)},
            "gcs_path_parts": (video_id, args.method),
        })
    return tasks


def generate_tasks_gcs_veo3(args: argparse.Namespace) -> List[Dict]:
    """Task generation for the specific GCS veo3-demo-set2 structure."""
    logging.info(f"Using 'gcs-veo3' profile. video_id is '<middle_folder>_<video_name>'.")
    tasks = []
    client = storage.Client()
    blobs = client.list_blobs(args.gcs_bucket, prefix=args.gcs_prefix)
    video_blobs = [b for b in blobs if b.name.lower().endswith((".mp4", ".mov", ".avi")) and not b.name.endswith('/')]

    for blob in video_blobs:
        p = Path(blob.name)
        if len(p.parts) < 3:  # Ensure there are enough parent directories
            logging.warning(f"Skipping blob with unexpected path structure: {blob.name}")
            continue

        # GCS path: .../<middle folder>/<video name>/<a single video>
        video_name_part = p.parent.name
        middle_folder_part = p.parent.parent.name
        video_id = f"{middle_folder_part}_{video_name_part}"

        tasks.append({
            "source_ref": {"gcs_bucket": blob.bucket.name, "gcs_blob": blob.name},
            "gcs_path_parts": (video_id, args.method),
        })
    return tasks


def generate_tasks_gcs_custom(args: argparse.Namespace) -> List[Dict]:
    """
    Task generation for a custom GCS structure like:
    <gcs_prefix>/<method>/<video_id>.mp4
    """
    logging.info(f"Using 'gcs-custom' profile. Structure: <prefix>/<method>/<video_id>.mp4")
    tasks = []
    client = storage.Client()

    # We list blobs with the specified prefix
    blobs = client.list_blobs(args.gcs_bucket, prefix=args.gcs_prefix)
    video_blobs = [b for b in blobs if b.name.lower().endswith((".mp4", ".mov", ".avi")) and not b.name.endswith('/')]

    if not video_blobs:
        logging.warning(f"No video files found in gs://{args.gcs_bucket}/{args.gcs_prefix}")
        return []

    for blob in video_blobs:
        p = Path(blob.name)

        # In this structure, the video filename is the ID
        video_id = p.stem

        # The parent directory is the method
        method_name = p.parent.name

        tasks.append({
            "source_ref": {"gcs_bucket": blob.bucket.name, "gcs_blob": blob.name},
            "gcs_path_parts": (video_id, method_name),  # Note: the script uses (video_id, method)
        })

    return tasks


# ──────────────────────────
# 🚀 Main Execution
# ──────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A flexible pipeline to process videos from various sources and upload cropped faces to GCS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--cores', type=int, default=NUM_CORES, help='Number of CPU cores to use.')
    subparsers = parser.add_subparsers(dest='profile', required=True, help='Select a processing profile.')

    # --- Profile 1: GCS Default (Original behavior) ---
    p_gcs_default = subparsers.add_parser('gcs-default',
                                          help='Process from GCS with default naming (method=bucket, id=parent_dir).')
    p_gcs_default.add_argument('--gcs-bucket', type=str, required=True, help='Source GCS bucket name.')
    p_gcs_default.add_argument('--gcs-prefix', type=str, default="",
                               help='Optional path prefix within the source GCS bucket.')
    p_gcs_default.add_argument('--label', type=str, required=True,
                               help='Top-level GCS prefix for data (e.g., "fake" or "real").')
    p_gcs_default.add_argument('--target-bucket', type=str, required=True,
                               help='Destination GCS bucket for processed faces.')
    p_gcs_default.set_defaults(func=generate_tasks_gcs_default)

    # --- Profile 2: Local VCD ---
    p_local_vcd = subparsers.add_parser('local-vcd', help='Process a local directory of VCD videos (id=filename).')
    p_local_vcd.add_argument('--source-path', type=str, required=True,
                             help='Path to the root directory of local videos.')
    p_local_vcd.add_argument('--label', type=str, required=True, help='Top-level GCS prefix (e.g., "real").')
    p_local_vcd.add_argument('--method', type=str, required=True, help='Method name for GCS path (e.g., "VCD").')
    p_local_vcd.add_argument('--target-bucket', type=str, required=True,
                             help='Destination GCS bucket for processed faces.')
    p_local_vcd.set_defaults(func=generate_tasks_local_vcd)

    # --- Profile 3: GCS VEO3 Demo-Set2 ---
    p_gcs_veo3 = subparsers.add_parser('gcs-veo3', help='Process GCS veo3-creations/demo-set2 with special naming.')
    p_gcs_veo3.add_argument('--gcs-bucket', type=str, required=True,
                            help='Source GCS bucket name (e.g., "veo3-creations").')
    p_gcs_veo3.add_argument('--gcs-prefix', type=str, default="demo-set2/", help='Path prefix within the bucket.')
    p_gcs_veo3.add_argument('--label', type=str, required=True, help='Top-level GCS prefix (e.g., "fake").')
    p_gcs_veo3.add_argument('--method', type=str, required=True,
                            help='Method name for GCS path (e.g., "veo3-demo-set2").')
    p_gcs_veo3.add_argument('--target-bucket', type=str, required=True, help='Destination GCS bucket name.')
    p_gcs_veo3.set_defaults(func=generate_tasks_gcs_veo3)

    p_gcs_custom = subparsers.add_parser('gcs-custom',
                                         help='Process a custom GCS structure (<prefix>/<method>/<video_id>.mp4).')
    p_gcs_custom.add_argument('--gcs-bucket', type=str, required=True, help='Source GCS bucket name.')
    p_gcs_custom.add_argument('--gcs-prefix', type=str, required=True,
                              help='The prefix path that contains the method folders (e.g., "Deep fake test 10.08.25/fake/").')
    p_gcs_custom.add_argument('--label', type=str, required=True,
                              help='Top-level GCS prefix (e.g., "fake" or "real"). This should match the folder in your prefix.')
    p_gcs_custom.add_argument('--target-bucket', type=str, required=True,
                              help='Destination GCS bucket for processed faces.')
    p_gcs_custom.set_defaults(func=generate_tasks_gcs_custom)

    args = parser.parse_args()
    NUM_CORES = args.cores

    # Generate tasks based on the selected profile
    tasks = args.func(args)

    if not tasks:
        logging.error("No compatible video files found for the selected profile and paths.")
        sys.exit(1)

    logging.info(f"Found {len(tasks)} videos to process using profile '{args.profile}'.")
    logging.info(f"Target Structure: gs://{args.target_bucket}/{args.label}/<method>/<video_id>/")
    logging.info(f"Initializing {NUM_CORES} workers (this may take a moment)...")

    with Manager() as manager:
        progress_data = manager.dict({'processed': 0, 'skipped': 0, 'failed': 0})
        worker_func = partial(process_task_worker,
                              progress_data=progress_data,
                              target_bucket_name=args.target_bucket,
                              gcs_root_prefix=args.label)

        with Pool(processes=NUM_CORES, initializer=worker_initializer) as pool:
            list(tqdm(pool.imap_unordered(worker_func, tasks), total=len(tasks), desc="Processing Videos"))

        logging.info("=" * 50 + "\n            PROCESSING COMPLETE\n" + "=" * 50)
        logging.info(f"Successfully processed: {progress_data['processed']}")
        logging.info(f"Skipped (already exist): {progress_data['skipped']}")
        logging.info(f"Failed (could not process): {progress_data['failed']}")
        logging.info("=" * 50)
