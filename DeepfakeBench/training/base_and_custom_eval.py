# local_evaluator.py

import os
import sys
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import cv2  # noqa
import numpy as np  # noqa
import pandas as pd  # noqa
import torch  # noqa
import yaml  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, \
    confusion_matrix  # noqa
from tqdm import tqdm  # noqa

# Assuming app2.py and video_preprocessor.py are in the same directory or accessible in PYTHONPATH
import video_preprocessor
from detectors import DETECTOR, EffortDetector  # noqa (Needed for model loading)

# --- CONFIGURATION ---
GCS_BUCKET_NAME = "deep-fake-test-10-08-25"
GCS_DATA_PREFIX = "Deep fake test 10.08.25/"

# --- LOCAL DIRECTORIES ---
BASE_DIR = Path("./evaluation_run")
LOCAL_DATA_DIR = BASE_DIR / "evaluation_data"
CACHE_DIR = BASE_DIR / "processed_cache"
MISCLASSIFIED_DIR = BASE_DIR / "misclassified_videos"
LOG_FILE = BASE_DIR / "evaluation_log.txt"

# --- EVALUATION PARAMETERS ---
MODEL_TYPES_TO_TEST = ["base", "custom"]
PRE_METHODS_TO_TEST = ["yolo", "yolo_haar"]

# --- FRAME SAMPLING PARAMETERS ---
# We will sample 40 frames and require at least 16 to proceed.
INITIAL_SAMPLE_COUNT = 40
TARGET_FRAME_COUNT = 32
MIN_FRAME_COUNT = 16

# --- GLOBAL VARIABLES ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger("local_evaluator")


def setup_logging():
    """Configures logging to both file and console."""
    BASE_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def download_videos_from_gcs() -> List[Dict]:
    """
    Downloads all videos from the specified GCS path to a local directory,
    preserving the folder structure and extracting labels.
    """
    logger.info(f"--- Phase 1: Downloading videos from GCS ---")
    logger.info(f"Bucket: {GCS_BUCKET_NAME}, Prefix: {GCS_DATA_PREFIX}")

    video_metadata = []
    try:
        storage_client = storage.Client()
        blobs = list(storage_client.list_blobs(GCS_BUCKET_NAME, prefix=GCS_DATA_PREFIX))
        if not blobs:
            raise RuntimeError("No files found at the specified GCS path.")

        for blob in tqdm(blobs, desc="Downloading GCS videos"):
            if not blob.name.lower().endswith(('.mp4', '.mov', '.avi', '.webm')):
                continue

            # Robustly handle nested paths and determine label
            relative_path_str = os.path.relpath(blob.name, GCS_DATA_PREFIX)
            path_parts = Path(relative_path_str).parts

            if "fake" in path_parts:
                label = "fake"
            elif "real" in path_parts:
                label = "real"
            else:
                logger.warning(f"Could not determine label for {blob.name}. Skipping.")
                continue

            # Extract the source method (e.g., 'tiktok', 'FaceSwap', 'Celeb-real')
            # This is usually the folder right after 'fake' or 'real'
            try:
                label_index = path_parts.index(label)
                method_source = path_parts[label_index + 1]
            except (ValueError, IndexError):
                method_source = "unknown"

            local_path = LOCAL_DATA_DIR / relative_path_str
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if not local_path.exists():
                blob.download_to_filename(local_path)

            video_metadata.append({
                "local_path": str(local_path),
                "true_label": label,
                "method_source": method_source,
                "gcs_path": blob.name
            })

    except exceptions.Forbidden:
        logger.error("FATAL: GCS Permission Denied. Have you run 'gcloud auth application-default login'?")
        return []
    except Exception as e:
        logger.error(f"FATAL: An error occurred during GCS download: {e}")
        return []

    logger.info(f"Successfully downloaded and indexed {len(video_metadata)} videos.")
    return video_metadata


def load_all_models() -> Dict[str, torch.nn.Module]:
    """
    Loads all models ('base', 'custom') and pre-processors needed for the evaluation.
    This function is a simplified version of the startup logic in app2.py.
    """
    logger.info("--- Phase 2: Loading all models and pre-processors ---")

    # This is a simplified copy of the logic from app2.py's startup
    def load_detector(cfg: dict, weights: str) -> torch.nn.Module:
        model_cls = DETECTOR[cfg["model_name"]]
        model = model_cls(cfg).to(device)
        ckpt = torch.load(weights, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        return model

    loaded_models = {}
    try:
        # 1. Load configurations
        repo_base = Path(".")
        cfg_path = repo_base / "config/detector/effort.yaml"
        train_cfg_path = repo_base / "config/train_config.yaml"
        with open(cfg_path, "r") as f:
            config = yaml.safe_load(f)
        with open(train_cfg_path, "r") as f:
            config.update(yaml.safe_load(f))

        # 2. Load Base Model
        base_weights_path = repo_base / "weights/effort_clip_L14_trainOn_FaceForensic.pth"
        if not base_weights_path.exists(): raise FileNotFoundError(f"Base weights not found at {base_weights_path}")
        loaded_models['base'] = load_detector(config, str(base_weights_path))
        logger.info(f"✅ Loaded 'base' model successfully.")

        # 3. Load Custom Model (if available)
        custom_checkpoint_path = os.getenv("CHECKPOINT_GCS_PATH")  # Assuming it's already downloaded by app2 logic
        if custom_checkpoint_path:
            local_filename = Path(custom_checkpoint_path.split("gs://", 1)[1]).name
            custom_weights_path = repo_base / "weights" / "custom" / local_filename
            if custom_weights_path.exists():
                loaded_models['custom'] = load_detector(config, str(custom_weights_path))
                logger.info(f"✅ Loaded 'custom' model successfully from {custom_weights_path}.")
            else:
                logger.warning("Custom model specified but weights not found locally. Skipping.")
        else:
            logger.info("No custom model specified. Skipping.")

        # 4. Initialize Pre-processors
        video_preprocessor.initialize_yolo_model()
        video_preprocessor.initialize_haar_cascades()
        logger.info("✅ Initialized YOLO and Haar pre-processors.")

    except Exception as e:
        logger.exception(f"FATAL: Failed to load models. Error: {e}")
        return {}

    return loaded_models


def process_and_cache_video_faces(video_info: Dict, pre_method: str) -> Optional[str]:
    """
    Processes a single video using the flexible frame sampling logic and caches
    the extracted faces as a .npz file.
    """
    video_path = Path(video_info["local_path"])

    # Define cache path to avoid re-processing
    cache_path = CACHE_DIR / pre_method / video_info['true_label'] / f"{video_path.stem}.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return str(cache_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}. Skipping.")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, INITIAL_SAMPLE_COUNT, dtype=int)

    collected_faces = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if pre_method == 'yolo':
            face = video_preprocessor.extract_yolo_face(frame)
        elif pre_method == 'yolo_haar':
            face = video_preprocessor.extract_yolo_haar_face(frame)
        else:
            face = None  # Should not happen

        if face is not None:
            collected_faces.append(face)

    cap.release()
    num_faces = len(collected_faces)

    if num_faces < MIN_FRAME_COUNT:
        # logger.debug(f"Skipped {video_path.name}: Found only {num_faces} faces (min {MIN_FRAME_COUNT}).")
        return None

    # Take the target number of frames, or all if fewer than target but more than min
    final_faces = collected_faces[:TARGET_FRAME_COUNT]

    # Save to cache
    np.savez_compressed(cache_path, faces=np.array(final_faces), num_frames=len(final_faces))

    return str(cache_path)


def run_single_evaluation(model_type: str, pre_method: str, models: Dict, all_videos: List[Dict]) -> Tuple[
    pd.DataFrame, int]:
    """
    Runs a full evaluation for one combination of model and pre-method.
    """
    logger.info(f"\n{'=' * 80}\nRunning Evaluation: model='{model_type}', pre_method='{pre_method}'\n{'=' * 80}")

    model = models.get(model_type)
    if not model:
        logger.error(f"Model '{model_type}' not loaded. Skipping this run.")
        return pd.DataFrame(), len(all_videos)

    transform = video_preprocessor._get_transform()
    results = []
    skipped_count = 0

    # --- Pre-processing Step ---
    logger.info("Preprocessing videos and caching faces...")
    processed_videos = []
    for video_info in tqdm(all_videos, desc=f"Preprocessing for {pre_method}"):
        cache_file = process_and_cache_video_faces(video_info, pre_method)
        if cache_file:
            processed_videos.append((video_info, cache_file))
        else:
            skipped_count += 1

    # --- Inference Step ---
    logger.info("Running inference on processed videos...")
    for video_info, cache_file in tqdm(processed_videos, desc=f"Inference with {model_type}"):
        data = np.load(cache_file)
        faces = data['faces']
        num_frames = data['num_frames'].item()

        if num_frames < MIN_FRAME_COUNT: continue  # Should be already filtered, but as a safeguard

        tensor_frames = [transform(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)) for face in faces]
        video_tensor = torch.stack(tensor_frames, dim=0).unsqueeze(0).to(device)

        with torch.inference_mode():
            preds = model({'image': video_tensor}, inference=True)
            avg_prob = preds["prob"].mean().cpu().item()

        pred_label = "fake" if avg_prob >= 0.5 else "real"

        results.append({
            "local_path": video_info["local_path"],
            "true_label": video_info["true_label"],
            "method_source": video_info["method_source"],
            "pred_label": pred_label,
            "fake_prob": avg_prob,
            "num_frames": num_frames
        })

        # Handle misclassifications
        if pred_label != video_info["true_label"]:
            misclassified_path_dir = MISCLASSIFIED_DIR / f"{model_type}_{pre_method}" / f"PRED_{pred_label.upper()}_WAS_{video_info['true_label'].upper()}"
            misclassified_path_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(video_info["local_path"], misclassified_path_dir)

    return pd.DataFrame(results), skipped_count


def generate_and_print_report(results_df: pd.DataFrame, model_type: str, pre_method: str, total_attempted: int,
                              skipped_count: int):
    """Calculates and logs a detailed performance report."""

    logger.info(f"\n--- Evaluation Report: model='{model_type}', pre_method='{pre_method}' ---")

    # --- Processing Summary ---
    processed_count = len(results_df)
    processed_percent = (processed_count / total_attempted) * 100 if total_attempted > 0 else 0

    logger.info("--- Processing Summary ---")
    logger.info(f"Total Videos Attempted: {total_attempted}")
    logger.info(f"Successfully Processed : {processed_count} ({processed_percent:.1f}%)")
    logger.info(f"Skipped (Not Enough Faces): {skipped_count}")

    if results_df.empty:
        logger.warning("No videos were successfully processed. Cannot generate metrics.")
        return

    # --- Overall Performance Metrics ---
    logger.info("\n--- Overall Performance Metrics (Positive Class: FAKE) ---")
    y_true = results_df['true_label'].map({'real': 0, 'fake': 1})
    y_pred = results_df['pred_label'].map({'real': 0, 'fake': 1})
    y_prob = results_df['fake_prob']

    logger.info(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_true, y_pred):.4f}")
    logger.info(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    logger.info(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    try:
        auc = roc_auc_score(y_true, y_prob)
        logger.info(f"AUC Score: {auc:.4f}")
    except ValueError as e:
        logger.warning(f"AUC Score not computable: {e}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=['True REAL', 'True FAKE'], columns=['Pred REAL', 'Pred FAKE'])
    logger.info("\nConfusion Matrix:\n" + str(cm_df))

    # --- Per-Method Accuracy ---
    logger.info("\n--- Per-Method Accuracy ---")
    per_method_stats = results_df.groupby('method_source').apply(
        lambda g: pd.Series({
            'accuracy': accuracy_score(g['true_label'], g['pred_label']),
            'count': len(g)
        })
    ).reset_index()

    for _, row in per_method_stats.iterrows():
        logger.info(f"{row['method_source']:<20}: {row['accuracy']:.2%} accuracy ({row['count']} processed)")

    logger.info("-" * 60)


def main():
    """Main script execution function."""
    setup_logging()
    logger.info("========= Starting Local Evaluation Script =========")

    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires a GPU for efficient processing.")
        return

    # Phase 1 & 2: Download data and load models
    all_videos = download_videos_from_gcs()
    if not all_videos:
        logger.error("Halting script: No videos were loaded from GCS.")
        return

    models = load_all_models()
    if not models:
        logger.error("Halting script: Models could not be loaded.")
        return

    # Phase 3 & 4: Run evaluations and report
    total_videos = len(all_videos)
    for model_type in MODEL_TYPES_TO_TEST:
        if model_type not in models:
            logger.warning(f"Model '{model_type}' not available for testing. Skipping.")
            continue
        for pre_method in PRE_METHODS_TO_TEST:
            results_df, skipped_count = run_single_evaluation(model_type, pre_method, models, all_videos)
            generate_and_print_report(results_df, model_type, pre_method, total_videos, skipped_count)

    logger.info(f"========= Evaluation Finished =========\n"
                f"Full log available at: {LOG_FILE}\n"
                f"Misclassified videos saved in: {MISCLASSIFIED_DIR}")


if __name__ == "__main__":
    main()
