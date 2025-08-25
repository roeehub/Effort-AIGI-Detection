import os
import sys
import logging
import shutil
import time
import argparse
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
MODEL_CACHE_DIR = BASE_DIR / "model_cache"
FACE_CACHE_DIR = BASE_DIR / "face_cache"
MISCLASSIFIED_DIR = BASE_DIR / "misclassified_videos"
LOG_FILE = BASE_DIR / "evaluation_log.txt"

# --- FRAME SAMPLING PARAMETERS ---
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

            relative_path_str = os.path.relpath(blob.name, GCS_DATA_PREFIX)
            path_parts = Path(relative_path_str).parts

            label = "fake" if "fake" in path_parts else "real" if "real" in path_parts else None
            if label is None:
                logger.warning(f"Could not determine label for {blob.name}. Skipping.")
                continue

            try:
                label_index = path_parts.index(label)
                method_source = path_parts[label_index + 1] if len(path_parts) > label_index + 1 else "unknown"
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


def _load_detector_from_file(weights_path: Path) -> torch.nn.Module:
    """Helper function to load a single detector model from a weights file."""
    repo_base = Path(".")
    cfg_path = repo_base / "config/detector/effort.yaml"
    train_cfg_path = repo_base / "config/train_config.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    with open(train_cfg_path, "r") as f:
        config.update(yaml.safe_load(f))

    model_cls = DETECTOR[config["model_name"]]
    model = model_cls(config).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_models_from_paths(model_paths: List[str]) -> Dict[str, torch.nn.Module]:
    """
    Loads models from a list of local or GCS paths.
    GCS models are downloaded and cached locally.
    Returns a dictionary mapping a unique model name to the loaded model object.
    """
    logger.info("--- Phase 2: Loading all specified models ---")
    loaded_models = {}
    MODEL_CACHE_DIR.mkdir(exist_ok=True)

    for path_str in model_paths:
        model_name = Path(path_str).name
        if model_name in loaded_models:
            logger.warning(f"Model '{model_name}' specified multiple times. Skipping duplicate.")
            continue

        local_weights_path = None
        try:
            if path_str.startswith("gs://"):
                local_weights_path = MODEL_CACHE_DIR / model_name
                if not local_weights_path.exists():
                    logger.info(f"Downloading model '{model_name}' from GCS...")
                    storage_client = storage.Client()
                    bucket_name, blob_path = path_str.replace("gs://", "").split("/", 1)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    blob.download_to_filename(str(local_weights_path))
                    logger.info(f"✅ Downloaded to {local_weights_path}")
                else:
                    logger.info(f"Model '{model_name}' found in local cache.")
            else:
                local_weights_path = Path(path_str)
                if not local_weights_path.exists():
                    raise FileNotFoundError(f"Local model file not found: {path_str}")

            logger.info(f"Loading model '{model_name}' into memory...")
            model = _load_detector_from_file(local_weights_path)
            loaded_models[model_name] = model
            logger.info(f"✅ Loaded '{model_name}' successfully.")

        except Exception as e:
            logger.error(f"❌ FAILED to load model from '{path_str}'. Error: {e}")

    try:
        logger.info("Initializing YOLO and Haar pre-processors...")
        video_preprocessor.initialize_yolo_model()
        video_preprocessor.initialize_haar_cascades()
        logger.info("✅ Pre-processors initialized.")
    except Exception as e:
        logger.exception(f"FATAL: Failed to initialize pre-processors. Error: {e}")
        return {}

    return loaded_models


def process_and_cache_video_faces(video_info: Dict, pre_method: str) -> Optional[str]:
    """Processes a single video and caches extracted faces as a .npz file."""
    video_path = Path(video_info["local_path"])
    cache_path = FACE_CACHE_DIR / pre_method / video_info['true_label'] / f"{video_path.stem}.npz"
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
        if not ret: continue

        face = None
        if pre_method == 'yolo':
            face = video_preprocessor.extract_yolo_face(frame)
        elif pre_method == 'yolo_haar':
            face = video_preprocessor.extract_yolo_haar_face(frame)

        if face is not None:
            collected_faces.append(face)

    cap.release()
    if len(collected_faces) < MIN_FRAME_COUNT:
        return None

    final_faces = collected_faces[:TARGET_FRAME_COUNT]
    np.savez_compressed(cache_path, faces=np.array(final_faces), num_frames=len(final_faces))
    return str(cache_path)


# MODIFIED: Added `save_misclassified` parameter
def run_single_evaluation(model_name: str, pre_method: str, model: torch.nn.Module, all_videos: List[Dict],
                          save_misclassified: bool = False) -> Tuple[pd.DataFrame, int]:
    """Runs a full evaluation for one combination of model and pre-method."""
    logger.info(f"\n{'=' * 80}\nRunning Evaluation: model='{model_name}', pre_method='{pre_method}'\n{'=' * 80}")

    transform = video_preprocessor._get_transform()
    results = []
    skipped_count = 0

    logger.info("Preprocessing videos and caching faces...")
    processed_videos = []
    for video_info in tqdm(all_videos, desc=f"Preprocessing ({pre_method})"):
        cache_file = process_and_cache_video_faces(video_info, pre_method)
        if cache_file:
            processed_videos.append((video_info, cache_file))
        else:
            skipped_count += 1

    if not processed_videos:
        logger.warning("No videos could be preprocessed for this run. Skipping inference.")
        return pd.DataFrame(), skipped_count

    logger.info("Running inference on processed videos...")
    for video_info, cache_file in tqdm(processed_videos, desc=f"Inference ({model_name})"):
        data = np.load(cache_file)
        faces, num_frames = data['faces'], data['num_frames'].item()
        if num_frames < MIN_FRAME_COUNT: continue

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

        # MODIFIED: Conditional saving of misclassified videos
        if save_misclassified and pred_label != video_info["true_label"]:
            misclassified_path_dir = MISCLASSIFIED_DIR / f"{model_name}_{pre_method}" / f"PRED_{pred_label.upper()}_WAS_{video_info['true_label'].upper()}"
            misclassified_path_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(video_info["local_path"], misclassified_path_dir)

    return pd.DataFrame(results), skipped_count


def generate_and_print_report(results_df: pd.DataFrame, model_name: str, pre_method: str, total_attempted: int,
                              skipped_count: int) -> Dict:
    """Calculates, logs a detailed report, and returns a summary dictionary."""
    logger.info(f"\n--- Evaluation Report: model='{model_name}', pre_method='{pre_method}' ---")

    processed_count = len(results_df)
    processed_percent = (processed_count / total_attempted) * 100 if total_attempted > 0 else 0
    logger.info("--- Processing Summary ---")
    logger.info(f"Total Videos Attempted: {total_attempted}")
    logger.info(f"Successfully Processed : {processed_count} ({processed_percent:.1f}%)")
    logger.info(f"Skipped (Not Enough Faces): {skipped_count}")

    summary = {"model_name": model_name, "pre_method": pre_method, "processed": f"{processed_count}/{total_attempted}"}

    if results_df.empty:
        logger.warning("No videos were processed. Cannot generate metrics.")
        return {**summary, "accuracy": 0, "f1_score": 0, "auc": 0, "precision": 0, "recall": 0}

    y_true = results_df['true_label'].map({'real': 0, 'fake': 1})
    y_pred = results_df['pred_label'].map({'real': 0, 'fake': 1})
    y_prob = results_df['fake_prob']

    logger.info("\n--- Overall Performance Metrics (Positive Class: FAKE) ---")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    logger.info(f"Accuracy:  {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    try:
        auc = roc_auc_score(y_true, y_prob)
        logger.info(f"AUC Score: {auc:.4f}")
    except ValueError as e:
        auc = 0.0
        logger.warning(f"AUC Score not computable: {e}")

    cm = confusion_matrix(y_true, y_pred)
    logger.info("\nConfusion Matrix:\n" + str(
        pd.DataFrame(cm, index=['True REAL', 'True FAKE'], columns=['Pred REAL', 'Pred FAKE'])))
    logger.info("-" * 60)

    return {**summary, "accuracy": accuracy, "f1_score": f1, "auc": auc, "precision": precision, "recall": recall}


def print_final_summary(summaries: List[Dict]):
    """Prints a consolidated table of results from all evaluation runs."""
    if not summaries:
        logger.info("No evaluation runs were completed to summarize.")
        return
    logger.info(f"\n{'=' * 30} FINAL EVALUATION SUMMARY {'=' * 30}")
    summary_df = pd.DataFrame(summaries)
    float_cols = summary_df.select_dtypes(include='float').columns
    summary_df[float_cols] = summary_df[float_cols].applymap(lambda x: f"{x:.4f}")
    logger.info("\n" + summary_df.to_string(index=False))
    logger.info("=" * 88)


def main(args):
    """Main script execution function."""
    setup_logging()
    logger.info("========= Starting Flexible Evaluation Script =========")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Using CPU. This will be very slow.")

    logger.info("Evaluation Plan:")
    logger.info(f"  Models to test: {', '.join([Path(p).name for p in args.models])}")
    logger.info(f"  Preprocessing methods: {', '.join(args.pre_methods)}")
    # NEW: Log whether misclassified videos will be saved
    logger.info(
        f"  Save misclassified videos: {'Yes (Debug Mode)' if args.save_misclassified else 'No (Standard Mode)'}")
    logger.info("-" * 50)

    all_videos = download_videos_from_gcs()
    if not all_videos:
        logger.error("Halting script: No videos were loaded from GCS.")
        return

    loaded_models = load_models_from_paths(args.models)
    if not loaded_models:
        logger.error("Halting script: No models could be loaded.")
        return

    total_videos = len(all_videos)
    all_run_summaries = []

    for model_name, model in loaded_models.items():
        for pre_method in args.pre_methods:
            # MODIFIED: Pass the new flag to the evaluation function
            results_df, skipped_count = run_single_evaluation(
                model_name, pre_method, model, all_videos, save_misclassified=args.save_misclassified
            )
            summary = generate_and_print_report(results_df, model_name, pre_method, total_videos, skipped_count)
            all_run_summaries.append(summary)

    print_final_summary(all_run_summaries)

    logger.info(f"========= Evaluation Finished =========\n"
                f"Full log available at: {LOG_FILE}")
    if args.save_misclassified:
        logger.info(f"Misclassified videos saved in: {MISCLASSIFIED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a flexible deepfake detection evaluation.")
    parser.add_argument(
        '--model', dest='models', action='append', required=True,
        help="Path to a model weights file (local or GCS). Specify multiple times for multiple models."
    )
    parser.add_argument(
        '--pre-method', dest='pre_methods', action='append', required=True, choices=['yolo', 'yolo_haar'],
        help="Preprocessing method for face extraction. Specify multiple times for multiple methods."
    )
    # NEW: Added the optional flag for saving misclassified videos
    parser.add_argument(
        '--save-misclassified', action='store_true',
        help="If specified, save copies of misclassified videos for debugging. "
             "Warning: This can use a large amount of disk space."
    )

    cli_args = parser.parse_args()
    main(cli_args)
