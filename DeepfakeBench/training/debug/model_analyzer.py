import os
import argparse
import random
import tempfile
import shutil
import time
import pickle
import sys
import warnings
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import concurrent.futures
from functools import partial
import glob

# --- Core ML/Data Science Libraries ---
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
import fsspec

# --- Visualization Libraries ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# --- Image & Augmentation Libraries ---
import cv2
import albumentations as A
from transformers import CLIPProcessor, CLIPModel

# --- Model Explanation & GCS ---
import shap
from google.cloud import storage
from google.api_core import exceptions

# --- Suppress common warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================================================================================
# ----------------------------- CONFIGURATION & SETUP ----------------------------------
# ======================================================================================

# --- GCS Buckets ---
BUCKETS_TO_ANALYZE = {
    "df40_train": "df40-frames-recropped-rfa85",
    "collected_train": "effort-collected-data",
    "test": "deep-fake-test-10-08-25-frames-yolo-recropped-rfa85",
}

# --- Sampling Parameters ---
# For the large df40 bucket, we sample to keep the analysis manageable.
DF40_VIDEOS_PER_METHOD = 50
# For all videos, we sample N frames.
FRAMES_PER_VIDEO = 16

PROCESSING_BATCH_SIZE = 200  # Process 200 videos, save, then do the next 200

# --- Analysis Parameters ---
# Define the specific methods to investigate for the anomaly
ANOMALY_METHOD_PATHS = {
    "validation": "effort-collected-data/fake/veo3-creations",
    "test": "deep-fake-test-10-08-25-frames-yolo-recropped-rfa85/fake/veo3",
}
# Define a few example videos for deep-dive plots
EXAMPLE_VIDEOS_FOR_PLOTS = {
    "high_conf_fake": "deep-fake-test-10-08-25-frames-yolo-recropped-rfa85/fake/SimSwap_256/id14_id6_0007",
    "low_conf_fake": "deep-fake-test-10-08-25-frames-yolo-recropped-rfa85/fake/veo3/video_13",
    "false_positive_candidate": "deep-fake-test-10-08-25-frames-yolo-recropped-rfa85/real/youtube-real/--2gZnd_a2A"
}

# --- Model & Preprocessing ---
# Import necessary components from your project structure
sys.path.append(str(Path(__file__).resolve().parents[1]))  # go up one level
from detectors import DETECTOR, EffortDetector
import video_preprocessor  # We only need its transform function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_detector(cfg: dict, weights: str) -> nn.Module:
    """Loads the EffortDetector model from config and weights."""
    model_cls = DETECTOR[cfg["model_name"]]
    model = model_cls(cfg).to(device)
    ckpt = torch.load(weights, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def download_gcs_blob(gcs_path: str, local_path: Path) -> bool:
    """Downloads a single blob from GCS."""
    try:
        bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        return True
    except Exception as e:
        print(f"ERROR: Failed to download {gcs_path}: {e}")
        return False


# ======================================================================================
# ------------------------- PHASE 1: DATA GATHERING & INFERENCE ------------------------
# ======================================================================================

def discover_and_sample_gcs_data(gcs_client, quick_test=False):
    """
    Scans GCS buckets, applies sampling logic, and returns a list of "video units".
    Each unit contains metadata and a list of frame paths to be processed.
    """
    print("--- Phase 1a: Discovering and sampling video data from GCS... ---")
    all_video_units = []

    for dataset_name, bucket_name in BUCKETS_TO_ANALYZE.items():
        print(f"  Scanning bucket: {bucket_name} ({dataset_name})")

        blobs = list(gcs_client.list_blobs(bucket_name))
        videos = defaultdict(list)
        for blob in tqdm(blobs, desc=f"Listing blobs in {bucket_name}"):
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                video_path = str(Path(blob.name).parent)
                videos[video_path].append(f"gs://{blob.bucket.name}/{blob.name}")

        methods = defaultdict(list)
        for video_path, frame_paths in videos.items():
            parts = video_path.split('/')
            if len(parts) >= 2:
                method_name = parts[1]
                methods[method_name].append((video_path, frame_paths))

        sampled_videos = []
        if dataset_name == 'df40_train':
            print(f"    Applying sampling: max {DF40_VIDEOS_PER_METHOD} videos per method.")
            for method_name, video_list in methods.items():
                num_to_sample = min(DF40_VIDEOS_PER_METHOD, len(video_list))
                sampled_videos.extend(random.sample(video_list, num_to_sample))
        else:
            for method_name, video_list in methods.items():
                sampled_videos.extend(video_list)

        if quick_test:
            print("    QUICK TEST MODE: Using only 5 videos total for this bucket.")
            sampled_videos = random.sample(sampled_videos, min(5, len(sampled_videos)))

        for video_path, frame_gcs_paths in sampled_videos:
            parts = video_path.split('/')
            label, method = parts[0], parts[1]
            video_id = "/".join(parts[2:])

            frame_path_sample = random.sample(frame_gcs_paths, min(FRAMES_PER_VIDEO, len(frame_gcs_paths)))

            all_video_units.append({
                "video_path": video_path,
                "dataset": dataset_name,
                "label": label,
                "method": method,
                "video_id": video_id,
                "frame_gcs_paths": frame_path_sample
            })

    print(f"--- Discovered and sampled a total of {len(all_video_units)} videos. ---")
    return all_video_units


def get_low_level_stats(image_np: np.ndarray) -> dict:
    """Calculates basic image statistics."""
    if image_np is None or image_np.size == 0: return {}
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))
    return {"sharpness": sharpness, "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b}


def process_video_unit(video_unit: Dict, fs, transform, model, clip_model, clip_processor) -> List[Dict]:
    """
    Processes a single video: downloads frames, runs inference, and extracts features.
    """
    model_features = []

    def hook(module, input, output):
        model_features.append(output[0].detach().cpu().numpy())

    handle = model.backbone.register_forward_hook(hook)

    results = []
    local_temp_dir = Path(tempfile.mkdtemp())

    try:
        image_tensors = []
        raw_images = []

        for gcs_path in video_unit['frame_gcs_paths']:
            local_frame_path = local_temp_dir / Path(gcs_path).name
            try:
                with fs.open(gcs_path, 'rb') as f_in, open(local_frame_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            except FileNotFoundError:
                print(f"WARNING: Frame not found on GCS: {gcs_path}. Skipping.")
                continue

            img_bgr = cv2.imread(str(local_frame_path))
            if img_bgr is None: continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_tensors.append(transform(img_rgb))
            raw_images.append(img_bgr)

        if not image_tensors:
            handle.remove()
            return []

        batch_tensor = torch.stack(image_tensors).to(device)
        with torch.no_grad():
            preds = model({'image': batch_tensor}, inference=True)
            probs = preds["prob"].cpu().numpy().tolist()

            clip_inputs = clip_processor(images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in raw_images],
                                         return_tensors="pt").to(device)
            clip_embeddings = clip_model.get_image_features(**clip_inputs).cpu().numpy()

        handle.remove()
        model_internal_embeddings = np.concatenate(model_features, axis=0)

        for i in range(len(probs)):
            stats = get_low_level_stats(raw_images[i])
            frame_gcs_path = video_unit['frame_gcs_paths'][i]
            result_unit = video_unit.copy()
            del result_unit['frame_gcs_paths']

            results.append({
                **result_unit,
                "frame_gcs_path": frame_gcs_path,
                "fake_prob": probs[i],
                "clip_embedding": clip_embeddings[i],
                "model_embedding": model_internal_embeddings[i],
                **stats
            })

    except Exception as e:
        print(f"WARNING: Failed to process video {video_unit['video_path']}. Error: {e}")
    finally:
        shutil.rmtree(local_temp_dir)
    return results


# ======================================================================================
# ----------------------------- PHASE 2: PLOTTING & ANALYSIS ---------------------------
# ======================================================================================

def plot_roc_and_pr_curves(df_video, output_dir):
    """Generates and saves ROC and Precision-Recall curves."""
    print("--- Phase 2a: Generating ROC and Precision-Recall curves... ---")
    test_df = df_video[df_video['dataset'] == 'test']
    y_true = (test_df['label'] == 'fake').astype(int)
    y_score = test_df['video_prob_mean']
    if len(np.unique(y_true)) < 2:
        print("  - WARNING: Cannot compute ROC/PR with only one class. Skipping.")
        return
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    optimal_threshold = thresholds_roc[optimal_idx]
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                label=f'Optimal Threshold = {optimal_threshold:.3f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_dir / "roc_curve.png")
    plt.close()
    print(f"  - ROC AUC: {roc_auc:.4f}")
    print(f"  - Optimal Threshold (Balanced): {optimal_threshold:.4f}")
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_dir / "pr_curve.png")
    plt.close()


def plot_performance_by_method(df_video, output_dir):
    """Plots the AUC for each method on the test set."""
    print("--- Phase 2b: Analyzing performance per method... ---")
    test_df = df_video[df_video['dataset'] == 'test'].copy()
    method_aucs, method_scores = {}, {}
    for method, group in test_df.groupby('method'):
        y_true = (group['label'] == 'fake').astype(int)
        if len(np.unique(y_true)) < 2:
            if np.all(y_true == 1):
                avg_score = group['video_prob_mean'].mean()
                method_scores[method] = avg_score
                print(f"  - Method '{method}' (fake only): Avg. Score = {avg_score:.3f}")
            continue
        fpr, tpr, _ = roc_curve(y_true, group['video_prob_mean'])
        method_aucs[method] = auc(fpr, tpr)
    if not method_aucs:
        print("  - Could not compute per-method AUCs (likely single-class methods).")
        return
    sorted_methods = sorted(method_aucs.items(), key=lambda x: x[1])
    plt.figure(figsize=(12, max(8, len(sorted_methods) * 0.5)))
    plt.barh([m[0] for m in sorted_methods], [m[1] for m in sorted_methods])
    plt.xlabel('AUC Score')
    plt.title('Model Performance (AUC) by Method on Test Set')
    plt.xlim([0, 1])
    plt.grid(axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / "performance_by_method.png")
    plt.close()


def analyze_anomaly_domain_shift(df_all_frames, output_dir):
    """Compares the 'veo3' data distributions between validation and test sets."""
    print("--- Phase 2c: Investigating 'veo3' anomaly via domain analysis... ---")
    val_path, test_path = ANOMALY_METHOD_PATHS["validation"], ANOMALY_METHOD_PATHS["test"]
    val_prefix, test_prefix = "gs://" + val_path, "gs://" + test_path
    df_val = df_all_frames[df_all_frames['frame_gcs_path'].str.startswith(val_prefix)].copy()
    df_val['source'] = 'veo3 (Validation)'
    df_test = df_all_frames[df_all_frames['frame_gcs_path'].str.startswith(test_prefix)].copy()
    df_test['source'] = 'veo3 (Test)'
    if df_val.empty or df_test.empty:
        print("  - WARNING: Could not find 'veo3' data for both validation and test. Skipping analysis.")
        return
    df_anomaly = pd.concat([df_val, df_test])
    metrics = ['sharpness', 'mean_r', 'mean_g', 'mean_b']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_anomaly, x=metric, hue='source', fill=True, common_norm=False)
        plt.title(f'Distribution of {metric.title()} for veo3')
        plt.grid(True)
        plt.savefig(output_dir / f"anomaly_veo3_{metric}_dist.png")
        plt.close()
    print("  - Running t-SNE on CLIP embeddings for 'veo3' data...")
    if 'clip_embedding' not in df_anomaly.columns:
        print("  - WARNING: 'clip_embedding' column not found. Skipping t-SNE.")
        return
    embeddings = np.stack(df_anomaly['clip_embedding'].values)
    labels = df_anomaly['source'].values
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, style=labels, s=50)
    plt.title('t-SNE of CLIP Embeddings: veo3 Validation vs. Test')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Source')
    plt.grid(True)
    plt.savefig(output_dir / "anomaly_veo3_tsne_clip.png")
    plt.close()


# ======================================================================================
# --------------------------- PHASE 3: BEHAVIOR DEEP DIVE ------------------------------
# ======================================================================================

def analyze_error_buckets(df_video, output_dir):
    """Identifies and visualizes examples from different error buckets."""
    print("--- Phase 3a: Analyzing error buckets (False Positives/Negatives)... ---")
    test_df = df_video[df_video['dataset'] == 'test'].copy()
    fp = test_df[(test_df['label'] == 'real') & (test_df['video_prob_mean'] > 0.8)]
    fn = test_df[(test_df['label'] == 'fake') & (test_df['video_prob_mean'] < 0.2)]
    ambiguous = test_df[(test_df['video_prob_mean'] >= 0.4) & (test_df['video_prob_mean'] <= 0.6)]
    print(f"  - Found {len(fp)} high-confidence False Positives.")
    print(f"  - Found {len(fn)} high-confidence False Negatives.")
    print(f"  - Found {len(ambiguous)} Ambiguous predictions.")
    fp[['video_path', 'video_prob_mean']].to_csv(output_dir / "false_positives.csv", index=False)
    fn[['video_path', 'video_prob_mean']].to_csv(output_dir / "false_negatives.csv", index=False)


def plot_intra_video_consistency(df_all_frames, output_dir):
    """Plots frame-by-frame probabilities for a few example videos."""
    print("--- Phase 3b: Plotting intra-video prediction consistency... ---")
    for name, video_path_prefix in EXAMPLE_VIDEOS_FOR_PLOTS.items():
        relative_path = "/".join(video_path_prefix.split('/')[1:])
        df_video_frames = df_all_frames[df_all_frames['video_path'] == relative_path]
        if df_video_frames.empty:
            print(f"  - WARNING: Could not find example video '{name}' with path prefix '{relative_path}'. Skipping.")
            continue
        probs = df_video_frames['fake_prob'].values
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(probs)), probs, marker='o', linestyle='-')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
        plt.ylim([0, 1])
        plt.xlabel('Frame Number')
        plt.ylabel('Predicted Fake Probability')
        plt.title(f'Frame-wise Predictions for Video: {name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"consistency_{name}.png")
        plt.close()


# ======================================================================================
# ----------------------------- PHASE 4: ROBUSTNESS PROBING ----------------------------
# ======================================================================================

def probe_augmentation_sensitivity(model, transform, output_dir):
    """Tests how model predictions change with increasing augmentation strength."""
    print("--- Phase 4a: Probing model sensitivity to augmentations... ---")
    video_path_prefix = EXAMPLE_VIDEOS_FOR_PLOTS["high_conf_fake"]
    gcs_client = storage.Client()
    bucket_name, blob_prefix = video_path_prefix.split('/', 1)
    blobs = list(gcs_client.list_blobs(bucket_name, prefix=blob_prefix))
    if not blobs:
        print("  - WARNING: Could not find frames for augmentation probe. Skipping.")
        return
    blob = blobs[0]
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
        blob.download_to_filename(tmp_file.name)
        image_bgr = cv2.imread(tmp_file.name)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = defaultdict(list)
    augmentations = {
        "JPEG Compression": [A.ImageCompression(quality_lower=q, quality_upper=q, p=1.0) for q in range(95, 20, -5)],
        "Gaussian Blur": [A.GaussianBlur(blur_limit=(s, s), p=1.0) for s in range(1, 15, 2)]
    }
    for aug_name, aug_list in augmentations.items():
        strengths = [p.quality_lower if hasattr(p, 'quality_lower') else p.blur_limit[0] for p in aug_list]
        for aug, strength in zip(aug_list, strengths):
            aug_image = aug(image=image_rgb)['image']
            img_tensor = transform(aug_image).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model({'image': img_tensor}, inference=True)["prob"].item()
            results[aug_name].append((strength, prob))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Robustness to Augmentations')
    strengths, probs = zip(*sorted(results["JPEG Compression"]))
    axes[0].plot(strengths, probs, 'o-')
    axes[0].set_title('JPEG Compression')
    axes[0].set_xlabel('JPEG Quality')
    axes[0].set_ylabel('Fake Probability')
    axes[0].invert_xaxis()
    axes[0].grid(True)
    strengths, probs = zip(*sorted(results["Gaussian Blur"]))
    axes[1].plot(strengths, probs, 'o-')
    axes[1].set_title('Gaussian Blur')
    axes[1].set_xlabel('Blur Sigma')
    axes[1].set_ylabel('Fake Probability')
    axes[1].grid(True)
    plt.savefig(output_dir / "augmentation_sensitivity.png")
    plt.close()


def explain_with_shap(model, transform, output_dir):
    """Generates SHAP explanations for a few key frames."""
    print("--- Phase 4b: Generating SHAP explanations for model decisions... ---")
    gcs_client = storage.Client()
    images_to_explain = {"correct_fake": EXAMPLE_VIDEOS_FOR_PLOTS["high_conf_fake"],
                         "potential_fp_real": EXAMPLE_VIDEOS_FOR_PLOTS["false_positive_candidate"]}
    image_tensors, image_rgbs, labels = [], [], []
    for name, path_prefix in images_to_explain.items():
        bucket_name, blob_prefix = path_prefix.split('/', 1)
        blob = next(iter(gcs_client.list_blobs(bucket_name, prefix=blob_prefix)), None)
        if blob:
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
                blob.download_to_filename(tmp_file.name)
                image_bgr = cv2.imread(tmp_file.name)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensors.append(transform(image_rgb))
            image_rgbs.append(cv2.resize(image_rgb, (224, 224)))
            labels.append(name)
    if len(image_tensors) < 2:
        print("  - WARNING: Could not load frames for SHAP analysis. Skipping.")
        return
    batch = torch.stack(image_tensors).to(device)

    def f(x):
        return model({'image': x}, inference=True)['prob']

    background = torch.zeros_like(batch[[0]])
    explainer = shap.DeepExplainer(f, background)
    shap_values = explainer.shap_values(batch)
    shap_values_transposed = [np.transpose(s, (1, 2, 0)) for s in shap_values]
    shap.image_plot(shap_values_transposed, -np.array(image_rgbs), show=False)
    plt.suptitle("SHAP Explanations (Red pixels increase fake score)")
    plt.savefig(output_dir / "shap_explanation.png")
    plt.close()


# ======================================================================================
# ------------------------------------ MAIN SCRIPT -------------------------------------
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis script for Effort-AIGI deepfake detector.")
    parser.add_argument('--model_gcs_path', type=str,
                        default='gs://training-job-outputs/best_checkpoints/k540e0ts/top_n_effort_20250910_ep2_auc0.9809_eer0.0800.pth',
                        help="GCS path to the model weights file (.pth).")
    parser.add_argument('--output_dir', type=str, default="model_analysis_output",
                        help="Directory to save all analysis plots and data files.")
    parser.add_argument('--skip_inference', action='store_true',
                        help="Skip the inference step and load data from output_dir.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers for data processing.")
    parser.add_argument('--use_discovery_cache', action='store_true',
                        help="Use cached GCS discovery results if available.")
    parser.add_argument('--quick_test', action='store_true', help="Run on a very small sample for debugging purposes.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    chunk_output_dir = output_dir / "inference_chunks"

    if not args.skip_inference:
        # --- PHASE 1: DATA GATHERING & BATCH INFERENCE ---
        discovery_cache_path = output_dir / "discovery_cache.pkl"
        video_units = []
        if args.use_discovery_cache and discovery_cache_path.exists():
            print(f"--- Loading cached GCS discovery data from {discovery_cache_path} ---")
            with open(discovery_cache_path, 'rb') as f:
                video_units = pickle.load(f)
            print(f"  Loaded {len(video_units)} video units from cache.")
        else:
            gcs_client = storage.Client()
            video_units = discover_and_sample_gcs_data(gcs_client, args.quick_test)
            print(f"--- Saving GCS discovery data to {discovery_cache_path} for future runs... ---")
            with open(discovery_cache_path, 'wb') as f:
                pickle.dump(video_units, f)

        print("\n--- Loading models for inference... ---")
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp_weights:
            model_weights_path = args.model_gcs_path
            if args.model_gcs_path.startswith("gs://"):
                print(f"  Downloading detector model from {args.model_gcs_path}...")
                assert download_gcs_blob(args.model_gcs_path, Path(tmp_weights.name)), "Model download failed."
                model_weights_path = tmp_weights.name

            config = {"model_name": "effort",
                      "backbone": {"arch": "ViT-L/14"},
                      'gcs_assets': {
                          'clip_backbone': {
                              'gcs_path': "gs://base-checkpoints/effort-aigi/models--openai--clip-vit-large-patch14/",
                              'local_path': "../weights/models--openai--clip-vit-large-patch14/"
                          }
                      }}
            model = load_detector(config, model_weights_path)
            print("  Detector model loaded.")

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("  CLIP model loaded.")

        transform = video_preprocessor._get_transform()
        fs = fsspec.filesystem('gcs')
        processing_func = partial(process_video_unit, fs=fs, transform=transform, model=model, clip_model=clip_model,
                                  clip_processor=clip_processor)

        # <<< FINALIZED: RESUMABLE AND MEMORY-EFFICIENT BATCH PROCESSING LOGIC >>>
        print(
            f"\n--- Starting resumable inference on {len(video_units)} videos in batches of {PROCESSING_BATCH_SIZE}... ---")
        chunk_output_dir.mkdir(exist_ok=True)

        video_chunks = [video_units[i:i + PROCESSING_BATCH_SIZE] for i in
                        range(0, len(video_units), PROCESSING_BATCH_SIZE)]

        for i, chunk in enumerate(video_chunks):
            chunk_csv_path = chunk_output_dir / f"chunk_{i}_data.csv"
            chunk_clip_path = chunk_output_dir / f"chunk_{i}_clip.npy"
            chunk_model_path = chunk_output_dir / f"chunk_{i}_model.npy"

            # Check if this chunk is already fully processed
            if chunk_csv_path.exists() and chunk_clip_path.exists() and chunk_model_path.exists():
                print(f"  Chunk {i + 1}/{len(video_chunks)} already processed. Skipping.")
                continue

            print(f"  Processing chunk {i + 1}/{len(video_chunks)} ({len(chunk)} videos)...")

            # Write header for the chunk's CSV file
            header = ['video_path', 'dataset', 'label', 'method', 'video_id', 'frame_gcs_path',
                      'fake_prob', 'sharpness', 'mean_r', 'mean_g', 'mean_b']
            pd.DataFrame(columns=header).to_csv(chunk_csv_path, index=False)

            # Open files for appending. numpy files use binary append ('ab').
            with open(chunk_clip_path, 'ab') as clip_f, \
                    open(chunk_model_path, 'ab') as model_f, \
                    open(chunk_csv_path, 'a', newline='') as csv_f:

                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_video = {executor.submit(processing_func, unit): unit for unit in chunk}

                    # Process results as they complete to keep memory usage low
                    for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(chunk),
                                       desc=f"Chunk {i + 1}"):
                        try:
                            video_results = future.result()
                            if not video_results:
                                continue

                            # Process one video's results at a time
                            df_video_results = pd.DataFrame(video_results)

                            clip_embeddings = np.stack(df_video_results.pop('clip_embedding').values)
                            model_embeddings = np.stack(df_video_results.pop('model_embedding').values)

                            # Append data to files immediately, without header
                            df_video_results.drop(columns=['frame_gcs_paths'], errors='ignore').to_csv(csv_f,
                                                                                                       header=False,
                                                                                                       index=False)
                            np.save(clip_f, clip_embeddings)
                            np.save(model_f, model_embeddings)

                        except Exception as exc:
                            video_unit = future_to_video[future]
                            print(f"\nWARNING: Video {video_unit['video_path']} generated an exception: {exc}")

        print("\n--- Inference complete. All chunks are processed and saved. ---")

    # --- CONSOLIDATE DATA FROM CHUNKS ---
    print(f"--- Consolidating data from chunk files in {chunk_output_dir} for analysis... ---")

    all_chunk_dfs = []
    for csv_file in sorted(glob.glob(str(chunk_output_dir / "chunk_*_data.csv"))):
        all_chunk_dfs.append(pd.read_csv(csv_file))

    all_clip_embeddings = []
    for npy_file in sorted(glob.glob(str(chunk_output_dir / "chunk_*_clip.npy"))):
        all_clip_embeddings.append(np.load(npy_file, allow_pickle=True))

    all_model_embeddings = []
    for npy_file in sorted(glob.glob(str(chunk_output_dir / "chunk_*_model.npy"))):
        all_model_embeddings.append(np.load(npy_file, allow_pickle=True))

    if not all_chunk_dfs:
        print("ERROR: No inference data found. Cannot proceed with analysis.")
        print("Please run the script without '--skip_inference' first.")
        sys.exit(1)

    df_all_frames = pd.concat(all_chunk_dfs, ignore_index=True)
    clip_embeddings = np.concatenate(all_clip_embeddings, axis=0)
    model_embeddings = np.concatenate(all_model_embeddings, axis=0)

    if len(df_all_frames) == len(clip_embeddings):
        df_all_frames['clip_embedding'] = list(clip_embeddings)
    else:
        print(
            f"WARNING: Mismatch! DataFrame has {len(df_all_frames)} rows, but found {len(clip_embeddings)} CLIP embeddings.")

    if len(df_all_frames) == len(model_embeddings):
        df_all_frames['model_embedding'] = list(model_embeddings)
    else:
        print(
            f"WARNING: Mismatch! DataFrame has {len(df_all_frames)} rows, but found {len(model_embeddings)} model embeddings.")

    print("  Data loaded and consolidated successfully.")

    # Save the consolidated files
    df_all_frames.to_csv(output_dir / "all_frame_data.csv", index=False)
    np.save(output_dir / "all_clip_embeddings.npy", clip_embeddings)
    np.save(output_dir / "all_model_embeddings.npy", model_embeddings)
    print(f"  Final consolidated data saved to {output_dir}")

    print("\n--- Aggregating frame data to video level... ---")
    df_video = df_all_frames.groupby(['video_path', 'dataset', 'label', 'method']).agg(
        video_prob_mean=('fake_prob', 'mean'),
        video_prob_median=('fake_prob', 'median'),
        frame_count=('fake_prob', 'count')
    ).reset_index()

    # --- RUN ALL ANALYSIS PHASES ---
    plot_roc_and_pr_curves(df_video, output_dir)
    plot_performance_by_method(df_video, output_dir)
    analyze_anomaly_domain_shift(df_all_frames, output_dir)
    analyze_error_buckets(df_video, output_dir)
    plot_intra_video_consistency(df_all_frames, output_dir)

    model_is_loaded = 'model' in locals()

    if not args.skip_inference and model_is_loaded:
        print("\n--- Model is loaded. Proceeding with live probes (Phase 4)... ---")
        transform = video_preprocessor._get_transform()
        probe_augmentation_sensitivity(model, transform, output_dir)
        explain_with_shap(model, transform, output_dir)
    else:
        print("\n--- SKIPPING Phase 4 (Robustness Probes) as model was not loaded (`--skip_inference` was used). ---")

    print(f"\n✅ Analysis complete! All outputs are saved in '{output_dir}'.")


if __name__ == "__main__":
    main()
