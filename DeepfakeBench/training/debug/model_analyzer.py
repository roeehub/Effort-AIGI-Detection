import os
import argparse
import random
import tempfile
import shutil
import time
import sys
import warnings
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import concurrent.futures
from functools import partial

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
DF40_VIDEOS_PER_METHOD = 100
# For all videos, we sample N frames.
FRAMES_PER_VIDEO = 16

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
# To make this script standalone, we copy the required functions here.
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

        # Group all blobs by their video folder path
        blobs = list(gcs_client.list_blobs(bucket_name))
        videos = defaultdict(list)
        for blob in blobs:
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                video_path = str(Path(blob.name).parent)
                videos[video_path].append(blob)

        # Group videos by their method
        methods = defaultdict(list)
        for video_path, video_blobs in videos.items():
            parts = video_path.split('/')
            if len(parts) >= 2:
                method_name = parts[1]
                methods[method_name].append((video_path, video_blobs))

        # Apply sampling logic
        sampled_videos = []
        if dataset_name == 'df40_train':
            print(f"    Applying sampling: max {DF40_VIDEOS_PER_METHOD} videos per method.")
            for method_name, video_list in methods.items():
                num_to_sample = min(DF40_VIDEOS_PER_METHOD, len(video_list))
                sampled_videos.extend(random.sample(video_list, num_to_sample))
        else:
            # For test and collected_train, use all videos
            for method_name, video_list in methods.items():
                sampled_videos.extend(video_list)

        if quick_test:
            print("    QUICK TEST MODE: Using only 5 videos total.")
            sampled_videos = random.sample(sampled_videos, min(5, len(sampled_videos)))

        # Create final "video units" with metadata
        for video_path, video_blobs in sampled_videos:
            parts = video_path.split('/')
            label, method = parts[0], parts[1]
            video_id = "/".join(parts[2:])

            # Sample frames within the video
            frame_sample = random.sample(video_blobs, min(FRAMES_PER_VIDEO, len(video_blobs)))

            all_video_units.append({
                "video_path": video_path,
                "dataset": dataset_name,
                "label": label,
                "method": method,
                "video_id": video_id,
                "frame_blobs": frame_sample
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
    # These will hold the features extracted from the model's penultimate layer
    model_features = []

    def hook(module, input, output):
        # The output of the ViT backbone before the head is what we want.
        # It's usually the first element of the output tuple.
        model_features.append(output[0].detach().cpu().numpy())

    # Register the hook on the backbone's output
    handle = model.backbone.register_forward_hook(hook)

    results = []
    local_temp_dir = Path(tempfile.mkdtemp())

    try:
        frame_paths = []
        image_tensors = []
        raw_images = []

        # Download and preprocess all frames for the video
        for blob in video_unit['frame_blobs']:
            local_frame_path = local_temp_dir / blob.name.replace('/', '_')
            with fs.open(f"{blob.bucket.name}/{blob.name}", 'rb') as f_in, open(local_frame_path, 'wb') as f_out:
                f_out.write(f_in.read())

            img_bgr = cv2.imread(str(local_frame_path))
            if img_bgr is None: continue

            # User confirmed frames are pre-cropped, so we just resize and transform
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            image_tensors.append(transform(img_rgb))
            raw_images.append(img_bgr)
            frame_paths.append(f"gs://{blob.bucket.name}/{blob.name}")

        if not image_tensors:
            return []

        # Batch inference
        batch_tensor = torch.stack(image_tensors).to(device)
        with torch.no_grad():
            preds = model({'image': batch_tensor}, inference=True)
            probs = preds["prob"].cpu().numpy().tolist()

            # Batch CLIP embedding
            clip_inputs = clip_processor(images=[cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in raw_images],
                                         return_tensors="pt").to(device)
            clip_embeddings = clip_model.get_image_features(**clip_inputs).cpu().numpy()

        # Unregister the hook
        handle.remove()
        model_internal_embeddings = np.concatenate(model_features, axis=0)

        # Collate results for each frame
        for i in range(len(probs)):
            stats = get_low_level_stats(raw_images[i])
            results.append({
                **video_unit,
                "frame_gcs_path": frame_paths[i],
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

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold (closest to top-left)
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

    # Precision-Recall Curve
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

    # Only calculate AUC for methods with both real and fake examples (if applicable)
    # or just fakes if no corresponding real method exists.
    method_aucs = {}
    for method, group in test_df.groupby('method'):
        y_true = (group['label'] == 'fake').astype(int)
        # Skip if only one class is present
        if len(np.unique(y_true)) < 2:
            # For fake-only methods, we can report average prediction score
            if np.all(y_true == 1):
                avg_score = group['video_prob_mean'].mean()
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

    # Construct full GCS path prefix for filtering
    val_prefix = "gs://" + val_path
    test_prefix = "gs://" + test_path

    df_val = df_all_frames[df_all_frames['video_path'].str.startswith(val_prefix.split('/', 3)[-1])]
    df_val['source'] = 'veo3 (Validation)'
    df_test = df_all_frames[df_all_frames['video_path'].str.startswith(test_prefix.split('/', 3)[-1])]
    df_test['source'] = 'veo3 (Test)'

    if df_val.empty or df_test.empty:
        print("  - WARNING: Could not find 'veo3' data for both validation and test. Skipping analysis.")
        return

    df_anomaly = pd.concat([df_val, df_test])

    # 1. Low-level stats comparison
    metrics = ['sharpness', 'mean_r', 'mean_g', 'mean_b']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_anomaly, x=metric, hue='source', fill=True, common_norm=False)
        plt.title(f'Distribution of {metric.title()} for veo3')
        plt.grid(True)
        plt.savefig(output_dir / f"anomaly_veo3_{metric}_dist.png")
        plt.close()

    # 2. t-SNE on CLIP embeddings
    print("  - Running t-SNE on CLIP embeddings for 'veo3' data...")
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

    # Save lists of problematic videos
    fp[['video_path', 'video_prob_mean']].to_csv(output_dir / "false_positives.csv", index=False)
    fn[['video_path', 'video_prob_mean']].to_csv(output_dir / "false_negatives.csv", index=False)


def plot_intra_video_consistency(df_all_frames, output_dir):
    """Plots frame-by-frame probabilities for a few example videos."""
    print("--- Phase 3b: Plotting intra-video prediction consistency... ---")
    for name, video_path_prefix in EXAMPLE_VIDEOS_FOR_PLOTS.items():
        # The video_path in the dataframe is relative to bucket, so we match on that
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
    # Let's pick a high-confidence fake video to test
    video_path_prefix = EXAMPLE_VIDEOS_FOR_PLOTS["high_conf_fake"]
    gcs_client = storage.Client()
    bucket_name = video_path_prefix.split('/')[0]
    blob_prefix = "/".join(video_path_prefix.split('/')[1:])
    blobs = list(gcs_client.list_blobs(bucket_name, prefix=blob_prefix))

    if not blobs:
        print("  - WARNING: Could not find frames for augmentation probe. Skipping.")
        return

    # Download one frame
    blob = blobs[0]
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
        blob.download_to_filename(tmp_file.name)
        image_bgr = cv2.imread(tmp_file.name)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = defaultdict(list)
    augmentations = {
        "JPEG Compression": [A.ImageCompression(quality_lower=q, quality_upper=q) for q in range(95, 20, -5)],
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

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Model Robustness to Augmentations')

    # Compression plot
    strengths, probs = zip(*sorted(results["JPEG Compression"]))
    axes[0].plot(strengths, probs, 'o-')
    axes[0].set_title('JPEG Compression')
    axes[0].set_xlabel('JPEG Quality')
    axes[0].set_ylabel('Fake Probability')
    axes[0].invert_xaxis()
    axes[0].grid(True)

    # Blur plot
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
    # Get a real and a fake image for explanation
    gcs_client = storage.Client()
    images_to_explain = {
        "correct_fake": EXAMPLE_VIDEOS_FOR_PLOTS["high_conf_fake"],
        "potential_fp_real": EXAMPLE_VIDEOS_FOR_PLOTS["false_positive_candidate"]
    }

    image_tensors = []
    image_rgbs = []
    labels = []

    for name, path_prefix in images_to_explain.items():
        bucket_name = path_prefix.split('/')[0]
        blob_prefix = "/".join(path_prefix.split('/')[1:])
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

    # We need to define a prediction function for SHAP
    def f(x):
        return model({'image': x}, inference=True)['prob']

    # Use a part of the test set as the background distribution for the explainer
    # For speed, we'll just use a black image as a simplified background
    background = torch.zeros_like(batch[[0]])
    explainer = shap.DeepExplainer(f, background)
    shap_values = explainer.shap_values(batch)

    # SHAP output for images is (N, H, W, C), but our model is C, H, W. Need to adapt.
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
                        # default='weights/effort_clip_L14_trainOn_FaceForensic.pth', # A local alternative
                        help="GCS path to the model weights file (.pth).")
    parser.add_argument('--output_dir', type=str, default="model_analysis_output",
                        help="Directory to save all analysis plots and data files.")
    parser.add_argument('--skip_inference', action='store_true',
                        help="Skip the inference step and load data from output_dir.")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help="Number of parallel workers for data processing.")
    parser.add_argument('--quick_test', action='store_true',
                        help="Run on a very small sample for debugging purposes.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Define paths for saved data
    frame_data_path = output_dir / "all_frame_data.csv"
    clip_emb_path = output_dir / "all_clip_embeddings.npy"
    model_emb_path = output_dir / "all_model_embeddings.npy"

    if not args.skip_inference:
        # --- PHASE 1: DATA GATHERING & INFERENCE ---
        gcs_client = storage.Client()
        video_units = discover_and_sample_gcs_data(gcs_client, args.quick_test)

        # Load models
        print("\n--- Loading models for inference... ---")
        # 1. Main Detector Model
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp_weights:
            if args.model_gcs_path.startswith("gs://"):
                print(f"  Downloading detector model from {args.model_gcs_path}...")
                assert download_gcs_blob(args.model_gcs_path, Path(tmp_weights.name)), "Model download failed."
                model_weights_path = tmp_weights.name
            else:
                model_weights_path = args.model_gcs_path  # Assume local path

            # We need a dummy config. Use the one from app2.py's context
            config = {"model_name": "EffortDetector", "backbone": {"arch": "ViT-L/14"}}
            model = load_detector(config, model_weights_path)
            print("  Detector model loaded.")

        # 2. CLIP Model for embeddings
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        print("  CLIP model loaded.")

        # Prepare for processing
        transform = video_preprocessor._get_transform()
        fs = fsspec.filesystem('gcs')
        processing_func = partial(process_video_unit, fs=fs, transform=transform, model=model,
                                  clip_model=clip_model, clip_processor=clip_processor)

        all_results = []
        print(f"\n--- Starting inference on {len(video_units)} videos with {args.num_workers} workers... ---")
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_video = {executor.submit(processing_func, unit): unit for unit in video_units}
            for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(video_units)):
                all_results.extend(future.result())

        print("\n--- Inference complete. Saving data... ---")
        df_all_frames = pd.DataFrame(all_results)

        # Separate embeddings for saving in numpy format
        clip_embeddings = np.stack(df_all_frames.pop('clip_embedding').values)
        model_embeddings = np.stack(df_all_frames.pop('model_embedding').values)

        df_all_frames.to_csv(frame_data_path, index=False)
        np.save(clip_emb_path, clip_embeddings)
        np.save(model_emb_path, model_embeddings)
        print(f"  Saved frame data and embeddings to {output_dir}")

    else:
        print(f"--- Skipping inference. Loading pre-computed data from {output_dir}... ---")
        df_all_frames = pd.read_csv(frame_data_path)
        clip_embeddings = np.load(clip_emb_path)
        model_embeddings = np.load(model_emb_path)
        # Add embeddings back to dataframe for easier filtering
        df_all_frames['clip_embedding'] = list(clip_embeddings)
        df_all_frames['model_embedding'] = list(model_embeddings)
        print("  Data loaded successfully.")

    # Aggregate frame-level data to video-level for relevant analyses
    print("\n--- Aggregating frame data to video level... ---")
    df_video = df_all_frames.groupby(['video_path', 'dataset', 'label', 'method']).agg(
        video_prob_mean=('fake_prob', 'mean'),
        video_prob_median=('fake_prob', 'median'),
        frame_count=('fake_prob', 'count')
    ).reset_index()

    # --- RUN ALL ANALYSIS PHASES ---

    # Phase 2: Foundational Performance & Anomaly Investigation
    plot_roc_and_pr_curves(df_video, output_dir)
    plot_performance_by_method(df_video, output_dir)
    analyze_anomaly_domain_shift(df_all_frames, output_dir)

    # Phase 3: Behavior Deep Dive
    analyze_error_buckets(df_video, output_dir)
    plot_intra_video_consistency(df_all_frames, output_dir)

    # Phase 4: Robustness Probing
    if not args.skip_inference:  # These require a live model
        print("\n--- Model is loaded. Proceeding with live probes (Phase 4)... ---")
        # We need to reload the model if we skipped inference
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp_weights:
            if args.model_gcs_path.startswith("gs://"):
                download_gcs_blob(args.model_gcs_path, Path(tmp_weights.name))
                model_weights_path = tmp_weights.name
            else:
                model_weights_path = args.model_gcs_path
            config = {"model_name": "EffortDetector", "backbone": {"arch": "ViT-L/14"}}
            model = load_detector(config, model_weights_path)
            transform = video_preprocessor._get_transform()

            probe_augmentation_sensitivity(model, transform, output_dir)
            explain_with_shap(model, transform, output_dir)
    else:
        print("\n--- SKIPPING Phase 4 (Robustness Probes) as model was not loaded (`--skip_inference` was used). ---")
        print("    Re-run without `--skip_inference` to generate these plots.")

    print(f"\n✅ Analysis complete! All outputs are saved in '{output_dir}'.")


if __name__ == "__main__":
    main()
