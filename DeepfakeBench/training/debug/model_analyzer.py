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
warnings.filterwarnings("ignore", category=UserWarning, module='shap')  # Suppress SHAP TF version warning

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

PROCESSING_BATCH_SIZE = 50

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
    """Generates and saves ROC and Precision-Recall curves for each dataset."""
    print("--- Phase 2a: Generating ROC and Precision-Recall curves for each dataset... ---")

    # We will analyze train and test datasets. Combine training buckets for a single 'train' ROC.
    df_video['dataset_group'] = df_video['dataset'].replace({'df40_train': 'train', 'collected_train': 'train'})

    for dataset_name, group_df in df_video.groupby('dataset_group'):
        print(f"  - Processing dataset: '{dataset_name}'")
        y_true = (group_df['label'] == 'fake').astype(int)
        y_score = group_df['video_prob_mean']

        if len(np.unique(y_true)) < 2:
            print(f"    - WARNING: Cannot compute ROC/PR for '{dataset_name}' with only one class. Skipping.")
            continue

        # ROC Curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Find optimal threshold based on G-Mean
        gmeans = np.sqrt(tpr * (1 - fpr))
        optimal_idx = np.argmax(gmeans)
        optimal_threshold = thresholds_roc[optimal_idx]

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                    label=f'Optimal Threshold (G-Mean) = {optimal_threshold:.3f}')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve - {dataset_name.title()} Set')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(output_dir / f"roc_curve_{dataset_name}.png")
        plt.close()

        print(f"    - ROC AUC: {roc_auc:.4f}")
        print(f"    - Optimal Threshold (from G-Mean): {optimal_threshold:.4f}")

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:0.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {dataset_name.title()} Set')
        plt.legend(loc="lower left")
        plt.grid(True)
        plt.savefig(output_dir / f"pr_curve_{dataset_name}.png")
        plt.close()


def plot_performance_by_method(df_video, output_dir):
    """
    Plots the AUC for methods with mixed labels and score distributions
    for fake-only methods on the test set.
    """
    print("--- Phase 2b: Analyzing performance per method... ---")
    test_df = df_video[df_video['dataset'] == 'test'].copy()

    method_aucs = {}
    fake_only_methods_df = []

    for method, group in test_df.groupby('method'):
        y_true = (group['label'] == 'fake').astype(int)

        if len(np.unique(y_true)) < 2:
            if np.all(y_true == 1):
                # This is a fake-only method
                fake_only_methods_df.append(group)
                avg_score = group['video_prob_mean'].mean()
                print(f"  - Method '{method}' (fake only): Avg. Score = {avg_score:.3f}")
            continue

        # This method has real and fake examples
        fpr, tpr, _ = roc_curve(y_true, group['video_prob_mean'])
        method_aucs[method] = auc(fpr, tpr)

    # Plot 1: AUC for methods with mixed labels
    if method_aucs:
        sorted_methods = sorted(method_aucs.items(), key=lambda x: x[1])
        plt.figure(figsize=(12, max(6, len(sorted_methods) * 0.5)))
        plt.barh([m[0] for m in sorted_methods], [m[1] for m in sorted_methods])
        plt.xlabel('AUC Score')
        plt.title('Model Performance (AUC) by Method on Test Set')
        plt.xlim([0, 1])
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / "performance_by_method_auc.png")
        plt.close()
    else:
        print("  - No test methods with mixed real/fake labels found for AUC plot.")

    # Plot 2: Score distribution for fake-only methods
    if fake_only_methods_df:
        df_fakes_only = pd.concat(fake_only_methods_df)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_fakes_only, x='video_prob_mean', y='method', orient='h')
        plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)')
        plt.xlabel('Predicted Fake Probability')
        plt.ylabel('Method')
        plt.title('Score Distributions for Fake-Only Methods on Test Set')
        plt.legend()
        plt.grid(axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / "performance_by_method_fakes_dist.png")
        plt.close()
    else:
        print("  - No fake-only test methods found for distribution plot.")


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
    tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42, max_iter=1000)
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


def plot_intra_video_consistency(df_all_frames, output_dir, example_videos: Dict[str, str]):
    """Plots frame-by-frame probabilities for a few example videos."""
    print("--- Phase 3b: Plotting intra-video prediction consistency... ---")
    for name, video_path_prefix in example_videos.items():
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

def probe_augmentation_sensitivity(model, transform, output_dir, example_videos: Dict[str, str]):
    """Tests how model predictions change with increasing augmentation strength."""
    print("--- Phase 4a: Probing model sensitivity to augmentations... ---")

    video_path_prefix = example_videos.get("high_conf_fake")
    if not video_path_prefix:
        print("  - WARNING: 'high_conf_fake' example not found for augmentation probe. Skipping.")
        return

    gcs_client = storage.Client()
    try:
        bucket_name, blob_prefix = video_path_prefix.split('/', 1)
        blobs = list(gcs_client.list_blobs(bucket_name, prefix=blob_prefix))
        if not blobs:
            print(f"  - WARNING: Could not find frames for augmentation probe at {video_path_prefix}. Skipping.")
            return
        blob = blobs[0]
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            blob.download_to_filename(tmp_file.name)
            image_bgr = cv2.imread(tmp_file.name)
        if image_bgr is None:
            print("  - WARNING: Failed to read image for augmentation probe. Skipping.")
            return
    except Exception as e:
        print(f"  - WARNING: Failed to download or process image for augmentation probe. Error: {e}. Skipping.")
        return

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = defaultdict(list)

    augmentations = {
        "JPEG Compression": {
            "augs_with_strength": [
                (A.ImageCompression(quality=q, p=1.0), q)  # Use 'quality' for older albumentations versions
                for q in range(95, 20, -5)
            ]
        },
        "Gaussian Blur": {
            "augs_with_strength": [
                (A.GaussianBlur(blur_limit=(s, s), p=1.0), s)
                for s in range(1, 15, 2)
            ]
        }
    }

    for aug_name, aug_info in augmentations.items():
        aug_list_with_strength = aug_info["augs_with_strength"]

        for aug, strength_val in aug_list_with_strength:
            try:
                aug_image = aug(image=image_rgb)['image']
                img_tensor = transform(aug_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    prob = model({'image': img_tensor}, inference=True)["prob"].item()
                results[aug_name].append((strength_val, prob))
            except Exception as e:
                print(f"  - WARNING: Augmentation '{aug_name}' failed at strength {strength_val}. Error: {e}")

    if not results:
        print("  - No augmentation results to plot.")
        return

    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 6), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Model Robustness to Augmentations')

    for ax, (aug_name, data) in zip(axes, results.items()):
        if not data: continue
        data.sort(key=lambda x: x[0])
        strengths, probs = zip(*data)
        ax.plot(strengths, probs, 'o-')
        ax.set_title(aug_name)
        ax.set_ylabel('Fake Probability')
        ax.grid(True)
        if "Compression" in aug_name:
            ax.set_xlabel('JPEG Quality')
            ax.invert_xaxis()
        elif "Blur" in aug_name:
            ax.set_xlabel('Blur Sigma')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / "augmentation_sensitivity.png")
    plt.close()


# --- SHAP WRAPPER FOR PYTORCH MODEL ---
class ShapModelWrapper(nn.Module):
    """
    Wrapper for a PyTorch model to make it compatible with SHAP's DeepExplainer.
    This wrapper handles the dictionary-based input/output and reshapes the
    output to be 2D for SHAP compatibility.
    """

    def __init__(self, model):
        super(ShapModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # The model expects a dictionary {'image': tensor}
        probs = self.model({'image': x}, inference=True)['prob']
        # SHAP expects a 2D output of shape [batch_size, num_classes] or [batch_size, 1].
        # Our model returns a 1D tensor [batch_size], so we unsqueeze it to [batch_size, 1].
        return probs.unsqueeze(1)


def explain_with_shap(model, transform, output_dir, example_videos: Dict[str, str]):
    """Generates SHAP explanations for a few key frames."""
    print("--- Phase 4b: Generating SHAP explanations for model decisions... ---")
    gcs_client = storage.Client()
    images_to_explain = {
        "correct_fake": example_videos.get("high_conf_fake"),
        "potential_fp_real": example_videos.get("false_positive_candidate")
    }
    images_to_explain = {k: v for k, v in images_to_explain.items() if v}
    if not images_to_explain:
        print("  - WARNING: No suitable example videos found for SHAP analysis. Skipping.")
        return

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

    # --- CORRECTED SHAP LOGIC ---
    wrapped_model = ShapModelWrapper(model)
    background = torch.zeros_like(batch[[0]])

    # Pass the wrapped model and background tensor to the explainer.
    explainer = shap.DeepExplainer(wrapped_model, background)
    # The explainer expects a PyTorch tensor for explanation.
    shap_values = explainer.shap_values(batch)

    # The output of shap_values for a single-output model is a single numpy array.
    # We need to reshape it for plotting if it comes out with an extra dimension.
    # The expected shape for shap.image_plot is (num_images, height, width, channels).
    if isinstance(shap_values, list):  # Should not happen for single output, but good practice
        shap_values = shap_values[0]

    # The SHAP values for a PyTorch model are often returned in (N, C, H, W) format.
    # We need to transpose them to (N, H, W, C) for plotting.
    if shap_values.ndim == 4 and shap_values.shape[1] in [1, 3]:  # Check if channels-first
        shap_values_transposed = np.transpose(shap_values, (0, 2, 3, 1))
    else:
        shap_values_transposed = shap_values

    shap.image_plot(shap_values_transposed, -np.array(image_rgbs), show=False)
    plt.suptitle("SHAP Explanations (Red pixels increase fake score)")
    plt.savefig(output_dir / "shap_explanation.png")
    plt.close()


# ======================================================================================
# ----------------------------- PHASE 5: DEEPFAKE METHOD ANALYSIS ----------------------
# ======================================================================================

def analyze_specific_method_distribution(df_all_frames, output_dir, method_name: str):
    """
    Analyzes a specific method from the test set by comparing its data distribution
    against the entire training set to identify potential domain shift.
    This is wrapped in a try-except block to prevent crashes.
    """
    clean_method_name = method_name.replace(" ", "_")  # for clean filenames
    print(f"--- Phase 5: Investigating '{method_name}' method distribution shift... ---")
    try:
        # 1. Isolate the data for comparison
        df_method = df_all_frames[
            (df_all_frames['dataset'] == 'test') &
            (df_all_frames['method'] == method_name)
            ].copy()
        df_method['source'] = f'{method_name} (Test)'

        # Combine both training buckets to form the training set distribution
        df_train = df_all_frames[
            df_all_frames['dataset'].isin(['df40_train', 'collected_train'])
        ].copy()
        df_train['source'] = 'Training Data'

        # 2. Robustness Check: Ensure we have data to analyze
        if df_method.empty or df_train.empty:
            print(
                f"  - WARNING: Could not find sufficient data for '{method_name}' or training sets. Skipping analysis.")
            return

        print(f"  - Comparing {len(df_method)} '{method_name}' frames against {len(df_train)} training frames.")
        df_comparison = pd.concat([df_method, df_train])

        # 3. Analyze low-level image statistics
        metrics_to_plot = ['sharpness', 'mean_r', 'mean_g', 'mean_b', 'fake_prob']
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=df_comparison, x=metric, hue='source', fill=True, common_norm=False, cut=0)
            plt.title(f"Distribution of '{metric.title()}' for {method_name.title()} vs. Training Data")
            plt.grid(True)
            plt.savefig(output_dir / f"{clean_method_name}_vs_train_{metric}_dist.png")
            plt.close()

        # 4. Analyze high-level CLIP embedding space
        if 'clip_embedding' not in df_comparison.columns:
            print(f"  - WARNING: 'clip_embedding' column not found. Skipping t-SNE analysis for {method_name}.")
            return

        print(f"  - Running t-SNE on CLIP embeddings for '{method_name}' vs. training data...")
        # To make t-SNE manageable, sample if the dataset is too large
        sample_size = 5000
        if len(df_comparison) > sample_size:
            print(
                f"    Dataset is large ({len(df_comparison)} samples), taking a random sample of {sample_size} for t-SNE.")
            # Stratified sampling to ensure both groups are represented
            df_comparison_sample = df_comparison.groupby('source', group_keys=False).apply(
                lambda x: x.sample(min(len(x), sample_size // 2)))
        else:
            df_comparison_sample = df_comparison

        embeddings = np.stack(df_comparison_sample['clip_embedding'].values)
        labels = df_comparison_sample['source'].values

        tsne = TSNE(n_components=2, perplexity=min(30, len(embeddings) - 1), random_state=42, max_iter=1000, init='pca')
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=labels, style=labels, s=40, alpha=0.7)
        plt.title(f't-SNE of CLIP Embeddings: {method_name.title()} (Test) vs. Training Data')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Data Source')
        plt.grid(True)
        plt.savefig(output_dir / f"{clean_method_name}_vs_train_tsne_clip.png")
        plt.close()

    except Exception as e:
        print(f"  - ERROR: An unexpected error occurred during the {method_name} analysis: {e}")
        print("  - Continuing with the rest of the script.")


# ======================================================================================
# ------------------------------------ MAIN SCRIPT -------------------------------------
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive analysis script for Effort-AIGI deepfake detector.")
    parser.add_argument('--model_gcs_path', type=str,
                        default='gs://training-job-outputs/best_checkpoints/w5cc8v6x/top_n_effort_20250910_ep1_auc0.9697_eer0.1055.pth',
                        # default='gs://training-job-outputs/best_checkpoints/k540e0ts/top_n_effort_20250910_ep2_auc0.9809_eer0.0800.pth',
                        help="GCS path to the model weights file (.pth).")
    parser.add_argument('--output_dir', type=str, default="model_analysis_output",
                        help="Directory to save all analysis plots and data files.")
    parser.add_argument('--skip_inference', action='store_true',
                        help="Skip the inference step and load data from output_dir.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers for data processing.")
    parser.add_argument('--use_discovery_cache', action='store_true',
                        help="Use cached GCS discovery results if available.")
    parser.add_argument('--quick_test', action='store_true', help="Run on a very small sample for debugging purposes.")
    # --- NEW ARGUMENT FOR FAST DEBUGGING ---
    parser.add_argument('--skip_to_probes', action='store_true',
                        help="Skip inference and analysis, load model and jump straight to live probes (Phase 4).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    chunk_output_dir = output_dir / "inference_chunks"

    # --- HANDLE --skip_to_probes FOR FAST DEBUGGING ---
    if args.skip_to_probes:
        print("--- Fast Debug Mode: Skipping to Phase 4 (Live Probes) ---")

        # 1. Load the model
        print("\n--- Loading model for probes... ---")
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

        # 2. Load aggregated data to find example videos
        print("\n--- Loading aggregated data to find example videos... ---")
        chunk_csv_files = sorted(glob.glob(str(chunk_output_dir / "chunk_*_data.csv")))
        if not chunk_csv_files:
            print("ERROR: No inference chunk data found. Cannot find examples for probes.")
            print("Please run the script without '--skip_inference' or '--skip_to_probes' at least once.")
            sys.exit(1)

        video_agg_data = []
        for csv_file in tqdm(chunk_csv_files, desc="Aggregating video stats"):
            df_chunk = pd.read_csv(csv_file)
            agg = df_chunk.groupby(['video_path', 'dataset', 'label', 'method']).agg(
                video_prob_mean=('fake_prob', 'mean')
            ).reset_index()
            video_agg_data.append(agg)
        df_video = pd.concat(video_agg_data, ignore_index=True)

        # 3. Find dynamic example videos
        print("\n--- Finding dynamic example videos for deep-dive analysis... ---")
        dynamic_example_videos = {}
        test_df_video = df_video[df_video['dataset'] == 'test']
        test_bucket_name = BUCKETS_TO_ANALYZE["test"]
        try:
            fakes = test_df_video[test_df_video['label'] == 'fake'].sort_values('video_prob_mean', ascending=False)
            if not fakes.empty:
                high_conf_path = fakes.iloc[0]['video_path']
                dynamic_example_videos["high_conf_fake"] = f"{test_bucket_name}/{high_conf_path}"
                print(f"  - Found high-confidence fake: {high_conf_path}")
        except Exception:
            pass
        try:
            reals = test_df_video[test_df_video['label'] == 'real'].sort_values('video_prob_mean', ascending=False)
            if not reals.empty:
                fp_cand_path = reals.iloc[0]['video_path']
                dynamic_example_videos["false_positive_candidate"] = f"{test_bucket_name}/{fp_cand_path}"
                print(f"  - Found FP candidate: {fp_cand_path}")
        except Exception:
            pass

        # 4. Run the probes
        print("\n--- Running live probes (Phase 4)... ---")
        transform = video_preprocessor._get_transform()
        probe_augmentation_sensitivity(model, transform, output_dir, example_videos=dynamic_example_videos)
        explain_with_shap(model, transform, output_dir, example_videos=dynamic_example_videos)
        print(f"\n✅ Probe analysis complete! Outputs saved in '{output_dir}'.")
        return  # Exit the script

    # --- NORMAL EXECUTION FLOW ---
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

        print(
            f"\n--- Starting resumable inference on {len(video_units)} videos in batches of {PROCESSING_BATCH_SIZE}... ---")
        chunk_output_dir.mkdir(exist_ok=True)
        video_chunks = [video_units[i:i + PROCESSING_BATCH_SIZE] for i in
                        range(0, len(video_units), PROCESSING_BATCH_SIZE)]

        for i, chunk in enumerate(video_chunks):
            chunk_csv_path = chunk_output_dir / f"chunk_{i}_data.csv"
            chunk_clip_path = chunk_output_dir / f"chunk_{i}_clip.npy"
            chunk_model_path = chunk_output_dir / f"chunk_{i}_model.npy"
            if chunk_csv_path.exists() and chunk_clip_path.exists() and chunk_model_path.exists():
                print(f"  Chunk {i + 1}/{len(video_chunks)} already processed. Skipping.")
                continue

            print(f"  Processing chunk {i + 1}/{len(video_chunks)} ({len(chunk)} videos)...")
            chunk_frame_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_to_video = {executor.submit(processing_func, unit): unit for unit in chunk}
                for future in tqdm(concurrent.futures.as_completed(future_to_video), total=len(chunk),
                                   desc=f"Chunk {i + 1}"):
                    try:
                        video_results = future.result()
                        if video_results:
                            chunk_frame_results.extend(video_results)
                    except Exception as exc:
                        video_unit = future_to_video[future]
                        print(f"\nWARNING: Video {video_unit['video_path']} generated an exception: {exc}")

            if not chunk_frame_results:
                print(f"  Chunk {i + 1} yielded no results. Moving to next.")
                continue

            df_chunk = pd.DataFrame(chunk_frame_results)
            clip_embeddings_chunk = np.stack(df_chunk.pop('clip_embedding').values)
            model_embeddings_chunk = np.stack(df_chunk.pop('model_embedding').values)
            df_chunk.to_csv(chunk_csv_path, index=False)
            np.save(chunk_clip_path, clip_embeddings_chunk)
            np.save(chunk_model_path, model_embeddings_chunk)
            print(f"  Chunk {i + 1} saved successfully.")
        print("\n--- Inference complete. All chunks are processed and saved. ---")

    # --- MEMORY-EFFICIENT CONSOLIDATION & ANALYSIS ---
    print(f"--- Consolidating data from chunk files in {chunk_output_dir} for analysis... ---")
    chunk_csv_files = sorted(glob.glob(str(chunk_output_dir / "chunk_*_data.csv")))
    if not chunk_csv_files:
        print("ERROR: No inference chunk data found. Cannot proceed with analysis.")
        print("Please run the script without '--skip_inference' first.")
        sys.exit(1)

    print("\n--- Aggregating frame data to video level... ---")
    video_agg_data = []
    for csv_file in tqdm(chunk_csv_files, desc="Aggregating video stats"):
        df_chunk = pd.read_csv(csv_file)
        agg = df_chunk.groupby(['video_path', 'dataset', 'label', 'method']).agg(
            video_prob_mean=('fake_prob', 'mean'),
            video_prob_median=('fake_prob', 'median'),
            frame_count=('fake_prob', 'count')
        ).reset_index()
        video_agg_data.append(agg)
    df_video = pd.concat(video_agg_data, ignore_index=True)
    print("  Video-level aggregation complete.")

    plot_roc_and_pr_curves(df_video, output_dir)
    plot_performance_by_method(df_video, output_dir)
    analyze_error_buckets(df_video, output_dir)

    print("\n--- Loading all frame data for deep-dive analysis... ---")
    all_frame_dfs = []
    required_cols = ['video_path', 'dataset', 'label', 'method', 'frame_gcs_path',
                     'fake_prob', 'sharpness', 'mean_r', 'mean_g', 'mean_b']
    for csv_file in tqdm(chunk_csv_files, desc="Loading frame data"):
        all_frame_dfs.append(pd.read_csv(csv_file, usecols=lambda c: c in required_cols))
    df_all_frames = pd.concat(all_frame_dfs, ignore_index=True)

    try:
        print("  Loading CLIP embeddings for domain shift analysis...")
        all_clip_embeddings = [np.load(f, allow_pickle=True, mmap_mode='r') for f in
                               sorted(glob.glob(str(chunk_output_dir / "chunk_*_clip.npy")))]
        clip_embeddings = np.concatenate(all_clip_embeddings, axis=0)
        if len(df_all_frames) == len(clip_embeddings):
            df_all_frames['clip_embedding'] = list(clip_embeddings)
            print("  CLIP embeddings attached successfully.")
        else:
            print(
                f"WARNING: Mismatch! DataFrame has {len(df_all_frames)} rows, but found {len(clip_embeddings)} CLIP embeddings.")
    except Exception as e:
        print(f"WARNING: Failed to load or attach embeddings. Error: {e}.")
    print("  Frame-level data loaded.")

    print("\n--- Finding dynamic example videos for deep-dive analysis... ---")
    dynamic_example_videos = {}
    test_df_video = df_video[df_video['dataset'] == 'test']
    test_bucket_name = BUCKETS_TO_ANALYZE["test"]
    try:
        fakes = test_df_video[test_df_video['label'] == 'fake'].sort_values('video_prob_mean', ascending=False)
        if not fakes.empty:
            high_conf_path, low_conf_path = fakes.iloc[0]['video_path'], fakes.iloc[-1]['video_path']
            dynamic_example_videos["high_conf_fake"] = f"{test_bucket_name}/{high_conf_path}"
            dynamic_example_videos["low_conf_fake"] = f"{test_bucket_name}/{low_conf_path}"
            print(f"  - Found high-confidence fake: {high_conf_path} (Score: {fakes.iloc[0]['video_prob_mean']:.3f})")
            print(f"  - Found low-confidence fake: {low_conf_path} (Score: {fakes.iloc[-1]['video_prob_mean']:.3f})")
    except Exception as e:
        print(f"  - WARNING: Could not find fake examples. {e}")
    try:
        reals = test_df_video[test_df_video['label'] == 'real'].sort_values('video_prob_mean', ascending=False)
        if not reals.empty:
            fp_cand_path = reals.iloc[0]['video_path']
            dynamic_example_videos["false_positive_candidate"] = f"{test_bucket_name}/{fp_cand_path}"
            print(f"  - Found FP candidate: {fp_cand_path} (Score: {reals.iloc[0]['video_prob_mean']:.3f})")
    except Exception as e:
        print(f"  - WARNING: Could not find real examples for FP analysis. {e}")

    analyze_anomaly_domain_shift(df_all_frames, output_dir)
    plot_intra_video_consistency(df_all_frames, output_dir, example_videos=dynamic_example_videos)
    analyze_specific_method_distribution(df_all_frames, output_dir, method_name="tiktok")
    analyze_specific_method_distribution(df_all_frames, output_dir, method_name="in the wild social media")

    model_is_loaded = 'model' in locals() or 'model' in globals()
    if not args.skip_inference and model_is_loaded:
        print("\n--- Model is loaded. Proceeding with live probes (Phase 4)... ---")
        transform = video_preprocessor._get_transform()
        probe_augmentation_sensitivity(model, transform, output_dir, example_videos=dynamic_example_videos)
        explain_with_shap(model, transform, output_dir, example_videos=dynamic_example_videos)
    else:
        print("\n--- SKIPPING Phase 4 (Robustness Probes) as model was not loaded (`--skip_inference` was used). ---")

    print(f"\n✅ Analysis complete! All outputs are saved in '{output_dir}'.")


if __name__ == "__main__":
    main()
