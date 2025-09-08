# domain_analyzer.py

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
import torch
from google.cloud import storage
from tqdm import tqdm
from scipy import fft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import CLIPProcessor, CLIPModel

import albumentations as A

# --- Configuration ---
# GCS Buckets
TRAIN_BUCKET_NAME = "df40-frames-recropped-rfa85"
TEST_BUCKET_NAME = "deep-fake-test-10-08-25-frames-yolo-recropped-rfa85"

# Analysis Parameters
SAMPLE_SIZE = 2500  # Number of images to sample from each domain (train/test)
OUTPUT_DIR = "domain_analysis_output"

# --- Augmentation Pipeline for Verification ---
# This is the "proposed solution" we want to test.
# It's an aggressive pipeline designed to mimic social media artifacts.
tiktok_simulation_pipeline = A.Compose([
    A.Downscale(scale_min=0.5, scale_max=0.8, interpolation=cv2.INTER_AREA, p=0.7),
    A.ImageCompression(quality_lower=25, quality_upper=50, p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 9), p=0.5),
        A.GaussNoise(var_limit=(20.0, 70.0), p=0.5),
    ], p=0.8),
    A.CoarseDropout(max_holes=5, max_height=40, max_width=120,
                    min_height=20, min_width=40, fill_value=0, p=0.4)
])


# --- Core Analysis Functions ---

def get_low_level_stats(image_np: np.ndarray) -> dict:
    """Calculates a dictionary of low-level statistics for a single image."""
    if image_np is None or image_np.size == 0:
        return {}

    # Sharpness (Laplacian Variance)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Color Stats (Mean RGB)
    mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))

    # Frequency Domain Ratio
    try:
        f = fft.fft2(gray)
        fshift = fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(crow, ccol) * 0.1)  # Use 10% radius for low freq
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        low_freq_energy = np.sum(magnitude_spectrum[mask == 1])
        high_freq_energy = np.sum(magnitude_spectrum[mask == 0])
        freq_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
    except Exception:
        freq_ratio = 0.0

    return {
        "sharpness": sharpness,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "freq_ratio": freq_ratio
    }


def get_clip_embedding(image_np: np.ndarray, model, processor, device) -> np.ndarray:
    """Generates a CLIP feature embedding for a single image."""
    if image_np is None:
        return None
    # CLIP expects RGB images
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()


def process_single_frame(gcs_path: str, fs, clip_model, clip_processor, device) -> dict | None:
    """
    Downloads a single frame from GCS and computes all required stats and embeddings.
    This is the target function for our parallel executor.
    """
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()
        image_np_raw = np.frombuffer(image_bytes, np.uint8)
        image_np_bgr = cv2.imdecode(image_np_raw, cv2.IMREAD_COLOR)

        if image_np_bgr is None:
            tqdm.write(f"\n[Warning] Failed to decode image: {gcs_path}")
            return None

        # Standardize image size to ensure consistent array shapes
        image_np_bgr = cv2.resize(image_np_bgr, (224, 224), interpolation=cv2.INTER_AREA)

        # Tier 1 & 2: Low-level stats
        stats = get_low_level_stats(image_np_bgr)
        stats['gcs_path'] = gcs_path
        stats['image_np'] = image_np_bgr  # Keep image in memory for augmentation step later

        # Tier 3: Deep features
        embedding = get_clip_embedding(image_np_bgr, clip_model, clip_processor, device)
        stats['embedding'] = embedding

        return stats
    except Exception as e:
        tqdm.write(f"\n[Error] Could not process {gcs_path}: {e}")
        return None


def run_data_processing(train_paths, test_paths, num_workers):
    """Orchestrates the parallel processing of train and test data."""
    print("--- Starting Data Processing ---")
    print("Loading CLIP model (this may take a moment)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    print("Connecting to GCS...")
    fs = fsspec.filesystem('gcs')

    processing_func = partial(
        process_single_frame, fs=fs, clip_model=model, clip_processor=processor, device=device
    )

    all_results = []
    for domain, paths in [("train", train_paths), ("test", test_paths)]:
        print(f"Processing {len(paths)} frames from the '{domain}' domain...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(processing_func, paths), total=len(paths)))

        for res in results:
            if res:
                res['domain'] = domain
                all_results.append(res)

    # Separate stats, embeddings, and images for easier handling
    df_list = []
    embeddings_list = []
    images_list = []
    for res in all_results:
        # Don't add the large numpy arrays to the dataframe
        embedding = res.pop('embedding')
        image_np = res.pop('image_np')
        df_list.append(res)
        embeddings_list.append(embedding)
        images_list.append(image_np)

    df = pd.DataFrame(df_list)
    embeddings = np.array(embeddings_list)

    return df, embeddings, images_list


# --- Plotting Functions ---

def plot_kde_distributions(df: pd.DataFrame, df_aug: pd.DataFrame):
    """Generates and saves KDE plots for low-level statistics."""
    print("\n--- Generating KDE Distribution Plots ---")
    metrics_to_plot = ["sharpness", "freq_ratio", "mean_r", "mean_g", "mean_b"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 7))
        sns.kdeplot(data=df[df['domain'] == 'train'], x=metric, fill=True, alpha=0.5, label="Train (Original)")
        sns.kdeplot(data=df[df['domain'] == 'test'], x=metric, fill=True, alpha=0.5, label="Test (In-the-Wild)")
        sns.kdeplot(data=df_aug, x=metric, fill=True, alpha=0.6, label="Train (Augmented)",
                    linestyle='--', color='green')

        plt.title(f'Distribution of {metric.replace("_", " ").title()}', fontsize=16)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        save_path = os.path.join(OUTPUT_DIR, f"kde_{metric}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  - Saved KDE plot for {metric} to {save_path}")


def plot_tsne_visualization(embeddings, labels):
    """Performs t-SNE and saves the resulting 2D scatter plot."""
    print("\n--- Performing t-SNE and Generating Visualization ---")
    print(f"Running t-SNE on {len(embeddings)} embeddings. This can take a few minutes...")

    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)

    df_tsne = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_tsne['domain'] = labels

    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=df_tsne, x='x', y='y', hue='domain',
        palette={"Train (Original)": "blue", "Test (In-the-Wild)": "red", "Train (Augmented)": "green"},
        alpha=0.7, s=30
    )
    plt.title("t-SNE Visualization of CLIP Feature Space", fontsize=18)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Data Source")
    plt.grid(True, linestyle='--', linewidth=0.5)

    save_path = os.path.join(OUTPUT_DIR, "tsne_visualization.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  - Saved t-SNE plot to {save_path}")


# --- Main Controller ---

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize domain shift between datasets.")
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE, help="Number of frames to sample per dataset.")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of parallel workers.")
    parser.add_argument('--skip_processing', action='store_true',
                        help="Skip GCS processing and use existing local files for plotting.")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_path = os.path.join(OUTPUT_DIR, "domain_stats.csv")
    embed_path = os.path.join(OUTPUT_DIR, "domain_embeddings.npy")
    images_path = os.path.join(OUTPUT_DIR, "original_train_images.npy")

    if not args.skip_processing:
        # --- 1. Discover and Sample Files ---
        print("--- Discovering Files in GCS Buckets ---")
        client = storage.Client()
        train_blobs = list(client.list_blobs(TRAIN_BUCKET_NAME, max_results=args.sample_size * 5))
        test_blobs = list(client.list_blobs(TEST_BUCKET_NAME, max_results=args.sample_size * 5))

        train_paths = [f"gs://{TRAIN_BUCKET_NAME}/{b.name}" for b in random.sample(train_blobs, args.sample_size)]
        test_paths = [f"gs://{TEST_BUCKET_NAME}/{b.name}" for b in random.sample(test_blobs, args.sample_size)]
        print(f"Sampled {len(train_paths)} train paths and {len(test_paths)} test paths.")

        # --- 2. Process Data ---
        df, embeddings, images_list = run_data_processing(train_paths, test_paths, args.workers)

        # Save results to disk
        print(f"\n--- Saving Processed Data to '{OUTPUT_DIR}' ---")
        df.to_csv(stats_path, index=False)
        np.save(embed_path, embeddings)
        # Separate original train images for augmentation simulation
        train_images = [img for img, domain in zip(images_list, df['domain']) if domain == 'train']
        np.save(images_path, np.array(train_images))
        print("✅ Data saved.")

    else:
        print(f"--- Skipping processing, loading data from '{OUTPUT_DIR}' ---")
        if not all(os.path.exists(p) for p in [stats_path, embed_path, images_path]):
            print(
                "[Error] Cannot skip processing. One or more required files not found. Run without --skip_processing first.")
            return
        df = pd.read_csv(stats_path)
        embeddings = np.load(embed_path)
        train_images = np.load(images_path, allow_pickle=True)
        print("✅ Data loaded.")

    # --- 3. Augmentation Verification ---
    print("\n--- Simulating Augmentations on Train Data ---")
    augmented_stats_list = []
    for img in tqdm(train_images, desc="Applying augmentations"):
        augmented_img = tiktok_simulation_pipeline(image=img)['image']
        stats = get_low_level_stats(augmented_img)
        augmented_stats_list.append(stats)
    df_aug = pd.DataFrame(augmented_stats_list)

    # --- 4. Generate Plots ---
    plot_kde_distributions(df, df_aug)

    # Prepare data for t-SNE
    print("\n--- Preparing data for t-SNE plot ---")
    train_embeddings = embeddings[df['domain'] == 'train']
    test_embeddings = embeddings[df['domain'] == 'test']

    # We need to generate embeddings for the augmented images
    print("Generating CLIP embeddings for augmented images...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    aug_embeddings = []
    for img in tqdm(train_images, desc="Embedding augmented images"):
        augmented_img = tiktok_simulation_pipeline(image=img)['image']
        aug_embeddings.append(get_clip_embedding(augmented_img, model, processor, device))

    all_embeddings_for_plot = np.vstack([train_embeddings, test_embeddings, np.array(aug_embeddings).squeeze()])
    labels = ["Train (Original)"] * len(train_embeddings) + \
             ["Test (In-the-Wild)"] * len(test_embeddings) + \
             ["Train (Augmented)"] * len(aug_embeddings)

    plot_tsne_visualization(all_embeddings_for_plot, labels)

    print(f"\n--- Analysis Complete! All outputs are in the '{OUTPUT_DIR}' directory. ---")


if __name__ == "__main__":
    main()
