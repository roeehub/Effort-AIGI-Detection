# domain_analyzer.py

import os
import random
import argparse
from pathlib import Path
import concurrent.futures
from functools import partial
from collections import defaultdict

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
TARGET_SAMPLES_PER_GROUP = 600
OUTPUT_DIR = "domain_analysis_output_v3_balanced"

# --- Augmentation Pipeline to Test ---
# revised_augmentation_pipeline = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.OneOf([
#         A.UnsharpMask(blur_limit=(3, 7), sigma_limit=0.5, alpha=(0.5, 1.0), threshold=10, p=0.7),
#         A.Compose([
#             A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_AREA, p=0.5),
#             A.ImageCompression(quality_lower=50, quality_upper=80, p=1.0)
#         ], p=0.3)
#     ], p=0.9),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
#     A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
# ])

# This is the pipeline designed to be the "final" version for maximum robustness.
revised_augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),

    # Step 1: CALIBRATED Quality Transformation
    # REASONING: We re-introduce NoOp. This is the most critical change.
    # It allows a fraction of the original, softer images to remain,
    # creating a mixture of soft and sharp images that should broaden
    # the distribution and pull the average back into alignment with the test set.
    A.OneOf([
        # We keep the aggressive sharpening, as it's good at creating the high-end values.
        A.UnsharpMask(blur_limit=(3, 9), sigma_limit=(0.3, 0.7), alpha=(0.5, 1.0), threshold=10, p=0.7),

        # We need a path for images to NOT be sharpened.
        A.NoOp(p=0.3),

    ], p=1.0),  # The OneOf itself still runs every time.

    # Step 2: Realistic Compression & Noise (Keep as is)
    A.OneOf([
        A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5),
        A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
    ], p=0.6),

    # Step 3: GENTLE Color Augmentation (This part is solved)
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
])


# --- Core Analysis Functions (Unchanged) ---
def get_low_level_stats(image_np: np.ndarray) -> dict:
    if image_np is None or image_np.size == 0: return {}
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))
    try:
        f = fft.fft2(gray)
        fshift = fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        radius = int(min(crow, ccol) * 0.1)
        mask = np.zeros((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), radius, 1, thickness=-1)
        low_freq_energy = np.sum(magnitude_spectrum[mask == 1])
        high_freq_energy = np.sum(magnitude_spectrum[mask == 0])
        freq_ratio = high_freq_energy / low_freq_energy if low_freq_energy > 0 else 0
    except Exception:
        freq_ratio = 0.0
    return {"sharpness": sharpness, "mean_r": mean_r, "mean_g": mean_g, "mean_b": mean_b, "freq_ratio": freq_ratio}


def get_clip_embedding(image_np: np.ndarray, model, processor, device) -> np.ndarray:
    if image_np is None: return None
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy().flatten()


def process_single_frame(path_info: dict, fs, clip_model, clip_processor, device) -> dict | None:
    gcs_path = path_info['gcs_path']
    try:
        with fs.open(gcs_path, 'rb') as f:
            image_bytes = f.read()
        image_np_raw = np.frombuffer(image_bytes, np.uint8)
        image_np_bgr = cv2.imdecode(image_np_raw, cv2.IMREAD_COLOR)
        if image_np_bgr is None: return None
        image_np_bgr = cv2.resize(image_np_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        stats = get_low_level_stats(image_np_bgr)
        stats.update(path_info)
        stats['image_np'] = image_np_bgr
        stats['embedding'] = get_clip_embedding(image_np_bgr, clip_model, clip_processor, device)
        return stats
    except Exception:
        return None


# --- CORRECTED BALANCED Sampling Function ---
def discover_and_sample_balanced(gcs_client, bucket_name, domain_name, target_samples_per_group):
    print(f"--- Discovering and sampling from {bucket_name} for domain '{domain_name}' ---")
    all_sampled_paths = []

    for label in ['real', 'fake']:
        print(f"  Scanning for label: '{label}'...")
        prefix = f"{label}/"

        # CORRECTED LOGIC: List all blobs flatly and infer methods from paths
        blobs = list(gcs_client.list_blobs(bucket_name, prefix=prefix))

        paths_by_method = defaultdict(list)
        for blob in blobs:
            # path is like 'label/method/video/frame.png'
            parts = blob.name.split('/')
            if len(parts) > 2:  # Ensure there is a method name
                method_name = parts[1]
                paths_by_method[method_name].append(blob)

        num_methods = len(paths_by_method)
        if num_methods == 0:
            print(f"    [Warning] No methods found under '{prefix}'. Skipping.")
            continue

        samples_per_method = int(np.ceil(target_samples_per_group / num_methods))
        print(f"    Found {num_methods} methods. Aiming for {samples_per_method} samples per method.")

        for method_name, method_blobs in paths_by_method.items():
            num_to_sample = min(samples_per_method, len(method_blobs))
            if num_to_sample == 0: continue

            selected_blobs = random.sample(method_blobs, num_to_sample)
            for blob in selected_blobs:
                all_sampled_paths.append({
                    'gcs_path': f"gs://{bucket_name}/{blob.name}",
                    'domain': domain_name, 'label': label, 'method': method_name
                })
    return all_sampled_paths


def run_data_processing(all_paths, num_workers):
    print("\n--- Starting Data Processing ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model to device: {device}")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    fs = fsspec.filesystem('gcs')
    processing_func = partial(process_single_frame, fs=fs, clip_model=model, clip_processor=processor, device=device)
    results = list(tqdm(concurrent.futures.ThreadPoolExecutor(max_workers=num_workers).map(processing_func, all_paths),
                        total=len(all_paths)))
    results = [res for res in results if res]
    df_list, embeddings_list, images_list, metadata_list = [], [], [], []
    for res in results:
        embeddings_list.append(res.pop('embedding'))
        images_list.append(res.pop('image_np'))
        df_list.append(res)
        metadata_list.append({'domain': res['domain'], 'label': res['label']})
    return pd.DataFrame(df_list), np.array(embeddings_list), images_list, metadata_list


def plot_kde_distributions(df, df_aug, quick_test=False):
    """Generates and saves KDE plots with enhanced visuals."""
    print("\n--- Generating KDE Distribution Plots (Detailed View) ---")
    metrics = ["sharpness", "freq_ratio", "mean_r", "mean_g", "mean_b"]

    df['plot_label'] = df['domain'] + " (" + df['label'] + ")"
    if not df_aug.empty:
        df_aug['plot_label'] = df_aug['domain'] + " (" + df_aug['label'] + ")"

    # --- ENHANCED VISUALS ---
    palette = {
        "Train (real)": "#0072B2",  # Strong Blue
        "Train (fake)": "#56B4E9",  # Lighter Blue
        "Test (real)": "#D55E00",  # Strong Orange/Red
        "Test (fake)": "#E69F00",  # Lighter Orange
    }
    hue_order = ['Train (real)', 'Train (fake)', 'Test (real)', 'Test (fake)']

    for metric in metrics:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        sns.kdeplot(
            data=df, x=metric, hue='plot_label', hue_order=hue_order,
            palette=palette, fill=True, alpha=0.3, linewidth=2.0, ax=ax  # Increased line width
        )

        if not df_aug.empty:
            sns.kdeplot(
                data=df_aug, x=metric, hue='plot_label', hue_order=hue_order,
                palette=palette, fill=False, alpha=0.9, linewidth=2.5, linestyle='--', ax=ax
            )
        # --- END ENHANCEMENTS ---

        plt.title(f'Distribution of {metric.replace("_", " ").title()} (Detailed View)', fontsize=16)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            from collections import OrderedDict
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title='Data Source')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        if quick_test:
            plt.show()
        else:
            save_path = os.path.join(OUTPUT_DIR, f"kde_{metric}_detailed.png")
            plt.savefig(save_path)
            print(f"  - Saved detailed KDE plot for {metric} to {save_path}")
        plt.close()


def plot_kde_distributions_by_split(df, df_aug, quick_test=False):
    """
    Generates and saves KDE plots, grouping by data split (Train, Test, Train-Augmented).
    This combines 'real' and 'fake' to show overall domain distributions.
    """
    print("\n--- Generating KDE Distribution Plots (Grouped by Data Split) ---")
    metrics = ["sharpness", "freq_ratio", "mean_r", "mean_g", "mean_b"]

    # --- Data Preparation: Create simplified labels ---
    df['plot_label_split'] = df['domain']  # 'Train' or 'Test'
    if not df_aug.empty:
        df_aug['plot_label_split'] = 'Train - Augmented'

    # Combine dataframes for easier plotting
    combined_df = pd.concat([df, df_aug], ignore_index=True)

    # --- Visual Enhancements for Readability ---
    palette = {
        "Train": "#0072B2",  # A strong blue
        "Test": "#D55E00",  # A strong orange/red
        "Train - Augmented": "#009E73"  # A strong green
    }
    hue_order = ["Train", "Test", "Train - Augmented"]

    for metric in metrics:
        plt.figure(figsize=(14, 8))
        ax = plt.gca()

        sns.kdeplot(
            data=combined_df, x=metric, hue='plot_label_split',
            hue_order=hue_order, palette=palette,
            fill=True, alpha=0.3, linewidth=2.0, ax=ax
        )

        plt.title(f'Distribution of {metric.replace("_", " ").title()} (Grouped by Split)', fontsize=16)

        # Use the robust legend fix from before
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            from collections import OrderedDict
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title='Data Split')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        if quick_test:
            plt.show()
        else:
            save_path = os.path.join(OUTPUT_DIR, f"kde_{metric}_by_split.png")
            plt.savefig(save_path)
            print(f"  - Saved grouped KDE plot for {metric} to {save_path}")
        plt.close()


def plot_tsne_visualization(embeddings_2d, labels, quick_test=False):
    """
    Generates and saves the t-SNE scatter plot with enhanced visuals for clarity.
    """
    print("\n--- Generating t-SNE Visualization (Detailed View) ---")

    df_tsne = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_tsne['Data Source'] = labels

    # --- ENHANCED VISUALS ---
    # A clearer palette with better color distinction
    palette = {
        "Train (Real)": "#0072B2",  # Strong Blue
        "Train (Fake)": "#56B4E9",  # Lighter Blue
        "Test (Real)": "#D55E00",  # Strong Orange/Red
        "Test (Fake)": "#E69F00",  # Lighter Orange
        "Train (Real) - Augmented": "#009E73",  # Strong Green
        "Train (Fake) - Augmented": "#02FAC6"  # Lighter Green (Cyan)
    }
    markers = {
        "Train (Real)": "o", "Train (Fake)": "o",
        "Test (Real)": "s", "Test (Fake)": "s",
        "Train (Real) - Augmented": "X", "Train (Fake) - Augmented": "X"
    }

    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        data=df_tsne,
        x='x', y='y',
        hue='Data Source',
        style='Data Source',
        palette=palette,
        markers=markers,
        alpha=0.9,
        s=50,  # Increased marker size
        edgecolor='black',  # Add outline to markers
        linewidth=0.5
    )
    # --- END ENHANCEMENTS ---

    plt.title("t-SNE Visualization of CLIP Feature Space (Detailed View)", fontsize=18)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Data Source")
    plt.grid(True, linestyle='--', linewidth=0.5)

    if quick_test:
        plt.show()
    else:
        save_path = os.path.join(OUTPUT_DIR, "tsne_visualization_detailed.png")
        plt.savefig(save_path, dpi=300)
        print(f"  - Saved detailed t-SNE plot to {save_path}")
    plt.close()


def plot_tsne_visualization_by_split(embeddings_2d, labels, quick_test=False):
    """
    Generates and saves a simplified t-SNE scatter plot, grouping by data split.
    This version combines 'real' and 'fake' labels for a clearer high-level view
    and uses enhanced visual styles for better readability.
    """
    print("\n--- Generating t-SNE Visualization (Grouped by Data Split) ---")

    # --- Data Preparation: Simplify labels ---
    simplified_labels = []
    for label in labels:
        if "Augmented" in label:
            simplified_labels.append("Train - Augmented")
        elif "Train" in label:
            simplified_labels.append("Train")
        elif "Test" in label:
            simplified_labels.append("Test")
        else:
            simplified_labels.append("Unknown")

    df_tsne = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_tsne['Data Split'] = simplified_labels

    # --- Visual Enhancements for Readability ---
    palette = {
        "Train": "#0072B2",  # A strong blue
        "Test": "#D55E00",  # A strong orange/red
        "Train - Augmented": "#009E73"  # A strong green
    }
    markers = {
        "Train": "o",
        "Test": "s",
        "Train - Augmented": "X"
    }
    # Ensure augmented points are plotted on top of original training points
    hue_order = ["Test", "Train", "Train - Augmented"]

    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        data=df_tsne,
        x='x', y='y',
        hue='Data Split',
        style='Data Split',
        hue_order=hue_order,
        palette=palette,
        markers=markers,
        alpha=0.9,
        s=50,  # Increased marker size
        edgecolor='black',  # Add outline to markers
        linewidth=0.5
    )

    plt.title("t-SNE Visualization of CLIP Feature Space (Grouped by Data Split)", fontsize=18)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Data Split")
    plt.grid(True, linestyle='--', linewidth=0.5)

    if quick_test:
        plt.show()
    else:
        save_path = os.path.join(OUTPUT_DIR, "tsne_visualization_by_split.png")
        plt.savefig(save_path, dpi=300)  # Higher DPI for better quality
        print(f"  - Saved grouped t-SNE plot to {save_path}")
    plt.close()


# --- Main Controller (Unchanged) ---
def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize domain shift between datasets.")
    parser.add_argument('--target_samples_per_group', type=int, default=TARGET_SAMPLES_PER_GROUP,
                        help="Target samples for each group (e.g., Train-Real).")
    parser.add_argument('--workers', type=int, default=os.cpu_count(), help="Number of parallel workers.")
    parser.add_argument('--skip_processing', action='store_true', help="Skip GCS processing and use local files.")
    parser.add_argument('--quick_test', action='store_true',
                        help="Run with a small sample size and show plots instead of saving.")
    args = parser.parse_args()

    target_samples = 10 if args.quick_test else args.target_samples_per_group
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = {k: os.path.join(OUTPUT_DIR, f) for k, f in
             {"stats": "domain_stats.csv", "embed": "domain_embeddings.npy", "images": "original_images.npy",
              "meta": "metadata.npy"}.items()}

    if not args.skip_processing:
        client = storage.Client()
        train_paths = discover_and_sample_balanced(client, TRAIN_BUCKET_NAME, "Train", target_samples)
        test_paths = discover_and_sample_balanced(client, TEST_BUCKET_NAME, "Test", target_samples)
        all_paths = train_paths + test_paths
        if not all_paths: print("[Error] No files were sampled. Exiting."); return

        print("\n--- Sampling Summary ---")
        counts = defaultdict(int)
        for p in all_paths: counts[f"{p['domain']} ({p['label']})"] += 1
        for group, count in sorted(counts.items()): print(f"  - {group}: {count} samples")
        print("------------------------")

        df, embeddings, images_list, metadata_list = run_data_processing(all_paths, args.workers)
        print(f"\n--- Saving Processed Data to '{OUTPUT_DIR}' ---")
        df.to_csv(paths['stats'], index=False)
        np.save(paths['embed'], embeddings)
        np.save(paths['images'], np.array(images_list))
        np.save(paths['meta'], np.array(metadata_list))
        print("✅ Data saved.")
    else:
        print(f"--- Skipping processing, loading data from '{OUTPUT_DIR}' ---")
        df, embeddings, images_list, metadata_list = pd.read_csv(paths['stats']), np.load(paths['embed']), np.load(
            paths['images'], allow_pickle=True), np.load(paths['meta'], allow_pickle=True)
        print("✅ Data loaded.")

    # =========================================================================
    # --- REORDERED WORKFLOW: All slow computations are performed upfront ---
    # =========================================================================
    print("\n--- Performing ALL expensive computations upfront ---")

    # --- Step 1: Augmentation Simulation (Medium-Slow) ---
    print("--> Simulating Augmentations and computing new embeddings...")
    train_real_imgs = [img for img, meta in zip(images_list, metadata_list) if
                       meta['domain'] == 'Train' and meta['label'] == 'real']
    train_fake_imgs = [img for img, meta in zip(images_list, metadata_list) if
                       meta['domain'] == 'Train' and meta['label'] == 'fake']
    aug_stats_list, aug_embeddings, aug_metadata = [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    for label, images in [('real', train_real_imgs), ('fake', train_fake_imgs)]:
        print(f"Applying augmentations and embedding for Train ({label})...")
        for img in tqdm(images):
            augmented_img = revised_augmentation_pipeline(image=img)['image']
            aug_stats_list.append(get_low_level_stats(augmented_img))
            aug_embeddings.append(get_clip_embedding(augmented_img, clip_model, clip_processor, device))
            aug_metadata.append({'domain': 'Train', 'label': label})

    df_aug = pd.DataFrame(aug_stats_list)
    if not df_aug.empty:
        df_aug['domain'] = [m['domain'] for m in aug_metadata]
        df_aug['label'] = [m['label'] for m in aug_metadata]

    # --- Step 2: t-SNE Calculation (Very Slow) ---
    print("\n--> Preparing data and computing t-SNE transformation...")
    labels = [f"{meta['domain']} ({meta['label'].title()})" for meta in metadata_list]
    all_embeddings_for_plot = embeddings
    aug_labels = [f"Train ({meta['label'].title()}) - Augmented" for meta in aug_metadata]
    if aug_embeddings:
        all_embeddings_for_plot = np.vstack([embeddings, np.array(aug_embeddings).squeeze()])
        labels.extend(aug_labels)

    print(f"Running t-SNE on {len(all_embeddings_for_plot)} total embeddings. This can take a while...")
    tsne_model = TSNE(n_components=2, perplexity=min(30, len(all_embeddings_for_plot) - 1), max_iter=1000,
                      random_state=42, verbose=1)
    embeddings_2d = tsne_model.fit_transform(all_embeddings_for_plot)

    print("\n--- ✅ All computations complete. Starting plotting. ---")

    # =========================================================================
    # --- All fast plotting happens last for easy debugging ---
    # =========================================================================

    # Plot KDEs
    plot_kde_distributions(df, df_aug, args.quick_test)

    # Plot t-SNE using the pre-computed data
    plot_tsne_visualization(embeddings_2d, labels, args.quick_test)

    # --- ADD THE CALLS TO THE NEW PLOTTING FUNCTIONS HERE ---
    print("\n--- Generating simplified plots by data split ---")
    plot_kde_distributions_by_split(df.copy(), df_aug.copy(), args.quick_test)  # Use .copy() to avoid warnings
    plot_tsne_visualization_by_split(embeddings_2d, labels, args.quick_test)
    # --- END OF ADDITION ---

    print(f"\n--- Analysis Complete! Outputs are in the '{OUTPUT_DIR}' directory. ---")


if __name__ == "__main__":
    main()
