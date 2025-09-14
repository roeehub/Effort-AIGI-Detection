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
from albumentations.core.transforms_interface import ImageOnlyTransform  # noqa
from albumentations.core.transforms_interface import BasicTransform  # noqa

# GCS Buckets
TRAIN_BUCKET_NAME = "df40-frames-recropped-rfa85"
SECOND_TRAIN_BUCKET_NAME = "effort-collected-data"
TEST_BUCKET_NAME = "deep-fake-test-10-08-25-frames-v2"

# Analysis Parameters
TARGET_SAMPLES_PER_GROUP = 600
OUTPUT_DIR = "domain_analysis_output"


# --- Custom Transforms ---
class CustomUnsharpMask(ImageOnlyTransform):
    def __init__(self, blur_limit=(3, 9), alpha=(0.5, 1.0), threshold=10, always_apply=False, p=0.5):
        super(CustomUnsharpMask, self).__init__(always_apply, p)
        if blur_limit[0] % 2 != 0 and blur_limit[1] % 2 != 0:
            self.blur_limit = blur_limit
        else:
            raise ValueError("blur_limit values must be odd integers.")
        self.alpha, self.threshold = alpha, threshold

    def apply(self, image, **params):
        ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 2, 2)
        current_alpha = random.uniform(self.alpha[0], self.alpha[1])
        blurred, image_float = cv2.GaussianBlur(image, (ksize, ksize), 0), image.astype(np.float32)
        mask = image_float - blurred.astype(np.float32)
        if self.threshold > 0:
            apply_condition = np.abs(mask) >= self.threshold
            sharpened_mask = mask * current_alpha
            image_float[apply_condition] += sharpened_mask[apply_condition]
        else:
            image_float += mask * current_alpha
        return np.clip(image_float, 0, 255).astype(np.uint8)


# --- Augmentation Pipelines ---
AUG_PIPELINE_V3 = A.Compose(
    [A.HorizontalFlip(p=0.5), CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.5, 1.0), threshold=10, p=0.7), A.OneOf(
        [A.ImageCompression(quality_lower=50, quality_upper=90, p=0.5), A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),
         A.GaussianBlur(blur_limit=(3, 7), p=0.2)], p=0.6),
     A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
     A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4)])
AUG_PIPELINE_V4 = A.Compose(
    [A.HorizontalFlip(p=0.5), CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.6, 1.2), threshold=10, p=0.75), A.OneOf(
        [A.ImageCompression(quality_lower=45, quality_upper=90, p=0.5), A.GaussNoise(var_limit=(10.0, 65.0), p=0.3),
         A.GaussianBlur(blur_limit=(3, 7), p=0.2)], p=0.7),
     A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
     A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=12, p=0.45)])

# A targeted pipeline designed to make 'Train-Primary' data look like 'Test' data.
AUG_PIPELINE_V6_FOR_PRIMARY = A.Compose([
    A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), A.GaussianBlur(blur_limit=(3, 5), p=0.5)], p=0.3),
    CustomUnsharpMask(blur_limit=(3, 9), alpha=(0.7, 1.5), threshold=10, p=0.9),
    A.ImageCompression(quality_lower=40, quality_upper=85, p=0.9),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
])

# A very light pipeline for the 'Train-Effort' data, which is already good.
AUG_PIPELINE_LIGHT_FOR_EFFORT = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
])


# --- Core Analysis Functions (Unchanged) ---
def get_low_level_stats(image_np: np.ndarray) -> dict:
    if image_np is None or image_np.size == 0: return {}
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_b, mean_g, mean_r = np.mean(image_np, axis=(0, 1))
    try:
        f = fft.fft2(gray);
        fshift = fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        rows, cols = gray.shape;
        crow, ccol = rows // 2, cols // 2
        radius = int(min(crow, ccol) * 0.1);
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
    with torch.no_grad(): embedding = model.get_image_features(**inputs)
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
        stats.update(path_info);
        stats['image_np'] = image_np_bgr
        stats['embedding'] = get_clip_embedding(image_np_bgr, clip_model, clip_processor, device)
        return stats
    except Exception:
        return None


def discover_and_sample_balanced(gcs_client, bucket_name, domain_name, target_samples_per_group):
    print(f"--- Discovering and sampling from {bucket_name} for domain '{domain_name}' ---")
    all_sampled_paths = []
    for label in ['real', 'fake']:
        print(f"  Scanning for label: '{label}'...")
        prefix, blobs = f"{label}/", list(gcs_client.list_blobs(bucket_name, prefix=prefix))
        paths_by_method = defaultdict(list)
        for blob in blobs:
            parts = blob.name.split('/')
            if len(parts) > 2: paths_by_method[parts[1]].append(blob)
        num_methods = len(paths_by_method)
        if num_methods == 0: print(f"    [Warning] No methods found under '{prefix}'. Skipping."); continue
        samples_per_method = int(np.ceil(target_samples_per_group / num_methods))
        print(f"    Found {num_methods} methods. Aiming for {samples_per_method} samples per method.")
        for method_name, method_blobs in paths_by_method.items():
            num_to_sample = min(samples_per_method, len(method_blobs))
            if num_to_sample == 0: continue
            for blob in random.sample(method_blobs, num_to_sample):
                all_sampled_paths.append(
                    {'gcs_path': f"gs://{bucket_name}/{blob.name}", 'domain': domain_name, 'label': label,
                     'method': method_name})
    return all_sampled_paths


def run_data_processing(all_paths, num_workers):
    print("\n--- Starting Data Processing ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model to device: {device}")
    model, processor = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
        device), CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval();
    fs = fsspec.filesystem('gcs')
    processing_func = partial(process_single_frame, fs=fs, clip_model=model, clip_processor=processor, device=device)
    results = [res for res in
               tqdm(concurrent.futures.ThreadPoolExecutor(max_workers=num_workers).map(processing_func, all_paths),
                    total=len(all_paths)) if res]
    df_list, embeddings_list, images_list, metadata_list = [], [], [], []
    for res in results:
        embeddings_list.append(res.pop('embedding'));
        images_list.append(res.pop('image_np'))
        df_list.append(res);
        metadata_list.append({'domain': res['domain'], 'label': res['label']})
    return pd.DataFrame(df_list), np.array(embeddings_list), images_list, metadata_list


# --- Plotting Functions (Unchanged) ---
def plot_kde_distributions(df, df_aug, pipeline_name, quick_test=False):
    print(f"\n--- [{pipeline_name}] Generating KDE Plots (Detailed Source View) ---")
    metrics = ["sharpness", "freq_ratio", "mean_r", "mean_g", "mean_b"]
    df['plot_label'] = df['domain'] + " (" + df['label'] + ")"
    if not df_aug.empty: df_aug['plot_label'] = "Augmented (" + df_aug['label'] + ")"
    palette = {"Train-Primary (real)": "#0072B2", "Train-Primary (fake)": "#56B4E9", "Train-Effort (real)": "#009E73",
               "Train-Effort (fake)": "#02FAC6", "Test (real)": "#D55E00", "Test (fake)": "#E69F00",
               "Augmented (real)": "#CC79A7", "Augmented (fake)": "#F0E442"}
    combined_df = pd.concat([df, df_aug], ignore_index=True)
    for metric in metrics:
        plt.figure(figsize=(14, 8));
        ax = plt.gca()
        sns.kdeplot(data=combined_df[~combined_df['plot_label'].str.contains("Augmented")], x=metric, hue='plot_label',
                    palette=palette, fill=True, alpha=0.3, ax=ax)
        if not df_aug.empty: sns.kdeplot(data=df_aug, x=metric, hue='plot_label', palette=palette, fill=False,
                                         linestyle='--', linewidth=2.5, ax=ax)
        plt.title(f'Detailed Source Distribution of {metric.title()} (Pipeline: {pipeline_name})', fontsize=16);
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if quick_test:
            plt.show()
        else:
            save_path = os.path.join(OUTPUT_DIR, f"kde_{metric}_detailed_source_{pipeline_name}.png");
            plt.savefig(
                save_path);
            print(f"  - Saved to {save_path}")
        plt.close()


def plot_kde_distributions_by_split(df, df_aug, pipeline_name, quick_test=False):
    print(f"\n--- [{pipeline_name}] Generating KDE Plots (Unified Train vs. Test View) ---")
    metrics = ["sharpness", "freq_ratio", "mean_r", "mean_g", "mean_b"]
    df['plot_label_split'] = df['domain_group']
    if not df_aug.empty: df_aug['plot_label_split'] = 'Train - Augmented'
    combined_df = pd.concat([df, df_aug], ignore_index=True)
    palette, hue_order = {"Train": "#0072B2", "Test": "#D55E00", "Train - Augmented": "#009E73"}, ["Train", "Test",
                                                                                                   "Train - Augmented"]
    for metric in metrics:
        plt.figure(figsize=(14, 8));
        ax = plt.gca()
        sns.kdeplot(data=combined_df, x=metric, hue='plot_label_split', hue_order=hue_order, palette=palette, fill=True,
                    alpha=0.3, ax=ax)
        plt.title(f'Unified Distribution of {metric.title()} (Pipeline: {pipeline_name})', fontsize=16);
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if quick_test:
            plt.show()
        else:
            save_path = os.path.join(OUTPUT_DIR, f"kde_{metric}_unified_split_{pipeline_name}.png");
            plt.savefig(
                save_path);
            print(f"  - Saved to {save_path}")
        plt.close()


def plot_tsne_visualization_by_split(embeddings_2d, labels, pipeline_name, quick_test=False):
    print(f"\n--- [{pipeline_name}] Generating t-SNE (Unified Train vs. Test View) ---")
    df_tsne = pd.DataFrame(embeddings_2d, columns=['x', 'y']);
    df_tsne['Data Split'] = labels
    palette, markers, hue_order = {"Train": "#0072B2", "Test": "#D55E00", "Train - Augmented": "#009E73"}, {
        "Train": "o", "Test": "s", "Train - Augmented": "X"}, ["Test", "Train", "Train - Augmented"]
    plt.figure(figsize=(16, 12))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='Data Split', style='Data Split', hue_order=hue_order,
                    palette=palette, markers=markers, alpha=0.9, s=50, edgecolor='black', linewidth=0.5)
    plt.title(f"t-SNE of Feature Space (Unified View, Pipeline: {pipeline_name})", fontsize=18);
    plt.grid(True, linestyle='--', linewidth=0.5)
    if quick_test:
        plt.show()
    else:
        save_path = os.path.join(OUTPUT_DIR, f"tsne_unified_split_{pipeline_name}.png");
        plt.savefig(save_path,
                    dpi=300);
        print(
            f"  - Saved to {save_path}")
    plt.close()


def plot_source_diagnostics_kde(df, quick_test=False):
    print("\n--- [DIAGNOSTIC] Generating KDE Plots for Original Data Sources ---")
    metrics, palette = ["sharpness", "freq_ratio", "mean_r", "mean_g", "mean_b"], {"Train-Primary": "#0072B2",
                                                                                   "Train-Effort": "#009E73",
                                                                                   "Test": "#D55E00"}
    for metric in metrics:
        plt.figure(figsize=(14, 8))
        sns.kdeplot(data=df, x=metric, hue='domain', palette=palette, fill=True, alpha=0.3)
        plt.title(f'Diagnostic: Initial Distribution of {metric.title()} by Source', fontsize=16);
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        if quick_test:
            plt.show()
        else:
            save_path = os.path.join(OUTPUT_DIR, f"diagnostic_kde_{metric}_by_source.png");
            plt.savefig(
                save_path);
            print(f"  - Saved to {save_path}")
        plt.close()


def plot_source_diagnostics_tsne(embeddings_2d, metadata_list, quick_test=False):
    print("\n--- [DIAGNOSTIC] Generating t-SNE for Original Data Sources ---")
    df_tsne = pd.DataFrame(embeddings_2d, columns=['x', 'y']);
    df_tsne['Source'] = [m['domain'] for m in metadata_list]
    palette, markers = {"Train-Primary": "#0072B2", "Train-Effort": "#009E73", "Test": "#D55E00"}, {
        "Train-Primary": "o", "Train-Effort": "^", "Test": "s"}
    plt.figure(figsize=(16, 12))
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='Source', style='Source', palette=palette, markers=markers,
                    alpha=0.9, s=50, edgecolor='black', linewidth=0.5)
    plt.title("Diagnostic: Initial t-SNE of Feature Space by Source", fontsize=18);
    plt.grid(True, linestyle='--', linewidth=0.5)
    if quick_test:
        plt.show()
    else:
        save_path = os.path.join(OUTPUT_DIR, "diagnostic_tsne_by_source.png");
        plt.savefig(save_path, dpi=300);
        print(
            f"  - Saved to {save_path}")
    plt.close()


# --- MODIFICATION START: Rewrite the simulation function for flexibility ---
def simulate_augmentations(images, metadata, pipeline_config, pipeline_name, clip_model, clip_processor, device):
    """
    Simulates augmentations based on a flexible pipeline configuration.
    pipeline_config can be a single pipeline or a dict mapping domain to a pipeline.
    """
    print(f"\n--- Simulating Augmentations for Strategy: {pipeline_name} ---")
    aug_stats_list, aug_embeddings, aug_metadata = [], [], []

    # Filter for only training images to be augmented
    train_images_meta = [(img, meta) for img, meta in zip(images, metadata) if 'Train' in meta['domain']]

    print(f"  --> Applying augmentations for {len(train_images_meta)} training images...")
    for img, meta in tqdm(train_images_meta):
        # Determine which pipeline to use
        if isinstance(pipeline_config, dict):
            # Use the pipeline specific to the image's domain
            pipeline_to_use = pipeline_config.get(meta['domain'])
        else:
            # Use the single pipeline provided
            pipeline_to_use = pipeline_config

        if pipeline_to_use is None:
            # If a domain has no specified pipeline in the dict, skip it.
            continue

        augmented_img = pipeline_to_use(image=img)['image']
        stats = get_low_level_stats(augmented_img)
        stats.update({'domain': 'Train - Augmented', 'label': meta['label']})
        aug_stats_list.append(stats)
        aug_embeddings.append(get_clip_embedding(augmented_img, clip_model, clip_processor, device))
        aug_metadata.append({'domain': 'Train - Augmented', 'label': meta['label']})

    df_aug = pd.DataFrame(aug_stats_list)
    return df_aug, np.array(aug_embeddings).squeeze(), aug_metadata


# --- MODIFICATION END ---


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize domain shift between datasets.")
    parser.add_argument('--target_samples_per_group', type=int, default=TARGET_SAMPLES_PER_GROUP)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--skip_processing', action='store_true', help="Skip GCS processing and load local data.")
    parser.add_argument('--quick_test', action='store_true')
    args = parser.parse_args()

    target_samples = 10 if args.quick_test else args.target_samples_per_group
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    paths = {k: os.path.join(OUTPUT_DIR, f) for k, f in
             {"stats": "domain_stats.csv", "embed": "domain_embeddings.npy", "images": "original_images.npy",
              "meta": "metadata.npy"}.items()}

    if not args.skip_processing:
        client = storage.Client()
        train_paths_primary = discover_and_sample_balanced(client, TRAIN_BUCKET_NAME, "Train-Primary", target_samples)
        train_paths_effort = discover_and_sample_balanced(client, SECOND_TRAIN_BUCKET_NAME, "Train-Effort",
                                                          target_samples)
        test_paths = discover_and_sample_balanced(client, TEST_BUCKET_NAME, "Test", target_samples)
        all_paths = train_paths_primary + train_paths_effort + test_paths
        if not all_paths: print("[Error] No files were sampled. Exiting."); return
        print("\n--- Sampling Summary ---")
        counts = defaultdict(int)
        for p in all_paths: counts[f"{p['domain']} ({p['label']})"] += 1
        for group, count in sorted(counts.items()): print(f"  - {group}: {count} samples")
        df, embeddings, images_list, metadata_list = run_data_processing(all_paths, args.workers)
        print(f"\n--- Saving Processed Data to '{OUTPUT_DIR}' ---")
        df.to_csv(paths['stats'], index=False)
        np.save(paths['embed'], embeddings);
        np.save(paths['images'], np.array(images_list, dtype=object));
        np.save(paths['meta'], np.array(metadata_list, dtype=object))
        print("✅ Data saved.")
    else:
        print(f"--- Skipping processing, loading data from '{OUTPUT_DIR}' ---")
        df, embeddings, images_list, metadata_list = pd.read_csv(paths['stats']), np.load(paths['embed']), np.load(
            paths['images'], allow_pickle=True), np.load(paths['meta'], allow_pickle=True).tolist()
        print("✅ Data loaded.")

    df['domain_group'] = df['domain'].apply(lambda x: 'Train' if 'Train' in x else 'Test')
    print("\n--- Performing ALL expensive computations upfront ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_processor = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(
        device), CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model.eval()

    # --- MODIFICATION START: Define the strategies to test, including our new targeted one ---
    pipelines_to_test = {
        # "V3": AUG_PIPELINE_V3, # Can uncomment to re-run old ones for comparison
        # "V4": AUG_PIPELINE_V4,
        "V6-Targeted": {
            "Train-Primary": AUG_PIPELINE_V6_FOR_PRIMARY,
            "Train-Effort": AUG_PIPELINE_LIGHT_FOR_EFFORT
        }
    }
    # --- MODIFICATION END ---

    all_aug_data, all_embeddings_list = {}, [embeddings]
    for name, pipeline_config in pipelines_to_test.items():
        df_aug, embeddings_aug, metadata_aug = simulate_augmentations(images_list, metadata_list, pipeline_config, name,
                                                                      clip_model, clip_processor, device)
        all_aug_data[name] = {'df': df_aug, 'metadata': metadata_aug}
        all_embeddings_list.append(embeddings_aug)

    print("\n--> Computing a single t-SNE transformation for all data points...")
    combined_embeddings = np.vstack(all_embeddings_list)
    tsne_model = TSNE(n_components=2, perplexity=min(30, len(combined_embeddings) - 1), max_iter=1000, random_state=42,
                      verbose=1)
    embeddings_2d_all = tsne_model.fit_transform(combined_embeddings)
    print("\n--- ✅ All computations complete. Starting plotting. ---")

    num_original_samples = len(embeddings)
    # The diagnostic plots are based on the initial data, so we can run them once here.
    # To avoid re-generating them every time, you can comment these out after the first run.
    print("\n--- Generating Diagnostic Plots (if not already present) ---")
    plot_source_diagnostics_kde(df, args.quick_test)
    plot_source_diagnostics_tsne(embeddings_2d_all[:num_original_samples], metadata_list, args.quick_test)

    start_index = num_original_samples
    for name in pipelines_to_test.keys():
        print(f"\n{'=' * 20} Generating Plots for Strategy: {name} {'=' * 20}")
        df_aug_current, num_aug_samples = all_aug_data[name]['df'], len(all_aug_data[name]['metadata'])
        plot_kde_distributions_by_split(df.copy(), df_aug_current.copy(), name, args.quick_test)
        plot_kde_distributions(df.copy(), df_aug_current.copy(), name, args.quick_test)
        indices_original, indices_aug_current = list(range(num_original_samples)), list(
            range(start_index, start_index + num_aug_samples))
        embeddings_2d_unified = np.vstack([embeddings_2d_all[indices_original], embeddings_2d_all[indices_aug_current]])
        labels_unified = list(df['domain_group']) + ['Train - Augmented'] * num_aug_samples
        plot_tsne_visualization_by_split(embeddings_2d_unified, labels_unified, name, args.quick_test)
        start_index += num_aug_samples

    print(f"\n--- Analysis Complete! Outputs are in the '{OUTPUT_DIR}' directory. ---")


if __name__ == "__main__":
    main()
