import argparse
import random
import datetime
import time
import yaml  # noqa
from datetime import timedelta
import math
import os
from collections import defaultdict, Counter
from tqdm import tqdm  # noqa

from collections import OrderedDict

import torch  # noqa
import torch.nn.parallel  # noqa
import torch.backends.cudnn as cudnn  # noqa
import torch.utils.data  # noqa
import torch.optim as optim  # noqa
from torch.utils.data.distributed import DistributedSampler  # noqa
import torch.distributed as dist  # noqa
import numpy as np  # noqa
import wandb  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
from google.cloud.storage import Bucket  # noqa
from sklearn import metrics 
from pathlib import Path
import gc


from trainer.trainer import Trainer
from detectors import DETECTOR  # noqa
from metrics.utils import parse_metric_for_print
from logger import create_logger
from PIL.ImageFilter import RankFilter  # noqa
from prepare_test_splits import prepare_video_splits
from dataset.test_dataloader import create_method_aware_dataloaders, collate_fn  # noqa

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--run_sanity_check', action='store_true', default=False,
                    help="Run the comprehensive sampler check and exit.")
parser.add_argument('--dataloader_config', type=str, default='./config/test_dataloader_config.yml',
                    help='Path to the dataloader configuration file')

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'adam':
        optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
        )
        return optimizer
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None: return None
    if config['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['nEpochs'], eta_min=config['optimizer']['adam']['lr'] / 100
        )
    raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def comprehensive_sampler_check(
        real_loaders,
        fake_loaders,
        real_method_names,
        full_batch_size,
        train_videos
):
    """
    Performs a deterministic, one-by-one check of each fake method.
    For each fake method, it attempts to load one batch and pairs it with a
    random real source to verify data integrity.
    """

    def _get_next_batch(method, dataloader_dict, iter_dict):
        """Gets the next batch, creating an iterator if one doesn't exist."""
        if method not in iter_dict:
            iter_dict[method] = iter(dataloader_dict[method])
        try:
            return next(iter_dict[method])
        except StopIteration:
            # Dataloader is exhausted.
            return None

    print("\n\n==================== COMPREHENSIVE SAMPLER CHECK ====================")
    if not fake_loaders or not real_loaders:
        print("❌ No fake or real loaders provided. Aborting check.")
        return

    fake_method_names = sorted(list(fake_loaders.keys()))
    num_fake_methods = len(fake_method_names)
    half_batch_size = full_batch_size // 2
    print(f"Deterministically checking {num_fake_methods} fake methods...")
    print(f"Expecting half-batches of size {half_batch_size}.\n")

    # Use persistent iterators to not re-test the same initial data
    real_method_iters = {}
    fake_method_iters = {}

    # --- Tracking ---
    # We use a simple dictionary to store the status of each fake method
    fake_method_status = {}
    real_source_usage = defaultdict(int)
    batch_times = []

    pbar = tqdm(fake_method_names, desc="Checking Methods")
    for chosen_fake_method in pbar:
        iteration_start_time = time.time()
        pbar.set_postfix_str(f"Testing: {chosen_fake_method}")

        # 1. Attempt to fetch a FAKE half-batch for the current method
        fake_data_dict = _get_next_batch(chosen_fake_method, fake_loaders, fake_method_iters)

        if not fake_data_dict or fake_data_dict['image'].shape[0] == 0:
            fake_method_status[chosen_fake_method] = "❌ EMPTY (Corrupt data or paths)"
            continue
        if fake_data_dict['image'].shape[0] < half_batch_size:
            fake_method_status[chosen_fake_method] = f"⚠️ PARTIAL (Not enough videos for a full batch)"
            continue

        # 2. Attempt to fetch a REAL half-batch to pair with it
        if not real_method_names:
            fake_method_status[chosen_fake_method] = "❌ SKIPPED (No real sources available)"
            continue

        chosen_real_method = random.choice(real_method_names)
        real_data_dict = _get_next_batch(chosen_real_method, real_loaders, real_method_iters)

        # If real source is exhausted, try to restart its iterator ONCE
        if real_data_dict is None:
            real_method_iters[chosen_real_method] = iter(real_loaders[chosen_real_method])
            real_data_dict = _get_next_batch(chosen_real_method, real_loaders, real_method_iters)

        if not real_data_dict or real_data_dict['image'].shape[0] == 0:
            fake_method_status[chosen_fake_method] = f"❌ SKIPPED (Paired real source '{chosen_real_method}' is empty)"
            continue
        if real_data_dict['image'].shape[0] < half_batch_size:
            fake_method_status[
                chosen_fake_method] = f"❌ SKIPPED (Paired real source '{chosen_real_method}' gave partial batch)"
            continue

        # 3. If both are successful
        fake_method_status[chosen_fake_method] = f"✅ OK (Paired with {chosen_real_method})"
        real_source_usage[chosen_real_method] += 1
        batch_times.append(time.time() - iteration_start_time)

    print("\n-------------------- CHECK COMPLETE: DIAGNOSTIC REPORT --------------------")
    total_time = sum(batch_times)
    print(f"Total time for {len(batch_times)} successful pairs: {total_time:.2f} seconds.")
    if batch_times:
        print(f"Avg batch creation time: {np.mean(batch_times):.3f}s | Max: {np.max(batch_times):.3f}s")

    print("\n--- Fake Methods Report ---")
    video_counts = Counter(v.method for v in train_videos)
    print(f"{'Method':<20} | {'Total Vids':>10} | Status & Details")
    print("-" * 85)
    for method in fake_method_names:
        total_vids = video_counts.get(method, 0)
        status = fake_method_status.get(method, "❔ NOT TESTED (Should not happen)")
        print(f"{method:<20} | {total_vids:>10} | {status}")

    print("\n--- Real Sources Usage Report ---")
    print("How many times each real source was successfully used for pairing:")
    for method in sorted(real_method_names):
        count = real_source_usage.get(method, 0)
        print(f"- {method:<20}: {count} times")

    print("\nRecommendations:")
    print("  - For '❌ EMPTY' methods, all videos are likely corrupt. Remove from `dataloader_config.yml`.")
    print(
        "  - For '⚠️ PARTIAL' methods, there aren't enough videos for one half-batch. Consider removing or adding more data.")
    print(
        "  - For '❌ SKIPPED' methods, the issue may be with the real sources, not the fake one. Check real source health.")
    print("================== END COMPREHENSIVE SAMPLER CHECK ==================\n")


# def download_checkpoint_from_gcs(config, logger):
#     """
#     Downloads a base checkpoint from a GCS bucket if specified in the config.
#
#     This function checks for 'base_checkpoint_bucket_path' in the config.
#     If present, it downloads the file to the local path specified by
#     'base_checkpoint_output_path' and 'base_checkpoint_name'.
#
#     It handles GCS authentication automatically in a Vertex AI environment.
#
#     Args:
#         config (dict): The main configuration dictionary.
#         logger: The logger instance for logging messages.
#
#     Returns:
#         str: The local path to the downloaded checkpoint file if successful,
#              otherwise None.
#     """
#     gcs_path = config.get('base_checkpoint_bucket_path')
#     local_dir = config.get('base_checkpoint_output_path')
#     file_name = config.get('base_checkpoint_name')
#
#     # first check if the checkpoint already exists in the local directory
#     if local_dir and file_name:
#         local_destination_path = os.path.join(local_dir, file_name)
#         if os.path.exists(local_destination_path):
#             logger.info(f"Base checkpoint already exists at {local_destination_path}. Skipping download.")
#             return local_destination_path
#
#     if not all([gcs_path, local_dir, file_name]):
#         logger.info("Base checkpoint download not configured. Skipping.")
#         return None
#
#     if not gcs_path.startswith('gs://'):
#         logger.error(f"Invalid GCS path: '{gcs_path}'. Must start with 'gs://'.")
#         return None
#
#     local_destination_path = os.path.join(local_dir, file_name)
#
#     logger.info("--- GCS Checkpoint Download ---")
#     logger.info(f"Attempting to download base checkpoint from GCS.")
#     logger.info(f"  Source: {gcs_path}")
#     logger.info(f"  Destination: {local_destination_path}")
#
#     try:
#         # Parse the GCS path
#         path_parts = gcs_path.replace('gs://', '').split('/', 1)
#         bucket_name = path_parts[0]
#         blob_name = path_parts[1]
#
#         # Create the local directory if it doesn't exist
#         os.makedirs(local_dir, exist_ok=True)
#
#         # In a Vertex AI/GCP environment, the client authenticates automatically
#         # using the service account associated with the job.
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(blob_name)
#
#         if not blob.exists():
#             logger.error(f"FAILED: Checkpoint file not found at {gcs_path}")
#             return None
#
#         logger.info("Checkpoint found. Starting download...")
#         start_time = time.time()
#         blob.download_to_filename(local_destination_path)
#         elapsed_time = time.time() - start_time
#         logger.info(f"✅ SUCCESS: Downloaded checkpoint in {elapsed_time:.2f}s.")
#         return local_destination_path
#
#     except exceptions.Forbidden as e:
#         logger.error(
#             "FAILED: GCP Permissions error. Ensure the Vertex AI job's service "
#             f"account has 'Storage Object Viewer' role on bucket '{bucket_name}'.")
#         logger.error(f"  Details: {e}")
#         return None
#     except exceptions.NotFound as e:
#         logger.error(f"FAILED: GCS bucket or path not found. Check your config.")
#         logger.error(f"  Details: {e}")
#         return None
#     except Exception as e:
#         logger.error(f"FAILED: An unexpected error occurred during download: {e}")
#         return None
#

def download_gcs_asset(bucket: Bucket, gcs_path: str, local_path: str, logger) -> bool:
    """
    Downloads a single blob or a directory of blobs from GCS.

    Args:
        bucket (storage.Bucket): The GCS bucket object.
        gcs_path (str): The path to the object or directory in GCS.
        local_path (str): The local path to download to.
        logger: The logger instance.

    Returns:
        bool: True if successful, False otherwise.
    """
    if gcs_path.endswith('/'):  # It's a directory
        prefix = gcs_path.split(bucket.name + '/', 1)[1]
        blobs = bucket.list_blobs(prefix=prefix)
        downloaded = False
        for blob in blobs:
            if blob.name.endswith('/'):  # Skip "directory" blobs
                continue
            destination_file_name = os.path.join(local_path, os.path.relpath(blob.name, prefix))
            os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
            try:
                blob.download_to_filename(destination_file_name)
                downloaded = True
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                return False
        if not downloaded:
            logger.error(f"Directory {gcs_path} is empty or does not exist.")
            return False
        return True
    else:  # It's a single file
        blob_name = gcs_path.split(bucket.name + '/', 1)[1]
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.error(f"File not found at {gcs_path}")
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        return True


def download_assets_from_gcs(config, logger):
    """
    Downloads specified assets (checkpoints, models) from a GCS bucket.

    This function reads a list of assets from the config, where each asset has
    a GCS path and a desired local path. It handles both individual files and
    entire directories.

    Args:
        config (dict): The main configuration dictionary.
        logger: The logger instance for logging messages.

    Returns:
        dict: A dictionary mapping asset keys to their local paths if successful,
              otherwise None.
    """
    assets_to_download = config.get('gcs_assets')
    if not assets_to_download:
        logger.info("No GCS assets configured for download. Skipping.")
        return None

    local_paths = {}

    # First, check if all assets already exist locally
    all_exist = True
    for key, asset_info in assets_to_download.items():
        local_path = asset_info.get('local_path')
        if not local_path or not os.path.exists(local_path):
            all_exist = False
            break
    if all_exist:
        logger.info("All GCS assets already exist locally. Skipping downloads.")
        for key, asset_info in assets_to_download.items():
            local_paths[key] = asset_info.get('local_path')
        return local_paths

    logger.info("--- GCS Asset Download ---")
    try:
        storage_client = storage.Client()
        start_time = time.time()

        for key, asset_info in assets_to_download.items():
            gcs_path = asset_info.get('gcs_path')
            local_path = asset_info.get('local_path')

            if not gcs_path or not local_path:
                logger.error(f"Asset '{key}' is missing 'gcs_path' or 'local_path' in config.")
                return None

            if not gcs_path.startswith('gs://'):
                logger.error(f"Invalid GCS path for asset '{key}': '{gcs_path}'. Must start with 'gs://'.")
                return None

            # Check if this specific asset already exists
            if os.path.exists(local_path):
                logger.info(f"Asset '{key}' already exists at {local_path}. Skipping.")
                local_paths[key] = local_path
                continue

            logger.info(f"Downloading asset '{key}'...")
            logger.info(f"  Source: {gcs_path}")
            logger.info(f"  Destination: {local_path}")

            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            bucket = storage_client.bucket(bucket_name)

            if not download_gcs_asset(bucket, gcs_path, local_path, logger):
                raise RuntimeError(f"Failed to download asset '{key}'.")

            local_paths[key] = local_path
            logger.info(f"✅ SUCCESS: Downloaded '{key}'.")

        elapsed_time = time.time() - start_time
        logger.info(f"✅ SUCCESS: All GCS assets downloaded in {elapsed_time:.2f}s.")
        return local_paths

    except exceptions.Forbidden as e:
        logger.error(
            "FAILED: GCP Permissions error. Ensure the Vertex AI job's service "
            "account has 'Storage Object Viewer' role on the relevant buckets.")
        logger.error(f"  Details: {e}")
        return None
    except exceptions.NotFound as e:
        logger.error("FAILED: GCS bucket or path not found. Check your config.")
        logger.error(f"  Details: {e}")
        return None
    except Exception as e:
        logger.error(f"FAILED: An unexpected error occurred during download: {e}")
        return None

def load_ckpt(model_path, model, logger):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in saved.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)
            logger.info(f'Model loaded from {model_path}')
        else:
            raise FileNotFoundError(f"=> no model found at '{model_path}'")


def get_test_metrics(y_pred, y_true, img_names=None):
    """
    Calculates frame-level and, optionally, video-level metrics.
    This version is robust to single-class inputs.

    Args:
        y_pred (np.ndarray): 1D array of frame-level prediction probabilities.
        y_true (np.ndarray): 1D array of frame-level ground truth labels.
        img_names (list, optional): List of frame paths. If provided, video-level
                                    metrics will be calculated by grouping frames.
                                    Defaults to None.

    Returns:
        dict: A dictionary containing calculated metrics.
    """
    # Ensure inputs are numpy arrays
    y_pred = np.array(y_pred).squeeze()
    y_true = np.array(y_true).squeeze()

    metrics_dict = {}

    # --- 1. Frame-level Metrics (Always Calculated) ---
    # --- START OF FRAME-LEVEL FIX ---
    unique_frame_labels = np.unique(y_true)
    if len(y_true) > 0 and len(unique_frame_labels) > 1:
        # Both classes are present, calculate all metrics
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
        fnr = 1 - tpr
        frame_auc = metrics.auc(fpr, tpr)
        frame_eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
        frame_ap = metrics.average_precision_score(y_true, y_pred)
    else:
        # Single class or empty input, cannot calculate AUC/EER/AP
        frame_auc = -1.0
        frame_eer = -1.0
        frame_ap = -1.0

    # Accuracy can always be calculated
    pred_class = (y_pred > 0.5).astype(int)
    correct = (pred_class == y_true).sum()
    frame_acc = correct / len(y_true) if len(y_true) > 0 else 0.0

    metrics_dict.update({
        'acc': frame_acc,
        'auc': frame_auc,
        'eer': frame_eer,
        'ap': frame_ap,
    })
    # --- END OF FRAME-LEVEL FIX ---

    # --- 2. Video-level Metrics (Calculated if img_names is provided) ---
    if img_names is not None and len(img_names) > 0:
        videos = defaultdict(lambda: {'preds': [], 'label': -1})
        for path, pred, label in zip(img_names, y_pred, y_true):
            # Using Path object correctly
            video_id = Path(path).parent.name
            videos[video_id]['preds'].append(pred)
            if videos[video_id]['label'] == -1:
                videos[video_id]['label'] = label

        video_preds = []
        video_labels = []
        for video_id, data in videos.items():
            if not data['preds']: continue
            video_preds.append(np.mean(data['preds']))
            video_labels.append(data['label'])

        # --- START OF VIDEO-LEVEL FIX ---
        if len(video_labels) > 1:
            video_preds = np.array(video_preds)
            video_labels = np.array(video_labels)

            unique_video_labels = np.unique(video_labels)
            if len(unique_video_labels) > 1:
                # Both classes are present at video level
                v_fpr, v_tpr, _ = metrics.roc_curve(video_labels, video_preds, pos_label=1)
                v_fnr = 1 - v_tpr
                metrics_dict['video_auc'] = metrics.auc(v_fpr, v_tpr)
                metrics_dict['video_eer'] = v_fpr[np.nanargmin(np.absolute(v_fnr - v_fpr))]
                metrics_dict['video_ap'] = metrics.average_precision_score(video_labels, video_preds)
            else:
                # Single class at video level
                metrics_dict['video_auc'] = -1.0
                metrics_dict['video_eer'] = -1.0
                metrics_dict['video_ap'] = -1.0

            # Video accuracy can always be calculated
            v_pred_class = (video_preds > 0.5).astype(int)
            v_correct = (v_pred_class == video_labels).sum()
            metrics_dict['video_acc'] = v_correct / len(video_labels) if len(video_labels) > 0 else 0.0
        # --- END OF VIDEO-LEVEL FIX ---

    return metrics_dict

@torch.no_grad()
def test_epoch(model,test_method_loaders, config, logger, wandb_run, metric_scoring='auc'):
    model.eval()

    test_iters = {name: iter(loader) for name, loader in test_method_loaders.items()}

    # --- Calculate total batches based on total number of videos and batch size ---
    total_videos = len(test_iters)
    batch_size = config['test_batchSize']
    # The number of batches is the total videos divided by batch size, rounded up.
    # We add a check for total_videos > 0 to avoid division by zero if val set is empty.
    total_batches = math.ceil(total_videos / batch_size) if total_videos > 0 else 0
    # total_batches = sum(len(loader) for loader in test_method_loaders.values())

    pbar = tqdm(total=total_batches, desc="Testing (Interleaved)", unit="batch")
    # Iterate through each method's DataLoader one by one.
    method_labels = defaultdict(list)
    method_preds = defaultdict(list)

    all_preds, all_labels = [], []

    # --- Loop until all iterators are exhausted ---
    while test_iters:
        # Iterate over a copy of keys, as we will modify the dict
        for method in list(test_iters.keys()):
            try:
                # --- NEW: Get the next batch from the current method's iterator ---
                data_dict = next(test_iters[method])

                # Move tensors to the correct device
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(model.device)

                # skip empty batches or invalid shapes
                if data_dict['image'].shape[0] == 0: continue
                if data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                predictions = model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)

                all_labels.extend(data_dict['label'].cpu().numpy())
                all_preds.extend(video_probs.cpu().numpy())

                # Add probabilities and labels by method
                method_labels[method].extend(data_dict['label'].cpu().numpy())
                method_preds[method].extend(video_probs.cpu().numpy())

                # (Future) Here you can accumulate per-method metrics:
                # per_method_results[method].append(...)

                pbar.update(1)

            except StopIteration:
                # This loader is finished, remove it ---
                del test_iters[method]

    pbar.close()

    if not all_labels:
        logger.error("Validation failed: No data was processed after iterating all loaders.")
        return

    logger.info("--- Calculating overall validation performance ---")
    overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))

    wandb_log_dict = {}
    for name, value in overall_metrics.items():
        if name not in ['pred', 'label']:
            wandb_log_dict[f'test/overall/{name}'] = value
            logger.info(f"Overall val {name}: {value:.4f}")

    # Metrics Per Method
    method_metrics = {}
    for method in method_preds.keys():
        method_metrics[method] = get_test_metrics(np.array(method_preds[method]), np.array(method_labels[method]))
        for name, value in method_metrics[method].items():
            if name in ['acc']:
                wandb_log_dict[f'test/{method}/{name}'] = value
                logger.info(f"Method {method} val {name}: {value:.4f}")

    # # Check for new best model and save
    # current_metric = overall_metrics.get(metric_scoring)
    # if current_metric is not None and current_metric > best_val_metric:
    #     best_val_metric = current_metric
    #     # best_val_epoch = epoch + 1
    #     logger.info(f"🎉 New best model found! Metric ({metric_scoring}): {current_metric:.4f}")
    #     if config['save_ckpt']:
    #         # --- NEW: Create the custom alias ---
    #         auc_metric = overall_metrics.get('auc', 0.0) # Default to 0.0 if not found
    #         # Format the string as requested
    #         custom_alias = f"AUC_{auc_metric:.4f}"
    #         # save_ckpt(epoch + 1, aliases=[custom_alias])
            
    #         # --- NEW: Logic to delete the previous artifact ---
    #         # 1. Store the previous artifact to be deleted
    #         # old_artifact = best_model_artifact

    #         # 2. Save the new artifact and get its object reference
    #         # best_model_artifact = save_ckpt(epoch + 1, aliases=[custom_alias])

    #         # 3. If there was an old artifact from this run, delete it now
            # if old_artifact:
            #     try:
            #         logger.info(f"Deleting previous best model artifact: {old_artifact.name}")
            #         old_artifact.delete(delete_aliases=True)
            #     except Exception as e:
            #         logger.warning(f"Failed to delete old artifact. It may need to be manually removed. Error: {e}")

    # wandb_log_dict['val/best_metric'] = best_val_metric
    # wandb_log_dict['val/best_epoch'] = best_val_epoch

    if wandb_run:
        wandb_run.log(wandb_log_dict)

    # Explicitly delete large variables and collect garbage
    del all_preds, all_labels, method_labels, method_preds, overall_metrics, method_metrics
    gc.collect()
    torch.cuda.empty_cache()

    logger.info("===> Evaluation Done!")
sweep_config = {
    'method': 'grid',  # Specifies a grid search
    'metric': {
        'name': 'test/overall/auc',  # The metric to optimize/track
        'goal': 'maximize'  # The goal for the metric (maximize AUC)
    },
    'parameters': {
        # Define the hyperparameter to sweep over and its values
        'frame_num_test': {
            'values': [8, 16, 32]
        }
    }
}


def main():
    ##################### ADAM CHANGED ###################
    os.chdir("/home/roee/repos/Effort-AIGI-Detection/DeepfakeBench/training")
    ##################### ADAM CHANGED ###################
    # Define the sweep configuration

    # Load all configurations
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/test_config.yaml', 'r') as f:
        config.update(yaml.safe_load(f))
    config['local_rank'] = args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = False

    dataloader_config_path = args.dataloader_config
    with open(dataloader_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    config.update(data_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # W&B and Logger Initialization
    run_name = f"{config['model_name']}_test_frames_{config['frame_num']['test']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb_run = wandb.init(
        name=run_name,
        mode="online",
        project="Test_Effort_Sweep", # Optional: Or set WANDB_PROJECT env var
        config=config,
        # entity="your_entity", # Optional: Or set WANDB_ENTITY env var
    )
    config.update(wandb.config)

    # create logger and path
    logger_path = os.path.join(wandb_run.dir, 'logs')  # Save logs inside wandb folder

    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'testing.log'))
    logger.info(f'Save log to {logger_path}')
    logger.info(f"Current config['frame_num']['test']: {config['frame_num']['test']}") # Log the current frame_num_test value
    config['ddp'] = args.ddp

    init_seed(config)

    if config['cudnn']: cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    # --- Download Base Checkpoint from GCS ---
    # This function will download a base model from GCS if configured.
    # It will also download the CLIP backbone.
    download_assets_from_gcs(config, logger)


    logger.info("------- Configuration & Data Loading -------")
    test_videos, _ = prepare_video_splits(dataloader_config_path)
    train_batch_size = data_config['dataloader_params']['batch_size']
    if train_batch_size % 2 != 0:
        raise ValueError(f"train_batchSize must be even for 50/50 split, but got {train_batch_size}")
    half_batch_size = train_batch_size // 2

    test_method_loaders = create_method_aware_dataloaders(
        test_videos, config, data_config, test_batch_size=1
    )

    real_source_names = data_config['methods']['use_real_sources']
    real_loaders, fake_loaders = {}, {}
    for name, loader in test_method_loaders.items():
        (real_loaders if name in real_source_names else fake_loaders)[name] = loader
    logger.info(
        f"Created {len(real_loaders)} real loaders, {len(fake_loaders)} fake loaders, and {len(test_method_loaders)} validation loaders.")

    # Prepare model, optimizer, scheduler, metric, trainer
    model = DETECTOR[config['model_name']](config)

    model.to(device)
    model.device = device
    # optimizer = choose_optimizer(model, config)
    # scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    # trainer = Trainer(
    #     config, model, optimizer, scheduler, logger, metric_scoring,
    #     wandb_run=wandb_run, val_videos=val_videos
    # )

    # # --- Training Loop Setup ---
    # real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
    # for v in train_videos:
    #     (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1

    # total_real_videos = sum(real_video_counts.values())
    # total_fake_videos = sum(fake_video_counts.values())

    # real_weights = [real_video_counts[m] / total_real_videos for m in
    #                 real_source_names] if total_real_videos > 0 else []
    # fake_method_names = list(fake_loaders.keys())
    # fake_weights = [fake_video_counts[m] / total_fake_videos for m in
    #                 fake_method_names] if total_fake_videos > 0 else []

    # total_train_videos = len(train_videos)
    # epoch_len = math.ceil(total_train_videos / train_batch_size) if total_train_videos > 0 else 0
    # logger.info(f"Total balanced training videos: {total_train_videos}, epoch length: {epoch_len} steps")

    # # NEW: Get evaluation frequency from config
    # eval_freq = data_config['data_params'].get('evaluation_frequency', 1)

    if config['gcs_assets']['base_checkpoint']['local_path']:
        load_ckpt(config['gcs_assets']['base_checkpoint']['local_path'], model, logger)

    test_epoch(model,test_method_loaders, config, logger, wandb_run, metric_scoring=metric_scoring)

    # # start training
    # for epoch in range(config['start_epoch'], config['nEpochs']):
    #     trainer.train_epoch(
    #         real_loaders=real_loaders,
    #         fake_loaders=fake_loaders,
    #         real_method_names=real_source_names,
    #         fake_method_names=fake_method_names,
    #         real_weights=real_weights,
    #         fake_weights=fake_weights,
    #         epoch=epoch,
    #         epoch_len=epoch_len,
    #         val_method_loaders=val_method_loaders,
    #         evaluation_frequency=eval_freq
    #     )
    #     if scheduler is not None: scheduler.step()

    wandb_run.finish()
    logger.info("Training complete.")


if __name__ == '__main__':
    start = time.time()
    # sweep_id = wandb.sweep(sweep=sweep_config, project="Test_Effort_Sweep") # Ensure project name is consistent
    # print(f"Starting W&B agent for sweep ID: {sweep_id}")
    # wandb.agent(sweep_id, function=main, count=len(sweep_config['parameters']['frame_num_test']['values']))
    main()
    end = time.time()
    elapsed = end - start
    print(f"Total training time in mn: {elapsed / 60:.2f} minutes")
    print("Testing complete.")
