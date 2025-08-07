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
from trainer.trainer import Trainer
from detectors import DETECTOR  # noqa
from metrics.utils import parse_metric_for_print
from logger import create_logger
from PIL.ImageFilter import RankFilter  # noqa
from prepare_splits import prepare_video_splits
from dataset.dataloaders import create_method_aware_dataloaders, collate_fn  # noqa

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

import torch  # noqa
import torch.nn.parallel  # noqa
import torch.backends.cudnn as cudnn  # noqa
import torch.utils.data  # noqa
import torch.optim as optim  # noqa
from torch.utils.data.distributed import DistributedSampler  # noqa
import torch.distributed as dist  # noqa
import numpy as np  # noqa
import wandb  # noqa

from trainer.trainer import Trainer
from detectors import DETECTOR  # noqa
from metrics.utils import parse_metric_for_print
from logger import create_logger
from PIL.ImageFilter import RankFilter  # noqa
from prepare_splits import prepare_video_splits
from dataset.dataloaders import create_method_aware_dataloaders, collate_fn  # noqa

# --- NEW: W&B SWEEP CONFIGURATION ---
# Define the hyperparameter sweep configuration
sweep_configuration = {
    'method': 'random',  # Can be 'grid', 'random', or 'bayes'
    'name': 'Effort-AIGI-Detection-Sweep',
    'metric': {
        'goal': 'maximize',
        'name': 'val/best_metric'
    },
    'parameters': {
        'lr': {
            'distribution': 'uniform',
            'min': 0.00001,
            'max': 0.01
        },
        'eps': {
            'distribution': 'uniform',
            'min': 1e-9,
            'max': 1e-5
        },
        'weight_decay': {
            'values': [0.00001, 0.0001, 0.001, 0.01, 0.1]
        },
        'train_batchSize': {
            # 'values': [1, 2, 4, 8, 16] # Reduced 256 to avoid potential OOM issues
            'values': [2] # Reduced 256 to avoid potential OOM issues
        }
}}
# --- END NEW SECTION ---

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
parser.add_argument('--dataloader_config', type=str, default='./config/dataloader_config.yml',
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
            eps=config['optimizer'][opt_name]['eps'], # Added eps
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


def main():
    # ##################### ADAM CHANGED ###################
    # os.chdir("/home/roee/repos/Effort-AIGI-Detection/DeepfakeBench/training")
    # ##################### ADAM CHANGED ###################
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/train_config.yaml', 'r') as f:
        config.update(yaml.safe_load(f))
        
    # --- NEW: W&B Initialization ---
    # W&B will automatically read entity/project from environment variables
    # (WANDB_ENTITY, WANDB_PROJECT) or your local wandb configuration.
    run_name = f"{config['model_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb_run = wandb.init(
        name=run_name,
        mode="online",
        # project="your_project_name", # Optional: Or set WANDB_PROJECT env var
        # entity="your_entity", # Optional: Or set WANDB_ENTITY env var
    )
        
    # Merge sweep hyperparameters into the main config dictionary
    config['optimizer']['adam']['lr'] = wandb.config.lr
    config['optimizer']['adam']['eps'] = wandb.config.eps
    config['optimizer']['adam']['weight_decay'] = wandb.config.weight_decay
        
    config['local_rank'] = args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt

    dataloader_config_path = args.dataloader_config
    with open(dataloader_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    config.update(data_config) # Merge data_config into config
    # --- NEW: Use train_batchSize from sweep config ---
    data_config['dataloader_params']['batch_size'] = wandb.config.train_batchSize
    config.update(data_config) # Merge data_config into config


    # create logger and path
    logger_path = os.path.join(wandb_run.dir, 'logs')  # Save logs inside wandb folder

    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info(f'Save log to {logger_path}')
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
    train_videos, val_videos, _ = prepare_video_splits(dataloader_config_path)
    train_batch_size = data_config['dataloader_params']['batch_size']
    if train_batch_size % 2 != 0:
        raise ValueError(f"train_batchSize must be even for 50/50 split, but got {train_batch_size}")
    half_batch_size = train_batch_size // 2

    all_train_loaders, val_method_loaders = create_method_aware_dataloaders(
        train_videos, val_videos, config, data_config, train_batch_size=half_batch_size
    )

    real_source_names = data_config['methods']['use_real_sources']
    real_loaders, fake_loaders = {}, {}
    for name, loader in all_train_loaders.items():
        (real_loaders if name in real_source_names else fake_loaders)[name] = loader
    logger.info(
        f"Created {len(real_loaders)} real loaders, {len(fake_loaders)} fake loaders, and {len(val_method_loaders)} validation loaders.")

    if args.run_sanity_check:
        logger.info("--- Running Sanity Check ---")
        # You can still run the check if you want by passing the argument
        comprehensive_sampler_check(...)  # Call the function if needed
        logger.info("Sanity check complete. Halting execution as planned.")
        wandb_run.finish()
        return

    # Prepare model, optimizer, scheduler, metric, trainer
    model = DETECTOR[config['model_name']](config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    trainer = Trainer(
        config, model, optimizer, scheduler, logger, metric_scoring,
        wandb_run=wandb_run, val_videos=val_videos
    )

    # --- Training Loop Setup ---
    real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
    for v in train_videos:
        (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1

    total_real_videos = sum(real_video_counts.values())
    total_fake_videos = sum(fake_video_counts.values())

    real_weights = [real_video_counts[m] / total_real_videos for m in
                    real_source_names] if total_real_videos > 0 else []
    fake_method_names = list(fake_loaders.keys())
    fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                    fake_method_names] if total_fake_videos > 0 else []

    total_train_videos = len(train_videos)
    epoch_len = math.ceil(total_train_videos / train_batch_size) if total_train_videos > 0 else 0
    logger.info(f"Total balanced training videos: {total_train_videos}, epoch length: {epoch_len} steps")

    # NEW: Get evaluation frequency from config
    eval_freq = data_config['data_params'].get('evaluation_frequency', 1)



    if config['gcs_assets']['base_checkpoint']['local_path']:
        trainer.load_ckpt(config['gcs_assets']['base_checkpoint']['local_path'])
    
    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.train_epoch(
            real_loaders=real_loaders,
            fake_loaders=fake_loaders,
            real_method_names=real_source_names,
            fake_method_names=fake_method_names,
            real_weights=real_weights,
            fake_weights=fake_weights,
            epoch=epoch,
            epoch_len=epoch_len,
            val_method_loaders=val_method_loaders,
            evaluation_frequency=eval_freq
        )
        if scheduler is not None: scheduler.step()

    wandb_run.finish()
    logger.info("Training complete.")


if __name__ == '__main__':
    start = time.time()
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="Effort-AIGI-Detection-Project" # Replace with your project name
    )
    # Start the sweep agent. It will call `run_training` for each set of hyperparameters.
    # `count` specifies how many runs to execute.
    wandb.agent(sweep_id, function=main, count=4) # Running 20 trials
    # main()
    end = time.time()
    elapsed = end - start
    print(f"Total training time in mn: {elapsed / 60:.2f} minutes")
    print("Training complete.")
