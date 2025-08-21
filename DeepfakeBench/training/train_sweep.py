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
from detectors import DETECTOR  # noqa
from PIL.ImageFilter import RankFilter  # noqa
from dataset.dataloaders import create_method_aware_dataloaders, collate_fn  # noqa

import argparse
import random
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
from logger import create_logger
from PIL.ImageFilter import RankFilter  # noqa
from prepare_splits import prepare_video_splits
from dataset.dataloaders import create_method_aware_dataloaders, collate_fn  # noqa

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
args, _ = parser.parse_known_args()
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
            eps=config['optimizer'][opt_name]['eps'],  # Added eps
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


# train_sweep.py - function replacement

def main():
    # ##################### ADAM CHANGED ###################
    # os.chdir("/home/roee/repos/Effort-AIGI-Detection/DeepfakeBench/training")
    # ##################### ADAM CHANGED ###################
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/train_config.yaml', 'r') as f:
        config.update(yaml.safe_load(f))

    dataloader_config_path = args.dataloader_config
    with open(dataloader_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # --- W&B Initialization ---
    # The agent provides the config for the run
    wandb_run = wandb.init(
        mode="online",
    )

    # --- MODIFIED: Map more hyperparameters from the sweep config ---
    # The single source of truth is now wandb.config, provided by the agent.

    # --- 1. Optimizer params (Search Space) ---
    config['optimizer']['adam']['lr'] = wandb.config.learning_rate
    config['optimizer']['adam']['eps'] = wandb.config.optimizer_eps
    config['optimizer']['adam']['weight_decay'] = wandb.config.weight_decay
    config['nEpochs'] = wandb.config.nEpochs
    config['lambda_reg'] = wandb.config.lambda_reg  # Pass the sweep value to the main config

    # --- 2. Data and Dataloader params (Search Space) ---
    data_config['dataloader_params']['batch_size'] = wandb.config.train_batchSize
    data_config['data_params']['num_frames_per_video'] = wandb.config.num_frames_per_video
    data_config['data_params']['val_split_ratio'] = wandb.config.val_split_ratio
    data_config['data_params']['evaluation_frequency'] = wandb.config.evaluation_frequency

    # --- 3. Data and Dataloader params (Fixed Context) ---
    data_config['data_params']['seed'] = wandb.config.seed
    data_config['data_params']['data_subset_percentage'] = wandb.config.data_subset_percentage
    data_config['dataloader_params']['test_batch_size'] = wandb.config.test_batch_size
    data_config['dataloader_params']['num_workers'] = wandb.config.num_workers
    data_config['dataloader_params']['prefetch_factor'] = wandb.config.prefetch_factor

    # Compensating. usually num_frames_per_video is 8.
    if data_config['data_params']['num_frames_per_video'] == 4:
        data_config['dataloader_params']['batch_size'] = data_config['dataloader_params']['batch_size'] * 2
    if data_config['data_params']['num_frames_per_video'] == 2:
        data_config['dataloader_params']['batch_size'] = data_config['dataloader_params']['batch_size'] * 4

    # Update the main config with the now-populated data_config
    config.update(data_config)

    # Log a curated snapshot of the config for filtering in W&B
    curated_config_log = {
        'metric_scoring': config.get('metric_scoring'),
        'nEpochs': config.get('nEpochs'),
        'model_name': config.get('model_name'),
        'data_params': data_config.get('data_params'),
        'dataloader_params': data_config.get('dataloader_params'),
        'gcs_base_checkpoint': config.get('gcs_assets', {}).get('base_checkpoint', {}).get('gcs_path', 'N/A'),
    }
    wandb.config.update(curated_config_log, allow_val_change=True)

    # Construct and set an informative run name
    model_name = config.get('model_name', 'model')
    batch_size = data_config['dataloader_params']['batch_size']
    lr = wandb.config.learning_rate
    wd = wandb.config.weight_decay
    eps = wandb.config.optimizer_eps
    num_frames = wandb.config.num_frames_per_video
    subset_pct = wandb.config.data_subset_percentage
    # subset_pct = int(data_config.get('data_params', {}).get('data_subset_percentage', 1.0) * 100)
    run_name = (
        f"{model_name}"
        f"_b{batch_size}"
        f"_lr{lr:.0e}"
        f"_wd{wd:.0e}"
        f"_eps{eps:.0e}"
        f"_frames{num_frames}"
        f"_subset{subset_pct}"
    ).replace("+", "")
    wandb.run.name = run_name

    # Standard setup
    config['local_rank'] = args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    logger_path = os.path.join(wandb_run.dir, 'logs')
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info(f'Save log to {logger_path}')
    config['ddp'] = args.ddp
    init_seed(config)
    if config['cudnn']: cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    # Download assets from GCS
    download_assets_from_gcs(config, logger)

    logger.info("------- Configuration & Data Loading -------")
    # MODIFIED: Capture the fourth return value: `data_split_stats`
    train_videos, val_videos, _, data_split_stats = prepare_video_splits(dataloader_config_path)

    # Count "external_youtube_avspeech" in train and validation sets
    train_avspeech_count = sum(1 for v in train_videos if v.method == "external_youtube_avspeech")
    val_avspeech_count = sum(1 for v in val_videos if v.method == "external_youtube_avspeech")

    logger.info(f"Count of 'external_youtube_avspeech' in train set: {train_avspeech_count}")
    logger.info(f"Count of 'external_youtube_avspeech' in val set: {val_avspeech_count}")

    # --- Create and log the comprehensive run overview ---
    real_methods = data_config.get('methods', {}).get('use_real_sources', [])
    fake_methods = data_config.get('methods', {}).get('use_fake_methods', [])
    overview_text = f"""
        ### Run Overview
        - **Model:** `{config.get('model_name', 'N/A')}`
        - **Base Checkpoint:** `{wandb.config.gcs_base_checkpoint}`
        - **Run ID:** `{wandb.run.id}`

        ### Data Split Details
        - **Discovered:** `{data_split_stats.get('discovered_videos'):,}` videos from `{data_split_stats.get('discovered_methods')}` methods.
        - **Subset:** Using `{wandb.config.data_params.get('data_subset_percentage'):.1%}` of data (`{data_split_stats.get('subset_video_count'):,}` videos).
        - **Unbalanced Split:** Train: `{data_split_stats.get('unbalanced_train_count'):,}` | Val: `{data_split_stats.get('unbalanced_val_count'):,}`
        - **Final Balanced Split:** Train: `{data_split_stats.get('balanced_train_count'):,}` | Val: `{data_split_stats.get('balanced_val_count'):,}`

        ### Datasets Used
        - **Real Sources ({len(real_methods)}):** `{', '.join(real_methods)}`
        - **Fake Methods ({len(fake_methods)}):** `{', '.join(fake_methods)}`

        ### Sweep Hyperparameters
        - **Batch Size:** `{wandb.config.train_batchSize}`
        - **Learning Rate:** `{wandb.config.learning_rate:.1e}`
        - **Weight Decay:** `{wandb.config.weight_decay:.1e}`
        - **Epsilon:** `{wandb.config.optimizer_eps:.1e}`
        - **Frames per Video:** `{wandb.config.num_frames_per_video}`
        - **Total Epochs:** `{config.get('nEpochs', 'N/A')}`
        - **Eval Frequency:** `{wandb.config.evaluation_frequency}` per epoch
        """
    wandb.run.summary["run_overview"] = overview_text.strip()

    # --- Log detailed dataset balance statistics ---
    real_source_names = data_config['methods']['use_real_sources']

    # Calculate per-method counts for the balanced training set
    train_counts = Counter(v.method for v in train_videos)
    train_real_count = sum(count for method, count in train_counts.items() if method in real_source_names)
    train_fake_count = sum(count for method, count in train_counts.items() if method not in real_source_names)

    # Create a W&B Table for detailed counts
    data_table = wandb.Table(columns=["Set", "Type", "Method", "Count"])
    for method, count in train_counts.items():
        data_type = "real" if method in real_source_names else "fake"
        data_table.add_data("train", data_type, method, count)

    # Also get validation counts and add them to the table
    val_counts = Counter(v.method for v in val_videos)
    val_real_count = sum(count for method, count in val_counts.items() if method in real_source_names)
    val_fake_count = sum(count for method, count in val_counts.items() if method not in real_source_names)
    for method, count in val_counts.items():
        data_type = "real" if method in real_source_names else "fake"
        data_table.add_data("val", data_type, method, count)

    # Log the table and scalar metrics
    wandb.log({
        "data/method_counts": data_table,
        "data/num_real_train": train_real_count,
        "data/num_fake_train": train_fake_count,
        "data/num_real_val": val_real_count,
        "data/num_fake_val": val_fake_count,
    })
    logger.info("Logged detailed dataset balance statistics to W&B.")

    train_batch_size = data_config['dataloader_params']['batch_size']
    if train_batch_size % 2 != 0:
        raise ValueError(f"train_batchSize must be even for 50/50 split, but got {train_batch_size}")
    half_batch_size = train_batch_size // 2

    all_train_loaders, val_method_loaders = create_method_aware_dataloaders(
        train_videos, val_videos, config, data_config, train_batch_size=half_batch_size
    )

    # The LazyDataLoaderManager is passed directly to the trainer.
    # We derive the method names for sampling from the manager's keys,
    # and the manager will handle on-demand loading of both real and fake batches.
    real_source_names = data_config['methods']['use_real_sources']
    all_method_names = all_train_loaders.keys()

    real_method_names = [m for m in all_method_names if m in real_source_names]
    fake_method_names = [m for m in all_method_names if m not in real_source_names]

    # The manager is used to retrieve both real and fake loaders as needed.
    real_loaders = all_train_loaders
    fake_loaders = all_train_loaders

    logger.info(
        f"Created manager for {len(real_method_names)} real methods, {len(fake_method_names)} fake methods, and {len(val_method_loaders.keys())} validation loaders.")

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
                    real_method_names] if total_real_videos > 0 else []
    fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                    fake_method_names] if total_fake_videos > 0 else []

    total_train_videos = len(train_videos)
    epoch_len = math.ceil(total_train_videos / train_batch_size) if total_train_videos > 0 else 0
    logger.info(f"Total balanced training videos: {total_train_videos}, epoch length: {epoch_len} steps")

    eval_freq = wandb.config.evaluation_frequency

    if config['gcs_assets']['base_checkpoint']['local_path']:
        trainer.load_ckpt(config['gcs_assets']['base_checkpoint']['local_path'])

    # --- Initial Validation before Training ---
    logger.info("--- Performing initial validation on the base model before training ---")
    if config['local_rank'] == 0:
        # Ensure there are validation loaders to test on.
        # test_epoch is safe to call with empty loaders, but this check is cleaner.
        if val_method_loaders and len(val_method_loaders.keys()) > 0:
            trainer.test_epoch(epoch=-1, val_method_loaders=val_method_loaders)
            logger.info("--- Initial validation complete. Starting training. ---")
        else:
            logger.warning("No validation loaders found. Skipping initial validation.")

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.train_epoch(
            real_loaders=real_loaders,
            fake_loaders=fake_loaders,
            real_method_names=real_method_names,
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
    # The W&B agent will call the main function directly.
    # No need for sweep initialization logic here.
    main()
    end = time.time()
    elapsed = end - start
    print(f"Total training time in mn: {elapsed / 60:.2f} minutes")
    print("Training complete.")
