from venv import logger

import yaml  # noqa
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
from dataset.dataloaders import create_dataloaders, collate_fn  # noqa
from transformers import get_cosine_schedule_with_warmup  # noqa

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
from dataset.dataloaders import create_dataloaders, collate_fn, create_ood_loader  # noqa
from prepare_splits import prepare_video_splits_v2, prepare_ood_videos

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
parser.add_argument('--param-config', type=str, default=None,
                    help='YAML for single-run; omit to run sweep mode.')

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
    scheduler_type = config.get('lr_scheduler')  # Use .get for safety
    if scheduler_type is None or scheduler_type.lower() == 'none' or scheduler_type.lower() == 'null':
        return None

    if scheduler_type == 'cosine':
        # This branch remains for backward compatibility but is epoch-based
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['nEpochs'], eta_min=config['optimizer']['adam']['lr'] / 100
        )
    elif scheduler_type == 'cosine_with_warmup':
        if not config.get('total_training_steps'):
            raise ValueError("'total_training_steps' must be configured for the 'cosine_with_warmup' scheduler.")

        warmup_steps = config.get('lr_scheduler_warmup_steps', 0)
        total_steps = config['total_training_steps']

        print(
            f"INFO: Using cosine_with_warmup scheduler with {warmup_steps} warmup steps and {total_steps} total steps.")

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    raise NotImplementedError(f"Scheduler '{scheduler_type}' is not implemented")


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


def main():
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
    single_cfg = None
    if args.param_config:
        with open(args.param_config, "r") as f:
            single_cfg = yaml.safe_load(f) or {}

    wandb_run = wandb.init(
        mode="online",
        config=single_cfg  # None -> sweep agent supplies config; dict -> single run
    )

    # --- 1. Optimizer params (Search Space) ---
    config['load_base_checkpoint'] = wandb.config.load_base_checkpoint
    config['optimizer']['adam']['lr'] = wandb.config.learning_rate
    config['optimizer']['adam']['eps'] = wandb.config.optimizer_eps
    config['optimizer']['adam']['weight_decay'] = wandb.config.weight_decay
    config['nEpochs'] = wandb.config.nEpochs
    config['lambda_reg'] = wandb.config.lambda_reg  # Pass the sweep value to the main config
    config['rank'] = wandb.config.rank
    config['lr_scheduler'] = wandb.config.get('lr_scheduler', None)
    config['total_training_steps'] = wandb.config.get('total_training_steps', 35000)  # e.g., 35k
    config['lr_scheduler_warmup_steps'] = wandb.config.get('lr_scheduler_warmup_steps', 1000)  # e.g., 1k

    # --- Focal Loss params (from wandb.config) ---
    config['use_focal_loss'] = wandb.config.get('use_focal_loss', False)
    config['focal_loss_gamma'] = wandb.config.get('focal_loss_gamma', 2.0)
    config['focal_loss_alpha'] = wandb.config.get('focal_loss_alpha', None)

    # --- Group DRO params (from wandb.config) ---
    config['use_group_dro'] = wandb.config.get('use_group_dro', False)
    if config['use_group_dro']:
        config['group_dro_params'] = {
            'beta': wandb.config.get('group_dro_beta', 3.0),
            'clip_min': wandb.config.get('group_dro_clip_min', 1.0),
            'clip_max': wandb.config.get('group_dro_clip_max', 4.0),
            'ema_alpha': wandb.config.get('group_dro_ema_alpha', 0.1)
        }

    # ArcFace margin loss params (from wandb.config)
    config['use_arcface_head'] = wandb.config.get('use_arcface_head', False)
    if config['use_arcface_head']:
        config['arcface_s'] = wandb.config.get('arcface_s', 30.0)
        config['arcface_m'] = wandb.config.get('arcface_m', 0.35)
        config['s_start'] = wandb.config.get('s_start', config['arcface_s'])
        config['s_end'] = wandb.config.get('s_end', config['arcface_s'])
        config['anneal_steps'] = wandb.config.get('anneal_steps', 0)

    # Robustly handle the 'null' case from W&B sweeps
    focal_alpha = wandb.config.get('focal_loss_alpha', None)
    if focal_alpha == 'null' or focal_alpha == 'None':
        focal_alpha = None
    config['focal_loss_alpha'] = focal_alpha

    config['early_stopping'] = {
        'enabled': wandb.config.get('early_stopping_enabled', False),
        'patience': wandb.config.get('early_stopping_patience', 3),
        'min_delta': wandb.config.get('early_stopping_min_delta', 0.0001)
    }
    config['metric_scoring'] = 'auc'

    # --- 2. Data and Dataloader params (Search Space) ---
    data_config['dataloader_params']['strategy'] = wandb.config.dataloader_strategy
    data_config['dataloader_params']['frames_per_batch'] = wandb.config.frames_per_batch
    data_config['dataloader_params']['videos_per_batch'] = wandb.config.get('videos_per_batch', 8)
    data_config['dataloader_params']['frames_per_video'] = wandb.config.get('frames_per_video', 8)
    data_config['data_params']['val_split_ratio'] = wandb.config.val_split_ratio
    data_config['data_params']['evaluation_frequency'] = wandb.config.evaluation_frequency
    data_config['property_balancing']['enabled'] = wandb.config.property_balancing_enabled

    # --- 3. Data and Dataloader params (Fixed Context) ---
    data_config['data_params']['seed'] = wandb.config.seed
    data_config['data_params']['data_subset_percentage'] = wandb.config.data_subset_percentage
    data_config['dataloader_params']['test_batch_size'] = wandb.config.test_batch_size
    data_config['dataloader_params']['num_workers'] = wandb.config.num_workers
    data_config['dataloader_params']['prefetch_factor'] = wandb.config.prefetch_factor

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
    if wandb.config.get("name"):
        # take wandb.config.name and append timestamp to ensure uniqueness, md-HM
        timestamp = time.strftime("%m%d-%H%M")
        run_name = f"{wandb.config.name}_{timestamp}"
        wandb.run.name = run_name
    else:
        model_name = config.get('model_name', 'model')
        strategy = wandb.config.dataloader_strategy
        if strategy == 'frame_level':
            batch_info = f"frames{wandb.config.frames_per_batch}"
        else:  # video_level or per_method
            batch_info = f"vids{wandb.config.videos_per_batch}x{wandb.config.frames_per_video}f"
        lr = wandb.config.learning_rate
        wd = wandb.config.weight_decay
        eps = wandb.config.optimizer_eps
        local_rank = wandb.config.rank
        # num_frames = wandb.config.num_frames_per_video -- This was causing an error, seems it was renamed.
        subset_pct = wandb.config.data_subset_percentage
        run_name = (
            f"{model_name}"
            f"_{strategy}"
            f"_{batch_info}"
            f"_lr{lr:.0e}"
            f"_wd{wd:.0e}"
            f"_r{local_rank}"
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

    # --- 4. Augmentation params (from wandb.config) ---
    # Check if augmentation_params are defined in the W&B config (from sweep or YAML)
    if 'augmentation_params' in wandb.config and wandb.config.augmentation_params:
        # Convert the W&B Config object to a standard Python dictionary
        config['augmentation_params'] = dict(wandb.config.augmentation_params)
        logger.info("Successfully loaded augmentation parameters from the run's configuration.")
        logger.info(f"Augmentation settings: {config['augmentation_params']}")
    else:
        # Fallback to a default configuration if not provided, with a clear warning.
        # logger.warning("`augmentation_params` not found in the run's configuration. Using a default set.")
        config['augmentation_params'] = {
            "use_geometric": True,
            "use_advanced_noise": False,
            "use_color_jitter": True,
            "use_occlusion": True,
            "sharpness_adjust_prob": 0.6,
            "occlusion_prob": 0.4,
        }

    # NEW: Inject the top-level augmentation version into the params dict.
    # This allows the dataloader to select the correct pipeline.
    # Defaults to 'surgical' to maintain backward compatibility.
    aug_version = wandb.config.get('augmentation_version')
    if aug_version:
        config['augmentation_params']['version'] = aug_version
        logger.info(f"SET augmentation version to: {aug_version}")
    else:
        # Set a default if not specified in the sweep config
        config['augmentation_params']['version'] = 'surgical'
        logger.info("`augmentation_version` not in wandb config, defaulting to 'surgical'.")

    # --- 5. Dataset Method Override (from wandb.config) ---
    # This logic checks if the user has provided a method override in their
    # run config (e.g., train_parameters.yaml) and applies it.
    if 'dataset_methods' in wandb.config and wandb.config.dataset_methods:
        logger.info("--- Overriding dataset methods from the run's configuration. ---")

        # Create a reference to the override config for cleaner code
        method_overrides = wandb.config.dataset_methods

        # Check and override each list individually for maximum flexibility
        if 'use_real_sources' in method_overrides:
            data_config['methods']['use_real_sources'] = list(method_overrides['use_real_sources'])
            logger.info(f"Overriding REAL sources with: {data_config['methods']['use_real_sources']}")

        if 'use_fake_methods_for_training' in method_overrides:
            data_config['methods']['use_fake_methods_for_training'] = list(
                method_overrides['use_fake_methods_for_training'])
            logger.info(
                f"Overriding FAKE TRAINING methods with: {data_config['methods']['use_fake_methods_for_training']}")

        if 'use_fake_methods_for_validation' in method_overrides:
            data_config['methods']['use_fake_methods_for_validation'] = list(
                method_overrides['use_fake_methods_for_validation'])
            logger.info(
                f"Overriding FAKE VALIDATION methods with: {data_config['methods']['use_fake_methods_for_validation']}")

    else:
        logger.info("--- Using default dataset methods from ./config/dataloader_config.yml ---")
        # Log the defaults for clarity
        logger.info(f"Default FAKE TRAINING methods: {data_config['methods']['use_fake_methods_for_training']}")
        logger.info(f"Default FAKE VALIDATION methods: {data_config['methods']['use_fake_methods_for_validation']}")

    # --- 6. Property Balancing Weights (from wandb.config) ---
    # This section transfers the hierarchical sampling weights from the W&B run config
    # to the data_config, which is used by the dataloader. This is necessary for the
    # 'property_balancing' strategy to work correctly.
    if data_config.get('property_balancing', {}).get('enabled', False):
        logger.info("--- Transferring property balancing weights from run configuration ---")
        if 'real_category_weights' in wandb.config:
            # --- CORRECTED PATH: Place weights in 'dataloader_params' ---
            data_config['dataloader_params']['real_category_weights'] = dict(wandb.config.real_category_weights)
            logger.info(
                f"Loaded real_category_weights: {data_config['dataloader_params']['real_category_weights']}")
        else:
            logger.warning("`real_category_weights` not found in run config. Dataloader will likely fail.")
            data_config['dataloader_params']['real_category_weights'] = {}

        if 'fake_category_weights' in wandb.config:
            # --- CORRECTED PATH: Place weights in 'dataloader_params' ---
            data_config['dataloader_params']['fake_category_weights'] = dict(wandb.config.fake_category_weights)
            logger.info(
                f"Loaded fake_category_weights: {data_config['dataloader_params']['fake_category_weights']}")
        else:
            logger.warning("`fake_category_weights` not found in run config. Dataloader will likely fail.")
            data_config['dataloader_params']['fake_category_weights'] = {}

    # Conditionally remove the base checkpoint from the download list if not needed
    if not config.get('load_base_checkpoint', False):
        if 'gcs_assets' in config and 'base_checkpoint' in config['gcs_assets']:
            logger.info("`load_base_checkpoint` is False. Skipping download of the base checkpoint.")
            del config['gcs_assets']['base_checkpoint']

    # Download assets from GCS
    download_assets_from_gcs(config, logger)

    # Programmatically set the parquet path for property balancing ---
    # This ensures that the dataloader config uses the same local path defined in the gcs_assets.
    if (data_config.get('property_balancing', {}).get('enabled', False) and
            'property_manifest_parquet' in config.get('gcs_assets', {})):
        local_path = config['gcs_assets']['property_manifest_parquet']['local_path']
        data_config['property_balancing']['frame_properties_parquet_path'] = local_path
        logger.info(
            "Programmatically set 'frame_properties_parquet_path' for property balancing "
            f"to: {local_path}"
        )

    logger.info("------- Configuration & Data Loading -------")
    # MODIFIED: Unpack the three data splits (train, val_in_dist, val_holdout)
    train_data, val_in_dist_videos, val_holdout_videos, data_split_stats = prepare_video_splits_v2(data_config)

    ## ++ Transfer method_mapping to config ++ ##
    if config.get('use_group_dro', False):
        if 'method_mapping' in data_split_stats:
            # The Trainer expects the mapping to be inside 'data_params'
            config['data_params']['method_mapping'] = data_split_stats['method_mapping']
            logger.info("Successfully transferred method_mapping from data prep to main config for Group-DRO.")
        else:
            # Fail loudly if DRO is on but the mapping is missing. This prevents cryptic errors later.
            raise ValueError("Group-DRO is enabled, but 'method_mapping' was not found in data_split_stats. "
                             "Ensure prepare_splits.py is adding it.")

    # MODIFIED: Pass the two validation sets and receive three dataloaders
    train_loader, val_in_dist_loader, val_holdout_loader = create_dataloaders(
        train_data, val_in_dist_videos, val_holdout_videos, config, data_config
    )

    logger.info(
        f"DEBUG: is property balancing enabled? {data_config.get('property_balancing', {}).get('enabled', False)}")

    # --- Create OOD Loader ---
    logger.info("------- OOD Data Loading -------")
    ood_loader = None
    if data_config.get('gcp', {}).get('ood_bucket_name'):
        ood_videos = prepare_ood_videos(data_config)
        if ood_videos:
            ood_loader = create_ood_loader(ood_videos, config, data_config)
    else:
        logger.info("No 'ood_bucket_name' in config, skipping OOD loader creation.")

    # NEW: Combine validation sets for accurate overall statistics
    all_val_videos = val_in_dist_videos + val_holdout_videos

    # --- Create and log the comprehensive run overview ---
    real_methods = data_config.get('methods', {}).get('use_real_sources', [])
    # MODIFIED: We now have two sets of validation methods, so we combine them for logging
    train_fake_methods = data_config.get('methods', {}).get('use_fake_methods_for_training', [])
    val_fake_methods = data_config.get('methods', {}).get('use_fake_methods_for_validation', [])
    all_fake_methods_used = sorted(list(set(train_fake_methods + val_fake_methods)))

    # --- Validate and gather data for the overview ---
    # MODIFIED: Update validation counts to use the combined list
    data_split_stats['val_video_count'] = len(all_val_videos)
    data_split_stats['val_frame_count'] = sum(len(v.frame_paths) for v in all_val_videos)

    # --- Validate and gather data for the overview ---
    overview_data = {
        "Model": config.get('model_name'),
        # "Base Checkpoint": wandb.config.get('gcs_base_checkpoint'),
        "Run ID": wandb.run.id,
        "Discovered Videos": data_split_stats.get('discovered_videos'),
        "Discovered Methods": data_split_stats.get('discovered_methods'),
        "Data Subset Percentage": wandb.config.get('data_subset_percentage'),
        "Unbalanced Train Frames": data_split_stats.get('unbalanced_train_count'),
        "Unbalanced Val Videos": data_split_stats.get('unbalanced_val_count'),
        "Final Train Videos": data_split_stats.get('train_video_count'),
        "Final Train Frames": data_split_stats.get('train_frame_count'),
        "Final Val Videos": data_split_stats.get('val_video_count'),
        "Final Val Frames": data_split_stats.get('val_frame_count'),
        "Dataloader Strategy": wandb.config.get('dataloader_strategy'),

        # [FIXED] Add default 'N/A' for optional parameters. This prevents the
        # script from crashing when a strategy that doesn't use these
        # parameters (e.g., 'property_balancing') is selected.
        "Frames per Video": wandb.config.get('frames_per_video', 8),
        "Videos per Batch": wandb.config.get('videos_per_batch', 8),

        "Frames per Batch": wandb.config.get('frames_per_batch'),
        "Learning Rate": wandb.config.get('learning_rate'),
        "Weight Decay": wandb.config.get('weight_decay'),
        "Epsilon": wandb.config.get('optimizer_eps'),
        "Total Epochs": config.get('nEpochs'),
        "Eval Frequency": wandb.config.get('evaluation_frequency'),
    }

    # Check for any None values that would cause formatting errors
    for key, value in overview_data.items():
        if value is None:
            raise ValueError(
                f"'{key}' is None. Cannot generate run overview. Check your config and data processing steps.")

    # --- Create and log the comprehensive run overview ---
    overview_text = f"""
            ### Run Overview
            - **Model:** `{overview_data["Model"]}`
            - **Run ID:** `{overview_data["Run ID"]}`

            ### Data Split Details
            - **Discovered:** `{overview_data["Discovered Videos"]:,}` videos from `{overview_data["Discovered Methods"]}` methods.
            - **Unbalanced Pools:** Train: `{overview_data["Unbalanced Train Frames"]:,}` frames | Val: `{overview_data["Unbalanced Val Videos"]:,}` videos
            - **Final Training Set (Unbalanced):** `{overview_data["Final Train Videos"]:,}` videos (`{overview_data["Final Train Frames"]:,}` frames)
            - **Final Validation Set (Balanced):** `{overview_data["Final Val Videos"]:,}` videos (`{overview_data["Final Val Frames"]:,}` frames)

            ### Datasets Used
            - **Real Sources ({len(real_methods)}):** `{', '.join(real_methods)}`
            - **Fake Methods ({len(all_fake_methods_used)}):** `{', '.join(all_fake_methods_used)}`

            ### Sweep Hyperparameters
            - **Dataloader Strategy:** `{overview_data["Dataloader Strategy"]}`
            - **Frames per Video:** `{overview_data["Frames per Video"]}`
            - **Videos per Batch:** `{overview_data["Videos per Batch"]}`
            - **Frames per Batch:** `{overview_data["Frames per Batch"]}`
            - **Learning Rate:** `{overview_data["Learning Rate"]:.1e}`
            - **Weight Decay:** `{overview_data["Weight Decay"]:.1e}`
            - **Epsilon:** `{overview_data["Epsilon"]:.1e}`
            - **Total Epochs:** `{overview_data["Total Epochs"]}`
            - **Eval Frequency:** `{overview_data["Eval Frequency"]}` per epoch
            """
    wandb.run.summary["run_overview"] = overview_text.strip()

    # --- Log detailed dataset balance statistics ---
    real_source_names = data_config['methods']['use_real_sources']
    is_property_balancing = data_config.get('property_balancing', {}).get('enabled', False)

    # conditionally handle list of dicts vs. list of objects
    # Calculate per-method counts for the balanced training set
    if is_property_balancing:
        # train_data is a list of frame dictionaries; count frames per method
        train_counts = Counter(frame['method'] for frame in train_data)  # Use train_data
    else:
        # train_data is a list of VideoInfo objects; count videos per method
        train_counts = Counter(v.method for v in train_data)  # Use train_data

    # Calculate per-method counts for the balanced training set
    train_real_count = sum(count for method, count in train_counts.items() if method in real_source_names)
    train_fake_count = sum(count for method, count in train_counts.items() if method not in real_source_names)

    # Create a W&B Table for detailed counts
    data_table = wandb.Table(columns=["Set", "Type", "Method", "Count"])
    for method, count in train_counts.items():
        data_type = "real" if method in real_source_names else "fake"
        data_table.add_data("train", data_type, method, count)

    # Also get validation counts and add them to the table
    # MODIFIED: Use the combined 'all_val_videos' list for validation counts
    val_counts = Counter(v.method for v in all_val_videos)
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

    # Prepare model, optimizer, scheduler, metric, trainer
    model = DETECTOR[config['model_name']](config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    # MODIFIED: Pass the two new validation loaders directly to the Trainer
    trainer = Trainer(
        config, model, optimizer, scheduler, logger,
        val_in_dist_loader=val_in_dist_loader,
        val_holdout_loader=val_holdout_loader,
        metric_scoring=metric_scoring,
        wandb_run=wandb_run,
        ood_loader=ood_loader,
        use_group_dro=config.get('use_group_dro', False)
    )

    if config.get('load_base_checkpoint', False):
        checkpoint_path = config.get('gcs_assets', {}).get('base_checkpoint', {}).get('local_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"--- Loading base checkpoint from {checkpoint_path} as requested by config. ---")
            trainer.load_ckpt(checkpoint_path)
        else:
            logger.warning(
                f"Configuration 'load_base_checkpoint' is True, but no valid checkpoint was found at '{checkpoint_path}'. "
                "The model will start from the base CLIP weights."
            )
    else:
        logger.info(
            "--- Configuration 'load_base_checkpoint' is False. "
            "Skipping checkpoint load. The model will start from the base CLIP weights. ---"
        )

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        # MODIFIED: The call is now simpler. The trainer handles its own validation loaders.
        trainer.train_epoch(
            train_loader=train_loader,
            epoch=epoch,
            train_videos=train_data
        )

        if trainer.early_stop_triggered:
            logger.info(f"Gracefully terminating training at epoch {epoch + 1} due to early stopping.")
            wandb.log({"train/status": "Early Stopped"})  # Log final status
            break

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
