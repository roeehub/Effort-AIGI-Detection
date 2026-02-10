from venv import logger

# =============================================================================
# CODE VERSION STAMP - Update this when making changes to verify deployment
# =============================================================================
CODE_VERSION = "2026-01-01-UNIFIED-V1"  # Unified training with data source factory
# =============================================================================

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
import pandas as pd  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
from google.cloud.storage import Bucket  # noqa

from detectors import DETECTOR  # noqa
from PIL.ImageFilter import RankFilter  # noqa
from dataset.dataloaders import create_dataloaders, collate_fn  # noqa
from transformers import get_cosine_schedule_with_warmup  # noqa

# ==============================================================================
# --- Utility Imports from Refactored Module ---
# ==============================================================================
from utils import (
    # GCS utilities
    download_gcs_asset,
    download_assets_from_gcs,
    # Setup utilities
    init_seed,
    choose_optimizer,
    choose_scheduler,
    choose_metric,
    # Config helpers (Phase 5 refactoring)
    load_base_configs,
    apply_all_wandb_overrides,
    generate_run_name,
    create_curated_config_log,
)

# ==============================================================================
# --- Data Source Factory (Unified Training) ---
# ==============================================================================
from data.sources import create_data_pipeline, DataPipelineResult

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
import torch.distributed as dist  # noqa
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset  # noqa
# --- add near other torch imports ---
from torch.utils.data import IterableDataset  # noqa

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


# ==============================================================================
# --- UTILITY FUNCTIONS MOVED TO utils/ MODULE ---
# ==============================================================================
# 
# The following functions have been refactored to:
#   - utils/setup.py: init_seed, choose_optimizer, choose_scheduler, choose_metric
#   - utils/gcs.py: download_gcs_asset, download_assets_from_gcs
#
# They are imported at the top of this file from the utils module.
# ==============================================================================


def main():
    # ===========================================================================
    # CODE VERSION VERIFICATION - This confirms which code is actually running
    # ===========================================================================
    print("=" * 70)
    print(f"🚀 TRAIN_SWEEP.PY CODE VERSION: {CODE_VERSION}")
    print(f"📋 args.param_config = {args.param_config}")
    print(f"📋 All args: {args}")
    print("=" * 70)
    
    # ===========================================================================
    # --- 1. Load Base Configs ---
    # ===========================================================================
    config, data_config = load_base_configs(
        detector_path=args.detector_path,
        dataloader_config_path=args.dataloader_config,
    )

    # --- W&B Initialization ---
    single_cfg = None
    if args.param_config:
        with open(args.param_config, "r") as f:
            single_cfg = yaml.safe_load(f) or {}
        # Debug: Log what we loaded from the param config file
        print(f"[DEBUG] Loaded single_cfg from {args.param_config}")
        print(f"[DEBUG] single_cfg keys: {list(single_cfg.keys()) if single_cfg else 'None'}")
        if single_cfg and 'dataset_methods' in single_cfg:
            print(f"[DEBUG] dataset_methods found in single_cfg: {list(single_cfg['dataset_methods'].keys())}")
        else:
            print(f"[DEBUG] WARNING: 'dataset_methods' NOT in single_cfg!")

    wandb_run = wandb.init(
        mode="online",
        config=single_cfg  # None -> sweep agent supplies config; dict -> single run
    )
    
    # Debug: Log what wandb.config received
    print(f"[DEBUG] wandb.config type: {type(wandb.config)}")
    print(f"[DEBUG] wandb.config keys (first 30): {list(wandb.config.keys())[:30]}")
    if 'dataset_methods' in wandb.config.keys():
        print(f"[DEBUG] dataset_methods in wandb.config: {wandb.config['dataset_methods']}")
    else:
        print(f"[DEBUG] WARNING: 'dataset_methods' NOT in wandb.config!")

    # ===========================================================================
    # --- 2. Apply All W&B Config Overrides (Refactored) ---
    # ===========================================================================
    # This replaces ~150 lines of manual W&B config mapping with a single call.
    # Individual helper functions are available in utils/config_helpers.py for
    # granular control if needed.
    # ===========================================================================
    
    # Create logger early so we can pass it to config helpers
    logger_path = os.path.join(wandb_run.dir, 'logs')
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info(f'Save log to {logger_path}')
    
    # ===========================================================================
    # CRITICAL FIX: Apply single_cfg directly to data_config BEFORE wandb overrides
    # W&B flattens nested dicts, so wandb.config.get('dataset_methods') returns None
    # even if single_cfg has it. We must apply these directly.
    # ===========================================================================
    if single_cfg:
        print("=" * 70)
        print("--- Applying single_cfg directly (bypassing W&B flattening) ---")
        logger.info("--- Applying single_cfg directly (bypassing W&B flattening) ---")
        
        # Apply dataset_methods directly
        if 'dataset_methods' in single_cfg:
            data_config['dataset_methods'] = single_cfg['dataset_methods']
            print(f"  ✅ Applied dataset_methods: {list(single_cfg['dataset_methods'].keys())}")
            logger.info(f"  Applied dataset_methods: {list(single_cfg['dataset_methods'].keys())}")
        else:
            print(f"  ⚠️ 'dataset_methods' NOT in single_cfg! Keys: {list(single_cfg.keys())}")
        
        # Apply lesson_data_control directly  
        if 'lesson_data_control' in single_cfg:
            config['lesson_data_control'] = single_cfg['lesson_data_control']
            print(f"  ✅ Applied lesson_data_control: enabled={single_cfg['lesson_data_control'].get('enabled')}")
            logger.info(f"  Applied lesson_data_control: enabled={single_cfg['lesson_data_control'].get('enabled')}")
        else:
            print(f"  ⚠️ 'lesson_data_control' NOT in single_cfg! Keys: {list(single_cfg.keys())}")
        
        # Apply lesson_gate directly
        if 'lesson_gate' in single_cfg:
            config['lesson_gate'] = single_cfg['lesson_gate']
            print(f"  ✅ Applied lesson_gate: enabled={single_cfg['lesson_gate'].get('enabled')}")
            logger.info(f"  Applied lesson_gate: enabled={single_cfg['lesson_gate'].get('enabled')}")
        else:
            print(f"  ⚠️ 'lesson_gate' NOT in single_cfg! Keys: {list(single_cfg.keys())}")
        
        # Apply augmentation config directly (W&B flattens nested dicts)
        # This is critical for DeepLive which uses config['augmentation'] for landmark occlusion
        if 'augmentation' in single_cfg:
            config['augmentation'] = single_cfg['augmentation']
            data_config['augmentation'] = single_cfg['augmentation']  # Also in data_config for deeplive
            aug_version = single_cfg['augmentation'].get('version', 'unknown')
            print(f"  ✅ Applied augmentation: version={aug_version}")
            logger.info(f"  Applied augmentation: version={aug_version}")
        else:
            print(f"  ⚠️ 'augmentation' NOT in single_cfg! Keys: {list(single_cfg.keys())}")
        
        # Apply deeplive config directly (for GCS bucket settings, etc.)
        if 'deeplive' in single_cfg:
            data_config['deeplive'] = single_cfg['deeplive']
            print(f"  ✅ Applied deeplive config")
            logger.info(f"  Applied deeplive config")
        
        # Apply backbone config directly
        if 'backbone' in single_cfg:
            config['backbone'] = single_cfg['backbone']
            print(f"  ✅ Applied backbone: {single_cfg['backbone'].get('name', 'unknown')}")
            logger.info(f"  Applied backbone: {single_cfg['backbone'].get('name', 'unknown')}")
        
        print("=" * 70)
    else:
        print("⚠️ single_cfg is None/empty - no direct config application!")
    
    # Apply all W&B overrides to config and data_config (handles flat keys)
    apply_all_wandb_overrides(config, data_config, wandb.config, logger)
    
    # IMPORTANT: Apply gcs_assets AFTER wandb overrides, because apply_wandb_backbone_params
    # constructs default GCS paths that we may need to override for OpenCLIP models
    if single_cfg and 'gcs_assets' in single_cfg:
        print("--- Applying gcs_assets override (post-wandb) ---")
        logger.info("--- Applying gcs_assets override (post-wandb) ---")
        if 'gcs_assets' not in config:
            config['gcs_assets'] = {}
        for asset_key, asset_config in single_cfg['gcs_assets'].items():
            config['gcs_assets'][asset_key] = asset_config
            print(f"  ✅ Override gcs_assets['{asset_key}']: {asset_config.get('gcs_path', 'no path')}")
            logger.info(f"  Override gcs_assets['{asset_key}']: {asset_config.get('gcs_path', 'no path')}")

    # Log curated config snapshot for W&B filtering
    curated_config_log = create_curated_config_log(config, data_config)
    wandb.config.update(curated_config_log, allow_val_change=True)

    # Generate and set run name
    wandb.run.name = generate_run_name(config, wandb.config)

    # ===========================================================================
    # --- 3. Standard Setup ---
    # ===========================================================================
    config['local_rank'] = args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['ddp'] = args.ddp
    init_seed(config)
    if config['cudnn']: cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    # ===========================================================================
    # --- 4. GCS Assets & Checkpoint Configuration ---
    # ===========================================================================
    # Conditionally remove the base checkpoint from the download list if not needed
    if not config.get('load_base_checkpoint', False):
        if 'gcs_assets' in config and 'base_checkpoint' in config['gcs_assets']:
            logger.info("`load_base_checkpoint` is False. Skipping download of the base checkpoint.")
            del config['gcs_assets']['base_checkpoint']

    # override the config gcs_assets - base_checkpoint - gcs_path with the wandb.config
    if wandb.config.get('gcs_base_checkpoint') and config.get('load_base_checkpoint', False):
        config['gcs_assets']['base_checkpoint']['gcs_path'] = wandb.config.get('gcs_base_checkpoint')
        logger.info(f"Overrode base checkpoint GCS path to: {config['gcs_assets']['base_checkpoint']['gcs_path']}")

    # Download assets from GCS
    downloaded_assets = download_assets_from_gcs(config, logger)
    if downloaded_assets is None and config.get('gcs_assets'):
        logger.error("Failed to download required GCS assets. Exiting.")
        raise RuntimeError("GCS asset download failed. Check logs for details.")

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

    # Final merge: Ensure the main config has the fully updated data_config
    # including all overrides from the wandb run.
    config.update(data_config)

    # ===========================================================================
    # --- 5. UNIFIED DATA PIPELINE (Factory Pattern) ---
    # ===========================================================================
    # This replaces ~130 lines of manual data loading with a single factory call.
    # The data source is determined by data_config['data_source']:
    #   - 'manifest' (default): Traditional GCS manifest-based loading
    #   - 'deeplive': DeepLive GCS bucket with paired real/fake frames
    #   - Future: 'hf_dataset' for HuggingFace datasets
    # ===========================================================================
    
    # Set default data source if not specified (backward compatibility)
    if 'data_source' not in data_config:
        data_config['data_source'] = 'manifest'
        logger.info("No 'data_source' specified, defaulting to 'manifest'")
    
    # Create the unified data pipeline
    pipeline_result: DataPipelineResult = create_data_pipeline(config, data_config, logger)
    
    # Extract results
    train_loader = pipeline_result.train_loader
    val_in_dist_loader = pipeline_result.val_in_dist_loader
    val_holdout_loader = pipeline_result.val_holdout_loader
    train_data = pipeline_result.train_samples
    data_split_stats = pipeline_result.data_stats
    ood_loader = pipeline_result.ood_loader
    
    # Handle Group DRO method mapping if it was set during pipeline creation
    if config.get('use_group_dro', False):
        if 'data_params' in config and 'method_mapping' in config.get('data_params', {}):
            logger.info("Group-DRO method_mapping already set by data pipeline.")
        elif 'method_mapping' in data_split_stats:
            if 'data_params' not in config:
                config['data_params'] = {}
            config['data_params']['method_mapping'] = data_split_stats['method_mapping']
            logger.info("Transferred method_mapping from data pipeline to config for Group-DRO.")
    
    # Get validation videos for statistics (backward compatibility)
    val_in_dist_videos = data_split_stats.get('val_in_dist_videos', [])
    val_holdout_videos = data_split_stats.get('val_holdout_videos', [])
    all_val_videos = data_split_stats.get('all_val_videos', val_in_dist_videos + val_holdout_videos)
    
    # Update stats if not already present
    if 'val_video_count' not in data_split_stats:
        data_split_stats['val_video_count'] = len(all_val_videos)
    if 'val_frame_count' not in data_split_stats and all_val_videos:
        try:
            data_split_stats['val_frame_count'] = sum(len(v.frame_paths) for v in all_val_videos)
        except AttributeError:
            # DeepLive samples don't have frame_paths attribute
            data_split_stats['val_frame_count'] = data_split_stats.get('val_samples', 0) * 16
    
    is_property_balancing = data_config.get('property_balancing', {}).get('enabled', False)
    current_data_source = data_config.get('data_source', 'manifest')
    logger.info(f"Data pipeline created successfully (data_source={current_data_source}, property_balancing={is_property_balancing})")

    # --- Create and log the comprehensive run overview ---
    # Handle different data sources - DeepLive uses 'strategies' instead of 'methods'
    if current_data_source == 'deeplive':
        # DeepLive: Get strategies from data_stats
        strategies = data_split_stats.get('strategies', {})
        real_methods = ['paired_real']  # DeepLive always has paired real frames
        all_fake_methods_used = sorted(list(strategies.keys()))
    elif current_data_source == 'df40_paired':
        # DF40 Paired: Get methods from data_stats
        methods = data_split_stats.get('methods', {})
        real_methods = ['paired_real']  # DF40 paired always has paired real frames
        all_fake_methods_used = sorted(list(methods.keys()))
    elif current_data_source == 'combined_paired':
        # Combined Paired: Get methods from data_stats
        methods = data_split_stats.get('methods', [])
        real_methods = ['paired_real']  # Combined paired always has paired real frames
        all_fake_methods_used = sorted(list(methods)) if isinstance(methods, (list, set)) else sorted(list(methods.keys()))
    else:
        # Manifest-based: Get methods from config
        real_methods = data_config.get('methods', {}).get('use_real_sources', [])
        train_fake_methods = data_config.get('methods', {}).get('use_fake_methods_for_training', [])
        val_fake_methods = data_config.get('methods', {}).get('use_fake_methods_for_validation', [])
        all_fake_methods_used = sorted(list(set(train_fake_methods + val_fake_methods)))


    # --- Validate and gather data for the overview ---
    # Only update counts if not already set by the data source (e.g., DeepLive sets its own)
    if 'val_video_count' not in data_split_stats or data_split_stats['val_video_count'] is None:
        data_split_stats['val_video_count'] = len(all_val_videos) if all_val_videos else 0
    if 'val_frame_count' not in data_split_stats or data_split_stats['val_frame_count'] is None:
        try:
            data_split_stats['val_frame_count'] = sum(len(v.frame_paths) for v in all_val_videos) if all_val_videos else 0
        except (AttributeError, TypeError):
            # DeepLive samples don't have frame_paths attribute
            data_split_stats['val_frame_count'] = data_split_stats.get('val_samples', 0) * 16

    # --- Validate and gather data for the overview ---
    # Determine data source for conditional defaults
    current_data_source = data_config.get('data_source', 'manifest')
    
    overview_data = {
        "Model": config.get('model_name'),
        # "Base Checkpoint": wandb.config.get('gcs_base_checkpoint'),
        "Run ID": wandb.run.id,
        "Discovered Videos": data_split_stats.get('discovered_videos'),
        "Discovered Methods": data_split_stats.get('discovered_methods'),
        # For DeepLive, use train_split as the "subset" equivalent; for manifest, use data_subset_percentage
        "Data Subset Percentage": wandb.config.get('data_subset_percentage') or data_split_stats.get('train_split', 1.0),
        "Unbalanced Train Frames": data_split_stats.get('unbalanced_train_count'),
        "Unbalanced Val Videos": data_split_stats.get('unbalanced_val_count'),
        "Final Train Videos": data_split_stats.get('train_video_count'),
        "Final Train Frames": data_split_stats.get('train_frame_count'),
        "Final Val Videos": data_split_stats.get('val_video_count'),
        "Final Val Frames": data_split_stats.get('val_frame_count'),
        "Dataloader Strategy": wandb.config.get('dataloader_strategy') or current_data_source,

        # [FIXED] Add default values for optional parameters. This prevents the
        # script from crashing when a strategy that doesn't use these
        # parameters (e.g., 'property_balancing', 'deeplive') is selected.
        "Frames per Video": wandb.config.get('frames_per_video', 8),
        "Videos per Batch": wandb.config.get('videos_per_batch', 8),

        "Frames per Batch": wandb.config.get('frames_per_batch') or config.get('frames_per_batch', 32),
        "Learning Rate": wandb.config.get('learning_rate') or config.get('learning_rate', 1e-4),
        "Weight Decay": wandb.config.get('weight_decay') or config.get('weight_decay', 0.05),
        "Epsilon": wandb.config.get('optimizer_eps') or config.get('optimizer_eps', 1e-8),
        "Total Epochs": config.get('nEpochs'),
        "Eval Frequency": wandb.config.get('evaluation_frequency') or config.get('evaluation_frequency', 1),
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
    real_source_names = data_config.get('dataset_methods', {}).get('use_real_sources', [])
    is_property_balancing = data_config.get('property_balancing', {}).get('enabled', False)

    # Handle different data sources for train_data structure
    if current_data_source == 'deeplive':
        # DeepLive: Each sample contains BOTH real and fake frames (it's a pair)
        # So num_real = num_fake = num_samples
        num_samples = len(train_data)
        train_counts = Counter(getattr(s, 'strategy', 'unknown') for s in train_data)
        # For DeepLive, we report paired counts
        train_real_count = num_samples  # Each sample has real frames
        train_fake_count = num_samples  # Each sample has fake frames
        real_source_names = []  # Not used for DeepLive counting
    elif current_data_source == 'df40_paired':
        # DF40 Paired: Each sample is a DF40PairedSample object with .method attribute
        # Similar to DeepLive - each sample contains BOTH real and fake frames
        num_samples = len(train_data)
        train_counts = Counter(getattr(s, 'method', 'unknown') for s in train_data)
        # For DF40 paired, we report paired counts
        train_real_count = num_samples  # Each sample has real frames
        train_fake_count = num_samples  # Each sample has fake frames
        real_source_names = []  # Not used for DF40 paired counting
    elif current_data_source == 'combined_paired':
        # Combined Paired: Each sample is a UnifiedPairedSample object with .method attribute
        # Similar to DeepLive/DF40 - each sample contains BOTH real and fake frames
        num_samples = len(train_data)
        train_counts = Counter(getattr(s, 'method', 'unknown') for s in train_data)
        # For combined paired, we report paired counts
        train_real_count = num_samples  # Each sample has real frames
        train_fake_count = num_samples  # Each sample has fake frames
        real_source_names = []  # Not used for combined paired counting
    elif is_property_balancing:
        # Property-balancing: train_data is a list of frame dictionaries; count frames per method
        train_counts = Counter(frame['method'] for frame in train_data)
        # Calculate per-method counts for the balanced training set
        train_real_count = sum(count for method, count in train_counts.items() if method in real_source_names)
        train_fake_count = sum(count for method, count in train_counts.items() if method not in real_source_names)
    else:
        # Manifest-based: train_data is a list of VideoInfo objects; count videos per method
        train_counts = Counter(v.method for v in train_data)
        # Calculate per-method counts for the balanced training set
        train_real_count = sum(count for method, count in train_counts.items() if method in real_source_names)
        train_fake_count = sum(count for method, count in train_counts.items() if method not in real_source_names)

    # Create a W&B Table for detailed counts
    data_table = wandb.Table(columns=["Set", "Type", "Method", "Count"])
    
    if current_data_source == 'deeplive':
        # DeepLive: Each sample is a PAIR (real + fake), so log both types per strategy
        for method, count in train_counts.items():
            data_table.add_data("train", "real", method, count)
            data_table.add_data("train", "fake", method, count)
    elif current_data_source == 'df40_paired':
        # DF40 Paired: Each sample is a PAIR (real + fake), log by method
        for method, count in train_counts.items():
            data_table.add_data("train", "real", method, count)
            data_table.add_data("train", "fake", method, count)
    elif current_data_source == 'combined_paired':
        # Combined Paired: Each sample is a PAIR (real + fake), log by method
        for method, count in train_counts.items():
            data_table.add_data("train", "real", method, count)
            data_table.add_data("train", "fake", method, count)
    else:
        for method, count in train_counts.items():
            data_type = "real" if method in real_source_names else "fake"
            data_table.add_data("train", data_type, method, count)

    # Also get validation counts and add them to the table
    # Handle different data sources for all_val_videos structure
    if current_data_source == 'deeplive':
        # DeepLive: Each validation sample also contains both real and fake
        val_sample_count = data_split_stats.get('val_samples', 0) + data_split_stats.get('test_samples', 0)
        val_real_count = val_sample_count  # Each sample has real frames
        val_fake_count = val_sample_count  # Each sample has fake frames
        val_counts = Counter()  # Strategy counts not needed for simple logging
    elif current_data_source == 'df40_paired':
        # DF40 Paired: Each validation sample also contains both real and fake
        val_sample_count = data_split_stats.get('val_samples', 0) + data_split_stats.get('test_samples', 0)
        val_real_count = val_sample_count  # Each sample has real frames
        val_fake_count = val_sample_count  # Each sample has fake frames
        val_counts = Counter()  # Method counts not needed for simple logging
    elif current_data_source == 'combined_paired':
        # Combined Paired: Each validation sample also contains both real and fake
        val_sample_count = data_split_stats.get('val_samples', 0) + data_split_stats.get('test_samples', 0)
        val_real_count = val_sample_count  # Each sample has real frames
        val_fake_count = val_sample_count  # Each sample has fake frames
        val_counts = Counter()  # Method counts not needed for simple logging
    elif all_val_videos:
        val_counts = Counter(v.method for v in all_val_videos)
        val_real_count = sum(count for method, count in val_counts.items() if method in real_source_names)
        val_fake_count = sum(count for method, count in val_counts.items() if method not in real_source_names)
        for method, count in val_counts.items():
            data_type = "real" if method in real_source_names else "fake"
            data_table.add_data("val", data_type, method, count)
    else:
        val_counts = Counter()
        val_real_count = 0
        val_fake_count = 0

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
        # ood_loader=None,
        use_group_dro=config.get('use_group_dro', False)
    )

    if config.get('load_base_checkpoint', False):
        checkpoint_path = config.get('gcs_assets', {}).get('base_checkpoint', {}).get('local_path')
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"--- Loading base checkpoint from {checkpoint_path} as requested by config. ---")
            
            # Load the checkpoint (trainer will handle ArcFace parameter validation/override based on train_arcface flag)
            trainer.load_ckpt(checkpoint_path, validate=False)
            
            # Apply ArcFace parameter overrides for curriculum learning if enabled
            if config.get('train_arcface', True) and config.get('use_arcface_head', False):
                logger.info("--- Applying ArcFace curriculum learning parameters after checkpoint load ---")
                
                # Get the model instance (handle DDP wrapper if needed)
                model_instance = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
                
                # Override ArcFace parameters with new config values
                if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                    # Update the ArcFace head parameters
                    device = model_instance.head.s.device
                    model_instance.head.s = torch.tensor(config['arcface_s'], device=device, dtype=torch.float32)
                    model_instance.head.m = config['arcface_m']
                    
                    # Update the model's configuration attributes for annealing
                    model_instance.s_start = config.get('s_start', config['arcface_s'])
                    model_instance.s_end = config.get('s_end', config['arcface_s'])
                    model_instance.anneal_steps = config.get('anneal_steps', 0)
                    
                    logger.info(f"   ✅ Applied arcface_s: {config['arcface_s']}")
                    logger.info(f"   ✅ Applied arcface_m: {config['arcface_m']}")
                    logger.info(f"   ✅ Set s_start: {model_instance.s_start}")
                    logger.info(f"   ✅ Set s_end: {model_instance.s_end}")
                    logger.info(f"   ✅ Set anneal_steps: {model_instance.anneal_steps}")
                else:
                    logger.warning("ArcFace head not found in model - cannot override parameters!")
            else:
                logger.info("--- Using checkpoint ArcFace parameters (train_arcface=False or ArcFace disabled) ---")
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

    # ===========================================================================
    # --- TRAINING SUMMARY: Key metrics for visibility ---
    # ===========================================================================
    logger.info("=" * 70)
    logger.info("📊 TRAINING CONFIGURATION SUMMARY")
    logger.info("=" * 70)
    
    # Data summary
    logger.info(f"📁 DATA:")
    logger.info(f"   - Data source: {config.get('data_source', 'standard')}")
    logger.info(f"   - Train samples: {train_real_count} real, {train_fake_count} fake")
    logger.info(f"   - Val samples: {val_real_count} real, {val_fake_count} fake")
    
    # Training schedule
    total_steps = config.get('total_training_steps', config.get('nEpochs', 50) * 100)
    warmup_steps = config.get('lr_scheduler_warmup_steps', 0)
    logger.info(f"📅 SCHEDULE:")
    logger.info(f"   - Epochs: {config.get('nEpochs', 'N/A')}")
    logger.info(f"   - Total steps: {total_steps}")
    logger.info(f"   - Warmup steps: {warmup_steps}")
    logger.info(f"   - Evaluate every: {config.get('evaluate_every_steps', 'N/A')} steps")
    
    # Model & Head
    logger.info(f"🧠 MODEL:")
    logger.info(f"   - Backbone: {config.get('backbone', {}).get('variant', 'N/A')} ({config.get('backbone', {}).get('source', 'N/A')})")
    logger.info(f"   - Hidden size: {config.get('backbone', {}).get('hidden_size', 'auto')}")
    logger.info(f"   - SVD rank: {config.get('rank', 'N/A')}")
    
    # ArcFace
    if config.get('use_arcface_head', False):
        logger.info(f"🎯 ARCFACE:")
        logger.info(f"   - s_start: {config.get('s_start', config.get('arcface_s', 30))}")
        logger.info(f"   - s_end: {config.get('s_end', config.get('arcface_s', 30))}")
        logger.info(f"   - anneal_steps: {config.get('anneal_steps', 0)}")
        logger.info(f"   - margin (m): {config.get('arcface_m', 0.35)}")
    
    # Augmentation
    aug_config = config.get('augmentation', {})
    logger.info(f"🎨 AUGMENTATION:")
    logger.info(f"   - Version: {aug_config.get('version', 'none')}")
    logger.info(f"   - Occlusion prob: {aug_config.get('occlusion_prob', 'N/A')}")
    
    # Optimizer
    logger.info(f"⚙️ OPTIMIZER:")
    logger.info(f"   - LR: {config.get('optimizer', {}).get('adam', {}).get('lr', config.get('learning_rate', 'N/A'))}")
    logger.info(f"   - Weight decay: {config.get('optimizer', {}).get('adam', {}).get('weight_decay', config.get('weight_decay', 'N/A'))}")
    logger.info(f"   - Scheduler: {config.get('lr_scheduler', 'none')}")
    
    # Early stopping
    logger.info(f"🛑 EARLY STOPPING:")
    logger.info(f"   - Enabled: {config.get('early_stopping_enabled', False)}")
    logger.info(f"   - Patience: {config.get('early_stopping_patience', 'N/A')}")
    logger.info(f"   - Min delta: {config.get('early_stopping_min_delta', 'N/A')}")
    
    logger.info("=" * 70)
    logger.info("🚀 Starting training...")
    logger.info("=" * 70)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        # If we’re in DDP with property balancing, rebuild a fresh, weighted per-epoch loader
        if (not is_property_balancing) and config.get('ddp', False) and config.get('_ddp_weight_helper'):
            train_loader = config['_ddp_weight_helper']['fn']()

        trainer.train_epoch(
            train_loader=train_loader,
            epoch=epoch,
            train_videos=train_data
        )

        if trainer.early_stop_triggered:
            logger.info(f"Gracefully terminating training at epoch {epoch + 1} due to early stopping.")
            wandb.log({"train/status": "Early Stopped"})
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
