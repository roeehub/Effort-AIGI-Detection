from venv import logger

# =============================================================================
# CODE VERSION STAMP - Update this when making changes to verify deployment
# =============================================================================
CODE_VERSION = "2026-04-19-PROPER-DATA-WTF-V1"
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
        
        # Apply combined_paired config directly (DF40/DeepLive/VisoMaster enabled flags, 
        # bucket settings, swap models, etc.). This is CRITICAL — without it,
        # combined_paired.py falls back to defaults and ignores enabled flags.
        if 'combined_paired' in single_cfg:
            data_config['combined_paired'] = single_cfg['combined_paired']
            cp = single_cfg['combined_paired']
            df40_en = cp.get('df40', {}).get('enabled', True)
            dl_en = cp.get('deeplive', {}).get('enabled', True)
            vm_en = cp.get('visomaster', {}).get('enabled', False)
            print(f"  ✅ Applied combined_paired config (df40={df40_en}, deeplive={dl_en}, visomaster={vm_en})")
            logger.info(f"  Applied combined_paired config (df40={df40_en}, deeplive={dl_en}, visomaster={vm_en})")

        # Apply deeplive config directly (for GCS bucket settings, etc.)
        # Note: This handles TOP-LEVEL deeplive config (legacy/non-combined mode).
        # For combined_paired mode, deeplive config is inside combined_paired above.
        if 'deeplive' in single_cfg:
            data_config['deeplive'] = single_cfg['deeplive']
            print(f"  ✅ Applied top-level deeplive config")
            logger.info(f"  Applied top-level deeplive config")
        
        # Apply visomaster config directly (for GCS bucket settings, swap models, tiers, etc.)
        # Note: Same as deeplive — this is for TOP-LEVEL visomaster config only.
        if 'visomaster' in single_cfg:
            data_config['visomaster'] = single_cfg['visomaster']
            print(f"  ✅ Applied top-level visomaster config")
            logger.info(f"  Applied top-level visomaster config")
        
        # Apply backbone config directly
        if 'backbone' in single_cfg:
            config['backbone'] = single_cfg['backbone']
            print(f"  ✅ Applied backbone: {single_cfg['backbone'].get('name', 'unknown')}")
            logger.info(f"  Applied backbone: {single_cfg['backbone'].get('name', 'unknown')}")
        
        # Apply checkpointing config directly (GCS prefix, save frequency, etc.)
        if 'checkpointing' in single_cfg:
            config['checkpointing'] = single_cfg['checkpointing']
            print(f"  ✅ Applied checkpointing: gcs_prefix={single_cfg['checkpointing'].get('gcs_prefix', 'N/A')}")
            logger.info(f"  Applied checkpointing: gcs_prefix={single_cfg['checkpointing'].get('gcs_prefix', 'N/A')}")

        # Apply stability / label-smoothing flat keys directly
        # (These are simple scalars — W&B won't flatten them, but
        # apply_wandb_stability_params also handles them as belt-and-suspenders.)
        for _flat_key in ('stability_lambda', 'stability_noise_std',
                          'stability_crop_jitter', 'label_smoothing'):
            if _flat_key in single_cfg:
                config[_flat_key] = single_cfg[_flat_key]
                print(f"  ✅ Applied {_flat_key}: {single_cfg[_flat_key]}")
                logger.info(f"  Applied {_flat_key}: {single_cfg[_flat_key]}")

        # Apply Group DRO config directly (W&B flattens group_dro_params.beta
        # to 'group_dro_params.beta' but config helper expects 'group_dro_beta')
        if 'use_group_dro' in single_cfg:
            config['use_group_dro'] = single_cfg['use_group_dro']
            print(f"  ✅ Applied use_group_dro: {single_cfg['use_group_dro']}")
            logger.info(f"  Applied use_group_dro: {single_cfg['use_group_dro']}")
        if 'group_dro_params' in single_cfg:
            config['group_dro_params'] = single_cfg['group_dro_params']
            print(f"  ✅ Applied group_dro_params: {single_cfg['group_dro_params']}")
            logger.info(f"  Applied group_dro_params: {single_cfg['group_dro_params']}")

        print("=" * 70)
    else:
        print("⚠️ single_cfg is None/empty - no direct config application!")
    
    # Apply all W&B overrides to config and data_config (handles flat keys)
    apply_all_wandb_overrides(config, data_config, wandb.config, logger)

    # Fail fast on config-propagation mismatches for critical Round-6+ fields.
    if single_cfg:
        if 'use_quality_domain_head' in single_cfg:
            requested_quality_head = bool(single_cfg.get('use_quality_domain_head'))
            effective_quality_head = bool(config.get('use_quality_domain_head', False))
            if requested_quality_head != effective_quality_head:
                raise RuntimeError(
                    "Config propagation mismatch: `use_quality_domain_head` "
                    f"requested={requested_quality_head}, effective={effective_quality_head}. "
                    "Refusing to launch with inconsistent GRL settings."
                )

        if bool(single_cfg.get('use_quality_domain_head', False)):
            for key in (
                'quality_domain_count',
                'quality_head_hidden_dim',
                'quality_domain_loss_weight',
            ):
                if key not in single_cfg:
                    continue
                requested_val = single_cfg.get(key)
                effective_val = config.get(key)
                if requested_val != effective_val:
                    raise RuntimeError(
                        f"Config propagation mismatch for `{key}`: "
                        f"requested={requested_val}, effective={effective_val}. "
                        "Refusing to launch with inconsistent GRL settings."
                    )
    
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

    # ===========================================================================
    # --- 3. Standard Setup ---
    # ===========================================================================
    config['local_rank'] = args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['ddp'] = args.ddp

    # Resolve a single canonical seed once and propagate it.
    canonical_seed = config.get('manualSeed')
    if canonical_seed is None:
        canonical_seed = config.get('seed')
    if canonical_seed is None:
        canonical_seed = data_config.get('data_params', {}).get('seed')
    if canonical_seed is None:
        canonical_seed = 737
    config['manualSeed'] = canonical_seed
    config['seed'] = canonical_seed
    if 'data_params' not in data_config:
        data_config['data_params'] = {}
    data_config['data_params']['seed'] = canonical_seed
    if isinstance(data_config.get('combined_paired'), dict):
        data_config['combined_paired'].setdefault('split_seed', canonical_seed)

    init_seed(config)
    logger.info(f"Canonical seed resolved: {canonical_seed}")
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

    # Override gcs_assets.base_checkpoint.gcs_path from the experiment YAML's
    # top-level gcs_base_checkpoint key. Read from single_cfg first (authoritative),
    # then fall back to wandb.config. This MUST happen BEFORE create_curated_config_log
    # so the curated log captures the correct (not stale) checkpoint path.
    if config.get('load_base_checkpoint', False):
        # Prefer single_cfg (experiment YAML) as the authoritative source
        gcs_ckpt_override = (
            (single_cfg or {}).get('gcs_base_checkpoint')
            or wandb.config.get('gcs_base_checkpoint')
        )
        if gcs_ckpt_override:
            base_ckpt = config.setdefault('gcs_assets', {}).setdefault('base_checkpoint', {})
            base_ckpt['gcs_path'] = gcs_ckpt_override
            base_ckpt.setdefault('local_path', './weights/base.pth')
            logger.info(f"Set base checkpoint GCS path to: {gcs_ckpt_override}")
        else:
            current_path = config.get('gcs_assets', {}).get('base_checkpoint', {}).get('gcs_path')
            if not current_path:
                logger.warning(
                    "load_base_checkpoint=True but no gcs_base_checkpoint provided "
                    "in experiment YAML or wandb.config, and gcs_assets.base_checkpoint.gcs_path is null. "
                    "The model will start from base CLIP weights."
                )

    # Log curated config snapshot for W&B filtering (AFTER checkpoint path is resolved)
    curated_config_log = create_curated_config_log(config, data_config)
    wandb.config.update(curated_config_log, allow_val_change=True)

    # Generate and set run name
    wandb.run.name = generate_run_name(config, wandb.config)

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
    #   - 'df40_paired': DF40 dataset with paired real/fake frames
    #   - 'combined_paired': Unified pipeline combining DF40 + DeepLive + VisoMaster
    #   - 'visomaster': VisoMaster standalone with paired real/fake frames
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
    test_loader = pipeline_result.test_loader
    ood_heldout_loader = pipeline_result.ood_heldout_loader
    
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

    # ===========================================================================
    # --- DATA VERIFICATION (added 2026-01-02) ---
    # Logs per-source sample counts, enabled flags, and method lists to W&B.
    # This makes it impossible to miss a misconfigured data pipeline.
    # ===========================================================================
    if current_data_source == 'combined_paired':
        cp_cfg = data_config.get('combined_paired', {})
        df40_en = cp_cfg.get('df40', {}).get('enabled', True)
        dl_en = cp_cfg.get('deeplive', {}).get('enabled', True)
        vm_en = cp_cfg.get('visomaster', {}).get('enabled', False)
        proper_en = cp_cfg.get('proper_data', {}).get('enabled', False)

        df40_n = data_split_stats.get('df40_samples', -1)
        dl_n = data_split_stats.get('deeplive_samples', -1)
        vm_n = data_split_stats.get('visomaster_samples', -1)
        proper_n = data_split_stats.get('proper_data_samples', -1)
        total_n = data_split_stats.get('total_samples', -1)
        detailed_source_counts = data_split_stats.get('source_counts', {})
        train_source_counts = data_split_stats.get('train_source_counts', {})
        proper_data_lane_counts = data_split_stats.get('proper_data_lane_counts', {})
        train_proper_data_lane_counts = data_split_stats.get('train_proper_data_lane_counts', {})

        # Log to console/file with highly visible formatting
        logger.info("=" * 70)
        logger.info("📊 DATA VERIFICATION — COMBINED PAIRED PIPELINE")
        logger.info("=" * 70)
        logger.info(f"  DF40:        enabled={str(df40_en):5s}  →  {df40_n:,} samples")
        logger.info(f"  DeepLive:    enabled={str(dl_en):5s}  →  {dl_n:,} samples")
        logger.info(f"  VisoMaster:  enabled={str(vm_en):5s}  →  {vm_n:,} samples")
        logger.info(f"  ProperData:  enabled={str(proper_en):5s}  →  {proper_n:,} samples")
        logger.info(f"  TOTAL:                           {total_n:,} samples")
        logger.info(f"  Methods: {data_split_stats.get('methods', [])}")
        if detailed_source_counts:
            logger.info(f"  Source counts (all): {detailed_source_counts}")
        if train_source_counts:
            logger.info(f"  Source counts (train): {train_source_counts}")
        if proper_data_lane_counts:
            logger.info(f"  Proper-data lane counts (all): {proper_data_lane_counts}")
        if train_proper_data_lane_counts:
            logger.info(f"  Proper-data lane counts (train): {train_proper_data_lane_counts}")
        logger.info("=" * 70)

        # Sanity checks — fail fast if config doesn't match reality
        if not df40_en and df40_n > 0:
            raise RuntimeError(f"DATA INTEGRITY ERROR: DF40 is DISABLED but got {df40_n} samples!")
        if not dl_en and dl_n > 0:
            raise RuntimeError(f"DATA INTEGRITY ERROR: DeepLive is DISABLED but got {dl_n} samples!")
        if not vm_en and vm_n > 0:
            raise RuntimeError(f"DATA INTEGRITY ERROR: VisoMaster is DISABLED but got {vm_n} samples!")
        if not proper_en and proper_n > 0:
            raise RuntimeError(f"DATA INTEGRITY ERROR: ProperData is DISABLED but got {proper_n} samples!")
        if df40_en and df40_n == 0:
            logger.warning("⚠️ DF40 is ENABLED but produced 0 samples!")
        if dl_en and dl_n == 0:
            logger.warning("⚠️ DeepLive is ENABLED but produced 0 samples!")
        if vm_en and vm_n == 0:
            logger.warning("⚠️ VisoMaster is ENABLED but produced 0 samples!")
        if proper_en and proper_n == 0:
            logger.warning("⚠️ ProperData is ENABLED but produced 0 samples!")

        # Persist to W&B summary for easy querying across runs
        wandb.run.summary["data/sources_enabled"] = {
            "df40": df40_en,
            "deeplive": dl_en,
            "visomaster": vm_en,
            "proper_data": proper_en,
        }
        wandb.run.summary["data/source_counts"] = {
            "df40": df40_n,
            "deeplive": dl_n,
            "visomaster": vm_n,
            "proper_data": proper_n,
            "total": total_n,
        }
        if detailed_source_counts:
            wandb.run.summary["data/source_counts_detailed"] = detailed_source_counts
        if train_source_counts:
            wandb.run.summary["data/train_source_counts"] = train_source_counts
        if proper_data_lane_counts:
            wandb.run.summary["data/proper_data_lane_counts"] = proper_data_lane_counts
        if train_proper_data_lane_counts:
            wandb.run.summary["data/train_proper_data_lane_counts"] = train_proper_data_lane_counts
        vm_models = cp_cfg.get('visomaster', {}).get('swap_models', [])
        if vm_models:
            wandb.run.summary["data/visomaster_swap_models"] = vm_models
        wandb.log({
            "data/df40_samples": df40_n,
            "data/deeplive_samples": dl_n,
            "data/visomaster_samples": vm_n,
            "data/proper_data_samples": proper_n,
        })

        strategy_counts = data_split_stats.get('strategy_counts', {})
        family_counts = data_split_stats.get('family_counts', {})
        train_strategy_counts = data_split_stats.get('train_strategy_counts', {})
        train_family_counts = data_split_stats.get('train_family_counts', {})
        deeplive_raw_strategy_counts = data_split_stats.get('deeplive_raw_strategy_counts', {})
        deeplive_effective_strategy_counts = data_split_stats.get('deeplive_effective_strategy_counts', {})
        deeplive_strategy_preflight = data_split_stats.get('deeplive_strategy_preflight', {})
        proper_data_discovery = data_split_stats.get('proper_data_discovery', {})
        sampling_strategy = data_split_stats.get('sampling_strategy')
        sampling_family_weights = data_split_stats.get('sampling_family_weights', {})
        holdout_mode = data_split_stats.get('holdout_mode')
        holdout_methods = data_split_stats.get('holdout_methods', [])
        holdout_method_counts = data_split_stats.get('holdout_method_counts', {})
        train_df40_methods = data_split_stats.get('train_df40_methods', [])
        holdout_df40_methods = data_split_stats.get('holdout_df40_methods', [])
        identity_split_mode = data_split_stats.get('identity_split_mode')
        ood_video_count = data_split_stats.get('ood_video_count', 0)
        ood_method_count = data_split_stats.get('ood_method_count', 0)

        wandb.run.summary["data/run_seed"] = data_split_stats.get('run_seed', canonical_seed)
        wandb.run.summary["data/split_seed"] = data_split_stats.get('split_seed', canonical_seed)
        if identity_split_mode:
            wandb.run.summary["data/identity_split_mode"] = identity_split_mode
        wandb.run.summary["data/strategy_counts"] = strategy_counts
        wandb.run.summary["data/family_counts"] = family_counts
        wandb.run.summary["data/train_strategy_counts"] = train_strategy_counts
        wandb.run.summary["data/train_family_counts"] = train_family_counts
        wandb.run.summary["data/deeplive_raw_strategy_counts"] = deeplive_raw_strategy_counts
        wandb.run.summary["data/deeplive_effective_strategy_counts"] = deeplive_effective_strategy_counts
        wandb.run.summary["data/deeplive_strategy_preflight"] = deeplive_strategy_preflight
        if proper_data_discovery:
            wandb.run.summary["data/proper_data_discovery"] = proper_data_discovery
        if sampling_strategy:
            wandb.run.summary["data/sampling_strategy"] = sampling_strategy
        if sampling_family_weights:
            wandb.run.summary["data/sampling_family_weights"] = sampling_family_weights
        if holdout_mode:
            wandb.run.summary["data/holdout_mode"] = holdout_mode
        if holdout_methods:
            wandb.run.summary["data/holdout_methods"] = holdout_methods
        if holdout_method_counts:
            wandb.run.summary["data/holdout_method_counts"] = holdout_method_counts
        if train_df40_methods:
            wandb.run.summary["data/train_df40_methods"] = train_df40_methods
        if holdout_df40_methods:
            wandb.run.summary["data/holdout_df40_methods"] = holdout_df40_methods
        wandb.run.summary["data/ood_video_count"] = ood_video_count
        wandb.run.summary["data/ood_method_count"] = ood_method_count
        proper_data_build_id = data_split_stats.get('proper_data_build_id') or ''
        if proper_data_build_id:
            wandb.run.summary["data/proper_data_build_id"] = proper_data_build_id
            try:
                wandb.config.update(
                    {"data_proper_data_build_id": proper_data_build_id},
                    allow_val_change=True,
                )
            except Exception as e:
                logger.warning(f"Failed to push proper_data_build_id to wandb.config: {e}")

        logger.info(f"Strategy counts (all): {strategy_counts}")
        logger.info(f"Family counts (all): {family_counts}")
        logger.info(f"Strategy counts (train): {train_strategy_counts}")
        logger.info(f"Family counts (train): {train_family_counts}")
        logger.info(f"DeepLive raw strategy counts: {deeplive_raw_strategy_counts}")
        logger.info(f"DeepLive effective strategy counts: {deeplive_effective_strategy_counts}")
        logger.info(f"DeepLive strategy preflight: {deeplive_strategy_preflight}")
        if proper_data_discovery:
            logger.info(f"Proper-data discovery summary: {proper_data_discovery}")
        if sampling_strategy:
            logger.info(f"Sampling strategy: {sampling_strategy}")
        if sampling_family_weights:
            logger.info(f"Sampling family weights: {sampling_family_weights}")
        if holdout_mode:
            logger.info(f"Holdout mode: {holdout_mode}")
        if identity_split_mode:
            logger.info(f"Identity split mode: {identity_split_mode}")
        if holdout_methods:
            logger.info(f"Holdout methods: {holdout_methods}")
        if holdout_method_counts:
            logger.info(f"Holdout method counts: {holdout_method_counts}")
        if train_df40_methods or holdout_df40_methods:
            logger.info(
                f"DF40 method placement: train={train_df40_methods}, holdout={holdout_df40_methods}"
            )
        logger.info(f"OOD monitoring videos: {ood_video_count} (methods={ood_method_count})")

        if isinstance(deeplive_strategy_preflight, dict):
            preflight_passed = deeplive_strategy_preflight.get("passed", True)
            preflight_mode = deeplive_strategy_preflight.get("mode")
            if preflight_mode and not preflight_passed:
                raise RuntimeError(
                    f"DeepLive strategy preflight failed in mode '{preflight_mode}': "
                    f"{deeplive_strategy_preflight.get('errors', [])}"
                )
        logger.info("✅ Data verification passed. Logged source counts to W&B.")

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
        real_methods = data_split_stats.get('overview_real_methods') or ['paired_real']
        all_fake_methods_used = (
            data_split_stats.get('overview_fake_methods')
            or (
                sorted(list(methods))
                if isinstance(methods, (list, set))
                else sorted(list(methods.keys()))
            )
        )
    elif current_data_source == 'visomaster':
        # VisoMaster standalone: Get swap_models from data_stats
        methods = data_split_stats.get('methods', [])
        real_methods = ['paired_real']  # VisoMaster always has paired real frames
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
    elif current_data_source == 'visomaster':
        # VisoMaster standalone: Each sample is a VisoMasterSample with .swap_model attribute
        # Each sample contains BOTH real and fake frames (paired)
        num_samples = len(train_data)
        train_counts = Counter(getattr(s, 'swap_model', 'unknown') for s in train_data)
        train_real_count = num_samples  # Each sample has real frames
        train_fake_count = num_samples  # Each sample has fake frames
        real_source_names = []  # Not used for visomaster counting
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
        ood_heldout_loader=ood_heldout_loader,
        test_loader=test_loader,
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

    # Quality-domain adversarial head (GRL)
    if config.get('use_quality_domain_head', False):
        logger.info(f"🧪 QUALITY DOMAIN HEAD:")
        logger.info(f"   - enabled: True")
        logger.info(f"   - domains: {config.get('quality_domain_count', 4)}")
        logger.info(f"   - hidden_dim: {config.get('quality_head_hidden_dim', 128)}")
        logger.info(f"   - loss_weight: {config.get('quality_domain_loss_weight', 0.1)}")
        logger.info(
            f"   - require_labels: {config.get('quality_domain_require_labels', True)}"
        )
    else:
        logger.info(f"🧪 QUALITY DOMAIN HEAD:")
        logger.info(f"   - enabled: False")
    
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

    # A2: dual-checkpoint final_eval — reload each best-step state dict and
    # run a clean pass over test_loader, ood_heldout_loader, val_holdout.
    try:
        trainer.run_final_eval()
    except Exception as final_eval_err:
        logger.warning(f"run_final_eval failed: {final_eval_err}")

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
