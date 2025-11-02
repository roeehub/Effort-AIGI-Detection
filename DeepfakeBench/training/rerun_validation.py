# File: rerun_validation.py

import argparse
import os
import random
import time
import yaml
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import wandb

from detectors import DETECTOR
from logger import create_logger
from trainer.trainer import Trainer
from train_sweep import download_assets_from_gcs, choose_metric

# Import the new helper functions
from prepare_splits import prepare_pure_validation, VideoInfo
from dataset.dataloaders import create_pure_validation_loader, LazyDataLoaderManager

# ==============================================================================
#                              CONFIGURATION
# ==============================================================================

# Define the methods you want to re-evaluate here.
# This list will be used to filter the data from the properties file.
METHODS_TO_VALIDATE = list({
    "vcd",
    "external_youtube_avspeech",
    "celeb_real",
    "youtube_real",
    "faceforensics++",
    "dfdc_real",
    "deepfakedetection",
    "deepfakes",
    "faceshifter",
    "faceswapff",
    "faceswap",
    "fsgan",
    "simswap",
    "mobileswap",
    "inswap",
    "facedancer",
    "blendface",
    "uniface",
    "face2face",
    "neuraltextures",
    "fomm",
    "facevid2vid",
    "danet",
    "lia",
    "mcnet",
    "one_shot_free",
    "pirender",
    "sadtalker",
    "tpsm",
    "wav2lip",
    "mraa",
    "stylegan2",
    "stylegan3",
    "styleganxl",
    "vqgan",
    "ddim",
    "dit",
    "rddm",
    "sit",
    "faceswap",
    "hey_gen",
    "veo3_creations",
    "deep_live_cam_fake",
    "celeb_synthesis",
    "dfdc_fake",
})

# METHODS_TO_VALIDATE = list({
#     "celeb_synthesis",
#     "dfdc_real",
#     "dfdc_fake",
# })

# Specify the GCS path of the checkpoint you want to load and test.
# This will be downloaded automatically.
# CHECKPOINT_GCS_PATH = "gs://training-job-outputs/best_checkpoints/o08s5u94/top_n_effort_20250927_ep1_auc0.9672_eer0.0905.pth"  # old one

# CHECKPOINT_GCS_PATH = "gs://training-job-outputs/best_checkpoints/dfsesrgu/top_n_effort_20251016_step9000_auc0.8621_eer0.2274.pth"  # DFDC
# CHECKPOINT_GCS_PATH = "gs://training-job-outputs/best_checkpoints/9rfa62j1/top_n_effort_20251009_step9000_auc0.8664_eer0.1765.pth"
CHECKPOINT_GCS_PATH = "gs://training-job-outputs/best_checkpoints/4vtny88m/top_n_effort_20251008_step6300_auc0.8694_eer0.2048.pth"
# CHECKPOINT_GCS_PATH = "gs://training-job-outputs/best_checkpoints/1mjgo9w1/top_n_effort_20251008_step9000_auc0.9422_eer0.1255.pth" # possibly EFS
LOCAL_CHECKPOINT_PATH = "./weights/rerun_checkpoint.pth"

# ==============================================================================

parser = argparse.ArgumentParser(description='Run a standalone validation pass.')
parser.add_argument('--detector_path', type=str, default='./config/detector/effort.yaml',
                    help='Path to detector YAML file')
parser.add_argument('--train_config_path', type=str, default='./config/train_config.yaml',
                    help='Path to the main training config')
parser.add_argument('--dataloader_config', type=str, default='./config/dataloader_config.yml',
                    help='Path to the dataloader configuration file')

args = parser.parse_args()


def init_seed(seed_val):
    """Initializes random seeds for reproducibility."""
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def load_model_weights_with_config_validation(model, model_path, config, logger):
    """Loads weights from a checkpoint file with configuration validation."""
    if os.path.isfile(model_path):
        logger.info(f"Loading model weights from: {model_path}")
        saved = torch.load(model_path, map_location='cpu')
        
        # Handle both old and new checkpoint formats
        if isinstance(saved, dict) and 'state_dict' in saved:
            # New format with configuration
            state_dict = saved['state_dict']
            model_config = saved.get('model_config', {})
            
            # Update current config with saved configuration for exact match
            if model_config:
                logger.info("📋 Restoring exact model configuration from checkpoint:")
                for key, value in model_config.items():
                    if key != 'current_arcface_s' :  # Skip dynamic parameter
                        old_value = config.get(key)
                        config[key] = value
                        if old_value != value:
                            logger.info(f"  {key}: {old_value} → {value}")
                
                # Restore dynamic ArcFace parameter if available
                if model_config.get('use_arcface_head', False) and 'current_arcface_s' in model_config:
                    if hasattr(model, 'head') and hasattr(model.head, 's'):
                        current_s = model_config['current_arcface_s']
                        model.head.s.data.fill_(current_s)
                        logger.info(f"  Restored ArcFace s parameter: {current_s}")
                
                logger.info(f"📊 Checkpoint info: Epoch {saved.get('epoch')}, AUC: {saved.get('auc', 0):.4f}")
            else:
                logger.warning("No model configuration found in checkpoint.")
        else:
            # Old format - issue warning
            state_dict = saved
            logger.warning("⚠️  Loading OLD checkpoint format. Configuration validation not possible!")
            logger.warning("⚠️  Results may be inconsistent due to missing configuration.")
        
        # Load state dict with module prefix handling
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        logger.info('✅ Model weights loaded successfully.')
        return model_config if 'model_config' in locals() else {}
    else:
        raise FileNotFoundError(f"=> No model found at '{model_path}'")


def load_model_weights_into_configured_model(model, checkpoint_data, saved_config, logger):
    """Load weights into a model that was created with the correct configuration."""
    
    # Extract state dict
    if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data
        
    # Restore dynamic ArcFace parameter if available
    if saved_config.get('use_arcface_head', False) and 'current_arcface_s' in saved_config:
        if hasattr(model, 'head') and hasattr(model.head, 's'):
            current_s = saved_config['current_arcface_s']
            model.head.s.data.fill_(current_s)
            logger.info(f"  ✅ Restored ArcFace s parameter: {current_s}")
    
    # Load state dict with module prefix handling
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Verify model architecture
    if hasattr(model, 'head'):
        head_type = type(model.head).__name__
        logger.info(f"✅ Model head type: {head_type}")
        if hasattr(model.head, 's') and hasattr(model.head, 'm'):
            logger.info(f"✅ ArcFace parameters - s: {float(model.head.s):.2f}, m: {model.head.m:.3f}")
        elif hasattr(model.head, 'weight'):
            logger.info(f"✅ Linear head - output features: {model.head.weight.shape[0]}")
    
    logger.info('✅ Model weights loaded into correctly configured model.')


def main():
    """
    Main execution function for the validation script.
    """
    # 1. Load Configurations
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.train_config_path, 'r') as f:
        config.update(yaml.safe_load(f))
    with open(args.dataloader_config, 'r') as f:
        data_config = yaml.safe_load(f)


    # merge all the configs into config
    config.update(data_config)


    # 2. Initialize Logger and W&B
    log_dir = './logs_rerun_validation'
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, 'validation.log'))

    wandb_run = wandb.init(
        project="my-project",  # <-- CHANGE THIS to your project name
        name=f"rerun_{os.path.basename(CHECKPOINT_GCS_PATH)[:30]}"
    )

    # --- Set default parameters that would normally come from a W&B sweep ---
    # These are mostly for model initialization and can be adjusted if needed.
    config['rank'] = 2023
    config['ddp'] = False
    config['local_rank'] = 0
    config['metric_scoring'] = 'auc'

    # Ensure nested dictionary structure exists to avoid KeyErrors
    if 'dataset_methods' not in config:
        config['dataset_methods'] = {}

    # Set default methods, required by test_epoch for metric calculation
    config['dataset_methods']['use_real_sources'] = (
        config.get('methods', {}).get('use_real_sources', ['youtube_real']))  # Example, adjust if needed

    # Set default augmentation params, needed by dataloader
    config['augmentation_params'] = config.get('augmentation_params', {'version': 'surgical'})

    # Set default lesson gate params to ensure it's safely disabled
    config['lesson_gate'] = config.get('lesson_gate', {'enabled': False})

    logger.info("--- Standalone Validation Rerun Script ---")
    init_seed(config['manualSeed'])
    if config['cudnn']:
        cudnn.benchmark = True

    # 3. Download the specific checkpoint from GCS
    logger.info("Overriding checkpoint path for on-demand validation.")
    config['gcs_assets']['base_checkpoint'] = {
        'gcs_path': CHECKPOINT_GCS_PATH,
        'local_path': LOCAL_CHECKPOINT_PATH
    }
    downloaded_assets = download_assets_from_gcs(config, logger)
    if not downloaded_assets or 'base_checkpoint' not in downloaded_assets:
        logger.error("Failed to download the specified checkpoint. Aborting.")
        return

    # Set parquet path for property balancing, required by prepare_pure_validation
    if 'property_manifest_parquet' in config.get('gcs_assets', {}):
        local_parquet_path = config['gcs_assets']['property_manifest_parquet']['local_path']
        config['property_balancing']['frame_properties_parquet_path'] = local_parquet_path
        logger.info(f"Set property manifest path to: {local_parquet_path}")

    # NOTE: ArcFace parameters will be restored from checkpoint configuration
    # No longer hardcoding them here to avoid mismatches

    # 4. Load checkpoint configuration FIRST, then create model with correct architecture
    logger.info(f"Loading checkpoint configuration from: {LOCAL_CHECKPOINT_PATH}")
    
    # Pre-load checkpoint to get configuration
    if os.path.isfile(LOCAL_CHECKPOINT_PATH):
        saved_checkpoint = torch.load(LOCAL_CHECKPOINT_PATH, map_location='cpu')
        
        if isinstance(saved_checkpoint, dict) and 'model_config' in saved_checkpoint:
            saved_config = saved_checkpoint['model_config']
            logger.info("📋 Updating config with checkpoint configuration BEFORE model creation:")
            
            # Update config with saved configuration for correct model architecture
            for key, value in saved_config.items():
                if key != 'current_arcface_s' :  # Skip dynamic parameter
                    old_value = config.get(key)
                    config[key] = value
                    if old_value != value:
                        logger.info(f"  {key}: {old_value} → {value}")
            
            logger.info(f"📊 Checkpoint: Epoch {saved_checkpoint.get('epoch')}, AUC: {saved_checkpoint.get('auc', 0):.4f}")
        else:
            logger.warning("⚠️  OLD checkpoint format - creating model with default config")
            saved_config = {}
    else:
        raise FileNotFoundError(f"Checkpoint not found: {LOCAL_CHECKPOINT_PATH}")
    
    # NOW create model with correct configuration
    logger.info(f"Creating model '{config['model_name']}' with restored configuration...")
    logger.info(f"  - ArcFace Head: {config.get('use_arcface_head', False)}")
    if config.get('use_arcface_head', False):
        logger.info(f"  - ArcFace s: {config.get('arcface_s', 30.0)}")
        logger.info(f"  - ArcFace m: {config.get('arcface_m', 0.35)}")
    
    model = DETECTOR[config['model_name']](config)
    
    # Load weights with the properly configured model
    logger.info("Loading model weights into correctly configured model...")
    load_model_weights_into_configured_model(model, saved_checkpoint, saved_config, logger)

    # 5. Prepare Data
    logger.info(f"Preparing pure validation set for methods: {METHODS_TO_VALIDATE}")
    validation_videos = prepare_pure_validation(config, methods=METHODS_TO_VALIDATE)

    if not validation_videos:
        logger.error("No videos found for the specified methods. Aborting validation.")
        wandb_run.finish()
        return

    # 6. Create Dataloader
    logger.info("Creating pure validation dataloader.")
    
    # Log the frames_per_video parameter explicitly
    frames_per_video = 8
    logger.info(f"🎬 Using {frames_per_video} frames per video for validation")
    
    validation_loader = create_pure_validation_loader(
        videos=validation_videos,
        config=config,
        data_config=config,
        detailed_reporting=True  # Enable detailed reporting for CSV generation
    )

    # 7. Instantiate a "dummy" Trainer for validation
    # We only need the model and logger. Optimizer, scheduler, and regular
    # dataloaders are not needed for this task.
    logger.info("Initializing Trainer in validation-only mode.")
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=None,
        scheduler=None,
        logger=logger,
        val_in_dist_loader=None,
        val_holdout_loader=None,
        metric_scoring=choose_metric(config),
        wandb_run=wandb_run,
        ood_loader=None  # Explicitly disable OOD monitoring
    )
    
    # 🔒 SAFETY: Disable OOD monitoring completely for rerun validation
    trainer.ood_loader = None
    logger.info("✅ OOD monitoring explicitly disabled for rerun validation.")

    # 8. Run the on-demand validation
    log_prefix = "rerun_validation"
    metrics = None
    try:
        metrics = trainer.run_validation_on_demand(
            validation_loader=validation_loader,
            log_prefix=log_prefix,
            generate_detailed_reports=True,
            run_name="validate full - Model 3 - 4vtny88m"
        )
        logger.info("✅ Validation completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error during validation: {e}")
        logger.error("However, detailed reports may have been generated. Check GCS.")
        # Don't fail completely - we want to at least log what we can
        metrics = {'overall': {'error': str(e)}}

    # 9. Log and Finish (with robust error handling)
    logger.info("--- Validation Rerun Complete ---")
    print("\nValidation Results:")
    if metrics and 'overall' in metrics:
        for metric_name, value in metrics['overall'].items():
            if metric_name not in ['pred', 'label']:
                try:
                    if isinstance(value, (int, float)) and value >= 0:
                        print(f"  Overall {metric_name.upper()}: {value:.4f}")
                    else:
                        print(f"  Overall {metric_name.upper()}: {value}")
                except Exception as e:
                    print(f"  Overall {metric_name.upper()}: {value} (formatting error: {e})")
    else:
        print("  No metrics available or validation failed.")

    try:
        wandb_run.finish()
        logger.info("✅ W&B run finished successfully.")
    except Exception as e:
        logger.error(f"❌ Error finishing W&B run: {e}")
    
    logger.info("Script finished.")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print(f"\nTotal validation time: {elapsed:.2f} seconds")
