import argparse
import os
import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import yaml
import wandb

from data.validation_sources import (
    load_df40_pairs_validation,
    load_external_fake_videos,
    load_external_real_videos,
    load_deeplive_validation,
    load_visomaster_validation,
)
from dataset.dataloaders import create_pure_validation_loader
from detectors import DETECTOR
from logger import create_logger
from trainer.trainer import Trainer
from utils import choose_metric, init_seed
from utils.gcs import download_assets_from_gcs


def _update_config_from_checkpoint(config, checkpoint_path, logger):
    """Load checkpoint and update config to match the saved model settings."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    saved_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = {}

    if isinstance(saved_checkpoint, dict) and "model_config" in saved_checkpoint:
        saved_config = saved_checkpoint["model_config"]
        logger.info("Updating config with checkpoint model_config BEFORE model creation:")
        for key, value in saved_config.items():
            if key == "current_arcface_s":
                continue
            old_value = config.get(key)
            config[key] = value
            if old_value != value:
                logger.info(f"  {key}: {old_value} -> {value}")
        logger.info(
            "Checkpoint metadata: epoch=%s auc=%s",
            saved_checkpoint.get("epoch"),
            saved_checkpoint.get("auc"),
        )
    else:
        logger.warning("Checkpoint missing model_config; using current config.")

    return saved_checkpoint, saved_config


def _merge_checkpoint_assets(config, saved_config, logger):
    """Merge checkpoint gcs_assets into config for backbone downloads."""
    saved_assets = (saved_config or {}).get("gcs_assets") or {}
    clip_backbone = saved_assets.get("clip_backbone")
    if not clip_backbone:
        return

    if "gcs_assets" not in config:
        config["gcs_assets"] = {}
    existing = config["gcs_assets"].get("clip_backbone")
    # Overwrite if missing entirely OR if present but has null/empty paths
    if not existing or (not existing.get("gcs_path") and not existing.get("local_path")):
        config["gcs_assets"]["clip_backbone"] = clip_backbone
        logger.info("Added clip_backbone asset from checkpoint config.")


def _load_state_into_model(model, checkpoint_data, saved_config, logger):
    """Load checkpoint weights into a model created with the right config."""
    if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
        state_dict = checkpoint_data["state_dict"]
    else:
        state_dict = checkpoint_data

    if saved_config.get("use_arcface_head", False) and "current_arcface_s" in saved_config:
        if hasattr(model, "head") and hasattr(model.head, "s"):
            current_s = saved_config["current_arcface_s"]
            model.head.s.data.fill_(current_s)
            logger.info("Restored ArcFace s parameter: %s", current_s)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    logger.info("Model weights loaded.")


def _ensure_dataset_methods(config):
    if "dataset_methods" not in config:
        config["dataset_methods"] = {}
    if "use_real_sources" not in config["dataset_methods"]:
        config["dataset_methods"]["use_real_sources"] = []


def _parse_list_arg(value):
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_external_real_location(bucket_arg: str, prefix_arg: str, logger):
    """
    Parse an external real location. Accepts:
      - bucket name: "effort-collected-data"
      - bucket + prefix: "effort-collected-data/real/external_youtube_avspeech"
      - gs://bucket/prefix
    """
    if bucket_arg.startswith("gs://"):
        bucket_arg = bucket_arg.replace("gs://", "", 1)
    if "/" in bucket_arg:
        bucket_name, prefix = bucket_arg.split("/", 1)
        if prefix_arg and prefix_arg != prefix:
            logger.info(
                "external_real_prefix overridden by external_real_bucket path: %s",
                prefix,
            )
        return bucket_name, prefix
    return bucket_arg, prefix_arg


def main():
    parser = argparse.ArgumentParser(description="Validate checkpoints on custom sources.")
    parser.add_argument("--detector_path", type=str, default="./config/detector/effort.yaml")
    parser.add_argument("--train_config_path", type=str, default="./config/train_config.yaml")
    parser.add_argument("--dataloader_config", type=str, default="./config/dataloader_config.yml")

    parser.add_argument("--checkpoint_gcs_path", type=str, required=True)
    parser.add_argument("--checkpoint_local_path", type=str, default="./weights/validation_checkpoint.pth")

    parser.add_argument("--df40_pair_json", type=str, default="dataset/df40_pairs/df40-pair-matching.json")
    parser.add_argument("--df40_orientation", type=str, default="all",
                        choices=["all", "target_source", "source_target"])
    parser.add_argument("--df40_mode", type=str, default="paired",
                        choices=["paired", "fake_only", "real_only", "none"])
    parser.add_argument("--df40_methods", type=str, default=None,
                        help="Comma-separated list of DF40 methods to include.")
    parser.add_argument("--df40_real_method", type=str, default="faceforensics++")

    parser.add_argument("--external_real_bucket", type=str, default=None,
                        help="GCS bucket name for external real data.")
    parser.add_argument("--external_real_prefix", type=str, default="real/external_youtube_avspeech")
    parser.add_argument("--external_real_method", type=str, default="external_youtube_avspeech")
    parser.add_argument("--external_real_cache", type=str, default=None,
                        help="Optional local JSON cache of frame paths.")
    parser.add_argument("--max_external_real", type=int, default=None,
                        help="Max number of external real videos to sample (default: all).")
    parser.add_argument("--external_real_seed", type=int, default=737,
                        help="Random seed for external real sampling (default 737).")
    parser.add_argument("--external_real_deterministic", action="store_true", default=False,
                        help="If set, disable random frame subsampling for external real by "
                             "deterministically shaping each sample to --frames_per_video frames.")
    parser.add_argument("--external_fake_bucket", type=str, default=None,
                        help="GCS bucket name for external fake data (e.g., WMA failure set).")
    parser.add_argument("--external_fake_prefix", type=str, default="wma_validation/enhanced_fake")
    parser.add_argument("--external_fake_method", type=str, default="wma_failure_fake")
    parser.add_argument("--external_fake_cache", type=str, default=None,
                        help="Optional local JSON cache of external fake frame paths.")
    parser.add_argument("--external_fake_grouping", type=str, default="by_folder",
                        choices=["by_folder", "per_image"],
                        help="How to group external fake images into samples. "
                             "'by_folder' keeps current behavior; 'per_image' treats every image "
                             "as its own sample (recommended for WMA flat reporting).")
    parser.add_argument("--external_fake_deterministic", action="store_true", default=False,
                        help="If set, disable random frame subsampling for external fake by "
                             "deterministically shaping each sample to --frames_per_video frames.")
    parser.add_argument("--max_external_fake", type=int, default=None,
                        help="Max number of external fake videos to sample (default: all).")
    parser.add_argument("--external_fake_seed", type=int, default=737,
                        help="Random seed for external fake sampling (default 737).")

    # DeepLive validation
    parser.add_argument("--deeplive_bucket", type=str, default=None,
                        help="GCS bucket for DeepLive data (enables DeepLive validation).")
    parser.add_argument("--deeplive_split", type=str, default="val",
                        choices=["train", "val", "all"],
                        help="Which DeepLive split to validate on.")
    parser.add_argument("--deeplive_train_split", type=float, default=0.9,
                        help="Train split ratio (default 0.9 to match training).")
    parser.add_argument("--deeplive_val_split", type=float, default=0.1,
                        help="Val split ratio (default 0.1 to match training).")
    parser.add_argument("--deeplive_seed", type=int, default=737,
                        help="Random seed for DeepLive split (default 737 to match training).")
    parser.add_argument("--deeplive_strategies", type=str, default=None,
                        help="Comma-separated strategies to include (e.g., 'quality_enhancement'). "
                             "Default: edge_cases,minimal_processing.")

    # VisoMaster validation
    parser.add_argument("--visomaster_bucket", type=str, default=None,
                        help="GCS bucket for VisoMaster cropped frames (enables VisoMaster validation).")
    parser.add_argument("--visomaster_frames_bucket", type=str,
                        default="live-deepfake-methods-real-and-fake-frames",
                        help="GCS bucket with full manifests + tier data.")
    parser.add_argument("--visomaster_swap_models", type=str, default=None,
                        help="Comma-separated swap models to include (e.g., 'CSCS,GhostFace-v1'). Default: all.")
    parser.add_argument("--visomaster_tiers", type=str, default=None,
                        help="Comma-separated tiers to include (e.g., 'STRONG,MODERATE'). Default: all.")
    parser.add_argument("--visomaster_include_tier_methods", action="store_true", default=True,
                        help="Emit additional per-tier fake methods for tier-level reporting.")
    parser.add_argument("--visomaster_no_tier_methods", dest="visomaster_include_tier_methods",
                        action="store_false",
                        help="Disable per-tier fake method entries.")
    parser.add_argument("--visomaster_held_out_models", type=str, default=None,
                        help="Comma-separated OOD swap models for held-out evaluation. "
                             "When set, runs VisoMaster validation TWICE: once for train models "
                             "(--visomaster_swap_models), once for held-out models (this flag). "
                             "Results are logged with separate prefixes for in-dist vs OOD.")


    parser.add_argument("--frames_per_video", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--eval_num_workers", type=int, default=None,
                        help="Override validation DataLoader worker count. "
                             "Use 0 to avoid iterable-worker duplication.")
    parser.add_argument("--log_prefix", type=str, default="custom_validation")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--detailed_reports", dest="detailed_reports", action="store_true")
    parser.add_argument("--no_detailed_reports", dest="detailed_reports", action="store_false")
    parser.set_defaults(detailed_reports=True)
    parser.add_argument("--disable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "my-project"))
    
    # Custom output path - append results to existing GCS folder
    parser.add_argument("--output_gcs_folder", type=str, default=None,
                        help="GCS folder path to append results to (e.g., 'gs://training-job-outputs/test_results/2026-01-20_14-28-06_B16_df40_source_target_extreal'). "
                             "If not specified, creates a new timestamped folder.")
    parser.add_argument("--output_filename_prefix", type=str, default=None,
                        help="Custom prefix for output files (e.g., 'target_source_' will create 'target_source_frames_report.csv'). "
                             "Useful when appending to existing results folder.")

    args = parser.parse_args()

    # --- Load config files ---
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open(args.train_config_path, "r") as f:
        config.update(yaml.safe_load(f))
    with open(args.dataloader_config, "r") as f:
        data_config = yaml.safe_load(f)
    config.update(data_config)

    config.setdefault("ddp", False)
    config.setdefault("local_rank", 0)
    config.setdefault("metric_scoring", "auc")
    config.setdefault("test_batchSize", args.test_batch_size)
    config.setdefault("manualSeed", 737)
    config.setdefault("cuda", True)
    config.setdefault("cudnn", True)

    dl_params = config.get("dataloader_params", {})
    dl_params["frames_per_video"] = args.frames_per_video
    if args.eval_num_workers is not None:
        dl_params["eval_num_workers"] = int(args.eval_num_workers)
    config["dataloader_params"] = dl_params
    config["test_batchSize"] = args.test_batch_size

    # --- Logging ---
    log_dir = "./logs_custom_validation"
    os.makedirs(log_dir, exist_ok=True)
    logger = create_logger(os.path.join(log_dir, "validation.log"))

    wandb_run = None
    if not args.disable_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"validate_{os.path.basename(args.checkpoint_gcs_path)[:30]}",
        )

    logger.info("--- Custom Validation Script ---")
    init_seed(config)
    if config.get("cudnn"):
        cudnn.benchmark = True

    # --- Download checkpoint ---
    config.setdefault("gcs_assets", {})
    config["gcs_assets"]["base_checkpoint"] = {
        "gcs_path": args.checkpoint_gcs_path,
        "local_path": args.checkpoint_local_path,
    }
    downloaded_assets = download_assets_from_gcs(config, logger)
    if not downloaded_assets or "base_checkpoint" not in downloaded_assets:
        logger.error("Failed to download checkpoint. Aborting.")
        return

    # --- Update config from checkpoint ---
    logger.info("Loading checkpoint configuration from: %s", args.checkpoint_local_path)
    saved_checkpoint, saved_config = _update_config_from_checkpoint(
        config, args.checkpoint_local_path, logger
    )

    # Merge gcs_assets from checkpoint (backbone, etc.), then download if needed
    _merge_checkpoint_assets(config, saved_config, logger)
    download_assets_from_gcs(config, logger)

    # --- Create model ---
    logger.info("Creating model '%s' with restored configuration", config.get("model_name"))
    model = DETECTOR[config["model_name"]](config)
    _load_state_into_model(model, saved_checkpoint, saved_config, logger)

    # --- Build validation videos ---
    all_videos = []

    if args.df40_mode != "none":
        df40_methods = _parse_list_arg(args.df40_methods)
        df40_pair_json = args.df40_pair_json
        if not os.path.isabs(df40_pair_json):
            df40_pair_json = os.path.join(os.path.dirname(__file__), df40_pair_json)
        df40_videos = load_df40_pairs_validation(
            pair_json_path=df40_pair_json,
            orientation=args.df40_orientation,
            mode=args.df40_mode,
            methods=df40_methods,
            real_method_override=args.df40_real_method,
        )
        all_videos.extend(df40_videos)

    if args.external_real_bucket:
        bucket_name, prefix = _parse_external_real_location(
            args.external_real_bucket, args.external_real_prefix, logger
        )
        external_real_deterministic_frames = (
            args.frames_per_video if args.external_real_deterministic else None
        )
        external_videos = load_external_real_videos(
            bucket_name=bucket_name,
            prefix=prefix,
            method_name=args.external_real_method,
            cache_manifest_path=args.external_real_cache,
            max_videos=args.max_external_real,
            seed=args.external_real_seed,
            deterministic_frame_count=external_real_deterministic_frames,
        )
        all_videos.extend(external_videos)

    if args.external_fake_bucket:
        bucket_name, prefix = _parse_external_real_location(
            args.external_fake_bucket, args.external_fake_prefix, logger
        )
        external_fake_deterministic_frames = None
        if args.external_fake_deterministic:
            external_fake_deterministic_frames = args.frames_per_video
        elif args.external_fake_grouping == "per_image" and args.frames_per_video > 1:
            # per_image sample has one frame; default frames_per_video=8 would otherwise
            # be dropped by validation loader. Auto-shape to keep behavior explicit/stable.
            external_fake_deterministic_frames = args.frames_per_video
            logger.info(
                "Auto-enabled deterministic frame shaping for external fake: "
                "grouping=per_image with frames_per_video=%d",
                args.frames_per_video,
            )
        external_fake_videos = load_external_fake_videos(
            bucket_name=bucket_name,
            prefix=prefix,
            method_name=args.external_fake_method,
            cache_manifest_path=args.external_fake_cache,
            max_videos=args.max_external_fake,
            seed=args.external_fake_seed,
            grouping=args.external_fake_grouping,
            deterministic_frame_count=external_fake_deterministic_frames,
        )
        all_videos.extend(external_fake_videos)

    # DeepLive validation
    if args.deeplive_bucket:
        deeplive_strategies = [s.strip() for s in args.deeplive_strategies.split(",")] if args.deeplive_strategies else None
        deeplive_videos = load_deeplive_validation(
            bucket_name=args.deeplive_bucket,
            gcs_project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'),
            split=args.deeplive_split,
            train_split=args.deeplive_train_split,
            val_split=args.deeplive_val_split,
            seed=args.deeplive_seed,
            strategies=deeplive_strategies,
        )
        all_videos.extend(deeplive_videos)

    # VisoMaster validation
    if args.visomaster_bucket:
        visomaster_swap_models = _parse_list_arg(args.visomaster_swap_models)
        visomaster_tiers = _parse_list_arg(args.visomaster_tiers)
        visomaster_videos = load_visomaster_validation(
            cropped_bucket_name=args.visomaster_bucket,
            frames_bucket_name=args.visomaster_frames_bucket,
            gcs_project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'),
            swap_models=visomaster_swap_models,
            tiers=visomaster_tiers,
            include_tier_methods=args.visomaster_include_tier_methods,
        )
        all_videos.extend(visomaster_videos)

        # OOD held-out models: load separately with "visomaster_ood_" prefix
        visomaster_held_out = _parse_list_arg(args.visomaster_held_out_models)
        if visomaster_held_out:
            logger.info(f"Loading VisoMaster OOD held-out models: {visomaster_held_out}")
            ood_videos = load_visomaster_validation(
                cropped_bucket_name=args.visomaster_bucket,
                frames_bucket_name=args.visomaster_frames_bucket,
                gcs_project=os.environ.get('GOOGLE_CLOUD_PROJECT', 'train-cvit2'),
                swap_models=visomaster_held_out,
                tiers=visomaster_tiers,
                include_tier_methods=args.visomaster_include_tier_methods,
            )
            # Re-tag OOD videos with "ood_" prefix so reports distinguish them
            for v in ood_videos:
                if v.method != "visomaster_real":
                    v.method = f"ood_{v.method}"
            all_videos.extend(ood_videos)
            logger.info(f"Added {len(ood_videos)} OOD VisoMaster validation videos")

    if not all_videos:
        logger.error("No validation videos were found. Aborting.")
        return

    # --- Ensure real sources are correctly registered ---
    _ensure_dataset_methods(config)
    real_sources = set(config["dataset_methods"].get("use_real_sources", []))
    if args.df40_mode in {"paired", "real_only"}:
        real_sources.add(args.df40_real_method)
    if args.external_real_bucket:
        real_sources.add(args.external_real_method)
    # DeepLive methods include both real and fake, register strategy-based methods
    if args.deeplive_bucket:
        # DeepLive strategies are the methods, they contain both real/fake
        deeplive_strategies = set(v.method for v in all_videos if v.method.startswith('deeplive_'))
        for strategy in deeplive_strategies:
            real_sources.add(strategy)
    # VisoMaster real source
    if args.visomaster_bucket:
        real_sources.add("visomaster_real")
    config["dataset_methods"]["use_real_sources"] = sorted(real_sources)

    # --- Create dataloader ---
    logger.info("Creating validation dataloader (videos=%d).", len(all_videos))
    validation_loader = create_pure_validation_loader(
        videos=all_videos,
        config=config,
        data_config=config,
        detailed_reporting=args.detailed_reports,
    )

    # --- Run validation ---
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
        ood_loader=None,
    )
    trainer.ood_loader = None

    metrics = trainer.run_validation_on_demand(
        validation_loader=validation_loader,
        log_prefix=args.log_prefix,
        generate_detailed_reports=args.detailed_reports,
        run_name=args.run_name,
        output_gcs_folder=args.output_gcs_folder,
        output_filename_prefix=args.output_filename_prefix,
    )

    logger.info("--- Validation Complete ---")
    overall = metrics.get("overall", {})
    for key, value in overall.items():
        if key in {"pred", "label"}:
            continue
        logger.info("Overall %s: %s", key.upper(), value)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nTotal validation time: {elapsed:.2f} seconds")
