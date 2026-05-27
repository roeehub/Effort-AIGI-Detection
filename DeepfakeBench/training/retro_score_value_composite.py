#!/usr/bin/env python
"""retro_score_value_composite.py — standalone retro-score runner.

Given a best-checkpoint GCS path and a training-style yaml, runs ONE OOD
monitoring pass so the trainer's in-loop `_compute_value_composite(...)` fires
and logs `summary/value_composite` (plus the TPR / stability / tau sub-components)
to W&B under the yaml's configured `value_composite` block.

The whole point: packet-3 checkpoints were trained+scored under the LEGACY
(0.02, 0.04, "max") gate. Packet 3.5 runs under the NEW (0.03, 0.05, "p95")
gate. This runner lets us re-compute the packet-3 checkpoints' value_composite
under the NEW gate (without retraining) so the packet-3 vs packet-3.5
leaderboard is apples-to-apples.

Usage:
    python retro_score_value_composite.py \
        --checkpoint_gcs_path gs://training-job-outputs/.../value_composite_effort_...pth \
        --config_yaml experiments/phase2_round13/R13_RLP3_02_FT_proper_main__NEW_GATES_FOR_RETRO.yaml \
        --wandb_run_name R13_RLP3_02_main_NEW_0422

Design notes:
  - The full training data pipeline is (re)built via `create_data_pipeline(...)`
    exactly like `train_sweep.py`. This is intentional — the OOD loader's
    `exclude_training_identities` filter depends on the training-side identity
    set, so we reproduce the training identity split end-to-end. Cost: 5-20 min
    of pipeline setup on top of the actual OOD scoring pass.
  - The checkpoint's `model_config` is restored into `config` BEFORE the model
    is instantiated, so ArcFace head shape / rank / etc. match the saved
    checkpoint exactly. This mirrors rerun_validation.py:278-315.
  - Nested yaml blocks (`dataset_methods`, `combined_paired`, `backbone`,
    `checkpointing`, `group_dro_params`, `value_composite`, ...) are copied
    from single_cfg -> config/data_config BEFORE any W&B roundtrip, because
    W&B flattens nested dicts. This is the same defensive pattern as
    train_sweep.py:171-282; missing the copy silently drops `value_composite`
    and would make the whole retro-score wrong.
"""

import argparse
import os
import sys
import time
import yaml
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import wandb

# Mirror train_sweep.py imports so we pick up the same behaviour and modules
from utils import (
    download_assets_from_gcs,
    init_seed,
    choose_metric,
    load_base_configs,
    apply_all_wandb_overrides,
)
from data.sources import create_data_pipeline, DataPipelineResult
from detectors import DETECTOR
from logger import create_logger
from trainer.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retro-score a checkpoint under configured value_composite gates."
    )
    parser.add_argument("--checkpoint_gcs_path", type=str, required=True,
                        help="gs:// path to the checkpoint to retro-score.")
    parser.add_argument("--config_yaml", type=str, required=True,
                        help="Training-style yaml (single_cfg). Contents feed W&B run config "
                             "and drive dataloaders / model config.")
    parser.add_argument("--wandb_run_name", type=str, required=True,
                        help="W&B display name; script prefixes with 'retro_'.")
    parser.add_argument("--wandb_project", type=str, default="enhanced-aug-test",
                        help="W&B project (default: enhanced-aug-test).")
    parser.add_argument("--wandb_entity", type=str,
                        default=os.environ.get("WANDB_ENTITY", "dtect-vision"),
                        help="W&B entity (default: from $WANDB_ENTITY or 'dtect-vision').")
    parser.add_argument('--detector_path', type=str, default='./config/detector/effort.yaml')
    parser.add_argument('--dataloader_config', type=str, default='./config/dataloader_config.yml')
    parser.add_argument('--local_rank', type=int, default=0)
    # Accept but ignore --param-config so this script is drop-in compatible with
    # the Vertex AI job template that always passes --param-config.
    parser.add_argument('--param-config', type=str, default=None, dest='param_config',
                        help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()
    # Allow --param-config as a fallback for --config_yaml when the launcher
    # pattern is reused as-is.
    if (not args.config_yaml) and args.param_config:
        args.config_yaml = args.param_config
    return args


def _load_state_dict_into_model(model, checkpoint_data, saved_config, logger):
    """Copy of rerun_validation.load_model_weights_into_configured_model.

    Handles both old (bare state_dict) and new (dict with 'state_dict' +
    'model_config') checkpoint formats, restores ArcFace `s`, and strips the
    DDP 'module.' prefix.
    """
    if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data:
        state_dict = checkpoint_data['state_dict']
    else:
        state_dict = checkpoint_data

    if saved_config.get('use_arcface_head', False) and 'current_arcface_s' in saved_config:
        if hasattr(model, 'head') and hasattr(model.head, 's'):
            current_s = saved_config['current_arcface_s']
            model.head.s.data.fill_(current_s)
            logger.info(f"Restored ArcFace s parameter: {current_s}")

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys while loading checkpoint ({len(missing)}): "
                       f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"Unexpected keys while loading checkpoint ({len(unexpected)}): "
                       f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info("Model weights loaded into configured model.")


def _apply_single_cfg_overrides(config, data_config, single_cfg, logger):
    """Mirror of train_sweep.py:171-282 nested-block copy. Must run BEFORE
    any W&B-config roundtrip to survive wandb.config flattening of nested dicts.
    """
    if not single_cfg:
        logger.warning("single_cfg is empty — no yaml overrides applied.")
        return

    # Data-side nested blocks
    if 'dataset_methods' in single_cfg:
        data_config['dataset_methods'] = single_cfg['dataset_methods']
    if 'augmentation' in single_cfg:
        config['augmentation'] = single_cfg['augmentation']
        data_config['augmentation'] = single_cfg['augmentation']
    if 'combined_paired' in single_cfg:
        data_config['combined_paired'] = single_cfg['combined_paired']
    if 'deeplive' in single_cfg:
        data_config['deeplive'] = single_cfg['deeplive']
    if 'visomaster' in single_cfg:
        data_config['visomaster'] = single_cfg['visomaster']

    # Training/model-side nested blocks
    if 'lesson_data_control' in single_cfg:
        config['lesson_data_control'] = single_cfg['lesson_data_control']
    if 'lesson_gate' in single_cfg:
        config['lesson_gate'] = single_cfg['lesson_gate']
    if 'backbone' in single_cfg:
        config['backbone'] = single_cfg['backbone']
    if 'checkpointing' in single_cfg:
        config['checkpointing'] = single_cfg['checkpointing']
    if 'use_group_dro' in single_cfg:
        config['use_group_dro'] = single_cfg['use_group_dro']
    if 'group_dro_params' in single_cfg:
        config['group_dro_params'] = single_cfg['group_dro_params']

    # ---- CRITICAL FOR THIS RUNNER ----
    # value_composite is the whole point of retro-scoring. Without this copy,
    # the trainer falls back to legacy defaults (0.02, 0.04, "max") and the
    # retro-score silently comes out under the LEGACY gate.
    if 'value_composite' in single_cfg:
        config['value_composite'] = single_cfg['value_composite']
        logger.info(f"Applied value_composite block: {single_cfg['value_composite']}")
    else:
        logger.warning("value_composite block missing from single_cfg — trainer will use "
                       "legacy defaults (0.02, 0.04, 'max').")

    # Flat scalars (simple str/float/int yaml keys). W&B doesn't flatten these
    # but copying them directly keeps this runner independent of wandb.config.
    flat_keys = (
        'name', 'description', 'seed', 'manualSeed', 'data_source',
        'model_name',
        'learning_rate', 'weight_decay', 'optimizer_eps', 'lambda_reg',
        'rank', 'gradient_clip_val',
        'nEpochs', 'total_training_steps',
        'lr_scheduler', 'lr_scheduler_warmup_steps',
        'dataloader_strategy', 'frames_per_batch', 'frames_per_video',
        'num_workers', 'prefetch_factor',
        'load_base_checkpoint', 'gcs_base_checkpoint',
        'label_smoothing', 'stability_lambda', 'stability_noise_std',
        'stability_crop_jitter',
        'use_arcface_head', 'arcface_m', 'arcface_s',
        's_start', 's_end', 'anneal_steps',
        'evaluate_every_steps', 'test_batch_size', 'test_batchSize',
        'early_stopping_enabled', 'early_stopping_patience',
    )
    for k in flat_keys:
        if k in single_cfg:
            config[k] = single_cfg[k]
            if k == 'data_source':
                data_config['data_source'] = single_cfg[k]


def main():
    args = parse_args()
    if not args.config_yaml:
        sys.stderr.write("ERROR: --config_yaml (or --param-config) is required.\n")
        sys.exit(2)

    # --- 1. Load base detector + dataloader configs (same as train_sweep.py) ---
    config, data_config = load_base_configs(
        detector_path=args.detector_path,
        dataloader_config_path=args.dataloader_config,
    )

    # --- 2. Load the experiment yaml ---
    with open(args.config_yaml, "r") as f:
        single_cfg = yaml.safe_load(f) or {}

    # --- 3. W&B init ---
    run_display_name = f"retro_{args.wandb_run_name}"
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_display_name,
        tags=[
            "retro-score",
            "packet-3",
            "new-metric-0305p95" if (single_cfg.get('value_composite') or {}).get('target_mean_fpr') == 0.03
            else "legacy-metric-0204max",
        ],
        config=single_cfg,
    )
    wandb.run.summary["retro/checkpoint_gcs_path"] = args.checkpoint_gcs_path
    wandb.run.summary["retro/config_yaml"] = os.path.abspath(args.config_yaml)

    # --- 4. Logger ---
    logger_path = os.path.join(wandb_run.dir, 'logs')
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'retro_score.log'))
    logger.info("=" * 70)
    logger.info(f"retro_score_value_composite runner  (run: {run_display_name})")
    logger.info(f"  checkpoint: {args.checkpoint_gcs_path}")
    logger.info(f"  yaml:       {args.config_yaml}")
    logger.info(f"  project:    {args.wandb_entity}/{args.wandb_project}")
    logger.info("=" * 70)

    # --- 5. Apply single_cfg into config/data_config (mirror train_sweep.py) ---
    _apply_single_cfg_overrides(config, data_config, single_cfg, logger)
    # Belt-and-suspenders: use train_sweep.py's flat-key helper for anything we missed.
    try:
        apply_all_wandb_overrides(config, data_config, wandb.config, logger)
    except Exception as wandb_override_err:
        logger.warning(f"apply_all_wandb_overrides raised: {wandb_override_err}. "
                       "Continuing with explicit-copy values only.")

    # --- 6. Standard setup ---
    config['local_rank'] = args.local_rank
    config['ddp'] = False
    config['save_ckpt'] = False  # retro-score never writes checkpoints

    canonical_seed = (
        config.get('manualSeed')
        or config.get('seed')
        or data_config.get('data_params', {}).get('seed')
        or 737
    )
    config['manualSeed'] = canonical_seed
    config['seed'] = canonical_seed
    if 'data_params' not in data_config:
        data_config['data_params'] = {}
    data_config['data_params']['seed'] = canonical_seed
    if isinstance(data_config.get('combined_paired'), dict):
        data_config['combined_paired'].setdefault('split_seed', canonical_seed)
    init_seed(config)
    if config.get('cudnn'):
        cudnn.benchmark = True

    # --- 7. Override base_checkpoint path, copy other gcs_assets from yaml ---
    if 'gcs_assets' not in config:
        config['gcs_assets'] = {}
    if 'gcs_assets' in single_cfg:
        for k, v in single_cfg['gcs_assets'].items():
            if k == 'base_checkpoint':
                continue  # CLI path wins
            config['gcs_assets'][k] = v
    config['gcs_assets']['base_checkpoint'] = {
        'gcs_path': args.checkpoint_gcs_path,
        'local_path': './weights/retro_checkpoint.pth',
    }
    config['load_base_checkpoint'] = True  # required for download_assets_from_gcs to pull it

    logger.info("Downloading GCS assets (backbone + retro-score checkpoint)...")
    downloaded = download_assets_from_gcs(config, logger)
    if downloaded is None:
        logger.error("GCS asset download failed — aborting.")
        wandb_run.finish(exit_code=1)
        sys.exit(1)

    # Mirror train_sweep.py:411-419 property-balancing parquet path wiring.
    if (data_config.get('property_balancing', {}).get('enabled', False) and
            'property_manifest_parquet' in config.get('gcs_assets', {})):
        local_path = config['gcs_assets']['property_manifest_parquet']['local_path']
        data_config['property_balancing']['frame_properties_parquet_path'] = local_path
        logger.info(f"Wired property_balancing parquet path: {local_path}")

    # --- 8. Build full data pipeline (same as training) ---
    logger.info("Building data pipeline (this takes several minutes)...")
    pipeline: DataPipelineResult = create_data_pipeline(config, data_config, logger)

    # Mirror train_sweep.py:423 — merge data_config into config.
    config.update(data_config)

    # --- 9. Load checkpoint -> restore saved model_config BEFORE creating model ---
    local_ckpt = config['gcs_assets']['base_checkpoint']['local_path']
    if not os.path.isfile(local_ckpt):
        logger.error(f"Checkpoint not found at {local_ckpt} after download — aborting.")
        wandb_run.finish(exit_code=1)
        sys.exit(1)
    logger.info(f"Reading checkpoint metadata from {local_ckpt}")
    saved = torch.load(local_ckpt, map_location='cpu')
    saved_cfg = {}
    if isinstance(saved, dict) and 'model_config' in saved:
        saved_cfg = saved['model_config'] or {}
        logger.info("Restoring saved model_config before model creation:")
        for k, v in saved_cfg.items():
            if k == 'current_arcface_s':
                continue  # dynamic; restored later via model.head.s
            old = config.get(k)
            if old != v:
                logger.info(f"  {k}: {old} -> {v}")
            config[k] = v
        logger.info(f"Checkpoint info: epoch={saved.get('epoch')} auc={saved.get('auc'):.4f}"
                    if saved.get('auc') is not None
                    else f"Checkpoint info: epoch={saved.get('epoch')}")
    else:
        logger.warning("OLD checkpoint format (no model_config) — model built from yaml only.")

    # --- 10. Instantiate model with correct config ---
    logger.info(f"Creating detector: {config['model_name']}")
    model = DETECTOR[config['model_name']](config)
    _load_state_dict_into_model(model, saved, saved_cfg, logger)

    # --- 11. Instantiate Trainer ---
    metric_scoring = choose_metric(config)
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=None,
        scheduler=None,
        logger=logger,
        val_in_dist_loader=pipeline.val_in_dist_loader,
        val_holdout_loader=pipeline.val_holdout_loader,
        ood_loader=pipeline.ood_loader,
        ood_heldout_loader=pipeline.ood_heldout_loader,
        test_loader=pipeline.test_loader,
        metric_scoring=metric_scoring,
        wandb_run=wandb_run,
        use_group_dro=config.get('use_group_dro', False),
    )

    # --- 12. Run ONE OOD monitoring pass. This is the whole point of the script ---
    if trainer.ood_loader is None:
        logger.error("ood_loader is None — cannot compute value_composite.")
        wandb_run.finish(exit_code=1)
        sys.exit(1)

    logger.info(">>> Running ood_monitoring_epoch (retro-score pass) <<<")
    t0 = time.time()
    trainer.ood_monitoring_epoch(epoch=0, step_cnt=0, indist_threshold=None)
    logger.info(f"OOD monitoring completed in {time.time() - t0:.1f}s")

    # --- 13. Mirror value_composite to retro/* summary keys for easy W&B pulls ---
    vc = getattr(trainer, "_last_value_composite", None)
    if vc is not None:
        def _set(key, val):
            if val is not None and not (isinstance(val, float) and val != val):  # skip NaN
                wandb.run.summary[key] = val
        _set("retro/value_composite", vc.get("value_composite"))
        _set("retro/tau", vc.get("tau"))
        _set("retro/teams_fakes_tpr", vc.get("teams_fakes_tpr"))
        _set("retro/other_fakes_tpr", vc.get("other_fakes_tpr"))
        _set("retro/stability", vc.get("stability"))
        _set("retro/mean_fpr", vc.get("mean_fpr"))
        _set("retro/max_fpr", vc.get("max_fpr"))
        _set("retro/blocked_by", vc.get("value_composite_blocked_by"))
        wandb.run.summary["retro/target_mean_fpr"] = trainer._vc_target_mean_fpr
        wandb.run.summary["retro/max_pool_fpr"] = trainer._vc_max_pool_fpr
        wandb.run.summary["retro/stability_jitter_stat"] = trainer._vc_stability_jitter_stat
        logger.info(
            "RETRO RESULT: value_composite=%s teams_tpr=%s other_tpr=%s stability=%.4f "
            "tau=%s  (gates: target_mean_fpr=%.3f max_pool_fpr=%.3f stat=%s)",
            vc.get("value_composite"),
            vc.get("teams_fakes_tpr"),
            vc.get("other_fakes_tpr"),
            vc["stability"],
            vc.get("tau"),
            trainer._vc_target_mean_fpr,
            trainer._vc_max_pool_fpr,
            trainer._vc_stability_jitter_stat,
        )
    else:
        logger.error("trainer._last_value_composite is missing — ood pass likely failed.")

    wandb_run.finish()
    logger.info("Retro-score run complete.")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed = (time.time() - start) / 60.0
    print(f"Total retro-score time: {elapsed:.2f} minutes")
