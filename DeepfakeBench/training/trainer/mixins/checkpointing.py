"""
Checkpointing mixin for Trainer.

Provides functionality for:
- Saving model checkpoints locally and to GCS
- Managing top-N checkpoints
- Computing model checksums for validation
- Loading checkpoints with config validation
"""
import hashlib
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
from google.api_core import exceptions
from google.cloud import storage


class CheckpointingMixin:
    """
    Mixin that provides checkpointing functionality for the Trainer.
    
    Assumes the base class has:
    - self.config: Training configuration dict
    - self.model: The PyTorch model
    - self.logger: Logger instance
    - self.wandb_run: W&B run instance (optional)
    - self.log_dir: Directory for local logs
    """
    
    def init_checkpointing(self):
        """Initialize checkpointing state. Call from __init__."""
        # List of dicts: [{'metric': float, 'epoch': int, 'gcs_path': str, ...}, ...]
        self.top_n_checkpoints: List[Dict] = []
        self.top_n_size = self.config.get('checkpointing', {}).get('top_n_size', 6)
        self.first_best_gcs_path: Optional[str] = None
        self.top_n_saved_count = 0
        
        self.best_val_metric = -1.0
        self.best_val_epoch = -1

        # --- OOD-composite checkpointing (R12) ---
        # Keeps a separate top-N list ranked by hmean(holdout_auc, ood_auc).
        # Enabled via config: checkpointing.ood_composite_enabled: true
        ckpt_cfg = self.config.get('checkpointing', {})
        self.ood_composite_enabled = bool(ckpt_cfg.get('ood_composite_enabled', False))
        self.ood_composite_top_n: List[Dict] = []
        self.ood_composite_top_n_size = ckpt_cfg.get('ood_composite_top_n_size', 3)
        self.best_ood_composite = -1.0
        self.best_ood_composite_step = -1

        # --- A2: dual-checkpoint tracking for deployment-aligned readout ---
        # best_value_composite is computed but does NOT drive packet-3 selection
        # (see R13 Packet 3 plan §8.7). Tracked alongside best_ood_composite so
        # A2 final_eval can compare the two choices. In-memory CPU snapshots
        # avoid depending on GCS round-trips at training end.
        self.value_composite_enabled = bool(ckpt_cfg.get('value_composite_enabled', True))
        self.value_composite_top_n: List[Dict] = []
        self.value_composite_top_n_size = ckpt_cfg.get('value_composite_top_n_size', 3)
        self.best_value_composite = -1.0
        self.best_value_composite_step = -1
        self._best_ood_composite_state_dict_cpu = None
        self._best_value_composite_state_dict_cpu = None

        # --- Anchor-pool monitor state (Phase 0.4 methodology fix) ---
        # ``value_composite`` does NOT reflect anchor-pool false-positive
        # behaviour (the actual deployment failure mode). The anchor monitor
        # tracks per-step pool metrics on ~180 cached real Dor/Roee frames and
        # GATES ``value_composite_*.pth`` checkpoint writes. See R13 plan §8.
        validation_cfg = self.config.get('validation', {}) or {}
        self.anchor_monitor_enabled = bool(
            validation_cfg.get('anchor_monitor_enabled', True)
        )
        self.anchor_cache_dir = str(
            validation_cfg.get('anchor_cache_dir', '~/.cache/anchor_pools')
        )
        # Allowed regression on anchor/composite when value_composite alone
        # justifies a save. Larger -> more permissive.
        self.anchor_regression_tolerance = float(
            validation_cfg.get('anchor_regression_tolerance', 0.02)
        )
        self.best_anchor_composite = -1e9  # sentinel: any real value beats it
        self.best_anchor_composite_step = -1
        self.last_anchor_composite = None  # set by _run_validation each step
        self._anchor_monitor_failures = 0
        self._anchor_monitor_disabled_after_failures = False
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str) -> bool:
        """Uploads a local file to a GCS path."""
        try:
            storage_client = storage.Client()
            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            blob_name = gcs_path.split(f'gs://{bucket_name}/', 1)[1]
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            self.logger.info(f"Uploading checkpoint to GCS: {gcs_path}")
            blob.upload_from_filename(local_path)
            self.logger.info(f"✅ SUCCESS: Uploaded to {gcs_path}")
            return True
        except exceptions.GoogleAPICallError as e:
            self.logger.error(f"FAILED to upload to GCS. Check permissions. Error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during GCS upload: {e}")
            return False

    def _delete_from_gcs(self, gcs_path: str) -> None:
        """Deletes a blob from a given GCS path."""
        if not gcs_path:
            return
        try:
            storage_client = storage.Client()
            bucket_name = gcs_path.split('gs://', 1)[1].split('/', 1)[0]
            blob_name = gcs_path.split(f'gs://{bucket_name}/', 1)[1]
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if blob.exists():
                self.logger.info(f"Deleting old GCS checkpoint: {gcs_path}")
                blob.delete()
                self.logger.info(f"✅ SUCCESS: Deleted {gcs_path}")
            else:
                self.logger.warning(f"Attempted to delete non-existent GCS blob: {gcs_path}")
        except Exception as e:
            self.logger.error(f"Failed to delete GCS blob {gcs_path}. Error: {e}")

    def compute_model_checksum(self) -> str:
        """
        Compute a checksum of the model state for validation.
        
        Returns:
            16-character hex string representing the model state hash
        """
        model_instance = self.model.module if self.config.get('ddp') else self.model
        state_dict = model_instance.state_dict()
        
        # Create deterministic hash from model parameters
        checksum_data = []
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            # Use tensor statistics for efficiency
            checksum_data.append(f"{key}:{tensor.shape}:{tensor.sum().item():.6f}")
        
        # Include critical config in checksum
        config_items = [
            f"use_arcface_head:{self.config.get('use_arcface_head', False)}",
            f"arcface_s:{self.config.get('arcface_s', 30.0)}",
            f"arcface_m:{self.config.get('arcface_m', 0.35)}",
            f"rank:{self.config.get('rank', 1023)}",
        ]
        checksum_data.extend(config_items)
        
        # Create hash
        combined_str = "|".join(checksum_data)
        checksum = hashlib.sha256(combined_str.encode()).hexdigest()[:16]
        
        return checksum

    def save_ckpt(
        self,
        epoch: int,
        auc: float,
        eer: float,
        ckpt_prefix: str = 'ckpt',
        step: Optional[int] = None
    ) -> Optional[str]:
        """
        Saves model checkpoint locally, uploads to GCS with a prefix, and cleans up.
        
        Args:
            epoch: The epoch number
            auc: The AUC score
            eer: The EER score
            ckpt_prefix: Prefix for the checkpoint name (e.g., 'first_best', 'top_n')
            step: Optional step count to use in naming instead of epoch
            
        Returns:
            The GCS path of the uploaded file, or None if upload failed
        """
        gcs_config = self.config.get('checkpointing')
        if not gcs_config or not gcs_config.get('gcs_prefix'):
            self.logger.warning("GCS checkpointing not configured. Skipping upload.")
            return None

        model_name = self.config.get('model_name', 'model')
        date_str = time.strftime("%Y%m%d")
        
        # Use step count for top_n checkpoints, epoch for others
        if step is not None:
            ckpt_name = f"{ckpt_prefix}_{model_name}_{date_str}_step{step}_auc{auc:.4f}_eer{eer:.4f}.pth"
        else:
            ckpt_name = f"{ckpt_prefix}_{model_name}_{date_str}_ep{epoch}_auc{auc:.4f}_eer{eer:.4f}.pth"

        local_save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(local_save_dir, exist_ok=True)
        local_save_path = os.path.join(local_save_dir, ckpt_name)

        model_state = self.model.module.state_dict() if self.config.get('ddp') else self.model.state_dict()
        
        # Save complete checkpoint with configuration for exact reconstruction
        checkpoint = {
            'state_dict': model_state,
            'model_config': {
                'model_name': self.config.get('model_name'),
                'use_arcface_head': self.config.get('use_arcface_head', False),
                'arcface_s': self.config.get('arcface_s', 30.0),
                'arcface_m': self.config.get('arcface_m', 0.35),
                's_start': self.config.get('s_start'),
                's_end': self.config.get('s_end'),
                'anneal_steps': self.config.get('anneal_steps', 0),
                'use_focal_loss': self.config.get('use_focal_loss', False),
                'focal_loss_gamma': self.config.get('focal_loss_gamma', 2.0),
                'focal_loss_alpha': self.config.get('focal_loss_alpha'),
                'lambda_reg': self.config.get('lambda_reg', 1.0),
                'rank': self.config.get('rank', 1023),
            },
            'epoch': epoch,
            'auc': auc,
            'eer': eer,
            'training_step': getattr(self, 'current_step', None),
        }

        gcs_assets = self.config.get('gcs_assets') or {}
        checkpoint['model_config'].update({
            'backbone': self.config.get('backbone', {}),
            'backbone_path': self.config.get('backbone_path'),
            'backbone_name': self.config.get('backbone_name'),
            'backbone_config': self.config.get('backbone_config'),
            'gcs_assets': {
                'clip_backbone': gcs_assets.get('clip_backbone'),
            },
            'mean': self.config.get('mean'),
            'std': self.config.get('std'),
            'metadata_version': 2,
            'metadata_updated_at_utc': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        
        # If using ArcFace with annealing, save current s value
        if self.config.get('use_arcface_head', False):
            model_instance = self.model.module if self.config.get('ddp') else self.model
            if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                checkpoint['model_config']['current_arcface_s'] = float(model_instance.head.s)
        
        # Add model checksum for validation
        checkpoint['model_checksum'] = self.compute_model_checksum()
        
        torch.save(checkpoint, local_save_path)
        self.logger.info(f"💾 Saved checkpoint with checksum: {checkpoint['model_checksum']}")

        gcs_prefix = gcs_config['gcs_prefix']
        if not gcs_prefix.endswith('/'):
            gcs_prefix += '/'
        run_id = self.wandb_run.id if self.wandb_run else "local_run"
        full_gcs_path = gcs_prefix + f"{run_id}/{ckpt_name}"

        upload_success = self._upload_to_gcs(local_save_path, full_gcs_path)

        try:
            os.remove(local_save_path)
        except OSError as e:
            self.logger.warning(f"Could not delete local temporary checkpoint: {e}")

        return full_gcs_path if upload_success else None

    def load_ckpt(self, model_path: str, validate: bool = True) -> None:
        """
        Load a model checkpoint from a file path.
        
        Args:
            model_path: Path to the checkpoint file
            validate: Whether to validate config compatibility
        """
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            
            # Handle both old (state_dict only) and new (complete checkpoint) formats
            if isinstance(saved, dict) and 'state_dict' in saved:
                # New format with configuration
                state_dict = saved['state_dict']
                model_config = saved.get('model_config', {})

                if validate:
                    self._validate_model_config(model_config, model_path)
                
                # Restore dynamic parameters if available
                if model_config.get('use_arcface_head', False) and 'current_arcface_s' in model_config:
                    model_instance = self.model.module if self.config.get('ddp') else self.model
                    if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                        current_s = model_config['current_arcface_s']
                        model_instance.head.s.data.fill_(current_s)
                        self.logger.info(f"Restored ArcFace s parameter to: {current_s}")
                
                self.logger.info(f"Loaded checkpoint from epoch {saved.get('epoch', 'unknown')} "
                               f"with AUC: {saved.get('auc', 'unknown'):.4f}")
            else:
                # Old format (state_dict only)
                state_dict = saved
                self.logger.warning(f"Loading old checkpoint format from {model_path}. "
                                   "Configuration validation not possible.")
            
            # Load state dict with module prefix handling
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Validate model checksum if available
            train_arcface = self.config.get('train_arcface', True)
            if isinstance(saved, dict) and 'model_checksum' in saved and not train_arcface:
                expected_checksum = saved['model_checksum']
                actual_checksum = self.compute_model_checksum()
                if expected_checksum == actual_checksum:
                    self.logger.info(f'✅ Model checksum validated: {actual_checksum}')
                else:
                    self.logger.error(f'❌ Model checksum mismatch! Expected: {expected_checksum}, Got: {actual_checksum}')
                    raise ValueError("Model checksum validation failed - model state may be corrupted")
            elif isinstance(saved, dict) and 'model_checksum' in saved and train_arcface:
                self.logger.info("⚠️  Skipping checksum validation - curriculum learning may modify parameters")
            else:
                self.logger.warning("⚠️  No checksum available for validation (old checkpoint format)")
            
            self.logger.info(f'Model loaded from {model_path}')
        else:
            raise FileNotFoundError(f"=> no model found at '{model_path}'")

    def _validate_model_config(self, saved_config: Dict, checkpoint_path: str) -> None:
        """Validate that the saved model configuration matches current configuration."""
        if not saved_config:
            self.logger.warning("No model configuration found in checkpoint. Skipping validation.")
            return
        
        # Critical parameters that must match exactly
        critical_params = [
            'model_name', 'use_arcface_head', 
            'use_focal_loss', 'focal_loss_gamma', 'focal_loss_alpha', 'rank'
        ]
        
        mismatches = []
        for param in critical_params:
            saved_value = saved_config.get(param)
            current_value = self.config.get(param)
            if saved_value != current_value and saved_value is not None:
                mismatches.append(f"  - {param}: saved={saved_value}, current={current_value}")
        
        if mismatches:
            self.logger.warning(
                f"Configuration mismatches detected for checkpoint {checkpoint_path}:\n" +
                "\n".join(mismatches)
            )

    def update_top_n_checkpoints(
        self,
        epoch: int,
        metric: float,
        eer: float,
        step: Optional[int] = None
    ) -> None:
        """
        Update the top-N checkpoint list and save if this is a top performer.
        
        Args:
            epoch: Current epoch
            metric: Primary metric value (e.g., AUC)
            eer: EER value
            step: Optional step count
        """
        # Determine if this should be saved as a top-N checkpoint
        if len(self.top_n_checkpoints) < self.top_n_size:
            should_save = True
        else:
            worst_in_top_n = min(self.top_n_checkpoints, key=lambda x: x['metric'])
            should_save = metric > worst_in_top_n['metric']
        
        if should_save:
            # Save the new checkpoint
            gcs_path = self.save_ckpt(epoch, metric, eer, ckpt_prefix='top_n', step=step)
            
            if gcs_path:
                self.top_n_saved_count += 1
                new_entry = {
                    'metric': metric,
                    'epoch': epoch,
                    'step': step,
                    'gcs_path': gcs_path,
                    'eer': eer,
                }
                self.top_n_checkpoints.append(new_entry)
                
                # If we exceeded top_n_size, remove the worst
                if len(self.top_n_checkpoints) > self.top_n_size:
                    worst = min(self.top_n_checkpoints, key=lambda x: x['metric'])
                    self.top_n_checkpoints.remove(worst)
                    self._delete_from_gcs(worst['gcs_path'])
                    self.logger.info(
                        f"📊 Removed checkpoint with AUC={worst['metric']:.4f} from top-{self.top_n_size}"
                    )
                
                self.logger.info(
                    f"📊 Top-{self.top_n_size} checkpoint saved: AUC={metric:.4f} "
                    f"(#{self.top_n_saved_count})"
                )
