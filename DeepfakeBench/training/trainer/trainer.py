import math
import os
import sys
import time
import gc
import contextlib

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import random
from collections import OrderedDict
import numpy as np  # noqa
from tqdm import tqdm  # noqa
import torch  # noqa
import torch.nn.functional as F  # noqa - Added Jan 10, 2026 for ArcFace diagnostics
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from metrics.utils import get_test_metrics, metrics_at_threshold  # noqa
from torch.cuda.amp import autocast, GradScaler  # noqa
import wandb  # noqa
from collections import defaultdict
from dataset.dataloaders import load_and_process_video, collate_fn  # noqa
from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
import gc
import csv
import tempfile
import shutil
from datetime import datetime
from sklearn.metrics import confusion_matrix
from utils.grouping import infer_group_and_family

# Import trainer mixins for modular functionality
from trainer.mixins import (
    CheckpointingMixin,
    EarlyStoppingMixin,
    GroupDROMixin,
    CurriculumMixin,
    ArcFaceMixin,
    ValidationMixin,
    ReportingMixin,
    StabilityRegMixin,
)

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trainer Is Using device: {device}")


class Trainer(
    CheckpointingMixin,
    EarlyStoppingMixin,
    GroupDROMixin,
    CurriculumMixin,
    ArcFaceMixin,
    ValidationMixin,
    ReportingMixin,
    StabilityRegMixin,
):
    """
    Main trainer class for DeepfakeBench training.
    
    Composes functionality from multiple mixins:
    - CheckpointingMixin: Model saving, top-N checkpoints, GCS upload
    - EarlyStoppingMixin: Patience-based early stopping
    - GroupDROMixin: Distributionally Robust Optimization
    - CurriculumMixin: Lesson gates and curriculum learning
    - ArcFaceMixin: ArcFace head parameter annealing
    - ValidationMixin: Validation state management
    - ReportingMixin: Report generation and GCS upload
    """
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            val_in_dist_loader,
            val_holdout_loader,
            metric_scoring='auc',
            wandb_run=None,
            ood_loader=None,
            use_group_dro=False  # Argument to activate the feature
    ):
        if config is None or model is None or logger is None:
            raise ValueError("config, model, and logger must be provided")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.wandb_run = wandb_run
        self.val_in_dist_loader = val_in_dist_loader
        self.val_holdout_loader = val_holdout_loader
        self.ood_loader = ood_loader  # Optional OOD loader
        self.unified_val_loader = None  # To cache the efficient loader

        # --- Initialize mixins ---
        # Checkpointing: manages model checkpoints and GCS uploads
        self.init_checkpointing()
        
        # Early stopping: patience-based training termination
        self.init_early_stopping()

        # --- Step-based training control ---
        self.max_train_steps = self.config.get('max_train_steps', None)
        self.evaluate_every_steps = self.config.get('evaluate_every_steps', None)
        if self.max_train_steps:
            self.logger.info(f"✅ Training will stop at a maximum of {self.max_train_steps} total steps.")
        if self.evaluate_every_steps:
            self.logger.info(
                f"✅ Evaluation will run every {self.evaluate_every_steps} steps, overriding epoch frequency.")

        # --- OOD monitoring cadence controls ---
        # Defaults preserve prior behavior: run on every validation call.
        self.ood_monitoring_enabled = bool(self.config.get('ood_monitoring_enabled', True))
        self.ood_monitoring_start_step = int(self.config.get('ood_monitoring_start_step', 0) or 0)
        cfg_ood_every = self.config.get('ood_monitoring_every_steps')
        if cfg_ood_every is None:
            if self.evaluate_every_steps and self.evaluate_every_steps > 0:
                self.ood_monitoring_every_steps = int(self.evaluate_every_steps)
            else:
                self.ood_monitoring_every_steps = 1
        else:
            self.ood_monitoring_every_steps = int(cfg_ood_every)
            if self.ood_monitoring_every_steps <= 0:
                self.ood_monitoring_every_steps = 1
        self._last_ood_monitor_step = None
        self._ood_warmup_logged = False
        self.logger.info(
            "OOD monitoring cadence: enabled=%s start_step=%d every_steps=%d",
            self.ood_monitoring_enabled,
            self.ood_monitoring_start_step,
            self.ood_monitoring_every_steps,
        )

        # Initialize AMP scaler for mixed precision training
        self.scaler = GradScaler()
        self.gradient_clip_val = self.config.get('gradient_clip_val')
        if self.gradient_clip_val:
            self.logger.info(f"✅ Gradient clipping enabled with max norm: {self.gradient_clip_val}")
        self.speed_up()

        self.log_dir = self.wandb_run.dir if self.wandb_run else './logs'
        # These are now only used for 'per_method' strategy, but are initialized here
        self.real_method_iters = {}
        self.fake_method_iters = {}

        # Group-DRO: distributionally robust optimization
        self.use_group_dro = use_group_dro
        if self.use_group_dro:
            self.init_group_dro()

        # Curriculum learning: lesson gate functionality
        self.init_curriculum()
        
        # ArcFace: parameter annealing (if using ArcFace head)
        self.init_arcface()

        # Stability regularisation: perturbation consistency loss
        self.init_stability_reg()

    # --- Group-DRO methods are now provided by GroupDROMixin ---
    # The mixin provides: init_group_dro(), calculate_group_dro_loss(), get_group_dro_stats()

    def _update_arcface_s(self, step_cnt):
        """Anneals the 's' parameter of the ArcFace head if configured."""
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model

        # Check if the model is an EffortDetector and has annealing configured
        if not hasattr(model_instance, 'use_arcface_head') or not model_instance.use_arcface_head:
            return
        if not hasattr(model_instance, 'anneal_steps') or model_instance.anneal_steps <= 0:
            return

        anneal_steps = model_instance.anneal_steps

        # Get the device from the existing buffer to ensure device consistency
        target_device = model_instance.head.s.device

        if step_cnt <= anneal_steps:
            # Linear annealing
            progress = step_cnt / anneal_steps
            current_s_float = model_instance.s_start + progress * (model_instance.s_end - model_instance.s_start)

            # Convert the float to a tensor on the correct device
            current_s_tensor = torch.tensor(current_s_float, device=target_device)
            model_instance.head.s = current_s_tensor
        else:
            # Ensure s is fixed at s_end after annealing is complete
            current_s_float = model_instance.s_end

            # Only update if necessary to avoid redundant operations
            if model_instance.head.s.item() != current_s_float:
                current_s_tensor = torch.tensor(current_s_float, device=target_device)
                model_instance.head.s = current_s_tensor

        # Log the change periodically during the annealing phase
        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        if self.wandb_run and step_cnt <= anneal_steps and (
                step_cnt % log_progress_steps == 0 or step_cnt == anneal_steps):
            self.wandb_run.log({'train/arcface_s': model_instance.head.s, 'train/step': step_cnt})

    def _update_lambda_reg(self, step_cnt):
        """Anneals lambda_reg (orthogonal constraint) if configured."""
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Check if the model supports lambda_reg annealing
        if not hasattr(model_instance, 'update_lambda_reg'):
            return
        if not hasattr(model_instance, 'lambda_reg_anneal_steps') or model_instance.lambda_reg_anneal_steps <= 0:
            return
        
        # Update lambda_reg via model method
        current_lambda = model_instance.update_lambda_reg(step_cnt)
        
        # Log the change periodically during the annealing phase
        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        anneal_steps = model_instance.lambda_reg_anneal_steps
        if self.wandb_run and step_cnt <= anneal_steps and (
                step_cnt % log_progress_steps == 0 or step_cnt == anneal_steps):
            self.wandb_run.log({'train/lambda_reg': current_lambda, 'train/step': step_cnt})

    def _update_quality_domain_lambda(self, step_cnt):
        """Anneal gradient-reversal lambda for the quality-domain head (DANN sigmoid schedule)."""
        model_instance = self.model.module if isinstance(self.model, DDP) else self.model
        if not getattr(model_instance, 'use_quality_head', False):
            return

        total_steps = self.config.get('total_training_steps', 65000)
        progress = min(step_cnt / max(total_steps, 1), 1.0)
        # Sigmoid schedule: 0→~1 with inflection at 50% of training
        lambda_val = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        model_instance.quality_head.set_lambda(lambda_val)

        log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
        if self.wandb_run and step_cnt % log_progress_steps == 0:
            self.wandb_run.log({
                'train/quality_grl_lambda': lambda_val,
                'train/step': step_cnt,
            })

    def _check_collapse_warning(self, predictions, data_dict, step_cnt):
        """
        Early warning system for model collapse.
        
        Added: Jan 10, 2026 - Critical diagnostic for ArcFace + SVD training
        
        Detects when model is outputting near-constant predictions, which indicates
        collapse into a degenerate solution (predicting all real or all fake).
        
        Returns:
            dict: Warning indicators and detailed collapse metrics
        """
        collapse_metrics = {}
        
        if 'raw_logits' not in predictions:
            return collapse_metrics
            
        raw_logits = predictions['raw_logits'].detach()
        probs = predictions['prob'].detach()
        labels = data_dict['label']
        
        # Handle expanded labels for video batches
        if raw_logits.shape[0] > labels.shape[0]:
            B = labels.shape[0]
            T = raw_logits.shape[0] // B
            labels = labels.repeat_interleave(T)
        
        # 1. Logit separation between classes
        logit_diff = raw_logits[:, 1] - raw_logits[:, 0]  # fake - real
        logit_std = logit_diff.std().item()
        logit_range = (logit_diff.max() - logit_diff.min()).item()
        
        # 2. Per-class logit statistics
        real_mask = (labels == 0)
        fake_mask = (labels == 1)
        
        if real_mask.any() and fake_mask.any():
            real_logit_mean = logit_diff[real_mask].mean().item()
            fake_logit_mean = logit_diff[fake_mask].mean().item()
            class_separation = fake_logit_mean - real_logit_mean  # Should be positive
            
            collapse_metrics['train/collapse/real_logit_diff_mean'] = real_logit_mean
            collapse_metrics['train/collapse/fake_logit_diff_mean'] = fake_logit_mean
            collapse_metrics['train/collapse/class_separation'] = class_separation
        
        # 3. Probability spread (should NOT be near 0)
        prob_spread = probs.std().item()
        prob_entropy = -((probs * (probs + 1e-8).log()) + ((1 - probs) * (1 - probs + 1e-8).log())).mean().item()
        
        collapse_metrics['train/collapse/logit_std'] = logit_std
        collapse_metrics['train/collapse/logit_range'] = logit_range
        collapse_metrics['train/collapse/prob_spread'] = prob_spread
        collapse_metrics['train/collapse/prob_entropy'] = prob_entropy
        
        # 4. Check for constant output (CRITICAL WARNING)
        is_constant_output = logit_std < 0.01
        collapse_metrics['train/collapse/is_constant_output'] = float(is_constant_output)
        
        if is_constant_output and step_cnt > 100:  # Only warn after warmup
            self.logger.warning(
                f"⚠️ COLLAPSE WARNING at step {step_cnt}: "
                f"Logit std={logit_std:.6f}, range={logit_range:.6f}. "
                f"Model may be outputting near-constant predictions!"
            )
            
            # Store collapse detection for potential intervention
            if not hasattr(self, '_collapse_warning_count'):
                self._collapse_warning_count = 0
            self._collapse_warning_count += 1
            collapse_metrics['train/collapse/warning_count'] = self._collapse_warning_count
        
        return collapse_metrics
    
    def _collect_arcface_diagnostics(self, step_cnt):
        """
        Collect ArcFace head diagnostics including gradient flow.
        
        Added: Jan 10, 2026 - For debugging ArcFace-induced collapse
        
        Returns:
            dict: ArcFace weight norms, gradient norms, and scale parameter
        """
        arcface_metrics = {}
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        
        if not hasattr(model_ref, 'head') or not hasattr(model_ref.head, 's'):
            return arcface_metrics
        
        head = model_ref.head
        
        # Current scale
        current_s = head.s.item() if hasattr(head.s, 'item') else float(head.s)
        arcface_metrics['train/arcface/scale'] = current_s
        
        # Weight statistics
        if hasattr(head, 'weight'):
            weight = head.weight.detach()
            arcface_metrics['train/arcface/weight_norm'] = weight.norm().item()
            
            # Class center separation (cosine distance between real and fake centers)
            if weight.shape[0] == 2:  # Binary classification
                real_center = F.normalize(weight[0:1], dim=1)
                fake_center = F.normalize(weight[1:2], dim=1)
                center_cosine = (real_center @ fake_center.T).item()
                arcface_metrics['train/arcface/center_cosine_sim'] = center_cosine
                # Ideally should be negative (opposite directions)
            
            # Gradient statistics (if available)
            if head.weight.grad is not None:
                grad = head.weight.grad
                arcface_metrics['train/arcface/weight_grad_norm'] = grad.norm().item()
                arcface_metrics['train/arcface/weight_grad_max'] = grad.abs().max().item()
        
        return arcface_metrics
    
    def _collect_svd_residual_stats(self):
        """
        Collects statistics from SVDResidualLinear layers for diagnostic logging.
        
        Added: Jan 4, 2026 (Task D3)
        Updated: Jan 10, 2026 - Added gradient flow diagnostics
        
        This helps diagnose the 'params_with_grad' drop observed in LAION B16 training,
        by tracking whether S_residual values are approaching zero (dying gradients).
        
        Returns:
            dict: Statistics including min/max/mean of S_residual across all layers,
                  and count of layers with very small S_residual values.
        """
        model_ref = self.model.module if isinstance(self.model, DDP) else self.model
        
        s_residual_values = []
        s_residual_mins = []
        s_residual_maxs = []
        layer_count = 0
        near_zero_count = 0
        
        # Threshold for considering S_residual as "near zero"
        near_zero_threshold = 1e-6
        
        for name, module in model_ref.named_modules():
            # Check for SVDResidualLinear (works for both HuggingFace and OpenCLIP)
            if hasattr(module, 'S_residual') and module.S_residual is not None:
                s_vals = module.S_residual.detach()
                layer_count += 1
                
                s_min = s_vals.min().item()
                s_max = s_vals.max().item()
                s_mean = s_vals.mean().item()
                
                s_residual_mins.append(s_min)
                s_residual_maxs.append(s_max)
                s_residual_values.append(s_mean)
                
                # Count layers where S_residual is very small (potential dying gradient)
                if s_vals.abs().max().item() < near_zero_threshold:
                    near_zero_count += 1
        
        if layer_count == 0:
            return {}
        
        stats = {
            'svd/S_residual_min': min(s_residual_mins),
            'svd/S_residual_max': max(s_residual_maxs),
            'svd/S_residual_mean': sum(s_residual_values) / len(s_residual_values),
            'svd/layer_count': layer_count,
            'svd/near_zero_layers': near_zero_count,
        }
        
        # Add per-layer detailed stats if requested (expensive, so optional)
        # This can be enabled via config if needed for deep debugging
        if self.config.get('log_svd_per_layer', False):
            for i, (s_min, s_max, s_mean) in enumerate(zip(s_residual_mins, s_residual_maxs, s_residual_values)):
                stats[f'svd/layer_{i}/S_min'] = s_min
                stats[f'svd/layer_{i}/S_max'] = s_max
        
        return stats

    def _check_lesson_gate(self, all_val_metrics: dict):
        """Checks if the curriculum lesson's gate conditions have been met."""
        self.logger.info("--- Checking Lesson Gate conditions...")

        # A helper to safely retrieve nested metric values
        def get_metric(metric_name, dataset):
            # This check is important: if config keys are missing, metric_name or dataset will be None.
            if not metric_name or not dataset:
                return None
            return all_val_metrics.get(dataset, {}).get(metric_name)

        # 1. Guardrail Check
        guardrail_ok = True
        guard_conf = self.gate_guardrail_config
        if guard_conf.get('enabled'):
            # FIX: Use .get() for safe dictionary access instead of ['key']
            guard_metric = get_metric(guard_conf.get('metric'), guard_conf.get('dataset'))
            if guard_metric is not None:
                if self.gate_guardrail_start_value is None:
                    self.gate_guardrail_start_value = guard_metric
                    self.logger.info(f"Guardrail initialized: {guard_conf.get('metric')} = {guard_metric:.4f}")

                drop = self.gate_guardrail_start_value - guard_metric
                if drop > guard_conf.get('max_drop', 0.01):
                    guardrail_ok = False
                    self.logger.warning(
                        f"🚨 GUARDRAIL FAILED: {guard_conf.get('metric')} dropped by {drop:.4f} (max allowed: {guard_conf.get('max_drop')})")
            else:
                self.logger.warning(
                    f"Guardrail check skipped: metric '{guard_conf.get('metric')}' not found for dataset '{guard_conf.get('dataset')}'. Check your lesson_gate config."
                )

        # 2. Threshold Checks
        all_thresholds_met = True
        for check in self.gate_checks:
            # FIX: Use .get() for safe dictionary access
            metric_val = get_metric(check.get('metric'), check.get('dataset'))
            if metric_val is None:
                all_thresholds_met = False
                self.logger.warning(
                    f"Gate check skipped: metric '{check.get('metric')}' not found for dataset '{check.get('dataset')}'.")
                break

            op = check.get('comparison')
            thresh = check.get('threshold')

            # FIX: Ensure comparison and threshold keys exist
            if op is None or thresh is None:
                all_thresholds_met = False
                self.logger.warning(
                    f"Gate check skipped: malformed check config (missing 'comparison' or 'threshold'): {check}")
                break

            passed = (op == 'ge' and metric_val >= thresh) or (op == 'le' and metric_val <= thresh)

            if not passed:
                all_thresholds_met = False
                self.logger.info(
                    f"Gate check FAILED: {check.get('dataset')}/{check.get('metric')} ({metric_val:.4f}) did not meet {op} {thresh}")
                break
            else:
                self.logger.info(
                    f"Gate check PASSED: {check.get('dataset')}/{check.get('metric')} ({metric_val:.4f}) met {op} {thresh}")

        # 3. Plateau Check (based on the primary validation metric)
        plateau_met = False
        plateau_conf = self.gate_plateau_config
        if plateau_conf.get('enabled') and all_thresholds_met:  # Only check plateau if thresholds are met
            primary_metric_val = all_val_metrics.get('val_holdout', {}).get('overall', {}).get(self.metric_scoring)
            if primary_metric_val is not None:
                self.gate_primary_metric_history.append(primary_metric_val)
                patience = plateau_conf.get('patience', 2)

                if len(self.gate_primary_metric_history) >= patience:
                    recent_history = self.gate_primary_metric_history[-patience:]
                    best_recent = max(recent_history)
                    improvement = best_recent - self.gate_primary_metric_history[-patience]

                    if improvement < plateau_conf.get('min_delta', 0.001):
                        plateau_met = True
                        self.logger.info(
                            f"✅ Plateau condition MET: Improvement ({improvement:.4f}) is less than min_delta over last {patience} evals.")

        # 4. Final Decision
        if guardrail_ok and all_thresholds_met and plateau_met:
            self.early_stop_triggered = True
            self.logger.critical("✅✅✅ LESSON GATE PASSED! All conditions met. Triggering stop.")
        else:
            self.logger.info("--- Lesson Gate conditions not yet met. Continuing training.")

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp']:
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True,
                             output_device=self.config['local_rank'])

    def setTrain(self):
        self.model.train()

    def setEval(self):
        self.model.eval()

    def _upload_to_gcs(self, local_path, gcs_path):
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

    def _delete_from_gcs(self, gcs_path):
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

    def load_ckpt(self, model_path, validate):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            
            # Handle both old (state_dict only) and new (complete checkpoint) formats
            if isinstance(saved, dict) and 'state_dict' in saved:
                # New format with configuration
                state_dict = saved['state_dict']
                model_config = saved.get('model_config', {})

                if validate:
                    # Validate critical configuration parameters
                    self._validate_model_config(model_config, model_path)
                
                # Restore dynamic parameters if available
                if model_config.get('use_arcface_head', False) and 'current_arcface_s' in model_config:
                    model_instance = self.model.module if self.config['ddp'] else self.model
                    if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                        current_s = model_config['current_arcface_s']
                        model_instance.head.s.data.fill_(current_s)
                        self.logger.info(f"Restored ArcFace s parameter to: {current_s}")
                
                self.logger.info(f"Loaded checkpoint from epoch {saved.get('epoch', 'unknown')} "
                               f"with AUC: {saved.get('auc', 'unknown'):.4f}")
            else:
                # Old format (state_dict only) - issue warning
                state_dict = saved
                self.logger.warning(f"Loading old checkpoint format from {model_path}. "
                                   "Configuration validation not possible.")
            
            # Load state dict with module prefix handling
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            
            # Pre-flight check: detect size mismatches before load_state_dict
            # This gives a clearer error message (e.g., wrong backbone checkpoint)
            model_state = self.model.state_dict()
            mismatches = []
            for key in new_state_dict:
                if key in model_state and new_state_dict[key].shape != model_state[key].shape:
                    mismatches.append(
                        f"  {key}: checkpoint={list(new_state_dict[key].shape)} "
                        f"vs model={list(model_state[key].shape)}"
                    )
            if mismatches:
                mismatch_str = "\n".join(mismatches)
                self.logger.error(
                    f"❌ CHECKPOINT SIZE MISMATCH loading {model_path}!\n"
                    f"This usually means the checkpoint was trained with a different backbone "
                    f"(e.g., ViT-L-14 vs ViT-B-16). Mismatched parameters:\n{mismatch_str}"
                )
                raise RuntimeError(
                    f"Checkpoint size mismatch: the checkpoint at {model_path} is incompatible "
                    f"with the current model architecture. {len(mismatches)} parameter(s) have "
                    f"different shapes. Check that gcs_base_checkpoint points to the correct "
                    f"backbone variant."
                )
            
            self.model.load_state_dict(new_state_dict, strict=False)
            
            # Validate model checksum if available (skip if curriculum learning might modify parameters)
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
                self.logger.info("⚠️  Skipping checksum validation - curriculum learning (train_arcface=True) may modify parameters")
            else:
                self.logger.warning("⚠️  No checksum available for validation (old checkpoint format)")
            
            self.logger.info(f'Model loaded from {model_path}')
        else:
            raise FileNotFoundError(f"=> no model found at '{model_path}'")

    def _validate_model_config(self, saved_config, checkpoint_path):
        """Validate that the saved model configuration matches current configuration."""
        if not saved_config:
            self.logger.warning("No model configuration found in checkpoint. Skipping validation.")
            return
        
        # Critical parameters that must match exactly
        critical_params = [
            'model_name', 'use_arcface_head', 
            'use_focal_loss', 'focal_loss_gamma', 'focal_loss_alpha', 'rank'
        ]
        
        # ArcFace parameters that can be overridden during curriculum learning
        arcface_params = ['arcface_s', 'arcface_m']
        
        # Check if ArcFace curriculum learning is enabled
        train_arcface = self.config.get('train_arcface', True)
        
        mismatches = []
        for param in critical_params:
            saved_val = saved_config.get(param)
            current_val = self.config.get(param)
            
            if saved_val != current_val:
                mismatches.append(f"{param}: saved={saved_val}, current={current_val}")
        
        # Only validate ArcFace parameters if curriculum learning is disabled
        if not train_arcface:
            for param in arcface_params:
                saved_val = saved_config.get(param)
                current_val = self.config.get(param)
                
                if saved_val != current_val:
                    mismatches.append(f"{param}: saved={saved_val}, current={current_val}")
        else:
            # Log that we're allowing ArcFace parameter override
            arcface_overrides = []
            for param in arcface_params:
                saved_val = saved_config.get(param)
                current_val = self.config.get(param)
                if saved_val != current_val:
                    arcface_overrides.append(f"{param}: saved={saved_val}, will_override_to={current_val}")
            
            if arcface_overrides:
                self.logger.info("ArcFace curriculum learning enabled - allowing parameter overrides:")
                for override in arcface_overrides:
                    self.logger.info(f"   - {override}")
        
        if mismatches:
            error_msg = (f"Critical configuration mismatch detected when loading {checkpoint_path}:\n" + 
                        "\n".join([f"  - {mm}" for mm in mismatches]))
            self.logger.error(error_msg)
            raise ValueError(f"Model configuration mismatch. {error_msg}")
        
        validation_mode = "strict" if not train_arcface else "curriculum learning enabled"
        self.logger.info(f"✅ Model configuration validation passed ({validation_mode}).")

    def compute_model_checksum(self):
        """Compute a checksum of the model's current state for validation."""
        import hashlib
        
        # Get model state dict
        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        
        # Create a deterministic string representation
        checksum_data = []
        for key in sorted(model_state.keys()):
            tensor = model_state[key]
            # Convert to numpy for consistent hashing across devices
            tensor_np = tensor.detach().cpu().numpy()
            checksum_data.append(f"{key}:{tensor_np.shape}:{tensor_np.sum():.10f}")
        
        # Add critical config parameters
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

    def save_ckpt(self, epoch, auc, eer, ckpt_prefix='ckpt', step=None):
        """
        Saves model checkpoint locally, uploads to GCS with a prefix, and cleans up.
        Returns the GCS path of the uploaded file.
        
        Args:
            epoch: The epoch number
            auc: The AUC score
            eer: The EER score
            ckpt_prefix: Prefix for the checkpoint name (e.g., 'first_best', 'top_n')
            step: Optional step count to use in naming instead of epoch (for top_n checkpoints)
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

        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        
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
            model_instance = self.model.module if self.config['ddp'] else self.model
            if hasattr(model_instance, 'head') and hasattr(model_instance.head, 's'):
                checkpoint['model_config']['current_arcface_s'] = float(model_instance.head.s)
        
        # Add model checksum for validation
        checkpoint['model_checksum'] = self.compute_model_checksum()
        
        torch.save(checkpoint, local_save_path)
        self.logger.info(f"💾 Saved checkpoint with checksum: {checkpoint['model_checksum']}")

        gcs_prefix = gcs_config['gcs_prefix']
        # <<< NEW: Ensure prefix ends with a slash for robust path joining
        if not gcs_prefix.endswith('/'):
            gcs_prefix += '/'
        run_id = self.wandb_run.id if self.wandb_run else "local_run"
        # <<< MODIFIED: Use os.path.join and then replace for cross-platform safety, though simple concatenation is fine for GCS.
        full_gcs_path = gcs_prefix + f"{run_id}/{ckpt_name}"

        upload_success = self._upload_to_gcs(local_save_path, full_gcs_path)

        try:
            os.remove(local_save_path)
        except OSError as e:
            self.logger.warning(f"Could not delete local temporary checkpoint: {e}")

        return full_gcs_path if upload_success else None

    def train_step(self, data_dict):
        with autocast():
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)

        # Fast-fail: detect NaN/Inf loss before wasting GPU hours
        if torch.isnan(losses['overall']) or torch.isinf(losses['overall']):
            raise RuntimeError(
                f"NaN/Inf loss detected in train_step "
                f"(global_step={getattr(self, 'global_step', '?')}). "
                "Aborting early. Check: learning rate, data pipeline, model init."
            )

        self.optimizer.zero_grad()
        self.scaler.scale(losses['overall']).backward()

        if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
            self.scaler.unscale_(self.optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler is not None:
            self.scheduler.step()

        return losses, predictions

    def _current_total_grad_norm(self) -> float:
        """Measure the current total grad norm with a single device sync."""
        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
        if torch.is_tensor(total_norm):
            return float(total_norm.item())
        return float(total_norm)

    def _next_batch_from_group(self, method_name, loaders, iters):
        """
        Gets the next batch from a specific method's dataloader.
        Manages a dictionary of iterators, creating or resetting them as needed.

        Args:
            method_name (str): The method to get a batch from.
            loaders (LazyDataLoaderManager): The manager holding all dataloader objects.
            iters (dict): The dictionary holding the active iterators for this group (real or fake).

        Returns:
            dict: The data dictionary for the next batch.
        """
        # 1. If we've never created an iterator for this method, create one.
        if method_name not in iters:
            loader = loaders[method_name]
            iters[method_name] = iter(loader)

        # 2. Try to get the next batch from the existing iterator.
        try:
            data_dict = next(iters[method_name])
        # 3. If the iterator is exhausted, create a new one and get the first batch.
        #    This allows us to loop over smaller datasets multiple times within one epoch.
        except StopIteration:
            loader = loaders[method_name]
            iters[method_name] = iter(loader)
            data_dict = next(iters[method_name])

        return data_dict

    def _train_per_method_step(self, real_loaders, fake_loaders, real_method_names, fake_method_names, real_weights,
                               fake_weights):
        """Helper to contain the logic for getting one 'per_method' batch."""
        chosen_fake_method = random.choices(fake_method_names, weights=fake_weights, k=1)[0]
        fake_data_dict = self._next_batch_from_group(chosen_fake_method, fake_loaders, self.fake_method_iters)

        chosen_real_method = random.choices(real_method_names, weights=real_weights, k=1)[0]
        real_data_dict = self._next_batch_from_group(chosen_real_method, real_loaders, self.real_method_iters)

        data_dict = {}
        for key in fake_data_dict.keys():
            f_val, r_val = fake_data_dict[key], real_data_dict[key]
            if torch.is_tensor(f_val) and torch.is_tensor(r_val):
                data_dict[key] = torch.cat((f_val, r_val), dim=0)
            elif isinstance(f_val, list):
                data_dict[key] = f_val + (r_val if r_val is not None else [])
            else:
                data_dict[key] = f_val

        batch_size = data_dict.get('label', torch.tensor([])).shape[0]
        if batch_size == 0: return None

        shuffle_indices = torch.randperm(batch_size)
        for key in data_dict.keys():
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key][shuffle_indices]
            elif isinstance(data_dict[key], list):
                data_dict[key] = [data_dict[key][i] for i in shuffle_indices.tolist()]

        return data_dict

    def train_epoch(
            self,
            train_loader,  # Now a single argument for the training data
            epoch,
            train_videos,  # Used for calculating epoch length
            val_method_loaders=None
    ):
        self.logger.info(f"===> Epoch[{epoch + 1}] start!")
        strategy = self.config.get('dataloader_params', {}).get('strategy', 'per_method')

        if strategy == 'property_balancing':
            dl_params = self.config.get('dataloader_params', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 64)
            # A logical batch is always composed of 32 unique videos.
            # Total logical frames = 32 videos * frames_per_video
            # Total GPU batches for one logical step = Total logical frames / frames_per_gpu_batch
            frames_per_video = dl_params.get('frames_per_video', 2)
            accumulation_steps = max(1, round((32.0 * frames_per_video) / gpu_batch_size))
            # `train_videos` is a list of frames (dicts) for this strategy
            total_items = len(train_videos)
            epoch_len = math.ceil(total_items / gpu_batch_size) if total_items > 0 else 0
        elif strategy == 'frame_level':
            gpu_batch_size = self.config.get('dataloader_params', {}).get('frames_per_batch')
            total_frames = sum(len(v.frame_paths) for v in train_videos)
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = 1  # No accumulation for this strategy
        elif strategy == 'deeplive':
            # DeepLive uses IterableDataset with paired real/fake frames
            # train_videos is a list of DeepLiveSample objects
            dl_params = self.config.get('dataloader_params', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            # Each sample produces frames_per_sample * 2 (real + fake) frames
            total_frames = len(train_videos) * frames_per_sample * 2
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = 1  # No accumulation for this strategy
        elif strategy == 'df40_paired':
            # DF40 Paired uses IterableDataset with paired real/fake frames (similar to deeplive)
            # train_videos is a list of DF40PairedSample objects
            dl_params = self.config.get('dataloader_params', {})
            df40_config = self.config.get('df40_paired', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            
            # Check if identity-balanced sampling is enabled
            identity_balanced = df40_config.get('identity_balanced_sampling', True)
            
            if identity_balanced:
                # With identity-balanced sampling, we sample ONE method per identity per epoch
                # So epoch length is based on unique identities, not total pairs
                unique_identities = set(s.target_identity for s in train_videos)
                num_samples_per_epoch = len(unique_identities)
                self.logger.info(f"DF40 identity-balanced sampling: {num_samples_per_epoch} unique identities (from {len(train_videos)} pairs)")
            else:
                # Legacy: iterate over all pairs
                num_samples_per_epoch = len(train_videos)
            
            # Each sample produces frames_per_sample * 2 (real + fake) frames
            total_frames = num_samples_per_epoch * frames_per_sample * 2
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = 1  # No accumulation for this strategy
        elif strategy == 'combined_paired':
            # Combined Paired uses IterableDataset with DF40, DeepLive, and/or VisoMaster samples
            # train_videos is a list of UnifiedPairedSample objects
            dl_params = self.config.get('dataloader_params', {})
            combined_config = self.config.get('combined_paired', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            
            # Check if identity-balanced sampling is enabled
            identity_balanced = combined_config.get('identity_balanced_sampling', True)
            
            if identity_balanced:
                # With identity-balanced sampling, we sample ONE method per identity per epoch
                unique_identities = set(s.identity for s in train_videos)
                num_samples_per_epoch = len(unique_identities)
                self.logger.info(f"Combined identity-balanced sampling: {num_samples_per_epoch} unique identities (from {len(train_videos)} samples)")
            else:
                num_samples_per_epoch = len(train_videos)
            
            # Each paired sample produces frames_per_sample * 2 (real + fake) frames.
            # Unpaired real samples produce frames_per_sample * 1 (real only).
            n_paired = sum(1 for s in train_videos if not getattr(s, 'is_unpaired_real', False))
            n_unpaired = sum(1 for s in train_videos if getattr(s, 'is_unpaired_real', False))
            if identity_balanced:
                paired_ids = set(
                    s.identity for s in train_videos if not getattr(s, 'is_unpaired_real', False)
                )
                unpaired_ids = set(
                    s.identity for s in train_videos if getattr(s, 'is_unpaired_real', False)
                )
                n_paired = len(paired_ids)
                n_unpaired = len(unpaired_ids)
            total_frames = (n_paired * frames_per_sample * 2) + (n_unpaired * frames_per_sample)
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            if n_unpaired > 0:
                self.logger.info(
                    f"Combined samples: {n_paired} paired + {n_unpaired} unpaired real -> {total_frames} total frames"
                )
            accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        elif strategy == 'visomaster':
            # Standalone VisoMaster uses IterableDataset (same pattern as combined_paired)
            dl_params = self.config.get('dataloader_params', {})
            viso_config = self.config.get('visomaster', {})
            gpu_batch_size = dl_params.get('frames_per_batch', 32)
            frames_per_sample = dl_params.get('frames_per_video', 8)
            identity_balanced = viso_config.get('identity_balanced_sampling', True)
            
            if identity_balanced:
                unique_identities = set(f"visomaster_{s.identity}" for s in train_videos)
                num_samples_per_epoch = len(unique_identities)
                self.logger.info(f"VisoMaster identity-balanced sampling: {num_samples_per_epoch} unique identities (from {len(train_videos)} samples)")
            else:
                num_samples_per_epoch = len(train_videos)
            
            total_frames = num_samples_per_epoch * frames_per_sample * 2
            epoch_len = math.ceil(total_frames / gpu_batch_size) if total_frames > 0 else 0
            accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        else:  # Handles 'per_method' and 'video_level'
            effective_batch_size = self.config.get('dataloader_params', {}).get('videos_per_batch')
            total_train_videos = len(train_videos)
            epoch_len = math.ceil(total_train_videos / effective_batch_size) if total_train_videos > 0 else 0
            accumulation_steps = 1  # No accumulation for these strategies

        self.logger.info(f"Training strategy: '{strategy}', Epoch length: {epoch_len} steps")
        if accumulation_steps > 1:
            self.logger.info(f"Using gradient accumulation with {accumulation_steps} steps.")

        # --- PRECISE EVALUATION SCHEDULING ---
        evaluation_frequency = self.config.get('data_params', {}).get('evaluation_frequency', 1)
        if evaluation_frequency <= 0: evaluation_frequency = 1
        eval_steps = set()
        # Only calculate epoch-based steps if step-based evaluation is not active
        if self.evaluate_every_steps is None or self.evaluate_every_steps <= 0:
            if epoch_len > 0:
                interval = max(1, epoch_len // evaluation_frequency)
                for i in range(1, evaluation_frequency): eval_steps.add(i * interval)
                eval_steps.add(epoch_len)
            self.logger.info(f"Scheduled evaluation at epoch steps: {sorted(list(eval_steps))}")
        else:
            # eval_steps remains empty; we will check step_cnt directly in the loop
            pass

        step_cnt = epoch * epoch_len
        epoch_start_time = time.time()

        # --- Conditional Training Loop based on strategy ---
        if strategy == 'per_method':
            # This logic remains self-contained as it doesn't use gradient accumulation
            real_source_names = self.config['methods']['use_real_sources']
            all_method_names = train_loader.keys()
            real_method_names = [m for m in all_method_names if m in real_source_names]
            fake_method_names = [m for m in all_method_names if m not in real_source_names]
            real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
            for v in train_videos:
                (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1
            total_real_videos, total_fake_videos = sum(real_video_counts.values()), sum(fake_video_counts.values())
            real_weights = [real_video_counts[m] / total_real_videos for m in
                            real_method_names] if total_real_videos > 0 else []
            fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                            fake_method_names] if total_fake_videos > 0 else []

            pbar = tqdm(range(epoch_len), desc=f"EPOCH (per_method): {epoch + 1}/{self.config['nEpochs']}")
            for iteration in pbar:
                data_dict = self._train_per_method_step(train_loader, train_loader, real_method_names,
                                                        fake_method_names, real_weights, fake_weights)
                if data_dict is None:
                    continue
                # This strategy uses the original, single-step logic.
                self._run_train_step(data_dict, step_cnt, epoch, epoch_len, epoch_start_time)

                # --- Evaluation and Stop Condition Check ---
                step_cnt += 1
                should_evaluate_now = False

                if self.evaluate_every_steps and self.evaluate_every_steps > 0:
                    if step_cnt > 0 and step_cnt % self.evaluate_every_steps == 0:
                        should_evaluate_now = True
                else:  # Fallback to epoch-based frequency
                    if (iteration + 1) in eval_steps:
                        should_evaluate_now = True

                if should_evaluate_now:
                    self._run_validation(epoch, iteration, step_cnt)

                    # Check for max steps termination
                if self.max_train_steps and step_cnt >= self.max_train_steps:
                    self.logger.critical(f"MAX STEPS REACHED: {step_cnt}/{self.max_train_steps}. Stopping training.")
                    if not should_evaluate_now and self.evaluate_every_steps and self.evaluate_every_steps > 0:
                        next_eval_step = math.ceil(step_cnt / self.evaluate_every_steps) * self.evaluate_every_steps
                        steps_until_next_eval = next_eval_step - step_cnt
                        threshold = self.evaluate_every_steps / 1.5
                        if steps_until_next_eval <= threshold:
                            self.logger.info(
                                f"Running one final evaluation before stopping (steps until next eval {steps_until_next_eval} <= threshold {threshold:.1f}).")
                            self._run_validation(epoch, iteration, step_cnt)
                    self.early_stop_triggered = True

                if self.early_stop_triggered:
                    break

        elif strategy in ['video_level', 'frame_level', 'property_balancing', 'deeplive', 'df40_paired', 'combined_paired']:
            # Set epoch on the dataset for identity-balanced sampling (if supported)
            # This allows the dataset to vary random method selection per epoch
            if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'set_epoch'):
                train_loader.dataset.set_epoch(epoch)
                self.logger.info(f"Set epoch {epoch} on train_loader.dataset for identity-balanced sampling")
            
            pbar = tqdm(train_loader, desc=f"EPOCH ({strategy}): {epoch + 1}/{self.config['nEpochs']}", total=epoch_len)
            self.optimizer.zero_grad()  # Zero gradients at the start of the epoch
            
            # Flag for first-batch gradient diagnostics
            _first_backward_logged = False

            for i, data_dict in enumerate(pbar):
                if i >= epoch_len: break

                self._update_arcface_s(step_cnt)
                self._update_lambda_reg(step_cnt)
                self._update_quality_domain_lambda(step_cnt)

                is_final_accumulation_step = (i + 1) % accumulation_steps == 0
                is_ddp = type(self.model) is DDP
                # Use DDP's no_sync context manager to avoid redundant gradient all-reduce calls.
                # This is a significant speed-up for DDP with gradient accumulation.
                context = self.model.no_sync() if is_ddp and not is_final_accumulation_step and accumulation_steps > 1 else contextlib.nullcontext()

                with context:
                    self.setTrain()
                    for key in data_dict.keys():
                        if isinstance(data_dict[key], torch.Tensor): data_dict[key] = data_dict[key].to(
                            self.model.device)
                    # --- FORWARD PASS ---
                    with autocast():
                        predictions = self.model(data_dict)
                        loss_fn_owner = self.model.module if type(self.model) is DDP else self.model

                        if self.use_group_dro:
                            # PREREQUISITE #1: Your model's get_losses must support reduction='none'
                            per_sample_losses_dict = loss_fn_owner.get_losses(
                                data_dict, predictions, reduction='none'
                            )
                            per_sample_loss = per_sample_losses_dict['overall']

                            # PREREQUISITE #2: The data_dict must contain 'method_id'
                            # Uses GroupDROMixin.calculate_group_dro_loss()
                            losses = self.calculate_group_dro_loss(data_dict, per_sample_loss)
                        else:
                            # Original behavior
                            losses = loss_fn_owner.get_losses(data_dict, predictions)

                    # --- Stability regularization loss ---
                    stability_loss = self.compute_stability_loss(
                        self.model, data_dict, predictions
                    )
                    losses['overall'] = losses['overall'] + stability_loss
                    losses['stability'] = stability_loss.detach()

                    # Store unscaled loss for accurate logging
                    unscaled_loss = losses['overall'].clone().detach()

                    # Fast-fail: detect NaN/Inf loss before wasting GPU hours
                    if torch.isnan(unscaled_loss) or torch.isinf(unscaled_loss):
                        raise RuntimeError(
                            f"NaN/Inf loss detected at step {step_cnt} "
                            f"(epoch {epoch}). "
                            "Aborting early. Check: learning rate, data pipeline, model init."
                        )

                    # Scale loss for accumulation
                    if accumulation_steps > 1:
                        losses['overall'] = losses['overall'] / accumulation_steps
                    # --- BACKWARD PASS ---
                    self.scaler.scale(losses['overall']).backward()

                # --- OPTIMIZER STEP (conditional) ---
                if is_final_accumulation_step or accumulation_steps == 1:
                    # Unscale gradients first (required for clipping and gradient logging)
                    self.scaler.unscale_(self.optimizer)
                    
                    # === FIRST STEP DIAGNOSTIC: Show gradient health on first optimizer step ===
                    if not _first_backward_logged and epoch == self.config.get('start_epoch', 0):
                        _first_backward_logged = True
                        model_ref = self.model.module if is_ddp else self.model
                        
                        total_grad_norm = 0.0
                        num_params_with_grad = 0
                        params_without_grad = []
                        params_with_tiny_grad = []
                        params_with_big_grad = []
                        
                        for name, param in model_ref.named_parameters():
                            if param.requires_grad:
                                if param.grad is not None:
                                    grad_norm = param.grad.data.norm(2).item()
                                    total_grad_norm += grad_norm ** 2
                                    num_params_with_grad += 1
                                    if grad_norm < 1e-8:
                                        params_with_tiny_grad.append((name, grad_norm))
                                    elif grad_norm > 10:
                                        params_with_big_grad.append((name, grad_norm))
                                else:
                                    params_without_grad.append(name)
                        
                        total_grad_norm = total_grad_norm ** 0.5
                        
                        self.logger.info("=" * 70)
                        self.logger.info("🔍 FIRST OPTIMIZER STEP GRADIENT DIAGNOSTIC")
                        self.logger.info("=" * 70)
                        self.logger.info(f"   Total gradient norm: {total_grad_norm:.6e}")
                        self.logger.info(f"   Parameters with gradients: {num_params_with_grad}")
                        self.logger.info(f"   Parameters WITHOUT gradients (requires_grad=True but grad=None):")
                        for pname in params_without_grad[:10]:
                            self.logger.info(f"      ❌ {pname}")
                        if len(params_without_grad) > 10:
                            self.logger.info(f"      ... and {len(params_without_grad) - 10} more")
                        
                        if params_with_tiny_grad:
                            self.logger.info(f"   Parameters with TINY gradients (<1e-8):")
                            for pname, gnorm in params_with_tiny_grad[:10]:
                                self.logger.info(f"      ⚠️ {pname}: {gnorm:.2e}")
                        
                        if params_with_big_grad:
                            self.logger.info(f"   Parameters with LARGE gradients (>10):")
                            for pname, gnorm in params_with_big_grad[:10]:
                                self.logger.info(f"      🔥 {pname}: {gnorm:.2e}")
                        
                        # Also log loss and prediction stats
                        probs = predictions['prob'].detach()
                        labels = data_dict['label']
                        self.logger.info(f"   First batch stats:")
                        self.logger.info(f"      Loss: {unscaled_loss.item():.4f}")
                        self.logger.info(f"      Prob mean: {probs.mean().item():.4f}, std: {probs.std().item():.4f}")
                        self.logger.info(f"      Label dist: {(labels==0).sum().item()} real, {(labels==1).sum().item()} fake")
                        self.logger.info("=" * 70)
                    
                    if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                    
                    # Compute gradient health metrics before they're cleared
                    model_ref = self.model.module if is_ddp else self.model
                    _grad_norm = self._current_total_grad_norm()
                    _num_params_with_grad = sum(1 for p in model_ref.parameters() if p.grad is not None)
                    # Store for later logging
                    self._last_grad_norm = _grad_norm
                    self._last_params_with_grad = _num_params_with_grad

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.optimizer.zero_grad()

                # --- LOGGING (every GPU batch, using unscaled loss) ---
                if self.wandb_run and self.config['local_rank'] == 0:
                    log_dict = {"train/step": step_cnt, "epoch": epoch + 1}
                    log_dict[f'train/loss/overall'] = unscaled_loss.item()
                    for name, value in losses.items():
                        if name == 'overall':
                            continue  # Already logged the unscaled version

                        # If the value is a scalar tensor, log it with .item()
                        if value.numel() == 1:
                            log_dict[f'train/loss/{name}'] = value.item()
                        # If it's a multi-element tensor (like group_weights), log it as a histogram
                        else:
                            # wandb.Histogram is the perfect tool for this
                            log_dict[f'train/diagnostic/{name}'] = wandb.Histogram(value.detach().cpu())

                    batch_metrics = self.model.module.get_train_metrics(data_dict,
                                                                        predictions) if is_ddp else self.model.get_train_metrics(
                        data_dict, predictions)
                    for name, value in batch_metrics.items(): log_dict[f'train/metric/{name}'] = value
                    
                    # === LEARNING VISIBILITY: Key metrics for tracking actual learning ===
                    model_ref = self.model.module if is_ddp else self.model
                    
                    # 1. ArcFace scale parameter (s) - tracks annealing progress
                    if hasattr(model_ref, 'head') and hasattr(model_ref.head, 's'):
                        current_s = model_ref.head.s.item() if hasattr(model_ref.head.s, 'item') else float(model_ref.head.s)
                        log_dict['train/arcface/s'] = current_s
                    
                    # 2. Learning rate tracking
                    if self.optimizer and len(self.optimizer.param_groups) > 0:
                        log_dict['train/lr'] = self.optimizer.param_groups[0]['lr']
                    
                    # 3. Prediction confidence (how decisive is the model?)
                    probs = predictions['prob'].detach()
                    log_dict['train/confidence/mean'] = probs.mean().item()
                    log_dict['train/confidence/std'] = probs.std().item()
                    # Fraction of confident predictions (>0.7 or <0.3)
                    confident_mask = (probs > 0.7) | (probs < 0.3)
                    log_dict['train/confidence/fraction_confident'] = confident_mask.float().mean().item()
                    
                    # 4. Class balance in predictions (should be ~0.5 for balanced data)
                    pred_fake_ratio = (probs > 0.5).float().mean().item()
                    log_dict['train/pred_balance/fake_ratio'] = pred_fake_ratio
                    
                    # 5. Raw logit statistics (key for ArcFace debugging)
                    if 'raw_logits' in predictions:
                        raw_logits = predictions['raw_logits'].detach()
                        logit_diff = raw_logits[:, 1] - raw_logits[:, 0]  # fake - real
                        log_dict['train/logits/diff_mean'] = logit_diff.mean().item()
                        log_dict['train/logits/diff_std'] = logit_diff.std().item()
                        
                        # 5b. Per-class logit statistics (Added Jan 10, 2026)
                        labels = data_dict['label']
                        if raw_logits.shape[0] > labels.shape[0]:
                            B = labels.shape[0]
                            T = raw_logits.shape[0] // B
                            labels = labels.repeat_interleave(T)
                        
                        real_mask = (labels == 0)
                        fake_mask = (labels == 1)
                        if real_mask.any():
                            log_dict['train/logits/real_mean'] = logit_diff[real_mask].mean().item()
                        if fake_mask.any():
                            log_dict['train/logits/fake_mean'] = logit_diff[fake_mask].mean().item()
                    
                    # 6. Collapse early warning system (Added Jan 10, 2026)
                    collapse_metrics = self._check_collapse_warning(predictions, data_dict, step_cnt)
                    log_dict.update(collapse_metrics)
                    
                    # 7. ArcFace detailed diagnostics (Added Jan 10, 2026)
                    arcface_metrics = self._collect_arcface_diagnostics(step_cnt)
                    log_dict.update(arcface_metrics)

                    log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
                    if (i % log_progress_steps == 0) or (i == epoch_len - 1):
                        time_elapsed = time.time() - epoch_start_time
                        steps_per_sec = (i + 1) / time_elapsed if time_elapsed > 0 else 0
                        log_dict['train/steps_per_sec'] = steps_per_sec
                        log_dict['train/probabilities'] = wandb.Histogram(predictions['prob'].detach().cpu().numpy())
                        if epoch_len > 0:
                            progress_pct = ((i + 1) / epoch_len) * 100
                            log_dict['train/epoch_progress'] = progress_pct
                            if steps_per_sec > 0:
                                time_remaining_sec = (epoch_len - (i + 1)) / steps_per_sec
                                log_dict['train/epoch_eta_min'] = time_remaining_sec / 60
                        
                        # 6. Gradient health - use cached values from before optimizer.zero_grad()
                        if hasattr(self, '_last_grad_norm') and hasattr(self, '_last_params_with_grad'):
                            if self._last_params_with_grad > 0:
                                log_dict['train/grad_norm'] = self._last_grad_norm
                                log_dict['train/params_with_grad'] = self._last_params_with_grad
                            else:
                                # Log zero to indicate NO gradients flowing - critical warning sign!
                                log_dict['train/grad_norm'] = 0.0
                                log_dict['train/params_with_grad'] = 0
                        
                        # 7. SVD Residual diagnostics (added Jan 4, 2026 - Task D3)
                        # Track S_residual values to diagnose params_with_grad drops
                        svd_stats = self._collect_svd_residual_stats()
                        if svd_stats:
                            log_dict.update(svd_stats)
                        
                        self.wandb_run.log(log_dict)
                        self.logger.info(
                            f"Epoch {epoch + 1}/{self.config['nEpochs']} | Step {i + 1}/{epoch_len} ({progress_pct:.1f}%) | Loss: {unscaled_loss.item():.4f} | Speed: {steps_per_sec:.2f} it/s")
                    else:
                        self.wandb_run.log({"train/loss/overall": unscaled_loss.item(), "train/step": step_cnt})

                # --- Evaluation and Stop Condition Check ---
                step_cnt += 1
                should_evaluate_now = False

                if self.evaluate_every_steps and self.evaluate_every_steps > 0:
                    if step_cnt > 0 and step_cnt % self.evaluate_every_steps == 0:
                        should_evaluate_now = True
                else:  # Fallback to epoch-based frequency
                    if (i + 1) in eval_steps:
                        should_evaluate_now = True

                if should_evaluate_now:
                    self._run_validation(epoch, i, step_cnt)

                    # Check for max steps termination
                if self.max_train_steps and step_cnt >= self.max_train_steps:
                    self.logger.critical(f"MAX STEPS REACHED: {step_cnt}/{self.max_train_steps}. Stopping training.")
                    if not should_evaluate_now and self.evaluate_every_steps and self.evaluate_every_steps > 0:
                        next_eval_step = math.ceil(step_cnt / self.evaluate_every_steps) * self.evaluate_every_steps
                        steps_until_next_eval = next_eval_step - step_cnt
                        threshold = self.evaluate_every_steps / 1.5
                        if steps_until_next_eval <= threshold:
                            self.logger.info(
                                f"Running one final evaluation before stopping (steps until next eval {steps_until_next_eval} <= threshold {threshold:.1f}).")
                            self._run_validation(epoch, i, step_cnt)
                    self.early_stop_triggered = True

                if self.early_stop_triggered:
                    break

            # Perform a final optimizer step for any remaining gradients at the end of the epoch
            try:
                # 'i' is the index of the last item processed by the loop.
                num_iterations_run = i + 1
            except NameError:
                num_iterations_run = 0

            # Perform a final optimizer step for any remaining gradients at the end of the epoch.
            if num_iterations_run > 0 and num_iterations_run % accumulation_steps != 0 and accumulation_steps > 1:
                self.logger.info("Performing final optimizer step for dangling gradients at end of epoch.")

                try:
                    # --- ATTEMPT THE OPTIMIZER STEP ---
                    if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    # On success, zero the gradients for the next epoch.
                    self.optimizer.zero_grad()

                except AssertionError as e:
                    # --- FAILSAFE: CATCH THE ERROR ---
                    self.logger.error(
                        "!!! FAILSAFE TRIGGERED: Caught AssertionError during final optimizer step. "
                        "This indicates a logic error in gradient accumulation handling. "
                        f"Error: {e}. "
                        "Skipping this optimizer step and discarding stale gradients to prevent a crash."
                    )
                    # Manually discard the stale gradients that caused the error.
                    self.optimizer.zero_grad()
                    if hasattr(self.scaler, '_per_optimizer_states'):
                        self.scaler._per_optimizer_states.clear()
        else:
            raise ValueError(f"Unsupported training strategy: {strategy}")

        self.logger.info(f"===> Epoch[{epoch + 1}] finished in {(time.time() - epoch_start_time) / 60:.2f} minutes.")

    def _run_train_step(self, data_dict, step_cnt, epoch, epoch_len, epoch_start_time):  # Add new args
        """Helper to avoid code duplication in the training loop."""
        self._update_arcface_s(step_cnt)
        self._update_lambda_reg(step_cnt)
        self._update_quality_domain_lambda(step_cnt)
        self.setTrain()
        for key in data_dict.keys():
            if isinstance(data_dict[key], torch.Tensor): data_dict[key] = data_dict[key].to(self.model.device)

        losses, predictions = self.train_step(data_dict)

        if self.wandb_run and self.config['local_rank'] == 0:
            log_dict = {"train/step": step_cnt, "epoch": epoch + 1}
            for name, value in losses.items():
                log_dict[f'train/loss/{name}'] = value.item()

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                log_dict[f'train/metric/{name}'] = value

            # === ENHANCED DIAGNOSTICS (Added Jan 10, 2026) ===
            # 1. Collapse early warning
            collapse_metrics = self._check_collapse_warning(predictions, data_dict, step_cnt)
            log_dict.update(collapse_metrics)
            
            # 2. ArcFace diagnostics
            arcface_metrics = self._collect_arcface_diagnostics(step_cnt)
            log_dict.update(arcface_metrics)
            
            # 3. Per-class logit statistics
            if 'raw_logits' in predictions:
                raw_logits = predictions['raw_logits'].detach()
                logit_diff = raw_logits[:, 1] - raw_logits[:, 0]
                log_dict['train/logits/diff_mean'] = logit_diff.mean().item()
                log_dict['train/logits/diff_std'] = logit_diff.std().item()

            # --- DETAILED PROGRESS LOGGING (REPLACES PBAR) ---
            log_progress_steps = self.config.get('wandb', {}).get('log_progress_steps', 50)
            current_iter_in_epoch = step_cnt - (epoch * epoch_len)

            # Log every N steps or on the very last step of the epoch
            if (current_iter_in_epoch % log_progress_steps == 0) or (current_iter_in_epoch == epoch_len - 1):
                time_elapsed = time.time() - epoch_start_time
                steps_per_sec = (current_iter_in_epoch + 1) / time_elapsed if time_elapsed > 0 else 0

                log_dict['train/steps_per_sec'] = steps_per_sec
                log_dict['train/probabilities'] = wandb.Histogram(predictions['prob'].detach().cpu().numpy())

                progress_pct = 0.0
                if epoch_len > 0:
                    progress_pct = ((current_iter_in_epoch + 1) / epoch_len) * 100
                    log_dict['train/epoch_progress'] = progress_pct

                    if steps_per_sec > 0:
                        time_remaining_sec = (epoch_len - (current_iter_in_epoch + 1)) / steps_per_sec
                        log_dict['train/epoch_eta_min'] = time_remaining_sec / 60

                # Log to WandB
                self.wandb_run.log(log_dict)

                # Log a simple text line to the logger for basic feedback
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config['nEpochs']} | "
                    f"Step {current_iter_in_epoch + 1}/{epoch_len} "
                    f"({progress_pct:.1f}%) | "
                    f"Loss: {losses['overall'].item():.4f} | "
                    f"Speed: {steps_per_sec:.2f} it/s"
                )
            else:
                # For other steps, just log the minimal required info to not miss loss spikes
                self.wandb_run.log({"train/loss/overall": losses['overall'].item(), "train/step": step_cnt})

    def _run_validation(self, epoch, iteration, step_cnt):
        """
        Helper to run the full validation suite and check the lesson gate conditions.
        """
        if self.config['local_rank'] != 0:
            return

        self.logger.info(f"\n===> Evaluation at epoch {epoch + 1}, step {step_cnt}")

        all_val_metrics = {}

        # 1. Run In-Distribution Validation
        if self.val_in_dist_loader:
            in_dist_metrics = self.test_epoch(
                epoch=epoch,
                step_cnt=step_cnt,
                validation_loader=self.val_in_dist_loader,
                log_prefix="val_in_dist",
                is_primary_metric=False
            )
            all_val_metrics['val_in_dist'] = in_dist_metrics

        # 2. Run Holdout Validation
        if self.val_holdout_loader:
            holdout_metrics = self.test_epoch(
                epoch=epoch,
                step_cnt=step_cnt,
                validation_loader=self.val_holdout_loader,
                log_prefix="val_holdout",
                is_primary_metric=True  # This is the primary metric set for checkpointing
            )
            all_val_metrics['val_holdout'] = holdout_metrics

        # 3. Extract in-distribution EER threshold — this is the SINGLE
        #    operating-point threshold that all other splits are measured against.
        indist_threshold = None
        indist_m = all_val_metrics.get('val_in_dist')
        if indist_m and 'overall' in indist_m:
            indist_threshold = indist_m['overall'].get('eer_threshold')
            if indist_threshold is not None and indist_threshold > 0:
                self.logger.info(
                    f"In-dist EER threshold: {indist_threshold:.4f} — "
                    "will apply to holdout & OOD for unified evaluation"
                )

        # 4. Compute at-indist-threshold metrics for val_holdout
        holdout_m = all_val_metrics.get('val_holdout')
        if (
            indist_threshold is not None
            and indist_threshold > 0
            and holdout_m
            and 'all_preds' in holdout_m
            and len(holdout_m['all_preds']) > 0
        ):
            at_indist = metrics_at_threshold(
                holdout_m['all_preds'], holdout_m['all_labels'], indist_threshold
            )
            if at_indist and self.wandb_run:
                log_dict = {'train/step': step_cnt}
                for k, v in at_indist.items():
                    log_dict[f'val_holdout/at_indist/{k}'] = v
                log_dict['val_holdout/at_indist/threshold'] = indist_threshold
                self.wandb_run.log(log_dict)
                self.logger.info(
                    f"val_holdout @ indist threshold {indist_threshold:.4f}: "
                    f"acc={at_indist['acc']:.4f}  f1={at_indist['f1']:.4f}  "
                    f"fpr={at_indist['fpr']:.4f}  fnr={at_indist['fnr']:.4f}"
                )

        # 5. Compute unified threshold across all validation pools (legacy metric)
        unified_preds_parts = []
        unified_labels_parts = []
        for key in ('val_in_dist', 'val_holdout'):
            m = all_val_metrics.get(key)
            if m and 'all_preds' in m and len(m['all_preds']) > 0:
                unified_preds_parts.append(m['all_preds'])
                unified_labels_parts.append(m['all_labels'])
        if unified_preds_parts:
            combined_preds = np.concatenate(unified_preds_parts)
            combined_labels = np.concatenate(unified_labels_parts)
            if len(np.unique(combined_labels)) > 1:
                unified_metrics = get_test_metrics(combined_preds, combined_labels)
                if unified_metrics and self.wandb_run:
                    self.wandb_run.log({
                        'unified/eer_threshold': unified_metrics.get('eer_threshold', 0),
                        'unified/eer': unified_metrics.get('eer', 0),
                        'unified/f1': unified_metrics.get('f1_at_eer', 0),
                        'unified/auc': unified_metrics.get('auc', 0),
                        'train/step': step_cnt,
                    })
                    self.logger.info(
                        f"Unified threshold: {unified_metrics.get('eer_threshold', 0):.4f}, "
                        f"EER: {unified_metrics.get('eer', 0):.4f}, "
                        f"F1@EER: {unified_metrics.get('f1_at_eer', 0):.4f}"
                    )

        # Free raw predictions from returned metrics to save memory
        for m in all_val_metrics.values():
            m.pop('all_preds', None)
            m.pop('all_labels', None)

        # 6. Check Lesson Gate
        if self.gate_enabled:
            self._check_lesson_gate(all_val_metrics)

        # 7. Run OOD monitoring (does not affect gating)
        ood_auc = self._run_ood_monitoring(epoch, step_cnt, indist_threshold=indist_threshold)

        # 8. OOD-composite checkpointing (R12+)
        # If enabled, save a separate checkpoint ranked by hmean(holdout_auc, ood_auc).
        # This checkpoint list is independent of the holdout-only top-N list.
        if (
            self.ood_composite_enabled
            and ood_auc is not None
            and ood_auc > 0
        ):
            holdout_m = all_val_metrics.get('val_holdout')
            holdout_auc = holdout_m['overall'].get('auc') if holdout_m and 'overall' in holdout_m else None
            holdout_eer = holdout_m['overall'].get('eer') if holdout_m and 'overall' in holdout_m else None
            if holdout_auc and holdout_auc > 0:
                # Harmonic mean — penalizes large divergence between holdout and OOD
                composite = 2.0 * holdout_auc * ood_auc / (holdout_auc + ood_auc)

                # Log to W&B
                if self.wandb_run:
                    self.wandb_run.log({
                        'val_primary/ood_composite': composite,
                        'val_primary/ood_auc_for_composite': ood_auc,
                        'val_primary/holdout_auc_for_composite': holdout_auc,
                        'train/step': step_cnt,
                    })

                self.logger.info(
                    f"OOD composite: {composite:.4f} "
                    f"(holdout_auc={holdout_auc:.4f}, ood_auc={ood_auc:.4f})"
                )

                is_composite_improvement = composite > self.best_ood_composite
                if is_composite_improvement:
                    self.logger.info(
                        f"🎯 OOD COMPOSITE IMPROVED! {composite:.4f} "
                        f"(prev best: {self.best_ood_composite:.4f})"
                    )
                    self.best_ood_composite = composite
                    self.best_ood_composite_step = step_cnt

                    if self.wandb_run:
                        self.wandb_run.summary['best_ood_composite/metric'] = composite
                        self.wandb_run.summary['best_ood_composite/holdout_auc'] = holdout_auc
                        self.wandb_run.summary['best_ood_composite/ood_auc'] = ood_auc
                        self.wandb_run.summary['best_ood_composite/step'] = step_cnt

                    if self.config.get('save_ckpt', True):
                        is_top = (
                            len(self.ood_composite_top_n) < self.ood_composite_top_n_size
                            or composite > self.ood_composite_top_n[-1]['metric']
                        )
                        if is_top:
                            gcs_path = self.save_ckpt(
                                epoch=epoch + 1,
                                auc=holdout_auc,
                                eer=holdout_eer,
                                ckpt_prefix='ood_composite',
                                step=step_cnt,
                            )
                            if gcs_path:
                                self.ood_composite_top_n.append({
                                    'metric': composite,
                                    'holdout_auc': holdout_auc,
                                    'ood_auc': ood_auc,
                                    'epoch': epoch + 1,
                                    'step': step_cnt,
                                    'gcs_path': gcs_path,
                                })
                                self.ood_composite_top_n.sort(
                                    key=lambda x: x['metric'], reverse=True
                                )
                                if len(self.ood_composite_top_n) > self.ood_composite_top_n_size:
                                    worst = self.ood_composite_top_n.pop()
                                    self._delete_from_gcs(worst['gcs_path'])

                                if self.wandb_run:
                                    self.wandb_run.summary['best_ood_composite/gcs_path'] = (
                                        self.ood_composite_top_n[0]['gcs_path']
                                    )

    @torch.no_grad()
    def test_epoch(self, epoch, step_cnt, validation_loader, log_prefix: str, is_primary_metric: bool,
                   generate_detailed_reports: bool = False, run_name: str = None,
                   output_gcs_folder: str = None, output_filename_prefix: str = None):
        """
        Performs a full evaluation on a given validation dataloader. This function is now
        general-purpose and can be used for any validation set.

        Args:
            epoch (int): The current epoch number.
            step_cnt (int): The current global training step.
            validation_loader (LazyDataLoaderManager): The dataloader to evaluate.
            log_prefix (str): The prefix for WandB logs (e.g., "val_in_dist", "val_holdout").
            is_primary_metric (bool): If True, the results from this run will be used for
                                      checkpointing and early stopping decisions.
            generate_detailed_reports (bool): If True, collects detailed per-frame and
                                              per-video data and uploads CSV/TXT reports to GCS.
            run_name (str): Optional custom name with generate_detailed_reports, used in reports to identify the run.
            output_gcs_folder (str): Optional GCS folder to write reports to (e.g., 'gs://bucket/path/folder').
                                     If specified, appends to existing folder instead of creating new timestamped one.
            output_filename_prefix (str): Optional prefix for report filenames (e.g., 'target_source_').
        """
        self.setEval()

        # Calculate total videos from the provided loader, not a stored class attribute.
        total_videos = sum(len(v) for v in validation_loader.videos_by_method.values())
        if total_videos == 0:
            self.logger.warning(f"No validation videos found for loader with prefix '{log_prefix}'. Skipping.")
            return {}  # Return an empty dict to avoid errors

        self.logger.info(f"--- Starting validation for '{log_prefix}' set ({total_videos} videos)...")
        val_start_time = time.time()
        videos_processed = 0

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        all_preds, all_labels = [], []
        all_losses = []
        method_id_to_name = getattr(validation_loader, "method_id_to_name", {}) or {}

        # --- NEW: Initialize lists for detailed reporting if flag is enabled ---
        if generate_detailed_reports:
            frame_report_data = []
            video_report_data = []

        # --- SANITY CHECK: Initialize data collection for model consistency verification ---
        sanity_check_data = []  # Will store first 2 frames per method for verification

        # Iterate through methods from the provided loader.
        for method in validation_loader.keys():
            loader = validation_loader[method]
            num_videos_in_method = len(validation_loader.videos_by_method[method])

            self.logger.info(f"Validating method: {method} ({num_videos_in_method} videos) for '{log_prefix}' set")

            batch_count = 0
            for data_dict in loader:
                batch_count += 1
                # Move tensors to the correct device
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.model.device)

                if data_dict['image'].shape[0] == 0 or data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                batch_method_names = [method] * B
                batch_method_ids = data_dict.get("method_id")
                if (
                    isinstance(batch_method_ids, torch.Tensor)
                    and batch_method_ids.numel() == B
                    and method_id_to_name
                ):
                    method_ids_list = batch_method_ids.detach().cpu().tolist()
                    batch_method_names = [
                        method_id_to_name.get(int(method_id), method)
                        for method_id in method_ids_list
                    ]
                
                # Debug logging to track potential duplicate processing
                if generate_detailed_reports and batch_count % 10 == 0:
                    self.logger.info(f"  Processing batch {batch_count} for method '{method}', B={B}, T={T}")
                predictions = self.model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)
                
                # --- SANITY CHECK: Collect first 2 frames from first batch of each method ---
                if batch_count == 1:  # Only from the first batch
                    try:
                        frame_probs = predictions['prob'].view(B, T)  # Shape: [B, T]
                        for video_idx in range(min(1, B)):  # Only first video
                            # Safely get video_id with bounds checking
                            if 'video_id' in data_dict and len(data_dict['video_id']) > video_idx:
                                video_id = str(data_dict['video_id'][video_idx])
                            else:
                                video_id = f"unknown_video_{video_idx}"
                            
                            for frame_idx in range(min(2, T)):  # Only first 2 frames
                                # Safely get frame path
                                frame_path = f"video_{video_id}_frame_{frame_idx}"
                                if ('frame_paths' in data_dict and 
                                    data_dict['frame_paths'] is not None and 
                                    len(data_dict['frame_paths']) > video_idx):
                                    try:
                                        if isinstance(data_dict['frame_paths'][video_idx], list) and len(data_dict['frame_paths'][video_idx]) > frame_idx:
                                            frame_path = data_dict['frame_paths'][video_idx][frame_idx]
                                        elif not isinstance(data_dict['frame_paths'][video_idx], list):
                                            frame_path = str(data_dict['frame_paths'][video_idx])
                                    except (IndexError, TypeError):
                                        pass  # Keep default frame_path
                                
                                # Safely get label
                                if 'label' in data_dict and len(data_dict['label']) > video_idx:
                                    label = int(data_dict['label'][video_idx].cpu())
                                else:
                                    label = -1  # Unknown label
                                
                                sanity_check_data.append({
                                    'method': batch_method_names[video_idx] if video_idx < len(batch_method_names) else method,
                                    'video_id': video_id,
                                    'frame_idx': frame_idx,
                                    'frame_path': frame_path,
                                    'probability': float(frame_probs[video_idx, frame_idx].cpu()),
                                    'label': label,
                                    'epoch': epoch + 1,
                                    'step': step_cnt
                                })
                    except Exception as e:
                        self.logger.warning(f"Failed to collect sanity check data for method {method}: {e}")
                        # Continue validation without crashing

                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                all_losses.append(losses['overall'].item())

                labels_np = data_dict['label'].cpu().numpy()
                probs_np = video_probs.cpu().numpy()

                all_labels.extend(labels_np)
                all_preds.extend(probs_np)
                for idx, method_name in enumerate(batch_method_names):
                    method_labels[method_name].append(labels_np[idx])
                    method_preds[method_name].append(probs_np[idx])
                videos_processed += data_dict['image'].shape[0]

                # --- NEW: Collect detailed data for reports if flag is enabled ---
                if generate_detailed_reports:
                    frame_level_probs = predictions['prob'].view(B, T)
                    for i in range(B):  # Iterate over each video in the batch
                        video_id = data_dict['video_id'][i]
                        method_name = batch_method_names[i] if i < len(batch_method_names) else method

                        label = labels_np[i]
                        avg_prob = probs_np[i]
                        prediction = 1 if avg_prob >= 0.5 else 0
                        is_correct = 1 if prediction == label else 0
                        group_key, family_key = infer_group_and_family(
                            label=label,
                            method=method_name,
                            source=None,
                        )

                        # Append data for the video-level report
                        video_report_data.append([
                            method_name, label, video_id, avg_prob, prediction, is_correct, group_key, family_key
                        ])

                        # Append data for the frame-level report
                        for j in range(T):  # Iterate over each frame in the video
                            frame_path = data_dict['frame_paths'][i][j]
                            frame_prob = frame_level_probs[i, j].item()
                            frame_report_data.append([
                                method_name, label, video_id, frame_path, frame_prob, group_key, family_key
                            ])

            if generate_detailed_reports:
                self.logger.info(
                    f"  Finished method '{method}'. Processed {batch_count} batches.")
            else:
                self.logger.info(
                    f"  Finished method '{method}'. Total progress: {videos_processed}/{total_videos} videos.")

        if not all_labels:
            self.logger.error(f"Validation failed for '{log_prefix}': No data was processed.")
            return {}  # Return an empty dict

        total_val_time = time.time() - val_start_time
        self.logger.info(f"Validation for '{log_prefix}' finished in {total_val_time:.2f}s. Calculating metrics...")

        # --- NEW: Generate and upload reports if the flag was set ---
        if generate_detailed_reports:
            try:
                # Deduplicate frame data by method+frame_path (keep the first occurrence)
                seen_frame_keys = set()
                deduplicated_frame_data = []
                for row in frame_report_data:
                    method = row[0]      # method is at index 0
                    frame_path = row[3]  # frame_path is at index 3
                    unique_key = f"{method}_{frame_path}"  # Create unique key per method
                    if unique_key not in seen_frame_keys:
                        seen_frame_keys.add(unique_key)
                        deduplicated_frame_data.append(row)
                
                # Deduplicate video data by method+video_id (keep the first occurrence)
                seen_video_keys = set()
                deduplicated_video_data = []
                for row in video_report_data:
                    method = row[0]   # method is at index 0
                    video_id = row[2] # video_id is at index 2
                    unique_key = f"{method}_{video_id}"  # Create unique key per method
                    if unique_key not in seen_video_keys:
                        seen_video_keys.add(unique_key)
                        deduplicated_video_data.append(row)
                
                self.logger.info(f"Deduplication: Frame data reduced from {len(frame_report_data)} to {len(deduplicated_frame_data)} entries")
                self.logger.info(f"Deduplication: Video data reduced from {len(video_report_data)} to {len(deduplicated_video_data)} entries")
                
                self._generate_and_upload_reports(
                    log_prefix,
                    deduplicated_frame_data,
                    deduplicated_video_data,
                    all_preds,
                    all_labels,
                    method_preds,
                    method_labels,
                    generate_detailed_reports,
                    run_name=run_name,
                    output_gcs_folder=output_gcs_folder,
                    output_filename_prefix=output_filename_prefix,
                )
                self.logger.info("✅ Detailed reports generated and uploaded successfully.")
            except Exception as e:
                self.logger.error(f"❌ Error generating detailed reports: {e}")
                self.logger.error("Continuing with metric calculation...")

        self.logger.info(f"--- Calculating overall performance for '{log_prefix}' ---")
        try:
            overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))
        except Exception as e:
            self.logger.error(f"❌ Error calculating test metrics: {e}")
            overall_metrics = {
                'loss': -1.0,
                'acc': -1.0, 
                'auc': -1.0,
                'eer': -1.0,
                'eer_threshold': -1.0,
                'ap': -1.0,
                'error': str(e)
            }

        # Use the log_prefix for all WandB metrics to create separate charts.
        wandb_log_dict = {f"{log_prefix}/epoch": epoch + 1, "train/step": step_cnt}

        if all_losses:
            avg_val_loss = np.mean(all_losses)
            wandb_log_dict[f'{log_prefix}/overall/loss'] = avg_val_loss
            self.logger.info(f"Overall {log_prefix} loss: {avg_val_loss:.4f}")

        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'{log_prefix}/overall/{name}'] = value
                self.logger.info(f"Overall {log_prefix} {name}: {value:.4f}")

        wandb_log_dict[f'{log_prefix}/probabilities'] = wandb.Histogram(np.array(all_preds))

        # --- THIS IS THE CRUCIAL CONTROL BLOCK ---
        # All checkpointing and early stopping logic is now conditional on this being the
        # designated primary validation set (i.e., the holdout set).
        if is_primary_metric:
            current_metric = overall_metrics.get(self.metric_scoring)
            if current_metric is None:
                self.logger.warning(
                    f"Primary metric '{self.metric_scoring}' not found. Skipping checkpointing and early stopping check.")
            else:
                is_improvement = current_metric > self.best_val_metric + self.early_stopping_min_delta
                if is_improvement:
                    self.logger.info(
                        f"🚀 PRIMARY METRIC IMPROVED! New best {self.metric_scoring}: {current_metric:.4f} (previously {self.best_val_metric:.4f})")
                    self.best_val_metric = current_metric
                    self.best_val_epoch = epoch + 1
                    self.epochs_without_improvement = 0

                    if self.wandb_run:
                        self.wandb_run.summary['best/epoch'] = self.best_val_epoch
                        self.wandb_run.summary['best/metric'] = self.best_val_metric
                        self.wandb_run.summary['best/auc'] = overall_metrics.get('auc', 0)
                        self.wandb_run.summary['best/eer'] = overall_metrics.get('eer', 0)
                        self.wandb_run.summary['best/eer_threshold'] = overall_metrics.get('eer_threshold', 0)
                        self.wandb_run.summary['best/acc'] = overall_metrics.get('acc', 0)
                        # FPR operating points at best epoch
                        for fpr_key in ['tpr_at_fpr1pct', 'tpr_at_fpr2pct', 'tpr_at_fpr5pct',
                                        'thresh_at_fpr1pct', 'thresh_at_fpr2pct', 'thresh_at_fpr5pct']:
                            if fpr_key in overall_metrics:
                                self.wandb_run.summary[f'best/{fpr_key}'] = overall_metrics[fpr_key]

                    if self.config.get('save_ckpt', True):
                        self.logger.info(f"✅ Saving new best checkpoint to GCS (Epoch {epoch + 1})...")
                        if self.first_best_gcs_path is None:
                            new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                          eer=overall_metrics.get('eer'), ckpt_prefix='first_best')
                            if new_gcs_path:
                                self.first_best_gcs_path = new_gcs_path

                        is_top_n = len(self.top_n_checkpoints) < self.top_n_size or current_metric > \
                                   self.top_n_checkpoints[-1]['metric']
                        if is_top_n:
                            new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                          eer=overall_metrics.get('eer'), ckpt_prefix='top_n',
                                                          step=step_cnt)
                            if new_gcs_path:
                                self.top_n_checkpoints.append(
                                    {'metric': current_metric, 'epoch': epoch + 1, 'gcs_path': new_gcs_path})
                                self.top_n_checkpoints.sort(key=lambda x: x['metric'], reverse=True)
                                if len(self.top_n_checkpoints) > self.top_n_size:
                                    worst_ckpt = self.top_n_checkpoints.pop()
                                    self._delete_from_gcs(worst_ckpt['gcs_path'])

                    if self.wandb_run:
                        self.wandb_run.summary['overall_best_ckpt_gcs'] = self.top_n_checkpoints[0]['gcs_path']
                        self.wandb_run.summary[
                            'bottom_line'] = f"best_epoch={self.best_val_epoch} AUC={self.wandb_run.summary['best/auc']:.4f}"

                else:  # No improvement
                    if self.early_stopping_enabled:
                        self.epochs_without_improvement += 1
                        self.logger.warning(
                            f"No primary metric improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                            f"Current {self.metric_scoring}: {current_metric:.4f}, Best: {self.best_val_metric:.4f}"
                        )

                if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                    self.early_stop_triggered = True
                    self.logger.critical(
                        f"🚨 EARLY STOPPING TRIGGERED! No improvement in '{self.metric_scoring}' for {self.early_stopping_patience} epochs.")

        # Log the tracked best metric state regardless of which validation set is running for consistent tracking.
        # We namespace it to make it clear this is the state of the *primary* validation metric.
        wandb_log_dict['val_primary/best_metric'] = self.best_val_metric
        wandb_log_dict['val_primary/best_epoch'] = self.best_val_epoch
        if self.early_stopping_enabled:
            wandb_log_dict['val_primary/epochs_without_improvement'] = self.epochs_without_improvement

        # --- Calculate and log MEANINGFUL per-method and derived metrics ---
        per_method_aucs = []
        real_source_names = self.config.get('dataset_methods', {}).get('use_real_sources', [])

        # Create a temporary dictionary for method metrics to build the table
        method_table_metrics = {}

        # 1. Pool all real predictions and labels from the validation set
        all_real_preds = []
        all_real_labels = []
        for method_name in real_source_names:
            if method_name in method_preds:
                all_real_preds.extend(method_preds[method_name])
                all_real_labels.extend(method_labels[method_name])

        # 2. Calculate TRUE per-method accuracy for each FAKE method (threshold-based)
        for method in sorted(method_preds.keys()):
            # Only calculate for fake methods
            if method not in real_source_names:
                method_preds_array = np.array(method_preds[method])
                method_labels_array = np.array(method_labels[method])
                
                # Calculate threshold-based accuracy: how many fake videos are correctly classified as fake (>=0.5)
                predictions_binary = (method_preds_array >= 0.5).astype(int)
                correct_predictions = np.sum(predictions_binary == method_labels_array)
                per_method_accuracy = correct_predictions / len(method_labels_array) if len(method_labels_array) > 0 else 0.0
                
                # Store simplified metrics (only accuracy for fake methods)
                method_table_metrics[method] = {'acc': per_method_accuracy, 'n_samples': len(method_labels_array)}
                
                # Log only the meaningful accuracy metric
                wandb_log_dict[f'{log_prefix}/method/{method}/acc'] = per_method_accuracy
                
                self.logger.info(f"Method '{method}' per-method accuracy: {per_method_accuracy:.4f} ({correct_predictions}/{len(method_labels_array)})")

        # 3. Calculate TRUE per-method accuracy for each REAL method (threshold-based)
        for method in sorted(method_preds.keys()):
            if method in real_source_names:
                method_preds_array = np.array(method_preds[method])
                method_labels_array = np.array(method_labels[method])
                
                # Calculate threshold-based accuracy: how many real videos are correctly classified as real (<0.5)
                predictions_binary = (method_preds_array >= 0.5).astype(int)
                correct_predictions = np.sum(predictions_binary == method_labels_array)
                per_method_accuracy = correct_predictions / len(method_labels_array) if len(method_labels_array) > 0 else 0.0
                
                # Store simplified metrics (only accuracy for real methods)
                method_table_metrics[method] = {'acc': per_method_accuracy, 'n_samples': len(method_labels_array)}
                
                # Log only the meaningful accuracy metric
                wandb_log_dict[f'{log_prefix}/method/{method}/acc'] = per_method_accuracy
                
                self.logger.info(f"Method '{method}' per-method accuracy: {per_method_accuracy:.4f} ({correct_predictions}/{len(method_labels_array)})")

        # Create and log a simplified W&B Table with only meaningful metrics
        # --- Weakest-method tracking ---
        real_method_accs = {
            m: metrics['acc'] for m, metrics in method_table_metrics.items()
            if m in real_source_names and metrics.get('acc') is not None
        }
        fake_method_accs = {
            m: metrics['acc'] for m, metrics in method_table_metrics.items()
            if m not in real_source_names and metrics.get('acc') is not None
        }
        if real_method_accs:
            worst_real = min(real_method_accs.items(), key=lambda x: x[1])
            wandb_log_dict[f'{log_prefix}/weakest/real_method'] = worst_real[0]
            wandb_log_dict[f'{log_prefix}/weakest/real_acc'] = worst_real[1]
            self.logger.info(f"Weakest real method: {worst_real[0]} ({worst_real[1]:.4f})")
        if fake_method_accs:
            worst_fake = min(fake_method_accs.items(), key=lambda x: x[1])
            wandb_log_dict[f'{log_prefix}/weakest/fake_method'] = worst_fake[0]
            wandb_log_dict[f'{log_prefix}/weakest/fake_acc'] = worst_fake[1]
            self.logger.info(f"Weakest fake method: {worst_fake[0]} ({worst_fake[1]:.4f})")

        if self.wandb_run:
            columns = ["epoch", "method", "acc", "n_samples"]
            table_data = []
            for method, metrics in method_table_metrics.items():
                table_data.append([
                    epoch + 1, method, metrics.get('acc'), metrics.get('n_samples')
                ])
            wandb_log_dict[f"{log_prefix}/method_table"] = wandb.Table(columns=columns, data=table_data)

        # Calculate macro accuracy: Unweighted average of per-method accuracies
        method_accuracies = [metrics.get('acc') for metrics in method_table_metrics.values() if metrics.get('acc') is not None]
        if method_accuracies:
            macro_accuracy = np.mean(method_accuracies)
            wandb_log_dict[f'{log_prefix}/derived/macro_accuracy'] = macro_accuracy
            self.logger.info(f"Macro per-method accuracy: {macro_accuracy:.4f}")
        else:
            macro_accuracy = None

        # Real-Real AUC calculation can remain the same
        real_preds_for_auc = []
        real_labels_for_auc = []
        for method_name in real_source_names:
            if method_name in method_preds:
                real_preds_for_auc.extend(method_preds[method_name])
                real_labels_for_auc.extend(method_labels[method_name])

        if len(real_labels_for_auc) > 1 and len(np.unique(real_labels_for_auc)) > 1:
            real_real_metrics = get_test_metrics(np.array(real_preds_for_auc), np.array(real_labels_for_auc))
            real_real_auc = real_real_metrics.get('auc', 0.0)
            wandb_log_dict[f'{log_prefix}/derived/real_real_auc'] = real_real_auc
        else:
            real_real_auc = None

        # --- SANITY CHECK: Log verification data as W&B table ---
        if self.wandb_run and sanity_check_data:
            sanity_check_columns = ['method', 'video_id', 'frame_idx', 'frame_path', 'probability', 'label', 'epoch', 'step']
            sanity_check_table_data = [[row[col] for col in sanity_check_columns] for row in sanity_check_data]
            wandb_log_dict[f'{log_prefix}/sanity_check_predictions'] = wandb.Table(
                columns=sanity_check_columns, 
                data=sanity_check_table_data
            )
            
            self.logger.info(f"✅ Logged {len(sanity_check_data)} sanity check predictions for verification")
            
            # Also log a summary for quick reference
            sanity_summary = {}
            for row in sanity_check_data:
                method_key = f"{row['method']}_frame_{row['frame_idx']}"
                sanity_summary[f"{log_prefix}/sanity/{method_key}"] = row['probability']
            wandb_log_dict.update(sanity_summary)

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)

        returned_metrics = {
            'overall': overall_metrics,
            'per_method': method_table_metrics,
            'macro_accuracy': macro_accuracy if 'macro_accuracy' in locals() else None,
            'real_real_auc': real_real_auc,
            'all_preds': np.array(all_preds),
            'all_labels': np.array(all_labels),
        }

        del method_labels, method_preds, overall_metrics, all_losses
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"===> Evaluation for '{log_prefix}' Done!")
        return returned_metrics

    def _run_ood_monitoring(self, epoch, step_cnt, indist_threshold=None):
        """Helper to run the OOD monitoring loop.

        Returns:
            float | None: Overall OOD AUC if monitoring ran, else None.
        """
        if self.ood_loader is None or self.config['local_rank'] != 0:
            return None
        if not self.ood_monitoring_enabled:
            return None
        if step_cnt < self.ood_monitoring_start_step:
            if not self._ood_warmup_logged:
                self.logger.info(
                    "OOD monitoring warmup active: first run at step %d",
                    self.ood_monitoring_start_step,
                )
                self._ood_warmup_logged = True
            return None

        steps_since_start = step_cnt - self.ood_monitoring_start_step
        if steps_since_start % self.ood_monitoring_every_steps != 0:
            return None
        if self._last_ood_monitor_step == step_cnt:
            return None

        self._last_ood_monitor_step = step_cnt
        self.logger.info(f"\n===> OOD Monitoring at epoch {epoch + 1}, step {step_cnt}")
        return self.ood_monitoring_epoch(epoch, step_cnt, indist_threshold=indist_threshold)

    @torch.no_grad()
    def ood_monitoring_epoch(self, epoch, step_cnt, indist_threshold=None):
        """
        Runs evaluation on the OOD set. Logs metrics with an 'ood/' prefix.

        Returns the overall OOD AUC so callers can use it for composite
        checkpointing (R12+).  Does NOT trigger checkpointing itself.

        Args:
            indist_threshold: If provided, also logs ``ood/at_indist/*`` metrics
                evaluated at this fixed threshold (the val_in_dist EER threshold).

        Returns:
            float | None: Overall OOD AUC, or None if evaluation failed.
        """
        self.setEval()

        total_videos = sum(len(v_list) for v_list in self.ood_loader.videos_by_method.values())
        if total_videos == 0:
            self.logger.warning("OOD loader is configured but contains no videos. Skipping.")
            return None

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        method_jitters = defaultdict(list)  # frame-to-frame score jitter per method
        all_preds, all_labels = [], []

        self.logger.info(f"Starting OOD Monitoring for {total_videos} videos...")
        ood_start_time = time.time()
        videos_processed = 0

        for method in self.ood_loader.keys():
            loader = self.ood_loader[method]

            for data_dict in loader:
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor): data_dict[key] = value.to(self.model.device)
                if data_dict['image'].shape[0] == 0 or data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                predictions = self.model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)

                # Frame-to-frame score jitter per video
                frame_probs = predictions['prob'].view(B, T).detach().cpu().numpy()
                for b in range(B):
                    fp = frame_probs[b]
                    if len(fp) >= 2:
                        method_jitters[method].append(float(np.mean(np.abs(np.diff(fp)))))

                labels_np = data_dict['label'].cpu().numpy()
                probs_np = video_probs.cpu().numpy()

                all_labels.extend(labels_np)
                all_preds.extend(probs_np)
                method_labels[method].extend(labels_np)
                method_preds[method].extend(probs_np)

                B = data_dict['image'].shape[0]
                videos_processed += B

            self.logger.info(
                f"  ... OOD method {method} done. Total progress: {videos_processed}/{total_videos} videos.")

        total_ood_time = time.time() - ood_start_time
        self.logger.info(f"OOD Monitoring finished in {total_ood_time:.2f}s. Calculating metrics...")

        if not all_labels:
            self.logger.error("OOD Monitoring failed: No data was processed.")
            return

        overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))
        wandb_log_dict = {"ood/epoch": epoch + 1, "train/step": step_cnt}

        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'ood/overall/{name}'] = value
                self.logger.info(f"OOD Overall {name}: {value:.4f}")

        # Log the probability distribution histogram
        wandb_log_dict['ood/probabilities'] = wandb.Histogram(np.array(all_preds))

        # Metrics Per Method
        method_metrics = {}
        for method in method_preds.keys():
            # Check if there are any labels for this method to avoid errors
            if not method_labels[method]:
                continue

            labels_for_method = np.array(method_labels[method])
            preds_for_method = np.array(method_preds[method])

            # All labels for a given method group (e.g., 'tiktok' from real videos) should be the same.
            # We use the first label to determine if it's a real (0) or fake (1) group.
            label_type = '_real' if labels_for_method[0] == 0 else '_fake'
            method_key = f"{method}{label_type}"

            method_metrics[method] = get_test_metrics(preds_for_method, labels_for_method)

            for name, value in method_metrics[method].items():
                if name in ['acc', 'auc', 'eer']:
                    wandb_log_dict[f'ood/method/{method_key}/{name}'] = value

        # Log per-method score jitter
        for method, jitters in method_jitters.items():
            if jitters:
                mean_jitter = float(np.mean(jitters))
                wandb_log_dict[f'ood/score_jitter/{method}'] = mean_jitter
                self.logger.info(f"OOD score jitter for {method}: {mean_jitter:.4f}")

        # --- At-indist-threshold metrics for OOD ---
        if indist_threshold is not None and indist_threshold > 0:
            ood_at_indist = metrics_at_threshold(
                np.array(all_preds), np.array(all_labels), indist_threshold
            )
            if ood_at_indist:
                for k, v in ood_at_indist.items():
                    wandb_log_dict[f'ood/at_indist/{k}'] = v
                wandb_log_dict['ood/at_indist/threshold'] = indist_threshold
                self.logger.info(
                    f"OOD @ indist threshold {indist_threshold:.4f}: "
                    f"acc={ood_at_indist['acc']:.4f}  f1={ood_at_indist['f1']:.4f}  "
                    f"fpr={ood_at_indist['fpr']:.4f}  fnr={ood_at_indist['fnr']:.4f}"
                )

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)

        # Capture OOD AUC before cleanup so callers can use it for
        # composite checkpointing (R12+).
        ood_auc = overall_metrics.get('auc')

        del all_preds, all_labels, method_labels, method_preds, method_jitters, overall_metrics
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("===> OOD Monitoring Done!")
        return ood_auc

    def _generate_and_upload_reports(self, log_prefix, frame_data, video_data, all_preds, all_labels, method_preds,
                                     method_labels, generate_detailed_reports=False, run_name="",
                                     output_gcs_folder: str = None, output_filename_prefix: str = None):
        """
        Generates and uploads detailed CSV and TXT reports to GCS.
        
        Args:
            output_gcs_folder: Optional GCS folder to write reports to (e.g., 'gs://bucket/path/folder').
                              If specified, appends to existing folder instead of creating new timestamped one.
            output_filename_prefix: Optional prefix for report filenames (e.g., 'target_source_').
        """
        self.logger.info("Generating detailed validation reports...")
        local_temp_dir = tempfile.mkdtemp()
        
        # Determine filename prefix
        prefix = output_filename_prefix or ""
        
        # Track what we successfully generated
        files_generated = []
        expected_files = 4

        def _resolve_group_family_from_row(row):
            if len(row) >= 8:
                return row[6], row[7]
            return infer_group_and_family(label=row[1], method=row[0], source=None)

        normalized_video_rows = []

        try:
            # 1. --- Create Frame-level CSV ---
            try:
                frame_filename = f'{prefix}frames_report.csv'
                frame_csv_path = os.path.join(local_temp_dir, frame_filename)
                frame_rows = []
                for row in frame_data:
                    if len(row) >= 7:
                        frame_rows.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6]])
                    else:
                        group_key, family_key = _resolve_group_family_from_row(row)
                        frame_rows.append([row[0], row[1], row[2], row[3], row[4], group_key, family_key])
                with open(frame_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['method', 'label', 'video_id', 'frame_path', 'frame_prob', 'group_key', 'family_key'])
                    writer.writerows(frame_rows)
                self.logger.info(f"Frame report generated with {len(frame_rows)} entries.")
                files_generated.append((frame_filename, frame_csv_path))
            except Exception as e:
                self.logger.error(f"Failed to generate frame report: {e}")

            # 2. --- Create Video-level CSV ---
            try:
                video_filename = f'{prefix}videos_report.csv'
                video_csv_path = os.path.join(local_temp_dir, video_filename)
                for row in video_data:
                    if len(row) >= 8:
                        normalized_video_rows.append([row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]])
                    else:
                        group_key, family_key = _resolve_group_family_from_row(row)
                        normalized_video_rows.append([
                            row[0], row[1], row[2], row[3], row[4], row[5], group_key, family_key
                        ])
                with open(video_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'method', 'label', 'video_id', 'avg_video_prob', 'prediction', 'is_correct',
                        'group_key', 'family_key'
                    ])
                    writer.writerows(normalized_video_rows)
                self.logger.info(f"Video report generated with {len(normalized_video_rows)} entries.")
                files_generated.append((video_filename, video_csv_path))
            except Exception as e:
                self.logger.error(f"Failed to generate video report: {e}")

            # 3. --- Create Per-Group Metrics CSV ---
            group_metrics_rows = []
            try:
                group_filename = f'{prefix}group_metrics.csv'
                group_csv_path = os.path.join(local_temp_dir, group_filename)
                group_to_rows = defaultdict(list)
                for row in normalized_video_rows:
                    group_to_rows[row[6]].append(row)

                for group_key in sorted(group_to_rows.keys()):
                    rows = group_to_rows[group_key]
                    labels = np.array([int(r[1]) for r in rows], dtype=np.int64)
                    preds = np.array([int(r[4]) for r in rows], dtype=np.int64)
                    probs = np.array([float(r[3]) for r in rows], dtype=np.float32)

                    tn = int(np.sum((labels == 0) & (preds == 0)))
                    fp = int(np.sum((labels == 0) & (preds == 1)))
                    fn = int(np.sum((labels == 1) & (preds == 0)))
                    tp = int(np.sum((labels == 1) & (preds == 1)))

                    n_videos = len(rows)
                    accuracy = (tp + tn) / n_videos if n_videos > 0 else 0.0
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

                    mean_prob = float(np.mean(probs)) if probs.size > 0 else 0.0
                    p50_prob = float(np.percentile(probs, 50)) if probs.size > 0 else 0.0
                    p90_prob = float(np.percentile(probs, 90)) if probs.size > 0 else 0.0
                    family_key = rows[0][7] if len(rows[0]) >= 8 else "unknown"

                    group_metrics_rows.append([
                        group_key,
                        family_key,
                        n_videos,
                        accuracy,
                        tp,
                        tn,
                        fp,
                        fn,
                        tpr,
                        fpr,
                        fnr,
                        mean_prob,
                        p50_prob,
                        p90_prob,
                    ])

                with open(group_csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'group_key', 'family_key', 'n_videos', 'accuracy', 'tp', 'tn', 'fp', 'fn',
                        'tpr', 'fpr', 'fnr', 'mean_prob', 'p50_prob', 'p90_prob'
                    ])
                    writer.writerows(group_metrics_rows)

                self.logger.info(f"Group metrics report generated with {len(group_metrics_rows)} groups.")
                files_generated.append((group_filename, group_csv_path))
            except Exception as e:
                self.logger.error(f"Failed to generate group metrics report: {e}")

            # 4. --- Create Summary TXT file ---
            try:
                summary_filename = f'{prefix}summary_report.txt'
                summary_txt_path = os.path.join(local_temp_dir, summary_filename)
                with open(summary_txt_path, 'w') as f:
                    f.write(f"Validation Summary Report for: {log_prefix}\n")
                    f.write(f"Run Name: {run_name}\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 40 + "\n")
                    f.write("Overall Performance\n")
                    f.write("-" * 40 + "\n")

                    # Calculate overall confusion matrix
                    # Note: fake=1 (positive), real=0 (negative)
                    
                    # When detailed reports are enabled, use video-level aggregated data for summary
                    if generate_detailed_reports and normalized_video_rows:
                        # Extract video-level predictions and labels from the detailed report data
                        video_labels = [row[1] for row in normalized_video_rows]  # Column 1 is label
                        video_preds = [row[4] for row in normalized_video_rows]   # Column 4 is prediction (binary)
                        
                        tn, fp, fn, tp = confusion_matrix(video_labels, video_preds, labels=[0, 1]).ravel()
                        total = len(video_labels)
                        acc = (tp + tn) / total if total > 0 else 0
                        f.write(f"Total Videos: {total}\n")
                    else:
                        # Fallback to frame-level data (original behavior)
                        overall_preds_binary = (np.array(all_preds) >= 0.5).astype(int)
                        tn, fp, fn, tp = confusion_matrix(all_labels, overall_preds_binary, labels=[0, 1]).ravel()
                        total = tn + fp + fn + tp
                        acc = (tp + tn) / total if total > 0 else 0
                        f.write(f"Total Videos: {total}\n")
                    f.write(f"Accuracy: {acc:.4f}\n")
                    f.write(f"True Positives (Correctly identified Fake): {tp}\n")
                    f.write(f"True Negatives (Correctly identified Real): {tn}\n")
                    f.write(f"False Positives (Real misclassified as Fake): {fp}\n")
                    f.write(f"False Negatives (Fake misclassified as Real): {fn}\n\n")

                    f.write("=" * 40 + "\n")
                    f.write("Per-Method Performance\n")
                    f.write("-" * 40 + "\n")

                    # Calculate per-method performance using deduplicated video data
                    method_video_data = defaultdict(list)
                    for row in normalized_video_rows:  # video_data is deduplicated
                        method = row[0]  # Column 0 is method
                        method_video_data[method].append(row)
                    
                    for method in sorted(method_video_data.keys()):
                        method_rows = method_video_data[method]
                        if not method_rows:
                            continue
                            
                        # Extract data from deduplicated video rows
                        labels = [row[1] for row in method_rows]      # Column 1 is label
                        predictions = [row[4] for row in method_rows] # Column 4 is prediction (binary)
                        
                        labels = np.array(labels)
                        predictions = np.array(predictions)

                        # Determine method type
                        is_real_method = (labels[0] == 0)
                        method_type = "REAL" if is_real_method else "FAKE"

                        f.write(f"----- Method: {method} ({method_type}) -----\n")
                        f.write(f"Total Videos: {len(labels)}\n")

                        # Calculate accuracy using deduplicated data
                        if len(np.unique(labels)) == 1:
                            correct_predictions = np.sum(labels == predictions)
                            accuracy = correct_predictions / len(labels)
                            f.write(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{len(labels)} correct)\n\n")
                        else:  # Mixed labels (rare case)
                            m_tn, m_fp, m_fn, m_tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
                            m_total = m_tn + m_fp + m_fn + m_tp
                            m_acc = (m_tp + m_tn) / m_total if m_total > 0 else 0
                            f.write(f"Accuracy: {m_acc:.4f}\n")
                            f.write(f"  TN: {m_tn}, FP: {m_fp}, FN: {m_fn}, TP: {m_tp}\n\n")

                    f.write("=" * 40 + "\n")
                    f.write("Per-Group Performance\n")
                    f.write("-" * 40 + "\n")
                    if group_metrics_rows:
                        best_group = max(group_metrics_rows, key=lambda row: row[3])
                        worst_group = min(group_metrics_rows, key=lambda row: row[3])
                        for row in group_metrics_rows:
                            f.write(
                                f"{row[0]} (family={row[1]}): "
                                f"n={row[2]}, acc={row[3]:.4f}, tpr={row[8]:.4f}, "
                                f"fpr={row[9]:.4f}, fnr={row[10]:.4f}\n"
                            )
                        gap = best_group[3] - worst_group[3]
                        f.write("\n")
                        f.write(
                            f"Best-group accuracy: {best_group[0]} = {best_group[3]:.4f}\n"
                        )
                        f.write(
                            f"Worst-group accuracy: {worst_group[0]} = {worst_group[3]:.4f}\n"
                        )
                        f.write(f"Best-vs-worst group accuracy gap: {gap:.4f}\n")
                    else:
                        f.write("No group-level rows were available.\n")
                files_generated.append((summary_filename, summary_txt_path))
            except Exception as e:
                self.logger.error(f"Failed to generate summary report: {e}")

            # 4. --- Upload files to GCS ---
            # Use custom folder if provided, otherwise create timestamped folder
            if output_gcs_folder:
                # Use provided folder path (e.g., 'gs://bucket/path/folder')
                gcs_base = output_gcs_folder.rstrip('/')
                self.logger.info(f"Appending reports to existing folder: {gcs_base}")
            else:
                # Create new timestamped folder
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                gcs_folder = f"{timestamp}_{log_prefix}"
                gcs_base = f"gs://training-job-outputs/test_results/{gcs_folder}"
                self.logger.info(f"Creating new results folder: {gcs_base}")

            # Only upload files that were successfully generated
            for filename, local_path in files_generated:
                try:
                    gcs_path = f"{gcs_base}/{filename}"
                    self._upload_to_gcs(local_path, gcs_path)
                except Exception as e:
                    self.logger.error(f"Failed to upload {filename} to GCS: {e}")
            
            if files_generated:
                self.logger.info(f"Successfully processed {len(files_generated)}/{expected_files} report files.")
            else:
                self.logger.warning("No report files were successfully generated.")

        finally:
            # 5. --- Clean up local temporary directory ---
            try:
                shutil.rmtree(local_temp_dir)
            except Exception as e:
                self.logger.error(f"Failed to clean up temporary directory: {e}")

    def run_validation_on_demand(
            self,
            validation_loader,
            log_prefix: str = "on_demand_validation",
            generate_detailed_reports: bool = False,
            run_name: str = None,
            output_gcs_folder: str = None,
            output_filename_prefix: str = None,
    ) -> dict:
        """
        Runs a full validation pass on a provided dataloader.

        This method is designed for post-training analysis. It assumes that the
        model held by the Trainer instance has already been loaded with the
        desired checkpoint weights. It reuses the standard `test_epoch` logic
        to ensure the validation process is identical to the one used during
        training.

        Args:
            validation_loader: A fully configured LazyDataLoaderManager instance
                               containing the videos to evaluate.
            log_prefix: A string to prefix all WandB logs, allowing for clear
                        separation of different validation runs.
            generate_detailed_reports (bool): If True, generates and uploads
                                              detailed frame, video, and summary
                                              reports to GCS for deep analysis.
            run_name: Optional name for the report summary, useful for identifying
                      different runs in GCS.
            output_gcs_folder (str): Optional GCS folder to write reports to (e.g., 'gs://bucket/path/folder').
                                     If specified, appends to existing folder instead of creating new timestamped one.
            output_filename_prefix (str): Optional prefix for report filenames (e.g., 'target_source_').

        Returns:
            A dictionary containing the calculated metrics for this validation run.
        """
        self.logger.info(f"--- Starting on-demand validation for '{log_prefix}' ---")
        if generate_detailed_reports:
            self.logger.info("Detailed report generation is ENABLED.")
            if output_gcs_folder:
                self.logger.info(f"Output folder: {output_gcs_folder}")
            if output_filename_prefix:
                self.logger.info(f"Filename prefix: {output_filename_prefix}")

        # Reuse the existing test_epoch logic completely.
        # We pass dummy values for epoch/step and critically set
        # is_primary_metric=False to prevent any checkpointing or
        # early stopping state changes from being triggered.
        # The `generate_detailed_reports` flag will activate the new logic
        # inside the `test_epoch` method.
        try:
            metrics = self.test_epoch(
                epoch=-1,
                step_cnt=-1,
                validation_loader=validation_loader,
                log_prefix=log_prefix,
                is_primary_metric=False,
                generate_detailed_reports=generate_detailed_reports,
                run_name=run_name,
                output_gcs_folder=output_gcs_folder,
                output_filename_prefix=output_filename_prefix,
            )
        except Exception as e:
            self.logger.error(f"Error during test_epoch: {e}")
            # Return a minimal metrics dict so the caller can continue
            metrics = {
                'overall': {
                    'loss': -1.0,
                    'acc': -1.0,
                    'auc': -1.0,
                    'eer': -1.0,
                    'ap': -1.0,
                    'error': str(e)
                }
            }
            # Still try to log the error but don't crash completely
            self.logger.error("Returning default metrics to prevent total failure.")

        overall_auc = metrics.get('overall', {}).get('auc', 'N/A')
        # Fix formatting to handle non-float values properly
        if isinstance(overall_auc, (int, float)) and overall_auc >= 0:
            auc_str = f"{overall_auc:.4f}"
        else:
            auc_str = str(overall_auc)
        self.logger.info(
            f"--- On-demand validation for '{log_prefix}' complete. Overall AUC: {auc_str} ---")
        return metrics
