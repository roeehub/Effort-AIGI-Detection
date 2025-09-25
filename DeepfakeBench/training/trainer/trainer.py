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
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa
from metrics.utils import get_test_metrics  # noqa
from torch.cuda.amp import autocast, GradScaler  # noqa
import wandb  # noqa
from collections import defaultdict
from dataset.dataloaders import load_and_process_video, collate_fn  # noqa
from torchdata.datapipes.iter import IterableWrapper, Mapper, Filter  # noqa
from google.cloud import storage  # noqa
from google.api_core import exceptions  # noqa
import gc

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trainer Is Using device: {device}")


class Trainer(object):
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
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizer, and logger must be implemented")

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

        # --- Comprehensive checkpoint tracking ---
        # List of dicts: [{'metric': float, 'epoch': int, 'gcs_path': str, ...}, ...]
        self.top_n_checkpoints = []
        self.top_n_size = self.config.get('checkpointing', {}).get('top_n_size', 3)
        self.first_best_gcs_path = None
        self.top_n_saved_count = 0

        self.best_val_metric = -1.0
        self.best_val_epoch = -1

        self.early_stopping_config = self.config.get('early_stopping', {})
        self.early_stopping_enabled = self.early_stopping_config.get('enabled', False)
        if self.early_stopping_enabled:
            self.early_stopping_patience = self.early_stopping_config.get('patience', 3)
            # What is the minimum change to be considered an improvement
            self.early_stopping_min_delta = self.early_stopping_config.get('min_delta', 0.0001)
            self.epochs_without_improvement = 0
            self.logger.info(
                f"✅ Early stopping enabled: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}")

        self.early_stop_triggered = False  # Flag to signal the main loop

        # --- Step-based training control ---
        self.max_train_steps = self.config.get('max_train_steps', None)
        self.evaluate_every_steps = self.config.get('evaluate_every_steps', None)
        if self.max_train_steps:
            self.logger.info(f"✅ Training will stop at a maximum of {self.max_train_steps} total steps.")
        if self.evaluate_every_steps:
            self.logger.info(
                f"✅ Evaluation will run every {self.evaluate_every_steps} steps, overriding epoch frequency.")

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

        self.use_group_dro = use_group_dro
        if self.use_group_dro:
            self._init_group_dro(config)

        # --- Lesson Gate for curriculum learning ---
        self.gate_config = self.config.get('lesson_gate', {})
        self.gate_enabled = self.gate_config.get('enabled', False)
        if self.gate_enabled:
            self.gate_checks = self.gate_config.get('checks', [])
            self.gate_plateau_config = self.gate_config.get('plateau_check', {})
            self.gate_guardrail_config = self.gate_config.get('guardrail_check', {})

            # State tracking
            self.gate_primary_metric_history = []
            self.gate_guardrail_start_value = None

            self.logger.info("✅ Lesson Gate enabled with the following configuration:")
            self.logger.info(f"   - Checks: {len(self.gate_checks)} conditions")
            self.logger.info(f"   - Plateau Check: {self.gate_plateau_config}")
            self.logger.info(f"   - Guardrail Check: {self.gate_guardrail_config}")

    def _init_group_dro(self, config):
        """Initializes parameters and state for Group-DRO loss."""
        self.logger.info("Initializing Group-DRO loss strategy.")
        dro_params = config.get('group_dro_params', {})
        self.group_dro_beta = dro_params.get('beta', 3.0)
        self.group_dro_clip_min = dro_params.get('clip_min', 1.0)
        self.group_dro_clip_max = dro_params.get('clip_max', 4.0)
        ema_alpha = dro_params.get('ema_alpha', 0.1)

        # PREREQUISITE: Your config must contain the method_mapping generated in Step 1.
        method_mapping = config.get('data_params', {}).get('method_mapping')
        if not method_mapping:
            raise ValueError("Group-DRO requires 'data_params.method_mapping' in the config.")
        self.num_methods = len(method_mapping)

        self.group_losses_ema = torch.zeros(self.num_methods, device=self.model.device)
        self.ema_alpha = ema_alpha

    def _calculate_group_dro_loss(self, data_dict, per_sample_loss):
        """Computes the Group-DRO loss for a given batch."""
        # The dataloader now provides the 'method_id' tensor.
        method_ids = data_dict['method_id']
        device = per_sample_loss.device

        # Step 1: Compute per-group average loss for the current batch
        batch_group_losses = torch.zeros(self.num_methods, device=device)
        batch_group_counts = torch.zeros(self.num_methods, device=device)

        # This scatter_add operation is efficient for calculating group means
        batch_group_counts.index_add_(0, method_ids, torch.ones_like(per_sample_loss))
        batch_group_losses.index_add_(0, method_ids, per_sample_loss.detach())

        valid_groups_mask = batch_group_counts > 0
        batch_group_losses[valid_groups_mask] /= batch_group_counts[valid_groups_mask]

        # Step 2: Update the EMA of group losses
        current_ema = self.group_losses_ema[valid_groups_mask]
        current_batch_loss = batch_group_losses[valid_groups_mask]
        updated_ema = (1 - self.ema_alpha) * current_ema + self.ema_alpha * current_batch_loss
        self.group_losses_ema[valid_groups_mask] = updated_ema

        # Step 3: Compute group weights `w_g` based on the EMA
        # Use only the EMA of groups present in the current batch for the average to be stable
        valid_ema_losses = self.group_losses_ema[self.group_losses_ema > 0]
        avg_ema_loss = valid_ema_losses.mean() if len(valid_ema_losses) > 0 else 0

        relative_losses = self.group_losses_ema - avg_ema_loss

        weights = torch.exp(self.group_dro_beta * relative_losses)
        weights = weights * (self.num_methods / torch.sum(weights))  # Normalize
        clipped_weights = torch.clip(weights, self.group_dro_clip_min, self.group_dro_clip_max)

        # Step 4: Calculate the final weighted loss for the batch
        sample_weights = clipped_weights[method_ids].detach()
        weighted_loss = (per_sample_loss * sample_weights).mean()

        return {'overall': weighted_loss, 'group_weights': clipped_weights}

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

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in saved.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
            self.logger.info(f'Model loaded from {model_path}')
        else:
            raise FileNotFoundError(f"=> no model found at '{model_path}'")

    def save_ckpt(self, epoch, auc, eer, ckpt_prefix='ckpt'):
        """
        Saves model checkpoint locally, uploads to GCS with a prefix, and cleans up.
        Returns the GCS path of the uploaded file.
        """
        gcs_config = self.config.get('checkpointing')
        if not gcs_config or not gcs_config.get('gcs_prefix'):
            self.logger.warning("GCS checkpointing not configured. Skipping upload.")
            return None

        model_name = self.config.get('model_name', 'model')
        date_str = time.strftime("%Y%m%d")
        ckpt_name = f"{ckpt_prefix}_{model_name}_{date_str}_ep{epoch}_auc{auc:.4f}_eer{eer:.4f}.pth"

        local_save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(local_save_dir, exist_ok=True)
        local_save_path = os.path.join(local_save_dir, ckpt_name)

        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        torch.save(model_state, local_save_path)

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

        elif strategy in ['video_level', 'frame_level', 'property_balancing']:
            pbar = tqdm(train_loader, desc=f"EPOCH ({strategy}): {epoch + 1}/{self.config['nEpochs']}", total=epoch_len)
            self.optimizer.zero_grad()  # Zero gradients at the start of the epoch

            for i, data_dict in enumerate(pbar):
                if i >= epoch_len: break

                self._update_arcface_s(step_cnt)

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

                            # PREREQUISITE #2: The data_dict must contain 'method_id' (fixed in Step 2)
                            losses = self._calculate_group_dro_loss(data_dict, per_sample_loss)
                        else:
                            # Original behavior
                            losses = loss_fn_owner.get_losses(data_dict, predictions)
                    # Store unscaled loss for accurate logging
                    unscaled_loss = losses['overall'].clone().detach()
                    # Scale loss for accumulation
                    if accumulation_steps > 1:
                        losses['overall'] = losses['overall'] / accumulation_steps
                    # --- BACKWARD PASS ---
                    self.scaler.scale(losses['overall']).backward()

                # --- OPTIMIZER STEP (conditional) ---
                if is_final_accumulation_step or accumulation_steps == 1:
                    if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

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
            if epoch_len > 0 and epoch_len % accumulation_steps != 0 and accumulation_steps > 1:
                self.logger.info("Performing final optimizer step for dangling gradients at end of epoch.")
                if hasattr(self, 'gradient_clip_val') and self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()
        else:
            raise ValueError(f"Unsupported training strategy: {strategy}")

        self.logger.info(f"===> Epoch[{epoch + 1}] finished in {(time.time() - epoch_start_time) / 60:.2f} minutes.")

    def _run_train_step(self, data_dict, step_cnt, epoch, epoch_len, epoch_start_time):  # Add new args
        """Helper to avoid code duplication in the training loop."""
        self._update_arcface_s(step_cnt)
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
                is_primary_metric=not self.gate_enabled
            )
            all_val_metrics['val_holdout'] = holdout_metrics

        # 3. Check Lesson Gate
        if self.gate_enabled:
            self._check_lesson_gate(all_val_metrics)

        # 4. Run OOD monitoring (does not affect gating)
        self._run_ood_monitoring(epoch, step_cnt)

    @torch.no_grad()
    def test_epoch(self, epoch, step_cnt, validation_loader, log_prefix: str, is_primary_metric: bool):
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
        """
        self.setEval()

        # Calculate total videos from the provided loader, not a stored class attribute.
        total_videos = sum(len(v) for v in validation_loader.videos_by_method.values())
        if total_videos == 0:
            self.logger.warning(f"No validation videos found for loader with prefix '{log_prefix}'. Skipping.")
            return

        self.logger.info(f"--- Starting validation for '{log_prefix}' set ({total_videos} videos)...")
        val_start_time = time.time()
        videos_processed = 0

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        all_preds, all_labels = [], []
        all_losses = []

        # Iterate through methods from the provided loader.
        for method in validation_loader.keys():
            loader = validation_loader[method]
            num_videos_in_method = len(validation_loader.videos_by_method[method])

            self.logger.info(f"Validating method: {method} ({num_videos_in_method} videos) for '{log_prefix}' set")

            for data_dict in loader:
                # Move tensors to the correct device
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.model.device)

                if data_dict['image'].shape[0] == 0 or data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                predictions = self.model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)

                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                all_losses.append(losses['overall'].item())

                labels_np = data_dict['label'].cpu().numpy()
                probs_np = video_probs.cpu().numpy()

                all_labels.extend(labels_np)
                all_preds.extend(probs_np)
                method_labels[method].extend(labels_np)
                method_preds[method].extend(probs_np)
                videos_processed += data_dict['image'].shape[0]

            self.logger.info(
                f"  Finished method '{method}'. Total progress: {videos_processed}/{total_videos} videos.")

        if not all_labels:
            self.logger.error(f"Validation failed for '{log_prefix}': No data was processed.")
            return

        total_val_time = time.time() - val_start_time
        self.logger.info(f"Validation for '{log_prefix}' finished in {total_val_time:.2f}s. Calculating metrics...")

        self.logger.info(f"--- Calculating overall performance for '{log_prefix}' ---")
        overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))

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

        # Metrics Per Method, logged with the correct prefix.
        # This first block is kept for non-AUC metrics like ACC and for populating the W&B table.
        method_metrics = {}
        for method in method_preds.keys():
            method_metrics[method] = get_test_metrics(np.array(method_preds[method]),
                                                      np.array(method_labels[method]))
            for name, value in method_metrics[method].items():
                if name in ['acc']:
                    wandb_log_dict[f'{log_prefix}/method/{method}/{name}'] = value
                    self.logger.info(f"Method {method} ({log_prefix}) val {name}: {value:.4f}")

        # Create and log a W&B Table, namespaced with the prefix.
        if self.wandb_run:
            columns = ["epoch", "method", "acc", "auc", "eer", "n_samples"]
            table_data = []
            for method, metrics in method_metrics.items():
                table_data.append([
                    epoch + 1, method, metrics.get('acc'), metrics.get('auc'),
                    metrics.get('eer'), len(method_labels[method])
                ])
            wandb_log_dict[f"{log_prefix}/method_table"] = wandb.Table(columns=columns, data=table_data)

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
                        self.wandb_run.summary['best/acc'] = overall_metrics.get('acc', 0)

                    if self.config.get('save_ckpt', True):
                        if self.first_best_gcs_path is None:
                            new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                          eer=overall_metrics.get('eer'), ckpt_prefix='first_best')
                            if new_gcs_path:
                                self.first_best_gcs_path = new_gcs_path

                        is_top_n = len(self.top_n_checkpoints) < self.top_n_size or current_metric > \
                                   self.top_n_checkpoints[-1]['metric']
                        if is_top_n:
                            new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                          eer=overall_metrics.get('eer'), ckpt_prefix='top_n')
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

        if self.wandb_run:
            self.wandb_run.log(wandb_log_dict)

        # --- Calculate and log additional metrics for Lesson Gate ---
        # START: MODIFIED LOGIC FOR AUC CALCULATION
        per_method_aucs = []
        real_source_names = self.config.get('dataset_methods', {}).get('use_real_sources', [])

        # 1. Pool all real predictions and labels from the validation set
        all_real_preds = []
        all_real_labels = []
        for method_name in real_source_names:
            if method_name in method_preds:
                all_real_preds.extend(method_preds[method_name])
                all_real_labels.extend(method_labels[method_name])

        # 2. Calculate AUC for each FAKE method against the complete pool of REAL samples
        if all_real_labels:
            for method in method_preds.keys():
                # Only calculate for fake methods
                if method not in real_source_names:
                    combined_preds = np.array(method_preds[method] + all_real_preds)
                    combined_labels = np.array(method_labels[method] + all_real_labels)

                    # Ensure both classes (0 and 1) are present
                    if len(np.unique(combined_labels)) > 1:
                        # Calculate metrics on this combined (fake vs. all real) set
                        fake_vs_real_metrics = get_test_metrics(combined_preds, combined_labels)
                        auc = fake_vs_real_metrics.get('auc', -1.0)
                        per_method_aucs.append(auc)
                        # Log this correct, meaningful AUC for monitoring
                        wandb_log_dict[f'{log_prefix}/per_fake_method_auc/{method}'] = auc
                    else:
                        self.logger.warning(
                            f"Skipping AUC for method '{method}' in '{log_prefix}': not enough class diversity.")
        else:
            self.logger.warning(
                f"No real samples found in validation set '{log_prefix}'. Cannot calculate per-method AUCs.")

        if per_method_aucs:
            # Macro AUC: Unweighted average of per-method AUCs
            macro_auc = np.mean(per_method_aucs)
            wandb_log_dict[f'{log_prefix}/derived/macro_auc'] = macro_auc

            # Hardest-k AUC
            k_config_key = 'hardest_k' if 'lesson_gate' in self.config else 'hardest_k_fallback'
            k = self.config.get('lesson_gate', {}).get('hardest_k', 5)
            sorted_aucs = sorted(per_method_aucs)
            hardest_k_aucs = sorted_aucs[:k]
            hardest_k_auc_mean = np.mean(hardest_k_aucs)
            wandb_log_dict[f'{log_prefix}/derived/hardest_{k}_auc'] = hardest_k_auc_mean

            # Real-Real AUC (if real methods are present in this validation set)
            real_preds = []
            real_labels = []
            for method_name in real_source_names:
                if method_name in method_preds:
                    real_preds.extend(method_preds[method_name])
                    real_labels.extend(method_labels[method_name])

            if len(real_labels) > 1 and len(
                    np.unique(real_labels)) > 1:  # Can only compute AUC with multiple classes
                real_real_metrics = get_test_metrics(np.array(real_preds), np.array(real_labels))
                real_real_auc = real_real_metrics.get('auc', 0.0)
                wandb_log_dict[f'{log_prefix}/derived/real_real_auc'] = real_real_auc
            else:
                real_real_auc = None  # Can't be computed with only one class (all reals are label 0)
        else:
            macro_auc, hardest_k_auc_mean, real_real_auc = None, None, None

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)

        returned_metrics = {
            'overall': overall_metrics,
            'per_method': method_metrics,
            'macro_auc': macro_auc,
            f'hardest_{k}_auc' if 'k' in locals() else 'hardest_k_auc': hardest_k_auc_mean if 'hardest_k_auc_mean' in locals() else None,
            'real_real_auc': real_real_auc
        }

        del all_preds, all_labels, method_labels, method_preds, overall_metrics, all_losses
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info(f"===> Evaluation for '{log_prefix}' Done!")
        return returned_metrics

    def _run_ood_monitoring(self, epoch, step_cnt):
        """Helper to run the OOD monitoring loop."""
        if self.ood_loader is not None and self.config['local_rank'] == 0:
            self.logger.info(f"\n===> OOD Monitoring at epoch {epoch + 1}")
            self.ood_monitoring_epoch(epoch, step_cnt)

    @torch.no_grad()
    def ood_monitoring_epoch(self, epoch, step_cnt):
        """
        Runs evaluation on the OOD set. Logs metrics with an 'ood/' prefix
        and does NOT affect checkpointing or early stopping.
        """
        self.setEval()

        total_videos = sum(len(v_list) for v_list in self.ood_loader.videos_by_method.values())
        if total_videos == 0:
            self.logger.warning("OOD loader is configured but contains no videos. Skipping.")
            return

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
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

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)
        del all_preds, all_labels, method_labels, method_preds, overall_metrics
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("===> OOD Monitoring Done!")
