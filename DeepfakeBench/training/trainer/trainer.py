import math
import os
import sys
import time

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
            metric_scoring='auc',
            wandb_run=None,
            val_videos=None
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
        self.val_videos = val_videos  # Store the validation videos
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

        # Initialize AMP scaler for mixed precision training
        self.scaler = GradScaler()
        self.speed_up()

        self.log_dir = self.wandb_run.dir if self.wandb_run else './logs'
        # These are now only used for 'per_method' strategy, but are initialized here
        self.real_method_iters = {}
        self.fake_method_iters = {}

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
        self.scaler.step(self.optimizer)
        self.scaler.update()
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

        # --- Determine Epoch Length based on strategy ---
        if strategy == 'frame_level':
            effective_batch_size = self.config.get('dataloader_params', {}).get('frames_per_batch')
            total_frames = sum(len(v.frame_paths) for v in train_videos)
            epoch_len = math.ceil(total_frames / effective_batch_size) if total_frames > 0 else 0
        else:  # For per_method and video_level
            effective_batch_size = self.config.get('dataloader_params', {}).get('videos_per_batch')
            total_train_videos = len(train_videos)
            epoch_len = math.ceil(total_train_videos / effective_batch_size) if total_train_videos > 0 else 0

        self.logger.info(f"Training strategy: '{strategy}', Epoch length: {epoch_len} steps")

        # --- PRECISE EVALUATION SCHEDULING ---
        evaluation_frequency = self.config.get('data_params', {}).get('evaluation_frequency', 1)
        if evaluation_frequency <= 0:
            evaluation_frequency = 1  # Ensure at least one evaluation

        eval_steps = set()
        if epoch_len > 0:
            # Calculate interval, ensuring it's at least 1
            interval = max(1, epoch_len // evaluation_frequency)
            # Add evaluation points based on the interval
            for i in range(1, evaluation_frequency):
                eval_steps.add(i * interval)
            # Always add the last step to guarantee a final evaluation
            eval_steps.add(epoch_len)
        self.logger.info(f"Scheduled evaluation at steps: {sorted(list(eval_steps))}")

        step_cnt = epoch * epoch_len

        # --- Conditional Training Loop based on strategy ---
        if strategy == 'per_method':
            real_source_names = self.config['methods']['use_real_sources']
            all_method_names = train_loader.keys()
            real_method_names = [m for m in all_method_names if m in real_source_names]
            fake_method_names = [m for m in all_method_names if m not in real_source_names]

            real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
            for v in train_videos:
                (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1
            total_real_videos = sum(real_video_counts.values())
            total_fake_videos = sum(fake_video_counts.values())
            real_weights = [real_video_counts[m] / total_real_videos for m in
                            real_method_names] if total_real_videos > 0 else []
            fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                            fake_method_names] if total_fake_videos > 0 else []

            pbar = tqdm(range(epoch_len), desc=f"EPOCH (per_method): {epoch + 1}/{self.config['nEpochs']}")
            for iteration in pbar:
                data_dict = self._train_per_method_step(train_loader, train_loader, real_method_names,
                                                        fake_method_names, real_weights, fake_weights)
                if data_dict is None: continue

                self._run_train_step(data_dict, step_cnt, epoch, pbar)
                step_cnt += 1

                # Use the precise evaluation schedule
                if (iteration + 1) in eval_steps:
                    self._run_validation(epoch, iteration, val_method_loaders)

        elif strategy in ['video_level', 'frame_level']:
            pbar = tqdm(train_loader, desc=f"EPOCH ({strategy}): {epoch + 1}/{self.config['nEpochs']}", total=epoch_len)
            # We use 'i' as the iteration counter from enumerate
            for i, data_dict in enumerate(pbar):
                self._run_train_step(data_dict, step_cnt, epoch, pbar)
                step_cnt += 1

                # Use the precise evaluation schedule. i is 0-based, so add 1.
                if (i + 1) in eval_steps:
                    self._run_validation(epoch, i, val_method_loaders)
        else:
            raise ValueError(f"Unsupported training strategy: {strategy}")

    def _run_train_step(self, data_dict, step_cnt, epoch, pbar):
        """Helper to avoid code duplication in the training loop."""
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
            self.wandb_run.log(log_dict)

            pbar.set_postfix_str(f"Loss: {losses['overall'].item():.4f}")

    def _run_validation(self, epoch, iteration, val_method_loaders):
        """Helper to run validation to keep the main loop clean."""
        if val_method_loaders is not None and self.config['local_rank'] == 0:
            self.logger.info(f"\n===> Evaluation at epoch {epoch + 1}, step {iteration + 1}")
            self.test_epoch(epoch, val_method_loaders)

    @torch.no_grad()
    def test_epoch(self, epoch, val_method_loaders):
        self.setEval()

        # ---
        # Create a single, accurate progress bar for the entire validation set.
        # We iterate by videos, which is a more intuitive unit of progress.
        # ---
        total_videos = len(self.val_videos) if self.val_videos else 0
        if total_videos == 0:
            self.logger.warning("No validation videos found. Skipping validation.")
            return

        pbar = tqdm(total=total_videos, desc="Validating", unit="video")

        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        all_preds, all_labels = [], []
        all_losses = []

        # ---
        #  We iterate through methods sequentially. It also works perfectly with the LazyDataLoaderManager
        # to keep memory usage low.
        # ---
        for method in val_method_loaders.keys():
            # The lazy manager will create the loader on-demand here
            loader = val_method_loaders[method]
            pbar.set_description(f"Validating: {method}")

            for data_dict in loader:
                # Move tensors to the correct device
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.model.device)

                # skip empty batches or invalid shapes
                if data_dict['image'].shape[0] == 0: continue
                if data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]
                predictions = self.model(data_dict, inference=True)
                video_probs = predictions['prob'].view(B, T).mean(dim=1)

                # Calculate and store validation loss for the batch
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                all_losses.append(losses['overall'].item())

                labels_np = data_dict['label'].cpu().numpy()
                probs_np = video_probs.cpu().numpy()

                all_labels.extend(labels_np)
                all_preds.extend(probs_np)

                # Add probabilities and labels by method
                method_labels[method].extend(labels_np)
                method_preds[method].extend(probs_np)

                # ---
                # Update the progress bar by the number of videos in the batch.
                # ---
                pbar.update(B)

        pbar.close()

        if not all_labels:
            self.logger.error("Validation failed: No data was processed after iterating all loaders.")
            return

        self.logger.info("--- Calculating overall validation performance ---")
        overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))

        wandb_log_dict = {"val/epoch": epoch + 1}

        # --- Calculate and log average validation loss ---
        if all_losses:
            avg_val_loss = np.mean(all_losses)
            wandb_log_dict['val/overall/loss'] = avg_val_loss
            self.logger.info(f"Overall val loss: {avg_val_loss:.4f}")
        # ---

        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'val/overall/{name}'] = value
                self.logger.info(f"Overall val {name}: {value:.4f}")

        # Metrics Per Method
        method_metrics = {}
        for method in method_preds.keys():
            method_metrics[method] = get_test_metrics(np.array(method_preds[method]),
                                                      np.array(method_labels[method]))
            for name, value in method_metrics[method].items():
                if name in ['acc']:
                    wandb_log_dict[f'val/method/{method}/{name}'] = value
                    self.logger.info(f"Method {method} val {name}: {value:.4f}")

        # --- Create and log a W&B Table for side-by-side comparison ---
        if self.wandb_run:
            columns = ["epoch", "method", "acc", "auc", "eer", "n_samples"]
            table_data = []
            for method, metrics in method_metrics.items():
                table_data.append([
                    epoch + 1,
                    method,
                    metrics.get('acc'),
                    metrics.get('auc'),
                    metrics.get('eer'),
                    len(method_labels[method])  # Get the sample count for the method
                ])

            # Add the created table to the dictionary that will be logged
            wandb_log_dict["val/method_table"] = wandb.Table(columns=columns, data=table_data)

        # --- Advanced Checkpoint Saving and Early Stopping Logic ---
        current_metric = overall_metrics.get(self.metric_scoring)
        if current_metric is None:
            self.logger.warning(
                f"Metric '{self.metric_scoring}' not found. Skipping checkpointing and early stopping check.")
        else:
            is_improvement = current_metric > self.best_val_metric + self.early_stopping_min_delta

            if is_improvement:
                self.logger.info(
                    f"🚀 Performance improved! New best {self.metric_scoring}: {current_metric:.4f} (previously {self.best_val_metric:.4f})")
                self.epochs_without_improvement = 0  # Reset counter

                # --- The entire checkpoint saving and best metric update logic now runs only on improvement ---
                if self.config.get('save_ckpt', True):
                    # 1. Handle the "First Best" checkpoint (one-time save)
                    if self.first_best_gcs_path is None:
                        self.logger.info(
                            f"🎉 Saving FIRST best model! Metric ({self.metric_scoring}): {current_metric:.4f}")
                        new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                      eer=overall_metrics.get('eer'), ckpt_prefix='first_best')
                        if new_gcs_path:
                            self.first_best_gcs_path = new_gcs_path
                            if self.wandb_run:
                                self.wandb_run.summary['first_best_ckpt_gcs'] = new_gcs_path
                                self.wandb_run.summary['first_best_metric'] = current_metric

                    # 2. Manage the Top-N checkpoint list
                    is_top_n = len(self.top_n_checkpoints) < self.top_n_size or current_metric > \
                               self.top_n_checkpoints[-1]['metric']
                    if is_top_n:
                        self.logger.info(
                            f"🏆 New Top-{self.top_n_size} performance! Metric ({self.metric_scoring}): {current_metric:.4f}")
                        new_gcs_path = self.save_ckpt(epoch=epoch + 1, auc=overall_metrics.get('auc'),
                                                      eer=overall_metrics.get('eer'), ckpt_prefix='top_n')
                        if new_gcs_path:
                            self.top_n_saved_count += 1
                            self.top_n_checkpoints.append({
                                'metric': current_metric,
                                'epoch': epoch + 1,
                                'gcs_path': new_gcs_path,
                                'all_metrics': overall_metrics
                            })
                            self.top_n_checkpoints.sort(key=lambda x: x['metric'], reverse=True)
                            if len(self.top_n_checkpoints) > self.top_n_size:
                                worst_ckpt = self.top_n_checkpoints.pop()
                                self.logger.info(
                                    f"Pruning checkpoint from epoch {worst_ckpt['epoch']} as it's no longer in top {self.top_n_size}.")
                                self._delete_from_gcs(worst_ckpt['gcs_path'])

                # 3. Update overall best metric and W&B summary
                if self.top_n_checkpoints and self.top_n_checkpoints[0]['metric'] > self.best_val_metric:
                    overall_best = self.top_n_checkpoints[0]
                    self.best_val_metric = overall_best['metric']
                    self.best_val_epoch = overall_best['epoch']

                    if self.wandb_run:
                        self.wandb_run.summary['overall_best_ckpt_gcs'] = overall_best['gcs_path']
                        self.wandb_run.summary['best/epoch'] = self.best_val_epoch
                        self.wandb_run.summary['best/metric'] = self.best_val_metric
                        best_metrics_dict = overall_best['all_metrics']
                        best_auc = best_metrics_dict.get('auc', 0)
                        best_eer = best_metrics_dict.get('eer', 0)
                        best_acc = best_metrics_dict.get('acc', 0)
                        self.wandb_run.summary['best/auc'] = best_auc
                        self.wandb_run.summary['best/eer'] = best_eer
                        self.wandb_run.summary['best/acc'] = best_acc
                        self.wandb_run.summary[
                            'bottom_line'] = f"best_epoch={self.best_val_epoch} AUC={best_auc:.4f} EER={best_eer:.4f} ACC={best_acc:.4f} ckpt=GCS"

            else:  # No improvement
                if self.early_stopping_enabled:
                    self.epochs_without_improvement += 1
                    self.logger.warning(
                        f"No improvement for {self.epochs_without_improvement}/{self.early_stopping_patience} epochs. "
                        f"Current {self.metric_scoring}: {current_metric:.4f}, Best: {self.best_val_metric:.4f}"
                    )

            # Check for early stopping condition after updating the counter
            if self.early_stopping_enabled and self.epochs_without_improvement >= self.early_stopping_patience:
                self.early_stop_triggered = True
                self.logger.critical(
                    f"🚨 EARLY STOPPING TRIGGERED! No improvement in '{self.metric_scoring}' for {self.early_stopping_patience} epochs."
                )
                if self.wandb_run:
                    self.wandb_run.summary[
                        'early_stop_reason'] = f"No improvement in {self.metric_scoring} for {self.early_stopping_patience} epochs."
                    self.wandb_run.summary['early_stop_epoch'] = epoch + 1

        wandb_log_dict['val/best_metric'] = self.best_val_metric
        wandb_log_dict['val/best_epoch'] = self.best_val_epoch
        if self.early_stopping_enabled:
            wandb_log_dict['val/epochs_without_improvement'] = self.epochs_without_improvement

        if self.wandb_run: self.wandb_run.log(wandb_log_dict)
        del all_preds, all_labels, method_labels, method_preds, overall_metrics, all_losses
        gc.collect()
        torch.cuda.empty_cache()
        self.logger.info("===> Evaluation Done!")
