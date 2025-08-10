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
            raise ValueError("config, model, optimizier, and logger must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.wandb_run = wandb_run
        self.val_videos = val_videos  # Store the validation videos
        self.unified_val_loader = None  # To cache the efficient loader

        # --- MODIFIED: Track GCS path instead of W&B artifact ---
        self.best_model_gcs_path = None  # Track the GCS path of the best model

        # Track the best metric for model saving
        self.best_val_metric = -1.0
        self.best_val_epoch = -1

        # Initialize AMP scaler for mixed precision training
        self.scaler = GradScaler()

        self.speed_up()

        # Checkpoint saving directory is now managed by W&B
        self.log_dir = self.wandb_run.dir if self.wandb_run else './logs'

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

    # --- Helper function to upload a file to GCS ---
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

    # --- Helper function to delete a file from GCS ---
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

    # --- MODIFIED: save_ckpt now uploads to GCS and returns the path ---
    def save_ckpt(self, epoch, auc, eer):
        """
        Saves the model checkpoint locally, uploads it to GCS, and then cleans up.
        Returns the GCS path of the uploaded file.
        """
        gcs_config = self.config.get('checkpointing')
        if not gcs_config or not gcs_config.get('gcs_prefix'):
            self.logger.warning("GCS checkpointing not configured. Skipping upload.")
            return None

        # Create a descriptive checkpoint name
        model_name = self.config.get('model_name', 'model')
        date_str = time.strftime("%Y%m%d")
        ckpt_name = f"ckpt_{model_name}_{date_str}_ep{epoch}_auc{auc:.4f}_eer{eer:.4f}.pth"

        # Save temporarily to local disk (within the W&B run directory)
        local_save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(local_save_dir, exist_ok=True)
        local_save_path = os.path.join(local_save_dir, ckpt_name)

        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        torch.save(model_state, local_save_path)

        # Construct the full GCS path, including the run ID for organization
        gcs_prefix = gcs_config['gcs_prefix']
        if not gcs_prefix.endswith('/'):
            gcs_prefix += '/'
        run_id = self.wandb_run.id if self.wandb_run else "local_run"
        full_gcs_path = os.path.join(gcs_prefix, run_id, ckpt_name)

        # Upload to GCS
        upload_success = self._upload_to_gcs(local_save_path, full_gcs_path)

        # Clean up the local file
        try:
            os.remove(local_save_path)
        except OSError as e:
            self.logger.warning(f"Could not delete local temporary checkpoint: {e}")

        if upload_success:
            return full_gcs_path
        else:
            return None

    def train_step(self, data_dict):
        # --- Use autocast for the forward pass ---
        with autocast():
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)

        self.optimizer.zero_grad()

        # --- Scale the loss and call backward and step via the scaler ---
        self.scaler.scale(losses['overall']).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return losses, predictions

    def _next_batch_from_group(self, method, dataloader_dict, iter_dict):
        it = iter_dict.get(method)
        if it is None:
            it = iter_dict[method] = iter(dataloader_dict[method])
        try:
            return next(it)
        except StopIteration:
            it = iter_dict[method] = iter(dataloader_dict[method])
            return next(it)

    def train_epoch(
            self,
            real_loaders,
            fake_loaders,
            real_method_names,
            fake_method_names,
            real_weights,
            fake_weights,
            epoch,
            epoch_len,
            val_method_loaders=None,
            evaluation_frequency=1  # NEW: frequency parameter
    ):
        self.logger.info(f"===> Epoch[{epoch + 1}] start!")
        # Calculate test step based on frequency
        test_step = epoch_len // evaluation_frequency if evaluation_frequency > 0 else epoch_len
        if test_step == 0: test_step = 1  # Ensure we test at least once if epoch is short

        step_cnt = epoch * epoch_len
        pbar = tqdm(range(epoch_len), desc=f"EPOCH: {epoch + 1}/{self.config['nEpochs']}")

        for iteration in pbar:
            self.setTrain()

            chosen_fake_method = random.choices(fake_method_names, weights=fake_weights, k=1)[0]
            fake_data_dict = self._next_batch_from_group(chosen_fake_method, fake_loaders, self.fake_method_iters)

            chosen_real_method = random.choices(real_method_names, weights=real_weights, k=1)[0]
            real_data_dict = self._next_batch_from_group(chosen_real_method, real_loaders, self.real_method_iters)

            # Combine batches
            data_dict = {}
            # This robustly combines batches, handling Tensors, lists, and None values.
            for key in fake_data_dict.keys():
                f_val, r_val = fake_data_dict[key], real_data_dict[key]
                if torch.is_tensor(f_val):
                    data_dict[key] = torch.cat((f_val, r_val), dim=0)
                elif isinstance(f_val, list):
                    data_dict[key] = f_val + r_val
                else:
                    # This correctly handles the NoneType for 'landmark' and 'mask'
                    data_dict[key] = f_val

            batch_size = data_dict['label'].shape[0]
            if batch_size == 0: continue  # Skip empty batches

            shuffle_indices = torch.randperm(batch_size)
            for key in data_dict.keys():
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key][shuffle_indices]
                elif isinstance(data_dict[key], list):
                    data_dict[key] = [data_dict[key][i] for i in shuffle_indices.tolist()]

            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor): data_dict[key] = data_dict[key].to(self.model.device)

            losses, predictions = self.train_step(data_dict)

            # --- W&B Logging for Training Step ---
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

            step_cnt += 1
            if (iteration + 1) % test_step == 0:
                if val_method_loaders is not None and self.config['local_rank'] == 0:
                    self.logger.info(f"\n===> Evaluation at epoch {epoch + 1}, step {iteration + 1}")
                    self.test_epoch(epoch, val_method_loaders)

    @torch.no_grad()
    def test_epoch(self, epoch, val_method_loaders):
        self.setEval()

        val_iters = {name: iter(val_method_loaders[name]) for name in val_method_loaders.keys()}

        # --- Calculate total batches based on total number of videos and batch size ---
        total_videos = len(self.val_videos)
        batch_size = self.config['test_batchSize']
        # The number of batches is the total videos divided by batch size, rounded up.
        # We add a check for total_videos > 0 to avoid division by zero if val set is empty.
        total_batches = math.ceil(total_videos / batch_size) if total_videos > 0 else 0

        pbar = tqdm(total=total_batches, desc="Validating (Interleaved)")
        # Iterate through each method's DataLoader one by one.
        method_labels = defaultdict(list)
        method_preds = defaultdict(list)

        all_preds, all_labels = [], []

        # --- Loop until all iterators are exhausted ---
        while val_iters:
            # Iterate over a copy of keys, as we will modify the dict
            for method in list(val_iters.keys()):
                try:
                    # --- NEW: Get the next batch from the current method's iterator ---
                    data_dict = next(val_iters[method])

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

                    all_labels.extend(data_dict['label'].cpu().numpy())
                    all_preds.extend(video_probs.cpu().numpy())

                    # Add probabilities and labels by method
                    method_labels[method].extend(data_dict['label'].cpu().numpy())
                    method_preds[method].extend(video_probs.cpu().numpy())

                    # (Future) Here you can accumulate per-method metrics:
                    # per_method_results[method].append(...)

                    pbar.update(1)

                except StopIteration:
                    # This loader is finished, remove it ---
                    del val_iters[method]

        pbar.close()

        if not all_labels:
            self.logger.error("Validation failed: No data was processed after iterating all loaders.")
            return

        self.logger.info("--- Calculating overall validation performance ---")
        overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))

        wandb_log_dict = {"val/epoch": epoch + 1}
        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'val/overall/{name}'] = value
                self.logger.info(f"Overall val {name}: {value:.4f}")

        # Metrics Per Method
        method_metrics = {}
        for method in method_preds.keys():
            method_metrics[method] = get_test_metrics(np.array(method_preds[method]), np.array(method_labels[method]))
            for name, value in method_metrics[method].items():
                if name in ['acc']:
                    wandb_log_dict[f'val/method/{method}/{name}'] = value
                    self.logger.info(f"Method {method} val {name}: {value:.4f}")

        # --- NEW: Create and log a W&B Table for side-by-side comparison ---
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

        # Check for new best model and save
        current_metric = overall_metrics.get(self.metric_scoring)
        if current_metric is not None and current_metric > self.best_val_metric:
            self.best_val_metric = current_metric
            self.best_val_epoch = epoch + 1
            self.logger.info(f"🎉 New best model found! Metric ({self.metric_scoring}): {current_metric:.4f}")

            best_auc = overall_metrics.get('auc', 0.0)
            best_eer = overall_metrics.get('eer', 1.0)
            best_acc = overall_metrics.get('acc', 0.0)

            if self.config['save_ckpt']:
                # --- MODIFIED: Save to GCS and prune the old one ---

                # Store the path of the previous best model to delete it after the new one succeeds
                old_gcs_path = self.best_model_gcs_path

                # Upload the new best model to GCS
                new_gcs_path = self.save_ckpt(
                    epoch=epoch + 1,
                    auc=best_auc,
                    eer=best_eer
                )

                if new_gcs_path:
                    # If upload was successful, update the best path and delete the old one
                    self.best_model_gcs_path = new_gcs_path
                    if old_gcs_path:
                        self._delete_from_gcs(old_gcs_path)

                    # Log the GCS path to W&B
                    if self.wandb_run:
                        self.wandb_run.summary['best_ckpt_gcs'] = new_gcs_path
                        wandb_log_dict['checkpoint/gcs_path'] = new_gcs_path  # Log at current step

                        # Also update the bottom line summary
                        self.wandb_run.summary['bottom_line'] = (
                            f"best_epoch={self.best_val_epoch} AUC={best_auc:.4f} "
                            f"EER={best_eer:.4f} ACC={best_acc:.4f} ckpt=GCS"
                        )

            # Update W&B summary fields for easy sorting
            if self.wandb_run:
                self.wandb_run.summary['best/epoch'] = self.best_val_epoch
                self.wandb_run.summary['best/metric'] = self.best_val_metric
                self.wandb_run.summary['best/auc'] = best_auc
                self.wandb_run.summary['best/eer'] = best_eer
                self.wandb_run.summary['best/acc'] = best_acc

        wandb_log_dict['val/best_metric'] = self.best_val_metric
        wandb_log_dict['val/best_epoch'] = self.best_val_epoch

        if self.wandb_run:
            self.wandb_run.log(wandb_log_dict)

        # Explicitly delete large variables and collect garbage
        del all_preds, all_labels, method_labels, method_preds, overall_metrics, method_metrics
        gc.collect()
        torch.cuda.empty_cache()

        self.logger.info("===> Evaluation Done!")
