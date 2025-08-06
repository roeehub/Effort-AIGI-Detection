import os
import sys
import time
import shutil

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
import functools
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

        # NEW: Track best metric for model saving
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

    def save_ckpt(self, epoch):
        save_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        torch.save(model_state, save_path)
        self.logger.info(f"Saved best model checkpoint at epoch {epoch} to {save_path}")
        # NEW: Also save as a W&B artifact
        if self.wandb_run:
            artifact = wandb.Artifact(f'{self.wandb_run.name}-best-model', type='model')
            artifact.add_file(save_path)
            self.wandb_run.log_artifact(artifact)
            
            # Wait for the checkpoint to be uploaded to the W&B cloud
            time.sleep(5)
            try:
                os.remove(save_path)
                self.logger.info(f"Deleted local checkpoint: {save_path}")
            except Exception as e:
                self.logger.warning(f"Failed to delete local checkpoint: {e}")
            # Clean W&B staging cache (optional but safe)
            wandb_cache_dir = os.path.expanduser("~/.local/share/wandb/artifacts/staging/")
            try:
                shutil.rmtree(wandb_cache_dir)
                self.logger.info(f"Cleaned W&B staging cache: {wandb_cache_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean W&B staging cache: {e}")

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
        # NEW: Calculate test step based on frequency
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

            # --- NEW: W&B Logging for Training Step ---
            if self.wandb_run and self.config['local_rank'] == 0:
                log_dict = {"train/step": step_cnt, "epoch": epoch + 1}
                for name, value in losses.items():
                    log_dict[f'loss/train/{name}'] = value.item()

                if type(self.model) is DDP:
                    batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
                else:
                    batch_metrics = self.model.get_train_metrics(data_dict, predictions)

                for name, value in batch_metrics.items():
                    log_dict[f'metric/train/{name}'] = value

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

        all_preds, all_labels = [], []
        # The 'desc' provides a description for the progress bar.
        pbar = tqdm(val_method_loaders.items(), desc="Validating (Slow but Stable)")
        
        
        # Iterate through each method's DataLoader one by one.
        method_labels = defaultdict(list)
        method_preds = defaultdict(list)
        for method, loader in pbar:
            for data_dict in loader:
                # Move tensors to the correct device
                for key, value in data_dict.items():
                    if isinstance(value, torch.Tensor):
                        data_dict[key] = value.to(self.model.device)

                # Skip empty batches
                if data_dict['image'].shape[0] == 0: continue
                # Skip batches with incorrect dimensions
                if data_dict['image'].dim() != 5: continue

                B, T = data_dict['image'].shape[:2]

                predictions = self.model(data_dict, inference=True)

                # Calculate video-level probability by averaging frame scores
                video_probs = predictions['prob'].view(B, T).mean(dim=1)

                all_labels.extend(data_dict['label'].cpu().numpy())
                all_preds.extend(video_probs.cpu().numpy())
                
                # Add probabilities and labels by method
                method_labels[method].extend(data_dict['label'].cpu().numpy())
                method_preds[method].extend(video_probs.cpu().numpy())
                

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
                    wandb_log_dict[f'val/{method}/{name}'] = value
                    self.logger.info(f"Method {method} val {name}: {value:.4f}")

        # Check for new best model and save
        current_metric = overall_metrics.get(self.metric_scoring)
        if current_metric is not None and current_metric > self.best_val_metric:
            self.best_val_metric = current_metric
            self.best_val_epoch = epoch + 1
            self.logger.info(f"🎉 New best model found! Metric ({self.metric_scoring}): {current_metric:.4f}")
            if self.config['save_ckpt']:
                self.save_ckpt(epoch + 1)

        wandb_log_dict['val/best_metric'] = self.best_val_metric
        wandb_log_dict['val/best_epoch'] = self.best_val_epoch

        if self.wandb_run:
            self.wandb_run.log(wandb_log_dict)
        
        ##################### ADAM CHANGED ###########################
        # Explicitly delete large variables and collect garbage
        del all_preds, all_labels, method_labels, method_preds, overall_metrics, method_metrics
        gc.collect()
        torch.cuda.empty_cache()
        ##################### ADAM CHANGED ###########################

        self.logger.info("===> Evaluation Done!")
