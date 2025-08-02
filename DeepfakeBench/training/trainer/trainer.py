import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)


import random
from collections import OrderedDict
import numpy as np # noqa
from tqdm import tqdm # noqa
import torch # noqa
from torch.nn.parallel import DistributedDataParallel as DDP # noqa
from metrics.utils import get_test_metrics # noqa
import wandb # noqa

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
            wandb_run=None  # NEW: Accept wandb run object
    ):
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, and logger must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.wandb_run = wandb_run  # NEW: Store wandb run object

        # NEW: Track best metric for model saving
        self.best_val_metric = -1.0
        self.best_val_epoch = -1

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

    def train_step(self, data_dict):
        predictions = self.model(data_dict)
        if type(self.model) is DDP:
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)
        self.optimizer.zero_grad()
        losses['overall'].backward()
        self.optimizer.step()
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
            data_dict = {key: torch.cat((fake_data_dict[key], real_data_dict[key]), dim=0) if torch.is_tensor(
                fake_data_dict[key]) else fake_data_dict[key] + real_data_dict[key] for key in fake_data_dict.keys()}

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
        wandb_log_dict = {"val/step": epoch + 1}

        self.logger.info("--- Evaluating on individual methods ---")
        for method, loader in tqdm(val_method_loaders.items(), desc="Validating", leave=False):
            if not loader: continue

            method_preds, method_labels, method_paths = [], [], []
            for data_dict in loader:
                for key in data_dict.keys():
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(self.model.device)

                if data_dict['image'].shape[0] == 0: continue

                predictions = self.model(data_dict, inference=True)
                method_labels.extend(data_dict['label'].cpu().numpy())
                method_preds.extend(predictions['prob'].cpu().numpy())
                method_paths.extend(data_dict['path'])

            if not method_labels:
                self.logger.warning(f"No valid data found for validation method: {method}")
                continue

            all_labels.extend(method_labels)
            all_preds.extend(method_preds)

            # Calculate and log metrics for this specific method
            method_metrics = get_test_metrics(np.array(method_preds), np.array(method_labels), method_paths)
            for name, value in method_metrics.items():
                if name not in ['pred', 'label']:
                    wandb_log_dict[f'val_method/{name}/{method}'] = value

        # --- Calculate and log OVERALL validation metrics ---
        if not all_labels:
            self.logger.error("Validation failed: No data was loaded for any validation method.")
            return

        self.logger.info("--- Calculating overall validation performance ---")
        overall_metrics = get_test_metrics(np.array(all_preds), np.array(all_labels))
        for name, value in overall_metrics.items():
            if name not in ['pred', 'label']:
                wandb_log_dict[f'val/overall/{name}'] = value
                self.logger.info(f"Overall val {name}: {value:.4f}")

        # --- NEW: Save Best Model Checkpoint ---
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

        self.logger.info("===> Evaluation Done!")
