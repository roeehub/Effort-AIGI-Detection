# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
# from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']  #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring='auc',
            time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            swa_model=None
    ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        # self.writers = {}
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf')
            if self.metric_scoring != 'eer' else float('inf'))
        )
        self.speed_up()

        self.timenow = time_now
        if 'task_target' not in config:
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + '_' + self.timenow
            )
        else:
            task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
            self.log_dir = os.path.join(
                self.config['log_dir'],
                self.config['model_name'] + task_str + '_' + self.timenow
            )
        os.makedirs(self.log_dir, exist_ok=True)

        self.real_method_iters = {}
        self.fake_method_iters = {}

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] == True:
            self.model = DDP(self.model, device_ids=[self.config['local_rank']], find_unused_parameters=True,
                             output_device=self.config['local_rank'])

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            new_state_dict = OrderedDict()
            for k, v in saved.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise FileNotFoundError(f"=> no model found at '{model_path}'")

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        model_state = self.model.module.state_dict() if self.config['ddp'] else self.model.state_dict()
        torch.save(model_state, save_path)

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
        """Return the next batch for `method`, restarting the iterator if needed."""
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
            test_data_loaders=None,
    ):
        self.logger.info(f"===> Epoch[{epoch}] start!")
        times_per_epoch = 2
        test_step = epoch_len // times_per_epoch if times_per_epoch > 0 else epoch_len
        step_cnt = epoch * epoch_len

        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration in tqdm(range(epoch_len), desc=f"EPOCH: {epoch + 1}"):
            self.setTrain()

            # 1. Get a half-batch of FAKE data
            chosen_fake_method = random.choices(fake_method_names, weights=fake_weights, k=1)[0]
            fake_data_dict = self._next_batch_from_group(chosen_fake_method, fake_loaders, self.fake_method_iters)

            # 2. Get a half-batch of REAL data
            chosen_real_method = random.choices(real_method_names, weights=real_weights, k=1)[0]
            real_data_dict = self._next_batch_from_group(chosen_real_method, real_loaders, self.real_method_iters)

            # 3. Combine them into a single batch
            data_dict = {}
            for key in fake_data_dict.keys():
                if torch.is_tensor(fake_data_dict[key]):
                    data_dict[key] = torch.cat((fake_data_dict[key], real_data_dict[key]), dim=0)
                elif isinstance(fake_data_dict[key], list):
                    data_dict[key] = fake_data_dict[key] + real_data_dict[key]
                else:
                    data_dict[key] = fake_data_dict[key]

            # 4. Shuffle the combined batch to mix reals and fakes
            batch_size = data_dict['label'].shape[0]
            shuffle_indices = torch.randperm(batch_size)
            for key in data_dict.keys():
                if torch.is_tensor(data_dict[key]):
                    data_dict[key] = data_dict[key][shuffle_indices]
                elif isinstance(data_dict[key], list):
                    data_dict[key] = [data_dict[key][i] for i in shuffle_indices.tolist()]

            # Move data to GPU
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.model.device)

            losses, predictions = self.train_step(data_dict)

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items(): train_recorder_metric[name].update(value)
            for name, value in losses.items(): train_recorder_loss[name].update(value)

            if iteration % 300 == 0 and self.config['local_rank'] == 0:
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items(): loss_str += f"training-loss, {k}: {v.average()}    "
                self.logger.info(loss_str)
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items(): metric_str += f"training-metric, {k}: {v.average()}    "
                self.logger.info(metric_str)
                for recorder in train_recorder_loss.values(): recorder.clear()
                for recorder in train_recorder_metric.values(): recorder.clear()

            if (step_cnt + 1) % test_step == 0:
                if test_data_loaders is not None and (not self.config['ddp'] or dist.get_rank() == 0):
                    self.logger.info("===> Test start!")
                    self.test_epoch(epoch, iteration, test_data_loaders, step_cnt)
            step_cnt += 1

    def test_one_dataset(self, data_loader):
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists, label_lists, feature_lists, path_lists = [], [], [], []

        for data_dict in tqdm(data_loader, desc="Testing", leave=False):
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].to(self.model.device)

            predictions = self.inference(data_dict)
            label_lists.extend(data_dict['label'].cpu().numpy())
            prediction_lists.extend(predictions['prob'].cpu().numpy())
            feature_lists.extend(predictions['feat'].cpu().numpy())
            path_lists.extend(data_dict['path'])

            with torch.no_grad():
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)
                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists), np.array(
            feature_lists), path_lists

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()
        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}

        for key, loader in test_data_loaders.items():
            _, predictions_nps, label_nps, _, path_nps = self.test_one_dataset(loader)
            metric_one_dataset = get_test_metrics(y_pred=predictions_nps, y_true=label_nps, img_names=path_nps)

            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name] += value
            avg_metric['dataset_dict'][key] = metric_one_dataset[self.metric_scoring]
            # self.save_best(...) logic can be called here if needed
            metric_str = f"dataset: {key}    step: {step}    "
            for k, v in metric_one_dataset.items():
                if k not in ['pred', 'label', 'dataset_dict']:
                    metric_str += f"testing-metric, {k}: {v:.4f}    "
            self.logger.info(metric_str)

        self.logger.info('===> Test Done!')

    @torch.no_grad()
    def inference(self, data_dict):
        return self.model(data_dict, inference=True)