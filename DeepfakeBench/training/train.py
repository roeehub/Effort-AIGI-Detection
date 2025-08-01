# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image
import re
from collections import Counter, defaultdict
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger
from PIL.ImageFilter import RankFilter
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from prepare_splits import prepare_video_splits
from dataset.dataloaders import create_method_aware_dataloaders

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument('--no-save_feat', dest='save_feat', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def prepare_training_data(config, train_videos):
    # Only use the blending dataset class in training
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
        train_videos=train_videos
    )
    if config['ddp']:
        sampler = DistributedSampler(train_set)
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
                sampler=sampler
            )
    else:
        train_data_loader = \
            torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=config['train_batchSize'],
                shuffle=True,
                num_workers=int(config['workers']),
                collate_fn=train_set.collate_fn,
            )
    return train_data_loader


def prepare_testing_data(config, val_videos):
    def get_test_data_loader(config, test_name, val_videos):
        # update the config dictionary with the specific testing dataset
        config = config.copy()  # create a copy of config to avoid altering the original one
        config['test_dataset'] = test_name  # specify the current test dataset

        test_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode='test',
            VideoInfo=val_videos,
        )

        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=config['test_batchSize'],
                shuffle=False,
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                drop_last=(test_name == 'DeepFakeDetection'),
            )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name, val_videos)
    return test_data_loaders


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None:
        return None
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min'],
        )
        return scheduler
    elif config['lr_scheduler'] == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            config['nEpochs'],
            int(config['nEpochs'] / 4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


# ================================================================
#  Sanity-check utility for DataLoaders
# ================================================================


def sanity_check_loaders(fake_loader_dict,
                         real_loader_dict,
                         num_batches=3,
                         max_paths_to_show=2):
    """
    Scan a handful of batches from every loader and print:

      • label distribution   (expect all-1 for fake loaders, all-0 for real)
      • unique tensor shapes (image, mask, landmark)
      • #paths whose string does *not* match the method rule
      • example frame paths

    Args
    ----
    fake_loader_dict : dict[str, DataLoader]
        keys are fake method names
    real_loader_dict : dict[str, DataLoader]
        keys are real-source names
    num_batches : int
        how many batches to read per loader
    max_paths_to_show : int
        how many sample paths to print per loader
    """

    def scan(loader, expect_fake, method_name):
        """Return stats gathered from ≤ num_batches of `loader`."""
        label_counter = Counter()
        img_shapes = Counter()
        bad_path_count = 0
        example_paths = []

        it = iter(loader)  # fresh iterator every call
        for _ in range(num_batches):
            try:
                batch = next(it)
            except StopIteration:
                break

            # 1) labels
            lbls = batch['label']
            label_counter.update(lbls.tolist())

            # 2) image shapes
            img_shapes.update([tuple(batch['image'].shape)])

            # 3) path substring rule
            if 'path' in batch:
                expect_sub = f"/{method_name}/"
                for p_group in batch['path']:
                    for p in p_group:  # p is now a str
                        cond = (expect_fake and expect_sub not in p) or \
                               (not expect_fake and expect_sub in p)
                        if cond:
                            bad_path_count += 1
                        if len(example_paths) < max_paths_to_show:
                            example_paths.append(p)

            # 4) masks / landmarks just type-check
            assert isinstance(batch['mask'], (type(None), torch.Tensor))
            assert isinstance(batch['landmark'], (type(None), torch.Tensor))

        return label_counter, img_shapes, bad_path_count, example_paths[:max_paths_to_show]

    print("\n==================== SANITY CHECK ====================")

    all_keys = list(fake_loader_dict.keys()) + list(real_loader_dict.keys())
    for method in all_keys:
        is_fake_loader = method in fake_loader_dict
        loader = fake_loader_dict[method] if method in fake_loader_dict else real_loader_dict[method]
        if loader is None:
            print(f"[{method}]  ⚠  loader missing – skipped")
            continue

        labels, shapes, bad_paths, samples = scan(
            loader,
            expect_fake=is_fake_loader,
            method_name=method
        )

        print(f"\n[{method}]   ({'fake' if is_fake_loader else 'real'})")
        print("  label counts:", dict(labels) or "(loader empty)")
        print("  image shapes:", list(shapes.keys()) or "(none)")
        if bad_paths:
            print(f"  ⚠  {bad_paths} paths failed the substring test")
        for sp in samples:
            print("  sample path :", sp)

    print("\n============ END SANITY CHECK – review above =========\n")


def quick_single_process_check(loader):
    print("\n--- single-process probe ---")
    sp_loader = torch.utils.data.DataLoader(
        loader.dataset,  # same dataset/DataPipe
        batch_size=loader.batch_size,
        num_workers=0,  # ← no subprocesses
        collate_fn=loader.collate_fn
    )
    try:
        batch = next(iter(sp_loader))
        print("✅  got batch:", {k: v.shape if torch.is_tensor(v) else type(v)
                                for k, v in batch.items()})
    except Exception as e:
        print("❌  raised:", repr(e))
        raise


def check_label_in_paths(video_infos, split_name):
    bad_videos = []
    for v in video_infos:
        lbl_tag = f"/{v.label}/"  # "/real/" or "/fake/"
        # every path should contain this tag
        if any(lbl_tag not in p for p in v.frame_paths):
            bad_videos.append(f"{v.method}/{v.video_id}")

    if bad_videos:
        print(f"[{split_name}] ❌  {len(bad_videos)} videos failed label-in-path check")
        # show first few for inspection
        for vid in bad_videos[:5]:
            print("   →", vid)
    else:
        print(f"[{split_name}] ✅  all {len(video_infos)} videos have consistent paths")


def main():
    print("We are in Before:", os.getcwd())
    os.chdir('../')
    print("We are in:", os.getcwd())
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    config['local_rank'] = args.local_rank
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat'] = False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt
    config['save_feat'] = args.save_feat
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    # create logger
    logger_path = os.path.join(
        config['log_dir'],
        config['model_name'] + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp'] = args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    config_path = "./training/config/dataloader_config.yml"
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # init seed
    init_seed(config)

    # set cudnn benchmark if needed
    if config['cudnn']:
        cudnn.benchmark = True
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=30)
        )
        logger.addFilter(RankFilter(0))
    # Split the dataset
    # Load train/val splits using prepare_video_splits
    train_videos, val_videos, _ = prepare_video_splits('./training/config/dataloader_config.yml')

    check_label_in_paths(train_videos, "train")
    check_label_in_paths(val_videos, "val")

    # Create a dataset object to include all the instances of the dataset to load
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='train',
        VideoInfo=train_videos,
    )

    test_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode='test',
        VideoInfo=val_videos,
    )

    # returns a method aware dataloader - A dictionary with keys as methods and values as their dataloader
    method_loaders = create_method_aware_dataloaders(train_set, data_config)
    test_method_loaders = create_method_aware_dataloaders(test_set, data_config, config=config, test=True)

    # we need to separate the real sources from the fake methods
    real_sources = data_config['methods']['use_real_sources']
    real_source_loaders = {}
    for real_source in real_sources:
        if real_source in method_loaders:
            real_source_loaders[real_source] = method_loaders[real_source]
            del method_loaders[real_source]

    quick_single_process_check(method_loaders['sadtalker'])  # or any loader
    breakpoint()

    # after you created method_loaders and real_source_loaders
    sanity_check_loaders(method_loaders, real_source_loaders,
                         num_batches=3,  # scan first 3 batches per loader
                         max_paths_to_show=2)  # print 2 example paths

    breakpoint()

    # num_fake_videos = sum(len(dl.dataset) for dl in method_loaders.values())
    # num_real_videos = sum(len(dl.dataset) for dl in real_source_loaders.values())

    # Count videos per fake method
    video_counts = defaultdict(int)
    for v in train_videos:  # train_videos is List[VideoInfo]
        video_counts[v.method] += 1
    total_videos = len(train_videos)

    # Compute weights for random method choice
    method_names = list(method_loaders.keys())
    weights = [video_counts[m] / total_videos for m in method_names]

    # Compute epoch length: number of steps to roughly see all videos once
    batch_size = data_config['dataloader_params']['batch_size']
    epoch_len = math.ceil(total_videos / batch_size)

    print(f"Total videos: {total_videos}, batch size: {batch_size}, epoch_len: {epoch_len}")

    # Training Loop for Method-Aware
    # to cycle through methods:
    method_names = list(method_loaders.keys())
    random.shuffle(method_names)
    print(f"Training on methods in this order: {method_names[:5]}...")

    # For a balanced approach, we create a weighted sampler over `method_names`
    method_sizes = {m: len(dl.dataset) for m, dl in method_loaders.items()}
    total_videos = sum(method_sizes.values())
    weights = [method_sizes[m] / total_videos for m in method_names]

    print("\n--- Example: Method-Aware BALANCED Training Loop ---")
    # In each step, you'd pick a method based on weights and get a batch from it
    # This requires creating iterators for each dataloader.
    # method_iters = {name: iter(loader) for name, loader in method_loaders.items()}

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config, val_videos)

    # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    scheduler = choose_scheduler(config, optimizer)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    # Define the path to your pretrained checkpoint
    checkpoint_path = './training/weights/effort_clip_L14_trainOn_FaceForensic.pth'
    # Check if a path is provided and then load the checkpoint
    if checkpoint_path:
        trainer.load_ckpt(checkpoint_path)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        trainer.model.epoch = epoch
        best_metric = trainer.train_epoch(
            method_names=method_names,  # A list of all the methods
            weights=weights,
            epoch=epoch,  # A dictionary of methods as keys and dataloaders iterators as values
            dataloader_dict=method_loaders,  # A dictionary of methods as keys and dataloaders as values
            epoch_len=epoch_len,  # The number of steps in an epoch
            train_set=train_set,  # The class dataset of test
            test_set=test_set,  # The class dataset of test
            test_data_loaders=test_method_loaders,  # Usual dataloader for the test
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!")
    logger.info("Stop Training on best Testing metric {}".format(parse_metric_for_print(best_metric)))
    # update
    if 'svdd' in config['model_name']:
        model.update_R(epoch)
    if scheduler is not None:
        scheduler.step()

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == '__main__':
    main()
