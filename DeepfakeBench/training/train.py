import argparse
import random
import datetime
import time
import yaml  # noqa
from datetime import timedelta
import math
import os
from collections import defaultdict, Counter
from tqdm import tqdm  # noqa

import torch  # noqa
import torch.nn.parallel  # noqa
import torch.backends.cudnn as cudnn  # noqa
import torch.utils.data  # noqa
import torch.optim as optim  # noqa
from torch.utils.data.distributed import DistributedSampler  # noqa
import torch.distributed as dist  # noqa
import numpy as np  # noqa
import wandb  # noqa

from trainer.trainer import Trainer
from detectors import DETECTOR  # noqa
from metrics.utils import parse_metric_for_print
from logger import create_logger
from PIL.ImageFilter import RankFilter  # noqa
from prepare_splits import prepare_video_splits
from dataset.dataloaders import create_method_aware_dataloaders, collate_fn  # noqa

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/effort.yaml',
                    help='path to detector YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--no-save_ckpt', dest='save_ckpt', action='store_false', default=True)
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--run_sanity_check', action='store_true', default=False, help="Run the comprehensive sampler check and exit.")
parser.add_argument('--dataloader_config', type=str, default='./training/config/dataloader_config.yml',
                    help='Path to the dataloader configuration file')

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    if config['cuda']:
        torch.manual_seed(config['manualSeed'])
        torch.cuda.manual_seed_all(config['manualSeed'])


def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'adam':
        optimizer = optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
        )
        return optimizer
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_scheduler(config, optimizer):
    if config['lr_scheduler'] is None: return None
    if config['lr_scheduler'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['nEpochs'], eta_min=config['optimizer']['adam']['lr'] / 100
        )
    raise NotImplementedError('Scheduler {} is not implemented'.format(config['lr_scheduler']))


def choose_metric(config):
    metric_scoring = config['metric_scoring']
    if metric_scoring not in ['eer', 'auc', 'acc', 'ap']:
        raise NotImplementedError('metric {} is not implemented'.format(metric_scoring))
    return metric_scoring


def comprehensive_sampler_check(
        real_loaders,
        fake_loaders,
        real_method_names,
        full_batch_size,
        train_videos
):
    """
    Performs a deterministic, one-by-one check of each fake method.
    For each fake method, it attempts to load one batch and pairs it with a
    random real source to verify data integrity.
    """

    def _get_next_batch(method, dataloader_dict, iter_dict):
        """Gets the next batch, creating an iterator if one doesn't exist."""
        if method not in iter_dict:
            iter_dict[method] = iter(dataloader_dict[method])
        try:
            return next(iter_dict[method])
        except StopIteration:
            # Dataloader is exhausted.
            return None

    print("\n\n==================== COMPREHENSIVE SAMPLER CHECK ====================")
    if not fake_loaders or not real_loaders:
        print("❌ No fake or real loaders provided. Aborting check.")
        return

    fake_method_names = sorted(list(fake_loaders.keys()))
    num_fake_methods = len(fake_method_names)
    half_batch_size = full_batch_size // 2
    print(f"Deterministically checking {num_fake_methods} fake methods...")
    print(f"Expecting half-batches of size {half_batch_size}.\n")

    # Use persistent iterators to not re-test the same initial data
    real_method_iters = {}
    fake_method_iters = {}

    # --- Tracking ---
    # We use a simple dictionary to store the status of each fake method
    fake_method_status = {}
    real_source_usage = defaultdict(int)
    batch_times = []

    pbar = tqdm(fake_method_names, desc="Checking Methods")
    for chosen_fake_method in pbar:
        iteration_start_time = time.time()
        pbar.set_postfix_str(f"Testing: {chosen_fake_method}")

        # 1. Attempt to fetch a FAKE half-batch for the current method
        fake_data_dict = _get_next_batch(chosen_fake_method, fake_loaders, fake_method_iters)

        if not fake_data_dict or fake_data_dict['image'].shape[0] == 0:
            fake_method_status[chosen_fake_method] = "❌ EMPTY (Corrupt data or paths)"
            continue
        if fake_data_dict['image'].shape[0] < half_batch_size:
            fake_method_status[chosen_fake_method] = f"⚠️ PARTIAL (Not enough videos for a full batch)"
            continue

        # 2. Attempt to fetch a REAL half-batch to pair with it
        if not real_method_names:
            fake_method_status[chosen_fake_method] = "❌ SKIPPED (No real sources available)"
            continue

        chosen_real_method = random.choice(real_method_names)
        real_data_dict = _get_next_batch(chosen_real_method, real_loaders, real_method_iters)

        # If real source is exhausted, try to restart its iterator ONCE
        if real_data_dict is None:
            real_method_iters[chosen_real_method] = iter(real_loaders[chosen_real_method])
            real_data_dict = _get_next_batch(chosen_real_method, real_loaders, real_method_iters)

        if not real_data_dict or real_data_dict['image'].shape[0] == 0:
            fake_method_status[chosen_fake_method] = f"❌ SKIPPED (Paired real source '{chosen_real_method}' is empty)"
            continue
        if real_data_dict['image'].shape[0] < half_batch_size:
            fake_method_status[
                chosen_fake_method] = f"❌ SKIPPED (Paired real source '{chosen_real_method}' gave partial batch)"
            continue

        # 3. If both are successful
        fake_method_status[chosen_fake_method] = f"✅ OK (Paired with {chosen_real_method})"
        real_source_usage[chosen_real_method] += 1
        batch_times.append(time.time() - iteration_start_time)

    print("\n-------------------- CHECK COMPLETE: DIAGNOSTIC REPORT --------------------")
    total_time = sum(batch_times)
    print(f"Total time for {len(batch_times)} successful pairs: {total_time:.2f} seconds.")
    if batch_times:
        print(f"Avg batch creation time: {np.mean(batch_times):.3f}s | Max: {np.max(batch_times):.3f}s")

    print("\n--- Fake Methods Report ---")
    video_counts = Counter(v.method for v in train_videos)
    print(f"{'Method':<20} | {'Total Vids':>10} | Status & Details")
    print("-" * 85)
    for method in fake_method_names:
        total_vids = video_counts.get(method, 0)
        status = fake_method_status.get(method, "❔ NOT TESTED (Should not happen)")
        print(f"{method:<20} | {total_vids:>10} | {status}")

    print("\n--- Real Sources Usage Report ---")
    print("How many times each real source was successfully used for pairing:")
    for method in sorted(real_method_names):
        count = real_source_usage.get(method, 0)
        print(f"- {method:<20}: {count} times")

    print("\nRecommendations:")
    print("  - For '❌ EMPTY' methods, all videos are likely corrupt. Remove from `dataloader_config.yml`.")
    print(
        "  - For '⚠️ PARTIAL' methods, there aren't enough videos for one half-batch. Consider removing or adding more data.")
    print(
        "  - For '❌ SKIPPED' methods, the issue may be with the real sources, not the fake one. Check real source health.")
    print("================== END COMPREHENSIVE SAMPLER CHECK ==================\n")


def main():
    os.chdir('../')
    # parse options and load config
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config.update(yaml.safe_load(f))
    config['local_rank'] = args.local_rank
    if args.train_dataset: config['train_dataset'] = args.train_dataset
    if args.test_dataset: config['test_dataset'] = args.test_dataset
    config['save_ckpt'] = args.save_ckpt

    dataloader_config_path = args.dataloader_config
    with open(dataloader_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    config.update(data_config) # Merge data_config into config

    # --- NEW: W&B Initialization ---
    # W&B will automatically read entity/project from environment variables
    # (WANDB_ENTITY, WANDB_PROJECT) or your local wandb configuration.
    run_name = f"{config['model_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}"
    wandb_run = wandb.init(
        name=run_name,
        config=config,
        # project="your_project_name", # Optional: Or set WANDB_PROJECT env var
        # entity="your_entity", # Optional: Or set WANDB_ENTITY env var
    )

    # create logger and path
    logger_path = os.path.join(wandb_run.dir, 'logs')  # Save logs inside wandb folder
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info(f'Save log to {logger_path}')
    config['ddp'] = args.ddp

    init_seed(config)

    if config['cudnn']: cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    logger.info("------- Configuration & Data Loading -------")
    train_videos, val_videos, _ = prepare_video_splits(dataloader_config_path)

    train_batch_size = data_config['dataloader_params']['batch_size']
    if train_batch_size % 2 != 0:
        raise ValueError(f"train_batchSize must be even for 50/50 split, but got {train_batch_size}")
    half_batch_size = train_batch_size // 2

    all_train_loaders, val_method_loaders = create_method_aware_dataloaders(
        train_videos, val_videos, config, data_config, train_batch_size=half_batch_size
    )

    real_source_names = data_config['methods']['use_real_sources']
    real_loaders, fake_loaders = {}, {}
    for name, loader in all_train_loaders.items():
        (real_loaders if name in real_source_names else fake_loaders)[name] = loader

    logger.info(
        f"Created {len(real_loaders)} real loaders, {len(fake_loaders)} fake loaders, and {len(val_method_loaders)} validation loaders.")

    if args.run_sanity_check:
        logger.info("--- Running Sanity Check ---")
        # You can still run the check if you want by passing the argument
        comprehensive_sampler_check(...)  # Call the function if needed
        logger.info("Sanity check complete. Halting execution as planned.")
        wandb_run.finish()
        return

    # prepare model, optimizer, scheduler, metric, trainer
    model = DETECTOR[config['model_name']](config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    trainer = Trainer(
        config, model, optimizer, scheduler, logger, metric_scoring,
        wandb_run=wandb_run, val_videos=val_videos
    )

    # --- Training Loop Setup ---
    real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
    for v in train_videos:
        (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1

    total_real_videos = sum(real_video_counts.values())
    total_fake_videos = sum(fake_video_counts.values())

    real_weights = [real_video_counts[m] / total_real_videos for m in
                    real_source_names] if total_real_videos > 0 else []
    fake_method_names = list(fake_loaders.keys())
    fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                    fake_method_names] if total_fake_videos > 0 else []

    total_train_videos = len(train_videos)
    epoch_len = math.ceil(total_train_videos / train_batch_size) if total_train_videos > 0 else 0
    logger.info(f"Total balanced training videos: {total_train_videos}, epoch length: {epoch_len} steps")

    # NEW: Get evaluation frequency from config
    eval_freq = data_config['data_params'].get('evaluation_frequency', 1)

    if config.get('checkpoint_path'):
        trainer.load_ckpt(config['checkpoint_path'])

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.train_epoch(
            real_loaders=real_loaders,
            fake_loaders=fake_loaders,
            real_method_names=real_source_names,
            fake_method_names=fake_method_names,
            real_weights=real_weights,
            fake_weights=fake_weights,
            epoch=epoch,
            epoch_len=epoch_len,
            val_method_loaders=val_method_loaders,
            evaluation_frequency=eval_freq
        )
        if scheduler is not None: scheduler.step()

    wandb_run.finish()
    logger.info("Training complete.")


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    elapsed = end - start
    print(f"Total training time in mn: {elapsed / 60:.2f} minutes")
    print("Training complete.")
