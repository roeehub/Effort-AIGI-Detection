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


# in train.py

def comprehensive_sampler_check(
        real_loaders,
        fake_loaders,
        real_method_names,
        fake_method_names,
        real_weights,
        fake_weights,
        full_batch_size
):
    """
    Simulates the training loop's data fetching to test performance and correctness.
    Ensures all dataloaders are used and combined batches are correctly formed.
    """

    # --- FIX: Define the helper function inside this function's scope ---
    def _get_next_batch(method, dataloader_dict, iter_dict):
        """Helper to fetch a batch from a specific dataloader, creating an iterator if needed."""
        it = iter_dict.get(method)
        if it is None:
            print(f"\n   ... Initializing iterator for '{method}' ...")
            it = iter_dict[method] = iter(dataloader_dict[method])
        try:
            return next(it)
        except StopIteration:
            print(f"\n   ... Restarting exhausted iterator for '{method}' ...")
            it = iter_dict[method] = iter(dataloader_dict[method])
            return next(it)

    print("\n==================== COMPREHENSIVE SAMPLER CHECK ====================")
    if not fake_loaders:
        print("❌ No fake loaders provided. Aborting check.")
        return

    num_iterations = len(fake_loaders)
    half_batch_size = full_batch_size // 2
    print(f"Running for {num_iterations} iterations to ensure all {len(fake_loaders)} fake loaders are likely sampled.")
    print(f"Expecting combined batches of size {full_batch_size} ({half_batch_size} real + {half_batch_size} fake).\n")

    # Mimic the state maintained by the Trainer
    real_method_iters = {}
    fake_method_iters = {}

    # Tracking for the report
    seen_real_methods = set()
    seen_fake_methods = set()
    batch_times = []

    overall_start_time = time.time()

    pbar = tqdm(range(num_iterations), desc="Testing Sampler")
    for i in pbar:
        iteration_start_time = time.time()

        # 1. Sample and fetch a FAKE half-batch
        chosen_fake_method = random.choices(fake_method_names, weights=fake_weights, k=1)[0]
        fake_data_dict = _get_next_batch(chosen_fake_method, fake_loaders, fake_method_iters)
        seen_fake_methods.add(chosen_fake_method)

        # 2. Sample and fetch a REAL half-batch
        chosen_real_method = random.choices(real_method_names, weights=real_weights, k=1)[0]
        real_data_dict = _get_next_batch(chosen_real_method, real_loaders, real_method_iters)
        seen_real_methods.add(chosen_real_method)

        # 3. Combine into a single batch dictionary
        combined_data_dict = {}
        for key in fake_data_dict.keys():
            if torch.is_tensor(fake_data_dict[key]):
                combined_data_dict[key] = torch.cat((fake_data_dict[key], real_data_dict[key]), dim=0)
            elif isinstance(fake_data_dict[key], list):
                combined_data_dict[key] = fake_data_dict[key] + real_data_dict[key]
            else:
                combined_data_dict[key] = fake_data_dict[key]

        # 4. Perform checks on the combined batch
        image_shape = combined_data_dict['image'].shape
        assert image_shape[0] == full_batch_size, f"Expected batch size {full_batch_size}, but got {image_shape[0]}"

        label_counts = Counter(combined_data_dict['label'].tolist())
        assert label_counts.get(0,
                                0) == half_batch_size, f"Expected {half_batch_size} real samples, but got {label_counts.get(0, 0)}"
        assert label_counts.get(1,
                                0) == half_batch_size, f"Expected {half_batch_size} fake samples, but got {label_counts.get(1, 0)}"

        iteration_time = time.time() - iteration_start_time
        batch_times.append(iteration_time)

        pbar.set_description(
            f"Batch {i + 1}/{num_iterations} | Last: {iteration_time:.2f}s | Real: {chosen_real_method} | Fake: {chosen_fake_method}")

    overall_time = time.time() - overall_start_time

    print("\n-------------------- CHECK COMPLETE: REPORT --------------------")
    print(f"Total time for {num_iterations} batches: {overall_time:.2f} seconds.")
    if batch_times:
        print(f"Batch creation time (sec):")
        print(f"  - Average: {np.mean(batch_times):.3f}")
        print(f"  - Min:     {np.min(batch_times):.3f}")
        print(f"  - Max:     {np.max(batch_times):.3f}")

    # Coverage Report
    print("\nLoader Coverage:")
    all_fakes_seen = set(fake_loaders.keys()) == seen_fake_methods
    print(f"  - Fake Methods: {'✅' if all_fakes_seen else '❌'} Seen {len(seen_fake_methods)}/{len(fake_loaders)}")
    if not all_fakes_seen:
        missing = set(fake_loaders.keys()) - seen_fake_methods
        print(f"    - Missing: {missing}")

    all_reals_seen = set(real_loaders.keys()) == seen_real_methods
    print(f"  - Real Sources: {'✅' if all_reals_seen else '❌'} Seen {len(seen_real_methods)}/{len(real_loaders)}")
    if not all_reals_seen:
        missing = set(real_loaders.keys()) - seen_real_methods
        print(f"    - Missing: {missing}")

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

    # create logger
    logger_path = os.path.join(
        config['log_dir'], config['model_name'] + '_' + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    logger.info('Save log to {}'.format(logger_path))
    config['ddp'] = args.ddp

    with open("./training/config/dataloader_config.yml", 'r') as f:
        data_config = yaml.safe_load(f)

    init_seed(config)

    if config['cudnn']: cudnn.benchmark = True
    if config['ddp']:
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        logger.addFilter(RankFilter(0))

    # --- pretty print a bunch of configuration data for clarity ---
    logger.info("------- Configuration: -------")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("Data Configuration:")
    for key, value in data_config.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"  {sub_key}: {sub_value}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("------- Starting training process ---")

    train_videos, val_videos, _ = prepare_video_splits('./training/config/dataloader_config.yml')

    # --- 50/50 BATCH SETUP ---
    train_batch_size = data_config['dataloader_params']['batch_size']
    if train_batch_size % 2 != 0:
        raise ValueError(f"train_batchSize must be even for 50/50 split, but got {train_batch_size}")
    half_batch_size = train_batch_size // 2
    logger.info(f"Using 50/50 real/fake split. Half-batch size: {half_batch_size}.")

    all_train_loaders, test_method_loaders = create_method_aware_dataloaders(
        train_videos, val_videos, config, data_config, train_batch_size=half_batch_size
    )

    real_source_names = data_config['methods']['use_real_sources']
    real_loaders, fake_loaders = {}, {}
    for name, loader in all_train_loaders.items():
        (real_loaders if name in real_source_names else fake_loaders)[name] = loader

    logger.info(f"Created {len(real_loaders)} real loaders and {len(fake_loaders)} fake loaders.")

    real_video_counts, fake_video_counts = defaultdict(int), defaultdict(int)
    for v in train_videos:
        (real_video_counts if v.method in real_source_names else fake_video_counts)[v.method] += 1

    total_real_videos = sum(real_video_counts.values())
    total_fake_videos = sum(fake_video_counts.values())

    if total_real_videos != total_fake_videos and total_real_videos > 0 and total_fake_videos > 0:
        logger.warning(
            f"Mismatch after balancing: {total_real_videos} real vs {total_fake_videos} fake. Balance not perfect."
        )

    real_method_names = list(real_loaders.keys())
    real_weights = [real_video_counts[m] / total_real_videos for m in
                    real_method_names] if total_real_videos > 0 else []

    fake_method_names = list(fake_loaders.keys())
    fake_weights = [fake_video_counts[m] / total_fake_videos for m in
                    fake_method_names] if total_fake_videos > 0 else []

    total_train_videos = len(train_videos)
    epoch_len = math.ceil(total_train_videos / train_batch_size)
    logger.info(f"Total balanced training videos: {total_train_videos}, epoch length: {epoch_len} steps")

    # --- Call the new sanity check ---
    comprehensive_sampler_check(
        real_loaders=real_loaders,
        fake_loaders=fake_loaders,
        real_method_names=real_method_names,
        fake_method_names=fake_method_names,
        real_weights=real_weights,
        fake_weights=fake_weights,
        full_batch_size=train_batch_size
    )

    logger.info("Sanity check complete. Halting execution as planned.")
    return  # Stop the script here after the check

    # prepare model, optimizer, scheduler, metric, trainer
    model = DETECTOR[config['model_name']](config)
    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring)

    if config.get('checkpoint_path'):
        trainer.load_ckpt(config['checkpoint_path'])

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs']):
        trainer.train_epoch(
            real_loaders=real_loaders,
            fake_loaders=fake_loaders,
            real_method_names=real_method_names,
            fake_method_names=fake_method_names,
            real_weights=real_weights,
            fake_weights=fake_weights,
            epoch=epoch,
            epoch_len=epoch_len,
            test_data_loaders=test_method_loaders,
        )
        if scheduler is not None: scheduler.step()


if __name__ == '__main__':
    main()
