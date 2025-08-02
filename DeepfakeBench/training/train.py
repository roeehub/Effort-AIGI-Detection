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
        full_batch_size,
        train_videos
):
    """
    Simulates data fetching to test performance, correctness, and data integrity.
    Provides a detailed report on which methods are successfully providing data
    and which are failing due to data corruption.
    """

    def _get_next_batch(method, dataloader_dict, iter_dict):
        it = iter_dict.get(method)
        if it is None:
            it = iter_dict[method] = iter(dataloader_dict[method])
        try:
            return next(it)
        except StopIteration:
            # Dataloader is exhausted. For this check, we don't restart it
            # to see if it runs out of data prematurely.
            return None  # Signal exhaustion

    print("\n\n==================== COMPREHENSIVE SAMPLER CHECK ====================")
    if not fake_loaders or not real_loaders:
        print("❌ No fake or real loaders provided. Aborting check.")
        return

    # Run for more iterations to ensure even low-weight methods are likely sampled
    num_iterations = max(200, len(fake_loaders) * 10)
    half_batch_size = full_batch_size // 2
    print(f"Simulating {num_iterations} batch creation steps...")
    print(f"Expecting combined batches of size {full_batch_size} ({half_batch_size} real + {half_batch_size} fake).\n")

    # Mimic Trainer state
    real_method_iters = {name: iter(loader) for name, loader in real_loaders.items()}
    fake_method_iters = {name: iter(loader) for name, loader in fake_loaders.items()}

    # --- Enhanced Tracking ---
    successful_batches = defaultdict(int)
    empty_batches = defaultdict(int)
    batch_times = []
    video_counts = Counter(v.method for v in train_videos)
    all_methods_in_check = set(real_loaders.keys()) | set(fake_loaders.keys())

    pbar = tqdm(range(num_iterations), desc="Testing Sampler")
    for i in pbar:
        iteration_start_time = time.time()

        # 1. Sample and fetch a FAKE half-batch
        chosen_fake_method = random.choices(fake_method_names, weights=fake_weights, k=1)[0]
        fake_data_dict = _get_next_batch(chosen_fake_method, fake_loaders, fake_method_iters)

        if not fake_data_dict or fake_data_dict['image'].shape[0] == 0:
            empty_batches[chosen_fake_method] += 1
            fake_method_iters[chosen_fake_method] = iter(fake_loaders[chosen_fake_method])  # Restart for next attempt
            continue

        # 2. Sample and fetch a REAL half-batch
        chosen_real_method = random.choices(real_method_names, weights=real_weights, k=1)[0]
        real_data_dict = _get_next_batch(chosen_real_method, real_loaders, real_method_iters)

        if not real_data_dict or real_data_dict['image'].shape[0] == 0:
            empty_batches[chosen_real_method] += 1
            real_method_iters[chosen_real_method] = iter(real_loaders[chosen_real_method])  # Restart
            continue

        successful_batches[chosen_fake_method] += 1
        successful_batches[chosen_real_method] += 1

        # 3. Combine and check the batch
        combined_labels = torch.cat((real_data_dict['label'], fake_data_dict['label']), dim=0)
        assert combined_labels.shape[0] == full_batch_size, f"Batch size mismatch!"
        label_counts = Counter(combined_labels.tolist())
        assert label_counts.get(0, 0) == half_batch_size, f"Real sample count mismatch!"
        assert label_counts.get(1, 0) == half_batch_size, f"Fake sample count mismatch!"

        batch_times.append(time.time() - iteration_start_time)
        pbar.set_postfix_str(f"OK | Real: {chosen_real_method}, Fake: {chosen_fake_method}")

    print("\n-------------------- CHECK COMPLETE: DIAGNOSTIC REPORT --------------------")
    print(f"Total time for {num_iterations} simulated batches: {sum(batch_times):.2f} seconds.")
    if batch_times:
        print(f"Avg batch creation time: {np.mean(batch_times):.3f}s | Max: {np.max(batch_times):.3f}s")

    print("\n--- Data Source Health ---")
    print(f"{'Method':<20} | {'Total Videos':>12} | {'Successful Batches':>20} | {'Empty/Failed Batches':>22}")
    print("-" * 80)

    for method in sorted(list(all_methods_in_check)):
        total_vids = video_counts.get(method, 0)
        success = successful_batches.get(method, 0)
        failed = empty_batches.get(method, 0)

        status = "✅ OK"
        if success == 0 and failed > 0:
            status = "❌ TOTAL FAILURE"
        elif failed > 0:
            status = f"⚠️ PARTIAL FAILURE ({failed / (success + failed):.0%} failed)"

        print(f"{method:<20} | {total_vids:>12} | {success:>20} | {failed:>22} | {status}")

    print("\nRecommendations:")
    print(
        "  - For methods with 'TOTAL FAILURE', all videos are likely corrupt or missing. Consider removing them from `dataloader_config.yml`.")
    print(
        "  - For methods with 'PARTIAL FAILURE', some videos are corrupt. The robust loader is handling this, but be aware of the data loss.")
    print(
        "  - If a method shows 0 successful and 0 failed batches, it was likely never sampled. Check its weight/video count.")
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

    logger.info("------- Configuration & Data Loading -------")
    train_videos, val_videos, _ = prepare_video_splits('./training/config/dataloader_config.yml')

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
        full_batch_size=train_batch_size,
        train_videos=train_videos  # Pass video info for reporting
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
