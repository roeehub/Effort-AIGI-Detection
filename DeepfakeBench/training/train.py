import argparse
import random
import datetime
import time
import yaml  # noqa
from datetime import timedelta
import math
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch  # noqa
import torch.nn.parallel  # noqa
import torch.backends.cudnn as cudnn  # noqa
import torch.utils.data  # noqa
import torch.optim as optim  # noqa
from torch.utils.data.distributed import DistributedSampler  # noqa
import torch.distributed as dist  # noqa

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR  # noqa
from dataset import *
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
                         num_batches=2,
                         max_paths_to_show=2):
    """
    Parallelized version of the sanity check.
    Scans a handful of batches from every loader in parallel threads.
    """
    print("\n==================== PARALLEL SANITY CHECK ====================")
    start_time = time.time()

    all_loaders = {**fake_loader_dict, **real_loader_dict}

    # This inner function will be run in a separate thread for each loader
    def scan_one_loader(method_name, loader):
        is_fake = method_name in fake_loader_dict
        label_counter = Counter()
        img_shapes = Counter()
        bad_path_count = 0
        example_paths = []

        try:
            it = iter(loader)
            for i in range(num_batches):
                # This is the slow part that will run in parallel
                batch = next(it)

                # --- Perform checks ---
                # 1) labels
                lbls = batch['label']
                label_counter.update(lbls.tolist())

                # 2) image shapes
                img_shapes.update([tuple(batch['image'].shape)])

                # 3) path substring rule
                if 'path' in batch and batch['path']:
                    expect_sub = f"/{method_name}/"
                    # For real sources, the method name might be different (e.g., YouTube-real)
                    # This check is less critical now but kept for consistency
                    for p in batch['path']:
                        # p is now a list of 8 paths, just check the first one
                        p_str = p[0] if isinstance(p, list) else p
                        # Simple check: if fake, method should be in path. If real, it shouldn't be.
                        # This is a loose check and might have false positives.
                        is_in_path = expect_sub in p_str
                        if (is_fake and not is_in_path) or (not is_fake and is_in_path and method_name != 'real'):
                            bad_path_count += 1
                        if len(example_paths) < max_paths_to_show:
                            example_paths.append(p_str)

            # --- Format results ---
            result_str = f"\n[{method_name}]   ({'fake' if is_fake else 'real'}) - OK\n"
            result_str += f"  label counts: {dict(label_counter) or '(loader empty)'}\n"
            result_str += f"  image shapes: {list(img_shapes.keys()) or '(none)'}\n"
            if bad_path_count:
                result_str += f"  ⚠  {bad_path_count} paths failed the substring test\n"
            for sp in example_paths:
                result_str += f"  sample path : {sp}\n"

            return result_str

        except Exception as e:
            return f"\n[{method_name}]   ({'fake' if is_fake else 'real'}) - ❌ FAILED\n  Error: {repr(e)}\n"

    # Use a ThreadPoolExecutor to run scans in parallel
    # max_workers can be tuned, but 10-15 is a good start for network I/O
    with ThreadPoolExecutor(max_workers=15) as executor:
        # Submit all loader scan jobs to the pool
        future_to_loader = {executor.submit(scan_one_loader, name, ldr): name for name, ldr in all_loaders.items()}

        # Process results as they complete
        for future in as_completed(future_to_loader):
            result = future.result()
            print(result, end='')

    total_time = time.time() - start_time
    print(f"\n============ END SANITY CHECK – took {total_time:.2f}s ============\n")


def quick_single_process_check(loader):
    print("\n--- single-process probe ---")
    start = time.time()
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
        print("⏱️  took {:.3f} seconds".format(time.time() - start))
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

    method_loaders, test_method_loaders = create_method_aware_dataloaders(
        train_videos, val_videos, config, data_config
    )

    # we need to separate the real sources from the fake methods
    real_sources = data_config['methods']['use_real_sources']
    real_source_loaders = {}
    for real_source in real_sources:
        if real_source in method_loaders:
            real_source_loaders[real_source] = method_loaders[real_source]
            del method_loaders[real_source]

    # quick_single_process_check(method_loaders['fomm'])  # or any loader
    # breakpoint()

    # after you created method_loaders and real_source_loaders
    sanity_check_loaders(method_loaders, real_source_loaders,
                         num_batches=2,  # scan first 2 batches per loader
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
