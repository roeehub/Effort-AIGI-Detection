import torch
from torch.utils.data import IterDataPipe
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
import random
import fsspec
from PIL import Image
from torchvision import transforms as T
import numpy as np
from copy import deepcopy
from collections import defaultdict
import albumentations as A
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from itertools import filterfalse


class DeepfakePipeDataset(IterDataPipe):
    def __init__(self, dataset_class: DeepfakeAbstractBaseDataset):
        self.dataset = dataset_class
        self.image_list = dataset_class.image_list
        self.label_list = dataset_class.label_list
        self.config = dataset_class.config
        self.mode = dataset_class.mode
        self.frame_num = dataset_class.frame_num
        self.video_level = dataset_class.video_level

    def __iter__(self):
        for index in range(len(self.dataset)):
            yield self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        return DeepfakeAbstractBaseDataset.collate_fn(batch)


def data_aug(img, landmark=None, mask=None, augmentation_seed=None):
    """
    Apply data augmentation to an image, landmark, and mask.

    Args:
        img: An Image object containing the image to be augmented.
        landmark: A numpy array containing the 2D facial landmarks to be augmented.
        mask: A numpy array containing the binary mask to be augmented.

    Returns:
        The augmented image, landmark, and mask.
    """

    # Set the seed for the random number generator
    if augmentation_seed is not None:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed)

    # Define augmentation pipeline (from init_data_aug_method)
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(p=0.3),
        A.ImageCompression(quality_lower=40, quality_upper=100, p=0.1),
        A.GaussNoise(p=0.1),
        A.MotionBlur(p=0.1),
        A.CLAHE(p=0.1),
        A.ChannelShuffle(p=0.1),
        A.Cutout(p=0.1),
        A.RandomGamma(p=0.3),
        A.GlassBlur(p=0.3),
    ])

    # Create a dictionary of arguments
    kwargs = {'image': img}

    # Check if the landmark and mask are not None
    if mask is not None:
        kwargs['mask'] = mask

    # Run transform
    transformed = transform(**kwargs)

    # Extract results
    augmented_img = transformed['image']
    augmented_mask = transformed.get('mask')
    augmented_landmark = None  # Not used here

    # Reset seed (optional)
    if augmentation_seed is not None:
        random.seed()
        np.random.seed()

    return augmented_img, augmented_landmark, augmented_mask


def safe_loader(sample, config, mode='train'):
    """
    Calls load_video_frames_as_dataset and swallows any exception.
    If something goes wrong (corrupt image, missing mask, etc.) the
    video is silently dropped.
    """
    try:
        return load_video_frames_as_dataset(sample, config, mode)
    except Exception as e:
        print(f"[WARN] Dropped video due to error: {e}")
        return None      # signals the DataPipe to skip this sample


def load_video_frames_as_dataset(sample, config, mode='train'):
    """
    • Needs `target_frames` (= frame_num) good images.
    • Each corrupt image triggers up to 3 retries with alternate frames.
    • If still < target_frames → return None  (caller drops video).
    """

    # ---- 1. how many frames do we need? -----------------------------
    fn_cfg = config.get("frame_num", 8)  # could be int OR dict
    if isinstance(fn_cfg, dict):  # YAML style: {train: 8, test: 8}
        fn_cfg = fn_cfg.get("train" if mode == "train" else "test", 8)
    target_frames = int(fn_cfg)  # now guaranteed int

    frame_paths_all = list(sample["frames"])  # 32 paths per video
    random.shuffle(frame_paths_all)
    max_retries = 3

    good_imgs, good_masks, good_lms, good_paths = [], [], [], []

    for path in frame_paths_all:
        # -------- 2. try to open / retry up to 3 alternates ----------
        success, orig_path = False, path
        for _ in range(max_retries):
            try:
                with fsspec.open(path, "rb") as stream:
                    img = Image.open(stream).convert("RGB")
                success = True
                break
            except Exception:
                print(f"[WARN] Failed to open image: {path} - trying alternate paths")
                alt = [p for p in frame_paths_all
                       if p not in good_paths and p != orig_path]
                if not alt:
                    break
                path = random.choice(alt)

        if not success:
            print(f"[ERROR] Could not open image after retries: {orig_path}")
            continue

        # -------- 3. preprocessing & augmentation -------------------
        img = img.resize((config["resolution"], config["resolution"]), Image.BICUBIC)
        img_np = np.array(img)

        mask_path = path.replace("frames", "masks")
        landmark_path = path.replace("frames", "landmarks").replace(".png", ".npy")

        try:
            with fsspec.open(mask_path, "rb") as mf:
                mask_img = Image.open(mf).convert("L").resize(
                    (config["resolution"], config["resolution"])
                )
                mask = np.expand_dims(np.array(mask_img) / 255.0, axis=2)
        except Exception:
            mask = np.zeros((config["resolution"], config["resolution"], 1))

        try:
            with fsspec.open(landmark_path, "rb") as lf:
                lms = np.load(lf)
                if config["resolution"] != 256:
                    lms = lms * (config["resolution"] / 256)
        except Exception:
            lms = np.zeros((81, 2))

        if mode == "train" and config["use_data_augmentation"]:
            img_np, _, mask = data_aug(img_np, lms, mask)

        img_tensor = T.Normalize(mean=config["mean"], std=config["std"])(
            T.ToTensor()(img_np)
        )

        good_imgs.append(img_tensor)
        good_masks.append(mask)
        good_lms.append(lms)
        good_paths.append(path)

        if len(good_imgs) == target_frames:
            break

    # -------- 4. drop video if not enough good frames ---------------
    if len(good_imgs) < target_frames:
        return None

    video_level = sample["video_level"]

    if video_level:
        img_tensor = torch.stack(good_imgs, dim=0)
        lm_tensor = torch.stack(good_lms, dim=0)
        mask_tensor = torch.stack(good_masks, dim=0)
    else:
        img_tensor, lm_tensor, mask_tensor = good_imgs[0], good_lms[0], good_masks[0]

    return (img_tensor, sample["label"], lm_tensor, mask_tensor, good_paths)


def create_base_videopipe(dataset, method, test=False, dataset_name=None):
    """
    Build a DataPipe that streams frames for ONE method (or dataset-name).

    • Training   (test=False, method='fomm' etc.)
        – keeps only label==1 videos whose paths contain "/fomm/"
    • Testing    (test=True, method=None, dataset_name='DFDC' etc.)
        – keeps videos whose path contains dataset_name
        – keeps both real & fake labels
    """
    samples = []
    for i in range(len(dataset)):
        # 0) always treat frame_paths as a list
        frame_paths = dataset.data_dict['image'][i]
        if not isinstance(frame_paths, list):
            frame_paths = [frame_paths]

        sample_label = dataset.data_dict['label'][i]  # 0 real | 1 fake

        # 1) TRAIN-time: fake-loader → drop real videos, then path filter
        if method and not test:
            if sample_label == 0:  # drop real
                continue
            if f"/{method}/" not in frame_paths[0]:  # path mismatch
                continue

        # 2) TEST-time: filter by dataset name if provided
        if dataset_name and dataset_name.lower() not in frame_paths[0].lower():
            continue

        samples.append({
            'frames': frame_paths,
            'label': sample_label,
            'config': dataset.config,
            'video_level': dataset.video_level,
        })

    # Wrap into a streaming DataPipe
    pipe = IterableWrapper(samples)
    pipe = pipe.shuffle()
    pipe = pipe.sharding_filter()
    pipe = (pipe
            .map(lambda s: safe_loader(s, dataset.config, dataset.mode))
            .filter(lambda x: x is not None))  # filter out None samples
    return pipe


def create_method_aware_dataloaders(dataset: DeepfakeAbstractBaseDataset, dataloader_config, test=False, config=None):
    # Group videos by method
    videos_by_method = defaultdict(list)
    for v in dataset.video_infos:
        videos_by_method[v.method].append(v)

    dataloaders = {}
    if not test:
        for method in videos_by_method.keys():
            # returns a data pipe that is used to load the data from the GCP
            pipe = create_base_videopipe(dataset, method, test)

            dataloaders[method] = DataLoader(
                pipe,
                batch_size=dataloader_config['dataloader_params']['batch_size'],
                num_workers=dataloader_config['dataloader_params']['num_workers'],
                collate_fn=DeepfakePipeDataset.collate_fn,
                prefetch_factor=1,  # ↓ RAM
                persistent_workers=True  # keep workers hot between batches
            )
    else:
        for dataset_name in config['test_dataset']:
            # returns a data pip that used to load the data from the GCP
            pipe = create_base_videopipe(dataset, None, test, dataset_name)

            dataloaders[dataset_name] = DataLoader(
                pipe,
                batch_size=dataloader_config['dataloader_params']['batch_size'],
                num_workers=dataloader_config['dataloader_params']['num_workers'],
                collate_fn=DeepfakePipeDataset.collate_fn,
                prefetch_factor=1,  # ↓ RAM
                persistent_workers=True  # keep workers hot between batches
            )
    return dataloaders
