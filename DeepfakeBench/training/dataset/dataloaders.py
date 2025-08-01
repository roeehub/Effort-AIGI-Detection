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


def load_video_frames_as_dataset(sample, config, mode='train'):
    """
    Use `open_files` to fetch frames from GCS and build full image/landmark/mask logic like in __getitem__.
    """
    frame_paths, label = sample['frames'], sample['label']
    video_level = sample['video_level']
    config = sample['config']

    image_tensors = []
    landmark_tensors = []
    mask_tensors = []
    augmentation_seed = random.randint(0, 2 ** 32 - 1) if video_level else None

    files = fsspec.open_files(frame_paths, mode='rb')
    for i, f in enumerate(files):
        with f as stream:
            image = Image.open(stream).convert('RGB')
        image = image.resize((config['resolution'], config['resolution']), Image.BICUBIC)
        image_np = np.array(image)

        # Load landmarks and masks (assuming GCS or local paths)
        mask_path = frame_paths[i].replace('frames', 'masks')
        landmark_path = frame_paths[i].replace('frames', 'landmarks').replace('.png', '.npy')

        try:
            with fsspec.open(mask_path, 'rb') as mf:
                mask = Image.open(mf).convert('L').resize((config['resolution'], config['resolution']))
                mask = np.expand_dims(np.array(mask) / 255.0, axis=2)
        except Exception:
            mask = np.zeros((config['resolution'], config['resolution'], 1))

        try:
            with fsspec.open(landmark_path, 'rb') as lf:
                landmarks = np.load(lf)
                if config['resolution'] != 256:
                    landmarks = landmarks * (config['resolution'] / 256)
        except Exception:
            landmarks = np.zeros((81, 2))

        if mode == 'train' and config['use_data_augmentation']:
            # Implement basic augmentation or skip to keep logic close
            image_aug, mask_aug, landmarks_aug = data_aug(image_np, landmarks, mask, augmentation_seed)
        else:
            image_aug = deepcopy(image_np)
            mask_aug = deepcopy(mask)
            landmarks_aug = deepcopy(landmarks)

        to_tensor = T.ToTensor()
        normalize = T.Normalize(mean=config['mean'], std=config['std'])
        image_tensor = normalize(to_tensor(image_aug))

        image_tensors.append(image_tensor)
        landmark_tensors.append(landmarks_aug)
        mask_tensors.append(mask_aug)

    if sample['video_level']:
        image_tensors = torch.stack(image_tensors, dim=0)
        # Stack landmark and mask tensors along a new dimension (time)
        if not any(
                landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
            landmark_tensors = torch.stack(landmark_tensors, dim=0)
        if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
            mask_tensors = torch.stack(mask_tensors, dim=0)

    else:
        # Get the first image tensor
        image_tensors = image_tensors[0]
        # Get the first landmark and mask tensors
        if not any(
                landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
            landmark_tensors = landmark_tensors[0]
        if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
            mask_tensors = mask_tensors[0]

    return image_tensors, label, landmark_tensors, mask_tensors, frame_paths


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
    pipe = pipe.map(
        lambda s: load_video_frames_as_dataset(
            s, dataset.config, dataset.mode
        )
    )
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
