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
    augmentation_seed = random.randint(0, 2**32 - 1) if video_level else None

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
            image_aug = image_np
            mask_aug = mask
            landmarks_aug = landmarks
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
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
            landmark_tensors = torch.stack(landmark_tensors, dim=0)
        if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
            mask_tensors = torch.stack(mask_tensors, dim=0)
    
    else:
        # Get the first image tensor
        image_tensors = image_tensors[0]
        # Get the first landmark and mask tensors
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmark_tensors):
            landmark_tensors = landmark_tensors[0]
        if not any(m is None or (isinstance(m, list) and None in m) for m in mask_tensors):
            mask_tensors = mask_tensors[0]

    

    return image_tensors, label, landmark_tensors, mask_tensors


def create_base_videopipe(dataset):
    """
    Converts a DeepfakeAbstractBaseDataset into a DataPipe that loads from GCS.
    """
    samples = []
    for i in range(len(dataset)):
        frame_paths = dataset.data_dict['image'][i]
        if not isinstance(frame_paths, list):
            frame_paths = [frame_paths]
        samples.append({
            'frames': frame_paths,
            'label': dataset.data_dict['label'][i],
            'config': dataset.config,
            'video_level': dataset.video_level
        })

    pipe = IterableWrapper(samples)
    pipe = pipe.shuffle()
    pipe = pipe.sharding_filter()
    pipe = pipe.map(lambda sample: load_video_frames_as_dataset(sample, dataset.config, dataset.mode))
    pipe = pipe.prefetch(10)
    return pipe


def create_method_aware_dataloaders(dataset: DeepfakeAbstractBaseDataset, config):
    # Group videos by method
    videos_by_method = defaultdict(list)
    for v in dataset.video_infos:
        videos_by_method[v.method].append(v)
    
    dataloaders = {}
    for method, videos in videos_by_method.items():
        pipe = create_base_videopipe(dataset)
        dataloaders[method] = DataLoader(
            pipe,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            collate_fn=DeepfakePipeDataset.collate_fn
        )
    return dataloaders
