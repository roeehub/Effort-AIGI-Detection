# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys

import lmdb

sys.path.append('.')

import os
import math
import yaml
import glob
import json
import dlib

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T
import albumentations as A

from dataset.albu import IsotropicResize

FFpp_pool=['FaceForensics++','FaceShifter','DeepFakeDetection','FF-DF','FF-F2F','FF-FS','FF-NT']#

import os.path as osp
from glob import glob
import random
from tqdm import tqdm
from glob import glob
import skimage
import numpy as np


def all_in_pool(inputs,pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    def __init__(self, config=None, mode='train', VideoInfo=None):
        self.config = config
        self.mode = mode
        self.compression = config['compression']
        self.frame_num = config['frame_num'][mode]
        self.video_level = config.get('video_mode', False)
        self.clip_size = config.get('clip_size', None)
        self.lmdb = config.get('lmdb', False)
        self.image_list = []
        self.label_list = []


        if VideoInfo:
            self.video_infos = VideoInfo
        else:
            raise NotImplementedError('Only train and test modes are supported.')

        image_list, label_list, _ = self.collect_img_and_label_for_one_dataset(self.video_infos)

        if self.lmdb:
            # Construct LMDB path based on dataset
            if mode == 'train':
                dataset_list = config['train_dataset']
            else:
                dataset_list = [config['test_dataset']]

            if len(dataset_list) > 1:
                if all_in_pool(dataset_list, FFpp_pool):
                    lmdb_path = os.path.join(config['lmdb_dir'], f"FaceForensics++_lmdb")
                else:
                    raise ValueError('Training with multiple dataset and lmdb is not implemented yet.')
            else:
                ds = dataset_list[0]
                lmdb_path = os.path.join(config['lmdb_dir'], f"{ds if ds not in FFpp_pool else 'FaceForensics++'}_lmdb")
            self.env = lmdb.open(lmdb_path, create=False, subdir=True, readonly=True, lock=False)

        assert len(image_list) != 0 and len(label_list) != 0, f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
        }

        self.transform = self.init_data_aug_method()

        self.face_detector = dlib.get_frontal_face_detector()

        self.error_path_list = []

    def init_data_aug_method(self):
        trans = A.Compose([
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

        ],
           # keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None
        )
        return trans

    def collect_img_and_label_for_one_dataset(self, video_infos):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.

        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []
        video_name_list = []

        for video in video_infos:
            # Video name based on "real/fake" and ID
            unique_video_name = f"{video.label}_{video.video_id}"
            
            # label: real - 0 / fake - 1
            label = 0 if video.label == 'real' else 1
            
            # List of all the frame paths of a specific video
            frame_paths = video.frame_paths

            if len(frame_paths)==0:
                    print(f"{unique_video_name} is None. Let's skip it.")
                    continue
            # sorted video path to the lists
            if self.video_level:
                if '\\' in frame_paths[0]:
                    frame_paths = sorted(frame_paths, key=lambda x: int(x.split('\\')[-1].split('.')[0]))
                else:
                    frame_paths = sorted(frame_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))


            # If video_level true, we take 'self.frame_num' frames one by one after randomely picked frame from the sorted frames. \
                # If false, we take 'self.frame_num' frames randomely sorted as in the original dataset with even step between them.
            total_frames = len(frame_paths)
            if self.frame_num < total_frames:
                # total_frames = self.frame_num
                if self.video_level:
                    # Select clip_size continuous frames
                    start_frame = random.randint(0, total_frames - self.frame_num)
                    frame_paths = frame_paths[start_frame:start_frame + self.frame_num]  # update total_frames
                else:
                    # Select self.frame_num frames evenly distributed throughout the video
                    step = total_frames // self.frame_num
                    frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]

            # If video-level methods, crop clips from the selected frames if needed
            if self.video_level:
                if self.clip_size is None:
                    raise ValueError('clip_size must be specified when video_level is True.')
                # Check if the number of total frames is greater than or equal to clip_size
                if total_frames >= self.clip_size:
                    # Initialize an empty list to store the selected continuous frames
                    selected_clips = []

                    # Calculate the number of clips to select
                    num_clips = total_frames // self.clip_size

                    if num_clips > 1:
                        # Calculate the step size between each clip
                        clip_step = (total_frames - self.clip_size) // (num_clips - 1)

                        # Select clip_size continuous frames from each part of the video
                        for i in range(num_clips):
                            # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                            start_frame = random.randrange(i * clip_step, min((i + 1) * clip_step, total_frames - self.clip_size + 1))
                            continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                            assert len(continuous_frames) == self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                            selected_clips.append(continuous_frames)

                    else:
                        start_frame = random.randrange(0, total_frames - self.clip_size + 1)
                        continuous_frames = frame_paths[start_frame:start_frame + self.clip_size]
                        assert len(continuous_frames)==self.clip_size, 'clip_size is not equal to the length of frame_path_list'
                        selected_clips.append(continuous_frames)

                    # Append the list of selected clips and append the label
                    label_list.extend([label] * len(selected_clips))
                    frame_path_list.extend(selected_clips)
                    # video name save
                    video_name_list.extend([unique_video_name] * len(selected_clips))

                else:
                    print(f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames}).")

            # Otherwise, extend the label and frame paths to the lists according to the number of frames
            else:
                # Extend the label and frame paths to the lists according to the number of frames
                label_list.extend([label] * total_frames)
                frame_path_list.extend(frame_paths)
                # video name save
                video_name_list.extend([unique_video_name] * len(frame_paths))

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)

        return frame_path_list, label_list, video_name_list


    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.
        This version is updated to handle both local paths and GCS (gs://) paths.

        Args:
            file_path: A string indicating the path to the image file (e.g., '/path/to/img.png' or 'gs://bucket/img.png').

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config['resolution']

        # The LMDB logic remains unchanged as it reads from a local database.
        if self.lmdb:
            with self.env.begin(write=False) as txn:
                if file_path.startswith('./datasets\\'):
                    file_path = file_path.replace('./datasets\\', '')
                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        else:
            # --- MODIFICATION START ---
            # Check if the file path is a Google Cloud Storage URI
            if file_path.startswith('gs://'):
                # Use fsspec to open the remote file in binary read mode ('rb')
                # This is the same mechanism used in your dataloaders.py
                with fsspec.open(file_path, 'rb') as f:
                    # Read the file's byte stream into a NumPy array
                    image_buf = np.frombuffer(f.read(), dtype=np.uint8)
                    # Decode the image buffer using OpenCV, which is consistent with the original logic
                    img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
            else:
                # This is the original logic for handling local files
                assert os.path.exists(file_path), f"{file_path} does not exist"
                img = cv2.imread(file_path)

                # Original fallback logic
                if img is None:
                    img_pil = Image.open(file_path)
                    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            # --- MODIFICATION END ---

        if img is None:
            raise ValueError('Loaded image is None: {}'.format(file_path))

        # The rest of the function remains the same, ensuring consistent processing
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))


    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config['resolution']
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                image_bin = txn.get(file_path.encode())
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0]=='.':
                    file_path=file_path.replace('./datasets\\','')
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
        return np.float32(landmark)

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config['mean']
        std = self.config['std']
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
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

        # Create a dictionary of arguments
        kwargs = {'image': img}

        # Check if the landmark and mask are not None
        #if landmark is not None:
        #    kwargs['keypoints'] = landmark
        #    kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask')

        # Convert the augmented landmark to a numpy array
        #if augmented_landmark is not None:
        augmented_landmark=None
        #    augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict['image'][index]
        label = self.data_dict['label'][index]

        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        # print(image_paths)

        for i, image_path in enumerate(image_paths):

            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace('frames', 'masks')  # Use .png for mask
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            try:
                # image_path = image_path.replace("frames", "frames_wocropface")
                image = self.load_rgb(image_path)
            except Exception as e:
                # Skip this image and return the first one
                print(f"Error loading image at index {index}: {e}")
                index_random = random.randint(0, len(self.image_list)-1)
                return self.__getitem__(index_random)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            if self.mode=='train' and self.config['with_mask']:
                mask = self.load_mask(mask_path)
            else:
                mask = None


            if self.config['with_landmark']:
                landmarks = self.load_landmark(landmark_path)
                if self.config['resolution'] != 256:
                    landmarks = landmarks * (self.config['resolution'] / 256)

                # if self.config['model_name']  == 'clip_adapter' and first_frame_lmk is None:
                #     first_frame_lmk = landmarks
            else:
                landmarks = None


            # Do Data Augmentation
            if self.mode == 'train' and self.config['use_data_augmentation']:
                image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask, augmentation_seed)
                # landmarks_trans = landmarks
            else:
                image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)


            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                #if self.mode == 'train' and self.config['with_landmark']:
                #    landmarks_trans = torch.from_numpy(landmarks_trans)
                #if self.mode == 'train' and self.config['with_mask']:
                #    mask_trans = torch.from_numpy(mask_trans)

            image_tensors.append(image_trans)
            landmark_tensors.append(landmarks_trans)
            mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
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

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if not any(landmark is None or (isinstance(landmark, list) and None in landmark) for landmark in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = landmarks
        data_dict['mask'] = masks
        return data_dict

    def __len__(self):
        """
        Return the length of the dataset.

        Args:
            None.

        Returns:
            An integer indicating the length of the dataset.

        Raises:
            AssertionError: If the number of images and labels in the dataset are not equal.
        """
        assert len(self.image_list) == len(self.label_list), 'Number of images and labels are not equal'
        return len(self.image_list)
