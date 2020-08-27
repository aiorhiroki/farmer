from typing import Tuple
import numpy as np
import cv2
from ..utils import ImageUtil
from ..augmentation import segmentation_aug, segmentation_alb, RandAugment


class SegmentationDataset:
    """for segmentation task
    read image and mask, apply augmentation
    """

    def __init__(
            self,
            annotations: list,
            input_shape: Tuple[int, int],
            nb_classes: int,
            N: int = None,
            M: int = None,
            mean: np.ndarray = np.zeros(3),
            std: np.ndarray = np.ones(3),
            augmentation: list = list(),
            train_colors: list = list(),
            **kwargs
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.train_colors = train_colors
        self.N = N
        self.M = M

    def __getitem__(self, i):

        # input_file is [image_path]
        # label is mask_image_path
        *input_file, label = self.annotations[i]
        # read images
        input_image = self.image_util.read_image(input_file[0])
        label = self.image_util.read_image(label, self.train_colors)

        # apply augmentations
        if self.augmentation and len(self.augmentation) > 0:
            input_image, label = segmentation_alb(
                input_image, label,
                self.mean, self.std,
                self.augmentation
            )
        # apply preprocessing
        # resize
        input_image = self.image_util.resize(input_image, anti_alias=True)
        label = self.image_util.resize(label, anti_alias=False)
        # normalization and onehot encoding
        input_image = self.image_util.normalization(input_image)
        label = self.image_util.cast_to_onehot(label)

        return input_image, label

    def __len__(self):
        return len(self.annotations)


class ClassificationDataset:
    """for classification
    read image/frame of video and class id
    """

    def __init__(
            self,
            annotations: list,
            input_shape: Tuple[int, int],
            nb_classes: int,
            augmentation: list = list(),
            input_data_type: str = "image",
            **kwargs
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.augmentation = augmentation
        self.input_data_type = input_data_type

    def __getitem__(self, i):

        # input_file is [image_path] or [video_path, frame_id]
        # label is class_id
        *input_file, label = self.annotations[i]

        if self.input_data_type == "video":
            # video data [video_path, frame_id]
            video_path, frame_id = input_file
            # read frame
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, input_image = video.read()
            # BGR -> RGB
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        elif self.input_data_type == "image":
            # image data [image_path]
            input_image = self.image_util.read_image(input_file[0])

        # apply preprocessing
        input_image = self.image_util.resize(input_image, anti_alias=True)
        input_image = self.image_util.normalization(input_image)
        label = self.image_util.cast_to_onehot(label)

        return input_image, label

    def __len__(self):
        return len(self.annotations)
