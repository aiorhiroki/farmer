from typing import Tuple
import numpy as np
from mrcnn import utils
from pathlib import Path
import cv2
from PIL import Image
from ..utils import ImageUtil
from ..augmentation import segmentation_aug, classification_aug


class SegmentationDataset:
    """for segmentation task
    read image and mask, apply augmentation
    """

    def __init__(
            self,
            annotations: list,
            input_shape: Tuple[int, int],
            nb_classes: int,
            mean: np.ndarray = np.zeros(3),
            std: np.ndarray = np.ones(3),
            augmentation: list = list(),
            augmix: bool = False,
            train_colors: list = list(),
            **kwargs
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.augmix = augmix
        self.train_colors = train_colors

    def __getitem__(self, i):

        # input_file is [image_path]
        # label is mask_image_path
        *input_file, label = self.annotations[i]
        # read images
        input_image = self.image_util.read_image(input_file[0])
        label = self.image_util.read_image(label, self.train_colors)

        # apply augmentations
        if self.augmentation and len(self.augmentation) > 0:
            input_image, label = segmentation_aug(
                input_image, label,
                self.mean, self.std,
                self.augmentation,
                self.augmix
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
            mean: np.ndarray = np.zeros(3),
            std: np.ndarray = np.ones(3),
            augmentation: list = list(),
            augmix: bool = False,
            input_data_type: str = "image",
            **kwargs
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.mean = mean
        self.std = std
        self.augmentation = augmentation
        self.augmix = augmix
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

        # apply augmentations
        if self.augmentation and len(self.augmentation) > 0:
            input_image = classification_aug(
                input_image,
                self.mean, self.std,
                self.augmentation,
                self.augmix
            )

        return input_image, label

    def __len__(self):
        return len(self.annotations)


class MaskrcnnDataset(utils.Dataset):

    def __init__(
            self,
            annotations: list,
            train_colors: dict,
            class_names: list,
            **kwargs
    ):

        self.annotations = annotations
        self.train_colors = train_colors
        self.class_names = class_names
        self.nb_classes = max(train_colors.values())

        super().__init__()

    def load_dataset(self):
        for class_id in range(self.nb_classes):
            self.add_class('Maskrcnn', class_id+1, self.class_names[class_id])

        for image_path, mask_path in self.annotations:
            mask = cv2.imread(mask_path)
            # height, width = mask.shape[:2]
            height, width = 640, 640

            self.add_image(
                'Maskrcnn',
                path=str(image_path),
                image_id=str(image_path),
                mask_path=mask_path,
                width=width, height=height
            )

    def load_image(self, image_id):
        image_info = self.image_info[image_id]
        width, height = image_info['width'], image_info['height']
        image = Image.open(str(image_info['path'])).resize((width, height))
        return np.array(image)

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        mask = np.array(Image.open(str(image_info['mask_path'])))
        for train_id, train_color in enumerate(self.train_colors.items()):
            train_id += 1
            mask[mask == train_color[0]] = train_id
        mask[mask > len(self.train_colors)] = 0

        count = len(np.unique(mask)[1:])
        width, height = image_info['width'], image_info['height']
        out = np.zeros([height, width, count], dtype=np.uint8)

        mask_ids = list(np.unique(mask)[1:])
        for nb_mask, mask_id in enumerate(mask_ids):
            m_o = cv2.resize(np.array(mask == mask_id, dtype=np.uint8), (width, height))
            out[:, :, nb_mask] = m_o

        class_ids = np.unique(mask)[1:]
        for train_id, train_color in enumerate(self.train_colors.items()):
            train_id += 1
            train_cls_id = train_color[1]
            if train_cls_id != train_id:
                class_ids[class_ids == train_id] = train_cls_id

        return out.astype(np.int32), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]