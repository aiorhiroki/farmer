import os
import cv2
from ..tasks import Task
from ..utils import ImageUtil
from ..augmentation import segmentation_aug
import numpy as np


# classes for data loading and preprocessing
class Dataset:
    """Read images, apply augmentation.
    """

    def __init__(
            self,
            annotations: list,
            input_shape: (int, int),
            nb_classes: int,
            task: str,
            augmentation=list(),
            train_colors=list(),
            input_data_type="image",
    ):

        self.annotations = annotations
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.task = task
        self.augmentation = augmentation
        self.train_colors = train_colors
        self.input_data_type = input_data_type

    def __getitem__(self, i):

        *input_file, label = self.annotations[i]

        # input_file is [image_path] or [video_path, frame_id]
        # label is mask_image_path or class_id
        if self.input_data_type == "video":
            video_path, frame_id = input_file
            video = cv2.VideoCapture(video_path)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, input_image = video.read()
            input_image = input_image/255.0
            # (with,height) for cv2.resize
            resize_shape = self.input_shape[::-1]
            if input_image.shape[:2] != resize_shape:
                input_image = cv2.resize(
                    input_image,
                    resize_shape,
                    interpolation=cv2.INTER_LANCZOS4
                )
        else:
            input_image = self.image_util.read_image(
                input_file[0], anti_alias=True
            )
        if self.task == Task.SEMANTIC_SEGMENTATION:
            label = self.image_util.read_image(
                label,
                normalization=False,
                train_colors=self.train_colors
            )
            if self.augmentation and len(self.augmentation) > 0:
                input_image, label = segmentation_aug(
                    input_image,
                    label,
                    self.input_shape,
                    self.augmentation
                )

        label = self.image_util.cast_to_onehot(label)

        return input_image, label

    def __len__(self):
        return len(self.annotations)
