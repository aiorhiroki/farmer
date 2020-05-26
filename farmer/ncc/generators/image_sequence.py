import math
import cv2

from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from ..augmentation import segmentation_aug, augment_and_mix
from ..tasks import Task
from ..utils import ImageUtil


class ImageSequence(Sequence):
    def __init__(
        self,
        annotations: list,
        input_shape: (int, int),
        nb_classes: int,
        task: str,
        batch_size: int,
        mean=np.zeros(3),
        std=np.ones(3),
        augmentation=list(),
        train_colors=list(),
        input_data_type="image"
    ):
        self.annotations = annotations
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.task = task
        self.augmentation = augmentation
        self.train_colors = train_colors
        self.input_data_type = input_data_type

    def __getitem__(self, idx):
        data = self.annotations[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_x = list()
        batch_y = list()
        for *input_file, label in data:
            # input_file is [image_path] or [video_path, frame_id]
            # label is mask_image_path or class_id
            if self.input_data_type == "video":
                video_path, frame_id = input_file
                video = cv2.VideoCapture(video_path)
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, input_image = video.read()
                if not ret:
                    continue
                input_image = input_image / 255.0
                # (width, height) for cv2.resize
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

                if self.augmentation and "augmix" in self.augmentation:
                    """AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
                    AugMixか独自のDAかどちらかのみ
                    TODO: ひとまずハードラベル
                    Affine変換系が施されたらソフトラベルにした方がいい？
                    """
                    input_image = augment_and_mix(
                        input_image,
                        self.mean, self.std
                    )
                elif self.augmentation and len(self.augmentation) > 0:
                    input_image, label = segmentation_aug(
                        input_image,
                        label,
                        self.augmentation
                    )
            batch_x.append(input_image)
            batch_y.append(label)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.image_util.cast_to_onehot(batch_y)

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.annotations) / self.batch_size)
