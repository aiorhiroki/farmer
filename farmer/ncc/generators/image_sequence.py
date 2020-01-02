import math

from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
from ..augmentation import segmentation_aug
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
        augmentation=list(),
        train_colors=list()
    ):
        self.annotations = annotations
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.task = task
        self.augmentation = augmentation
        self.train_colors = train_colors

    def __getitem__(self, idx):
        data = self.annotations[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_x = list()
        batch_y = list()
        for input_file, label in data:
            input_image = self.image_util.read_image(
                input_file, anti_alias=True
            )
            if self.task == Task.SEMANTIC_SEGMENTATION:
                label = self.image_util.read_image(
                    label,
                    normalization=False,
                    train_colors=self.train_colors
                )
                if len(self.augmentation) > 0:
                    input_image, label = segmentation_aug(
                        input_image, label, self.input_shape, self.augmentation
                    )
            batch_x.append(input_image)
            batch_y.append(label)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.image_util.cast_to_onehot(batch_y)

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.annotations) / self.batch_size)
