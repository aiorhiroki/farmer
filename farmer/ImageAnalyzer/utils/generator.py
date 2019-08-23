import math

from keras.utils import Sequence
import numpy as np
from farmer.ImageAnalyzer.task import Task
from ncc.utils import ImageUtil


class ImageSequence(Sequence):
    def __init__(
        self,
        annotations: list,
        input_shape: (int, int),
        nb_classes: int,
        task: str,
        batch_size: int,
        augmentation=False
    ):
        self.annotations = annotations
        self.batch_size = batch_size
        self.image_util = ImageUtil(nb_classes, input_shape)
        self.task = task
        self.augmentation = augmentation

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
                    label, normalization=False
                )
                if self.augmentation:
                    input_image, label = self.image_util.augmentation(
                        input_image, label
                    )
            batch_x.append(input_image)
            batch_y.append(label)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_y = self.image_util.cast_to_onehot(batch_y)

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.annotations) / self.batch_size)
