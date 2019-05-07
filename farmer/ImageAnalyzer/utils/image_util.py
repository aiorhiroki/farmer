from PIL import Image
import numpy as np


class ImageUtil:

    def __init__(
        self,
        nb_classes: int,
        size: (int, int)
    ):
        self.nb_classes = nb_classes
        self.size = size

    @staticmethod
    def read_image(
        self,
        file_path: str,
        normalization=True,
        anti_alias=False
    ):
        image = Image.open(file_path)
        if self.size != image.size:
            resample = Image.LANCZOS if anti_alias else Image.NEAREST
            image = image.resize(self.size, resample)
        # delete alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = np.asarray(image)
        if normalization:
            image = image / 255.0

        return image

    def cast_to_onehot(
        self,
        labels: list
    ):
        labels = np.asarray(labels, dtype=np.uint8)
        # Classification
        if len(labels.shape) == 1:
            one_hot = np.eye(self.nb_classes, dtype=np.uint8)
        # Segmentation
        else:
            one_hot = np.identity(self.nb_classes, dtype=np.uint8)
        return one_hot[labels]
