from PIL import Image
import numpy as np
import cv2
from ncc.utils import palette
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    Normalize
)


class ImageUtil:

    def __init__(
        self,
        nb_classes: int,
        size: (int, int)
    ):
        self.nb_classes = nb_classes
        self.size = size[::-1]
        self.current_raw_size = None
        self.current_raw_frame = None

    def read_image(
        self,
        file_path: str,
        normalization=True,
        anti_alias=False
    ):
        image = Image.open(file_path)
        self.current_raw_frame = image
        self.current_raw_size = image.size
        if self.size != self.current_raw_size:
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

    def _cast_to_frame(
        self,
        prediction,
        size
    ):
        res = np.argmax(prediction, axis=2)
        image = Image.fromarray(np.uint8(res), mode="P")
        image.putpalette(palette.palettes)
        image = image.resize(self.current_raw_size, Image.LANCZOS)
        image = image.convert("RGB")
        return np.asarray(np.asarray(image)*255, dtype=np.uint8)

    def blend_image(
        self,
        output_image,
        size
    ):
        input_frame = np.array(self.current_raw_frame, dtype=np.uint8)
        output_frame = self._cast_to_frame(output_image, size)
        blended = cv2.addWeighted(
            src1=input_frame,
            src2=output_frame,
            alpha=0.7,
            beta=0.9,
            gamma=2.2
        )
        return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    def augmentation(self, image, mask):
        width, height = self.size
        aug_list = [
            RandomSizedCrop(
                min_max_height=(height//2, height),
                height=height,
                width=width,
                p=1
            ),
            HorizontalFlip(
                p=0.5
            )
        ]

        aug = Compose(aug_list, p=1)
        image = np.array(image*255, dtype=np.uint8)
        augmented = aug(image=image, mask=mask)
        return augmented['image']/255, augmented['mask']
