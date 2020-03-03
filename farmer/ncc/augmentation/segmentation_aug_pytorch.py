import numpy as np
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
    Blur,
    RandomScale,
    Resize,
)


def segmentation_aug_pytorch(input_image, label, size, augmentation_list):

    height, width = size
    transforms = list()

    # This is a substitute function instead of zoom in ImageDataGenerator on Tensorflow
    if "zoom" in augmentation_list:
        transforms.append(RandomScale(scale_limit=(0.75, 1.25)))
        transforms.append(Resize(height, width))

    if "vertical_flip" in augmentation_list:
        transforms.append(VerticalFlip(p=0.5))

    if "horizontal_flip" in augmentation_list:
        transforms.append(HorizontalFlip(p=0.5))

    if len(transforms) > 0:
        aug = Compose(transforms, p=1)
        augmented = aug(image=input_image, label=label)
        return augmented['image'], augmented["label"]

    else:
        return input_image, label
