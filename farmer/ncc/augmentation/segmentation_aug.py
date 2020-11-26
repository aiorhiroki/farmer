from tensorflow.keras.preprocessing import image
import numpy as np
from .aug_utils import get_aug
from .augment_and_mix import dual_augment_and_mix
import albumentations


def segmentation_aug(
        input_image,
        label,
        mean,
        std,
        augmentation_dict,
        augmix,
):
    transforms = get_aug(augmentation_dict)

    if len(transforms) > 0:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        if augmix:
            augmented = dual_augment_and_mix(
                augmented['image'],
                augmented['mask'],
                transforms,
                mean, std
            )
        return augmented['image'], augmented["mask"]

    else:
        return input_image, label
