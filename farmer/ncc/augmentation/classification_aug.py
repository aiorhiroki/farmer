from .aug_utils import get_aug
from .augment_and_mix import augment_and_mix
import albumentations


def classification_aug(
        input_image,
        mean,
        std,
        augmentation_dict,
        augmix,
):
    transforms = get_aug(augmentation_dict)

    if len(transforms) > 0:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image)
        if augmix is True:
            augmented['image'] = augment_and_mix(
                augmented['image'],
                transforms,
                mean,
                std
            )
        return augmented['image']

    else:
        return input_image
