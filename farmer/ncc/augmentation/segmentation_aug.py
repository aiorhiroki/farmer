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
        augmentation = aug(image=input_image, mask=label)
        image, mask = augmentation["image"], augmentation["mask"]
        if augmix:
            image, mask = dual_augment_and_mix(
                image, mask, transforms, mean, std)
        return image, mask

    else:
        return input_image, label
