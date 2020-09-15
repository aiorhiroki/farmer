from tensorflow.keras.preprocessing import image
import numpy as np
from .augment_and_mix import augment_and_mix
import albumentations


def segmentation_alb(
        input_image,
        label, mean,
        std,
        augmentation_dict, 
        aug_stat,
        augmix,
        ):
    transforms = get_aug(augmentation_dict)

    if aug_stat is None:
        return input_image, label

    elif "albumentation" in aug_stat:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        if augmix:
            augmented = augment_and_mix(augmented[image], mean, std,)
        return augmented['image'], augmented["mask"]

    elif "imagegenerator" in aug_stat:
    # will be modified with keras_image_generator
        if augmix:
            augmented = augment_and_mix(input_image, mean, std)
            label = label
        return input_image, label


def get_aug(augmentation_dict):
    transforms = list()
    for aug_command, aug_param in augmentation_dict.items():
        if aug_command.startswith("OneOf"):
            augs = get_aug(aug_param)
            augmentation = albumentations.OneOf(augs, aug_param['p'])
            transforms.append(augmentation)
        elif aug_command == 'p':
            continue
        else:
            if aug_param is None:
                augmentation = getattr(albumentations, aug_command)()
            else:
                aug_list = sorted(aug_param.items(), key=lambda x: x[0])
                new_param = dict()
                for k, v in aug_list:
                    if "-" in k:
                        tuple_name, tuple_id = k.split("-")
                        if int(tuple_id) == 1:
                            new_param[tuple_name] = (v,)
                        else:
                            new_param[tuple_name] += (v,)
                    else:
                        new_param[k] = v
                augmentation = getattr(
                    albumentations, aug_command)(**new_param)

            transforms.append(augmentation)

    return transforms
