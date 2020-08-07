# from keras.preprocessing import image
import numpy as np
import albumentations
from albumentations import Compose


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def segmentation_aug(input_image, label, size, augmentation_list):
    
    transforms = list()
    height, width = size

    for augmentation_command in augmentation_list:

        if isinstance(augmentation_command,str):
            print('str')
            augmentation = getattr(albumentations, augmentation_command)()

        elif isinstance(augmentation_command,dict):
            print('dict')
            augmentation = getattr(
                albumentations, 
                list(augmentation_command.keys())[0]
            )(
                **list(augmentation_command.values())[0]
            )

        elif isinstance(augmentation_command, list):
            print('list')
            one_of_list = list()  # prepare list of input inside OneOf function 
            for augmentation in augmentation_command:
                augmentation = getattr(albumentations, augmentation)()
                one_of_list.append(augmentation)
            augmentation = albumentations.OneOf(one_of_list)

        transforms.append(augmentation)

    

    if len(transforms) > 0:
        aug = Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        return augmented['image'], augmented["mask"]
    else:
        return input_image, label
