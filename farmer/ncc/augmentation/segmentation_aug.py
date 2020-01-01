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


def segmentation_aug(image, mask, size, augmentation_list):
    width, height = size
    transforms = list()
    for augmentation in augmentation_list:
        if augmentation == "HorizontalFlip":
            transforms.append(HorizontalFlip(p=0.5))
        elif augmentation == "RandomSizedCrop":
            transforms.append(
                RandomSizedCrop(
                    min_max_height=(height*3//4, height),
                    height=height,
                    width=width,
                    p=0.5
                )
            )

    aug = Compose(transforms, p=1)
    augmented = aug(image=image, mask=mask)
    return augmented['image'], augmented['mask']
