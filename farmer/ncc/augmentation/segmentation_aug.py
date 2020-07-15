from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
)


def segmentation_aug(input_image, label, size, augmentation_list):
    transforms = list()
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
