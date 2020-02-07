from keras.preprocessing import image
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
    Blur
)


def segmentation_aug(input_image, label, size, augmentation_list):
    if "zoom" in augmentation_list:
        data_gen_args = dict(zoom_range=[0.75, 1.25])
        seed = np.random.randint(100)
        label = np.expand_dims(label, axis=0)
        label = np.expand_dims(label, axis=-1)
        image_datagen = image.ImageDataGenerator(**data_gen_args)
        mask_datagen = image.ImageDataGenerator(**data_gen_args)

        image_datagen.fit(input_image[np.newaxis], augment=True, seed=seed)
        mask_datagen.fit(label, augment=True, seed=seed)

        image_gen = image_datagen.flow(input_image[np.newaxis], batch_size=1, seed=seed)
        mask_gen = mask_datagen.flow(label, batch_size=1, seed=seed)

        gen = zip(image_gen, mask_gen)
        img_batches, mask_batches = next(gen)
        input_image = img_batches[0]
        label = mask_batches[0][..., 0]

    height, width = size
    transforms = list()
    if "vertical_flip" in augmentation_list:
        transforms.append(VerticalFlip(p=0.5))
    if "horizontal_flip" in augmentation_list:
        transforms.append(HorizontalFlip(p=0.5))
    if len(transforms) > 0:
        aug = Compose(transforms, p=1)
        augmented = aug(image=input_image ,label=label)
        return augmented['image'], augmented["label"]
    else:
        return input_image, label
