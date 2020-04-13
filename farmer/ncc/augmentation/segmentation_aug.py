from keras.preprocessing import image
import numpy as np


def segmentation_aug(input_image, label, size, augmentation_list):
    data_gen_args = dict()
    if "zoom" in augmentation_list:
        data_gen_args["zoom_range"] = [0.5, 1.0]
    if "vertical_flip" in augmentation_list:
        data_gen_args["vertical_flip"] = True
    if "horizontal_flip" in augmentation_list:
        data_gen_args["horizontal_flip"] = True

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

    return input_image, label
