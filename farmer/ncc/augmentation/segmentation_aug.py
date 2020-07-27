from keras.preprocessing import image
import numpy as np


def segmentation_aug(input_image, label, augmentation_list):
    data_gen_args = dict()
    if "zoom" in augmentation_list:
        data_gen_args["zoom_range"] = [0.5, 1.0]
    if "vertical_flip" in augmentation_list:
        data_gen_args["vertical_flip"] = True
    if "horizontal_flip" in augmentation_list:
        data_gen_args["horizontal_flip"] = True

    # (H,W[,C]) => (N,H,W,C)
    input_image = input_image[np.newaxis]
    label = label[np.newaxis, ..., np.newaxis]

    image_datagen = image.ImageDataGenerator(**data_gen_args)
    mask_datagen = image.ImageDataGenerator(**data_gen_args)

    seed = np.random.randint(100)
    image_datagen.fit(input_image, augment=True, seed=seed)
    mask_datagen.fit(label, augment=True, seed=seed)

    image_gen = image_datagen.flow(input_image, batch_size=1, seed=seed)
    mask_gen = mask_datagen.flow(label, batch_size=1, seed=seed)

    gen = zip(image_gen, mask_gen)
    img_batches, mask_batches = next(gen)
    input_image = img_batches.squeeze()
    label = mask_batches.squeeze()

    return input_image, label
