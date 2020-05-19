from keras.preprocessing import image
import numpy as np


def segmentation_aug(input_image, label, size, augmentation_list):
    data_gen_args = augmentation_list
    data_gen_args["fill_mode"] = "constant"  # cvalの値で埋める
    data_gen_args["cval"] = 0  # 黒で埋める

    if "zoom_range" in data_gen_args:
        # args: zoom_range: [0.8, 1.3]  # x0.8 ~ x1.3
        zoom_range_min, zoom_range_max = data_gen_args["zoom_range"]
        zoom_range_lower = 2 - zoom_range_max
        zoom_range_upper = 2 - zoom_range_min
        data_gen_args["zoom_range"] = [zoom_range_lower, zoom_range_upper]

    seed = np.random.randint(100)
    input_image = input_image[np.newaxis]
    label = label[np.newaxis, ..., np.newaxis]
    image_datagen = image.ImageDataGenerator(**data_gen_args)
    mask_datagen = image.ImageDataGenerator(**data_gen_args)

    image_datagen.fit(input_image, augment=True, seed=seed)
    mask_datagen.fit(label, augment=True, seed=seed)

    image_gen = image_datagen.flow(input_image, batch_size=1, seed=seed)
    mask_gen = mask_datagen.flow(label, batch_size=1, seed=seed)

    gen = zip(image_gen, mask_gen)
    img_batches, mask_batches = next(gen)
    input_image_processed = img_batches[0]
    label_processed = mask_batches[0][..., 0]

    return input_image_processed, label_processed
