from keras.preprocessing import image
import numpy as np
from .augment_and_mix import augment_and_mix


def segmentation_aug(input_image, label, mean, std, augmentation_dict):
    """apply augmentation to one image respectively
    """
    # For Keras ImageDataGenerator
    data_gen_args = dict()
    data_gen_args["fill_mode"] = "constant"  # cvalの値で埋める
    data_gen_args["cval"] = 0  # 黒で埋める

    if "zoom_range" in augmentation_dict:
        # convert zoom range to fit keras.ImageDataGenerator
        # args: zoom_range: [0.8, 1.3]  # x0.8 ~ x1.3
        # -> [0.7, 1.2]
        zoom_range_min, zoom_range_max = augmentation_dict["zoom_range"]
        zoom_range_lower = 2 - zoom_range_max
        zoom_range_upper = 2 - zoom_range_min
        data_gen_args["zoom_range"] = [zoom_range_lower, zoom_range_upper]
    if "vertical_flip" in augmentation_dict:
        data_gen_args["vertical_flip"] = augmentation_dict["vertical_flip"]
    if "horizontal_flip" in augmentation_dict:
        data_gen_args["horizontal_flip"] = augmentation_dict["horizontal_flip"]
    if "rotation_range" in augmentation_dict:
        data_gen_args["rotation_range"] = augmentation_dict["rotation_range"]
    if "width_shift_range" in augmentation_dict:
        data_gen_args["width_shift_range"] = augmentation_dict["width_shift_range"]
    if "height_shift_range" in augmentation_dict:
        data_gen_args["height_shift_range"] = augmentation_dict["height_shift_range"]
    if "shear_range" in augmentation_dict:
        data_gen_args["shear_range"] = augmentation_dict["shear_range"]
    if "channel_shift_range" in augmentation_dict:
        data_gen_args["channel_shift_range"] = augmentation_dict["channel_shift_range"]

    # expand dimension for batch channel and color channel
    input_image = input_image[np.newaxis]
    label = label[np.newaxis, ..., np.newaxis]

    image_datagen = image.ImageDataGenerator(**data_gen_args)
    mask_datagen = image.ImageDataGenerator(**data_gen_args)
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = np.random.randint(100)
    image_datagen.fit(input_image, augment=True, seed=seed)
    mask_datagen.fit(label, augment=True, seed=seed)

    image_gen = image_datagen.flow(input_image, batch_size=1, seed=seed)
    mask_gen = mask_datagen.flow(label, batch_size=1, seed=seed)
    # combine generators into one which yields image and masks
    gen = zip(image_gen, mask_gen)
    img_batches, mask_batches = next(gen)
    input_image_processed = img_batches.squeeze()  # batch次元を捨てる
    label_processed = mask_batches.squeeze()  # batchとchannel次元を捨てる

    # Not Keras ImageDataGenerator
    if "augmix" in augmentation_dict:
        """AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
        AugMixか独自のDAかどちらかのみ
        TODO: ひとまずハードラベル
        Affine変換系が施されたらソフトラベルにした方がいい？
        """
        input_image_processed = augment_and_mix(
            input_image_processed,
            mean, std,
        )

    return input_image_processed, label_processed
