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
                augmentation = getattr(
                    albumentations, aug_command)(**aug_param)

            transforms.append(augmentation)

    return transforms



'''
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
    # combine generators into one which yields image and masks
    gen = zip(image_gen, mask_gen)
    img_batches, mask_batches = next(gen)
    input_image_processed = img_batches.squeeze()  # batch次元を捨てる
    label_processed = mask_batches.squeeze()  # batchとchannel次元を捨てる

    return input_image_processed, label_processed
'''