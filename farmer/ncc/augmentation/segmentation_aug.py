from keras.preprocessing import image
import numpy as np
from .augment_and_mix import augment_and_mix
import albumentations
from .randaugmentation import RandAugment

def segmentation_alb(input_image, label, mean, std, augmentation_dict):
    transforms, rand_aug_params = get_aug(augmentation_dict)
    
    if len(transforms) > 0:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        if rand_aug_params:
            augmented = RandAugment(**rand_aug_params)(**augmented)
        return augmented['image'], augmented["mask"]

    else:
        return input_image, label

def get_aug(augmentation_dict):
    transforms = list()
    rand_aug_params = dict()
    for aug_command, aug_param in augmentation_dict.items():
        if aug_command.startswith("OneOf"):
            augs, _ = get_aug(aug_param)
            augmentation = albumentations.OneOf(augs, aug_param['p'])
            transforms.append(augmentation)
        elif aug_command == 'p':
            continue
        elif aug_command == 'RandAugment':
            rand_aug_params = aug_param
            print('rand_aug_params: ', rand_aug_params)
        else:
            if aug_param is None:
                augmentation = getattr(albumentations, aug_command)()
            else:
                augmentation = getattr(albumentations, aug_command)(**aug_param)
        
            transforms.append(augmentation)
            
    return transforms, rand_aug_params



def segmentation_aug(input_image, label, mean, std, augmentation_dict):
    """apply augmentation to one image respectively
    """
    # Cut off black mask of left and right edge
    if "cut_black" in augmentation_dict and augmentation_dict["cut_black"] is True:
        width = input_image.shape[1]
        input_image = input_image[:, int(width * 0.05):int(width * 0.95), :]
        label = label[:, int(width * 0.05):int(width * 0.95)]

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

    # Not Keras ImageDataGenerator
    if "augmix" in augmentation_dict and augmentation_dict["augmix"] is True:
        """AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty
        AugMixは最後に行う
        TODO: ひとまずハードラベル
        Affine変換系が施されたらソフトラベルにした方がいい？
        """
        input_image_processed = augment_and_mix(
            input_image_processed,
            mean, std,
        )

    return input_image_processed, label_processed
