from tensorflow.keras.preprocessing import image
import numpy as np
from .augment_and_mix import augment_and_mix
import albumentations


def segmentation_alb(input_image, label, mean, std, augmentation_dict):
    transforms = get_aug(augmentation_dict)

    if len(transforms) > 0:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        return augmented['image'], augmented["mask"]

    else:
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
                aug_list = sorted(aug_param.items(), key=lambda x: x[0])
                new_param = dict()
                for k, v in aug_list:
                    if "-" in k:
                        tuple_name, tuple_id = k.split("-")
                        if int(tuple_id) == 1:
                            new_param[tuple_name] = (v,)
                        else:
                            new_param[tuple_name] += (v,)
                    else:
                        new_param[k] = v
                augmentation = getattr(
                    albumentations, aug_command)(**new_param)

            transforms.append(augmentation)

    return transforms


def segmentation_aug(input_image, label, mean, std, augmentation_dict):
    """apply augmentation to one image respectively
    """

    # For Keras ImageDataGenerator
    data_gen_args = dict()
    data_gen_args["fill_mode"] = "constant"  # cvalの値で埋める
    data_gen_args["cval"] = 0  # 黒で埋める

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
        """AugMix: to Improve Robustness and Uncertainty
        AugMixは最後に行う
        TODO: ひとまずハードラベル
        Affine変換系が施されたらソフトラベルにした方がいい？
        """
        input_image_processed = augment_and_mix(
            input_image_processed,
            mean, std,
        )

    return input_image_processed, label_processed
