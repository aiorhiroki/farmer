import numpy as np
import albumentations


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
