import albumentations

def segmentation_aug(input_image, label, input_shape, mean, std, augmentation_list):
    transforms = get_aug(augmentation_list)

    if len(transforms) > 0:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        return augmented['image'], augmented["mask"]

    else:
        return input_image, label

def get_aug(augmentation_list):
    transforms = list()
    for aug_command, aug_param in augmentation_list.items():
        print(type(aug_command)) # str
        if aug_command.startswith("OneOf"):
            print('Oneof')
            augs = get_aug(aug_param)
            augmentation = albumentations.OneOf(augs, aug_param['p'])
            transforms.append(augmentation)
        elif aug_command == 'p':
            print('p_assigned')
            continue
        else:
            if aug_param is None:
                augmentation = getattr(albumentations, aug_command)()
                print('augmentation_in_process')
            else:
                augmentation = getattr(albumentations, aug_command)(aug_param)
                print('augmentation_with_dict_in_process')
            transforms.append(augmentation)
    return transforms
