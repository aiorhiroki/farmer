import albumentations


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def segmentation_aug(input_image, label, size, augmentation_list):
    
    transforms = list()

    for augmentation_command in augmentation_list:
        
        if isinstance(augmentation_command, str):
            augmentation = getattr(albumentations, augmentation_command)()

        elif isinstance(augmentation_command, dict):
            augmentation = getattr(
                albumentations, 
                list(augmentation_command.keys())[0]
            )(
                **list(augmentation_command.values())[0]
            )

        elif isinstance(augmentation_command, list):
            one_of_list = list()  # prepare list of input inside OneOf function 
            for augmentation in augmentation_command:
                if isinstance(augmentation, dict):
                    augmentation = getattr(
                        albumentations, 
                        list(augmentation.keys())[0]
                    )(
                        **list(augmentation.values())[0]
                    )
                    one_of_list.append(augmentation)

                else:
                    augmentation = getattr(albumentations, augmentation)()
                    one_of_list.append(augmentation)
            augmentation = albumentations.OneOf(one_of_list)

        transforms.append(augmentation)


    if len(transforms) > 0:
        aug = albumentations.Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        return augmented['image'], augmented["mask"]

    else:
        return input_image, label