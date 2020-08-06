# from keras.preprocessing import image
import numpy as np
import albumentations
from albumentations import Compose
# from albumentations import (
#     Compose,
#     HorizontalFlip,
#     VerticalFlip,

#     Blur,
#     CLAHE,
#     ElasticTransform,
#     GaussianBlur,
#     GaussNoise,
#     GlassBlur,
#     GridDistortion,
#     GridDropout,
#     HueSaturationValue,
#     Lambda,
#     MotionBlur,
#     MedianBlur,
#     MultiplicativeNoise,
#     Normalize,
#     RandomBrightness,
#     RandomCrop,
#     RandomContrast,
#     RandomGamma,
#     MultiplicativeNoise,
#     ShiftScaleRotate,
#     IAAAdditiveGaussianNoise,
#     IAASharpen,
#     IAAPerspective,
#     PadIfNeeded,

#     #used in modules
#     OneOf,
#     DualTransform,
#     # ISONoise,
# )

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

def segmentation_aug(input_image, label, size, augmentation_list):
    transforms = list()
    # print(augmentation_list) # ['Gaussiannoise', 'Perspecitve', 'Blur']

    for augmentation_command in augmentation_list:
        # augmentation= f"{augmentation_command}()"
        print(augmentation_command)

        augmentation= getattr(albumentations, augmentation_command) #+ '()'   
        print(augmentation)
        # augmentation = augmentation + '()'
        # transforms.append(augmentation)
        transforms.append(augmentation)
        print(transforms)
        print(transforms[0])

    #     augmentation= getattr(albumentations, augmentation_command)    
    #     print(augmentation)
    #     print(transforms)
    # print(transforms)
    #     if type(augmentation_command) == list:
    #         continue
    #         # transforms.append(OneOf(getattr(albumentations,str(augmentation_command)for augmentation_ in augmentation_command)))
    #     else:
    #         augmentation_command= getattr(albumentations, augmentation_command)
    #         transforms.append(augmentation_command)
    #         print(transforms)
    #         print(augmentation_command)
    #         # while augmentation_command ==list:
    #         #     augmentation_command = getattr()
    #     print(augmentation_command)
    #     augmentation_command = getattr(albumentations, augmentation_command)

# def segmentation_aug(input_image, label, size, augmentation_list):
#     transforms = list()

#     for augmentation_command in augmentation_list:
#         if type(augmentation_command) == list:
#             augmentation_command = f"OneOf({augmentation_command},p=0.9)"
#         else:
#             augmentation_command = augmentation_command    
#     # print(size)
#     height, width = size
#     # print(height)
#     # print(width)
#     # transforms.append(
#     #     getattr(albu,augmentation_command)(
#     #     **self.config.augmentation_command
#     #     )
#     # )
#     # print(transforms)

#     if "vertical_flip" in augmentation_command:
#         transforms.append(VerticalFlip(p=0.5))
#     if "HorizontalFlip" in augmentation_command:
#         transforms.append(HorizontalFlip(p=0.5))
#     # if "blur" in augmentation_command:
#     #     transforms.append(Blur(blur_limit=3,p=0.5))
#     #newly added
#     if "motion_blur" in augmentation_command:
#         transforms.append(MotionBlur(blur_limit=3,p=0.5))
#     if "median_blur" in augmentation_command:
#         transforms.append(MedianBlur(blur_limit=3,p=0.5))
#     if "gaussian_blur" in augmentation_command:
#         transforms.append(GaussianBlur(blur_limit=3,p=0.5))
#     if "glass_blur" in augmentation_command:
#         transforms.append(GlassBlur(sigma=0.7, max_delta=4, iterations=2,p=0.5))
#     if "gauss_noise" in augmentation_command:
#         transforms.append(GaussNoise(p=0.5))
#     if "normalize" in augmentation_command:
#         transforms.append(Normalize(p=0.5))
#     if "grid_distortion" in augmentation_command:
#         transforms.append(GridDistortion
#             (num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, p=0.5)
#             )
#     if "RandomContrast" in augmentation_command:
#         transforms.append(RandomContrast(limit=0.2, p=0.5))   
#     if "multiplicative_noise" in augmentation_command:
#         transforms.append(MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=0.5))   
#     if "grid_dropout" in augmentation_command:
#         transforms.append(GridDropout(p=0.5))            
#     if "elastic_transform" in augmentation_command:
#         transforms.append(ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, p=0.5))
#     if "isonoise" in augmentation_command:
#         transforms.append(ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5))    
#     if "CLAHE" in augmentation_command:
#         transforms.append(CLAHE(p=1))
#     if "glass_blur" in augmentation_command:
#         transforms.append(GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.5))
    
#     #Module:happy-set
#     if "ShiftScaleRotate" in augmentation_command:
#         transforms.append(ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, border_mode=0, p=1))
    
#     if "PadIfNeeded" in augmentation_command:
#         transforms.append(PadIfNeeded(min_height=256, min_width=512, always_apply=True, border_mode=0))
#     if "RandomCrop" in augmentation_command:
#         transforms.append(RandomCrop(height=256, width=512, always_apply=True))
    
#     if "IAAAdditiveGaussianNoise" in augmentation_command:
#         transforms.append(IAAAdditiveGaussianNoise(p=0.2))
#     if "IAAPerspective" in augmentation_command:
#         transforms.append(IAAPerspective(p=0.5))

#     if "RandomBrightness" in augmentation_command:
#         transforms.append(RandomBrightness(p=1))
#     if "RandomGamma" in augmentation_command:
#         transforms.append(RandomGamma(p=1))
#     if "IAASharpen" in augmentation_command:
#         transforms.append(IAASharpen(p=1))
#     if "Blur" in augmentation_command:
#         transforms.append(Blur(blur_limit=3,p=1))
#     if "MotionBlur" in augmentation_command:
#         transforms.append(MotionBlur(blur_limit=3,p=1))

#     if "RandomContrast" in augmentation_command:
#         transforms.append(RandomContrast(limit=0.2, p=1))   
#     if "HueSaturationValue" in augmentation_command:
#         transforms.append(HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1))
#     if "Lambda" in augmentation_command:
#         transforms.append(Lambda(mask=round_clip_0_1))

    if len(transforms) > 0:
        # print(transforms[0])
        aug = Compose(transforms, p=1)
        # print(transforms)
        # print(type(transforms[0]))
        augmented = aug(image=input_image, mask=label)
        return augmented['image'], augmented["mask"]
    else:
        return input_image, label

