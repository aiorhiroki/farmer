# from keras.preprocessing import image
import numpy as np
from skimage import img_as_ubyte

# def segmentation_aug(input_image, label, size, augmentation_list):
#     data_gen_args = dict()
#     if "zoom" in augmentation_list:
#         data_gen_args["zoom_range"] = [0.5, 1.0]
#     if "vertical_flip" in augmentation_list:
#         data_gen_args["vertical_flip"] = True
#     if "horizontal_flip" in augmentation_list:
#         data_gen_args["horizontal_flip"] = True

#     seed = np.random.randint(100)
#     label = np.expand_dims(label, axis=0)
#     label = np.expand_dims(label, axis=-1)
#     image_datagen = image.ImageDataGenerator(**data_gen_args)
#     mask_datagen = image.ImageDataGenerator(**data_gen_args)

#     image_datagen.fit(input_image[np.newaxis], augment=True, seed=seed)
#     mask_datagen.fit(label, augment=True, seed=seed)

#     image_gen = image_datagen.flow(input_image[np.newaxis], batch_size=1, seed=seed)
#     mask_gen = mask_datagen.flow(label, batch_size=1, seed=seed)

#     gen = zip(image_gen, mask_gen)
#     img_batches, mask_batches = next(gen)
#     input_image = img_batches[0]
#     label = mask_batches[0][..., 0]

#     return input_image, label

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Blur,
    MotionBlur,
    # MedianBlur,
    GaussianBlur,
    GlassBlur,
    # CLAHE,
    # HueSaturationValue,
    GaussNoise,
    Normalize,
    GridDistortion,
    RandomContrast,
    MultiplicativeNoise,
    GridDropout,
    ElasticTransform,
    ISONoise,
)


def segmentation_aug(input_image, label, size, augmentation_list):
    transforms = list()
    
    label = img_as_ubyte(label)
    if "vertical_flip" in augmentation_list:
        transforms.append(VerticalFlip(p=0.5))
    if "horizontal_flip" in augmentation_list:
        transforms.append(HorizontalFlip(p=0.5))
    if "blur" in augmentation_list:
        transforms.append(Blur(blur_limit=3,p=0.5))
    #newly added
    if "motion_blur" in augmentation_list:
        transforms.append(MotionBlur(blur_limit=3,p=0.5))
    if "median_blur" in augmentation_list:
        transforms.append(MedianBlur(blur_limit=3,p=0.5))
    if "gaussian_blur" in augmentation_list:
        transforms.append(GaussianBlur(blur_limit=3,p=0.5))
    if "glass_blur" in augmentation_list:
        transforms.append(GlassBlur(sigma=0.7, max_delta=4, iterations=2,p=0.5))
    if "gauss_noise" in augmentation_list:
        transforms.append(GaussNoise(p=0.5))
    if "normalize" in augmentation_list:
        transforms.append(Normalize(p=0.5))
    if "grid_distortion" in augmentation_list:
        transforms.append(GridDistortion
            (num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, p=0.5)
            )
    if "random_contrast" in augmentation_list:
        transforms.append(RandomContrast(limit=0.2, p=0.5))   
    if "multiplicative_noise" in augmentation_list:
        transforms.append(MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=False, p=0.5))   
    if "grid_dropout" in augmentation_list:
        transforms.append(GridDropout(p=0.5))            
    if "elastic_transform" in augmentation_list:
        transforms.append(ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, p=0.5))
    if "isonoise" in augmentation_list:
        transforms.append(ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5))    
   # if "clahe" in augmentation_list:
    #     transforms.append(CLAHE(p=0.5))
    # if "hsv" in augmentation_list:
    #     transforms.append(HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,p=0.5))

    if len(transforms) > 0:
        print('augmentation!!!')
        aug = Compose(transforms, p=1)
        augmented = aug(image=input_image, mask=label)
        return augmented['image'], augmented["mask"]
    else:
        return input_image, label