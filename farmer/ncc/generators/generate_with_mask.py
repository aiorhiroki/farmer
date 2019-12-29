from keras.preprocessing.image import ImageDataGenerator


def generate_with_mask(x_in, mask, batch_size):
    data_gen_args = dict(
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.1,
        fill_mode='reflect',
        horizontal_flip=True,
        vertical_flip=True
                         )
    # we create two instances with the same arguments
    image_data_gen = ImageDataGenerator(**data_gen_args)
    mask_data_gen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_data_gen.fit(x_in, augment=True, seed=seed)
    mask_data_gen.fit(mask, augment=True, seed=seed)

    image_generator = image_data_gen.flow(x_in, batch_size=batch_size)
    mask_generator = mask_data_gen.flow(mask, batch_size=batch_size)

    generator = zip(image_generator, mask_generator)

    return generator
