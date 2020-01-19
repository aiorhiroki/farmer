# coding: utf-8
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
import keras.backend as K

from .util import inst_layers


def create_encoding_layer(sequence, filter_count):
    new_sequence = LeakyReLU(0.2)(sequence)
    new_sequence = ZeroPadding2D((1, 1))(new_sequence)
    new_sequence = Conv2D(filter_count, 4, strides=2)(new_sequence)
    new_sequence = BatchNormalization()(new_sequence)
    return new_sequence


def create_decoding_layer(sequence, filter_count, add_drop_layer=True):
    new_sequence = Activation(activation='relu')(sequence)
    new_sequence = Conv2DTranspose(
        filter_count, 2, strides=2,
        kernel_initializer='he_uniform'
    )(new_sequence)
    new_sequence = BatchNormalization()(new_sequence)
    if add_drop_layer:
        new_sequence = Dropout(0.5)(new_sequence)
    return new_sequence


def Unet(input_shape, output_channel_count):

    # will be divided by 2(strides)
    feature_map_height, feature_map_width = input_shape[0], input_shape[1]
    input_height, input_width = (1, 1)  # input_shape may be resize
    min_size = min(input_shape[:2])
    num_layers = 0
    while min_size > 5:
        min_size //= 2
        num_layers += 1
        feature_map_height //= 2
        feature_map_width //= 2
        input_height *= 2
        input_width *= 2

    input_height *= feature_map_height
    input_width *= feature_map_width
    if input_shape != (input_height, input_width, input_shape[2]):
        input_shape = (input_height, input_width, input_shape[2])
        print('input_shape has changed to {}'.format(input_shape))

    with tf.device("/cpu:0"):
        x_in = Input(shape=input_shape, name='input')

        #################
        # encoder layer #
        #################
        first_layers = [
            ZeroPadding2D((1, 1)),
            Conv2D(filters=8,
                   kernel_size=4,
                   strides=2,
                   input_shape=input_shape)
        ]

        first_tensor = inst_layers(first_layers, x_in)
        encoder = first_tensor
        encoders = [encoder]
        for layer_id in range(1, num_layers):
            encoder = create_encoding_layer(
                sequence=encoder,
                filter_count=8 * 2**layer_id if layer_id <= 3 else 8 * 2**3
            )
            encoders.append(encoder)

        #################
        # decoder layer #
        #################
        decoder = encoder
        for layer_id in range(1, num_layers):
            decoder = create_decoding_layer(
                sequence=decoder,
                filter_count=8 * 2**(
                    num_layers-layer_id - 1
                ) if (
                    num_layers-layer_id
                ) <= 3 else 8 * 2**3,
                add_drop_layer=True if layer_id <= 3 else False
            )
            decoder = concatenate(
                [decoder, encoders[num_layers-layer_id-1]], axis=-1)

        final_layers = [
            Activation(activation='relu'),
            Conv2DTranspose(output_channel_count, (2, 2), strides=2),
            Activation(activation='softmax')
        ]

        segmentation = inst_layers(final_layers, decoder)
        model = Model(x_in, segmentation)

    return model, input_shape


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
