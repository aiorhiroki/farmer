import tensorflow as tf

from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def efficientnetb0(nb_classes, height=224, width=224):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB0(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb1(nb_classes, height=240, width=240):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB1(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb2(nb_classes, height=260, width=260):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB2(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb3(nb_classes, height=300, width=300):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB3(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb4(nb_classes, height=380, width=380):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB4(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb5(nb_classes, height=456, width=456):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB5(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb6(nb_classes, height=528, width=528):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB6(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


def efficientnetb7(nb_classes, height=600, width=600):
    with tf.device("/cpu:0"):
        base_model = EfficientNetB7(
            input_shape=(height, width, 3),
            weights='imagenet',
            include_top=False
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model


if __name__ == '__main__':
    model = efficientnetb7(10)
    print(model.summary())