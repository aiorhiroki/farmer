import tensorflow as tf
import tensorflow.keras.applications.efficientnet as en

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


def EfficientNet(model_name, nb_classes, height, width):
    model_name_converted = model_name.replace('efficientnetb', 'EfficientNetB')

    with tf.device("/cpu:0"):
        base_model = getattr(en, model_name_converted)(
            include_top=False,
            input_shape=(height, width, 3),
            weights='imagenet'
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(base_model.input, predictions)

    return model