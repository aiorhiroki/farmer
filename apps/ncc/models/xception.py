import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.xception import Xception
from keras.models import Model

import pretrainedmodels


def xception(nb_classes, height=299, width=299, framework="tensorflow"):
    if framework == "tensorflow":
        with tf.device("/cpu:0"):
            base_model = Xception(
                input_shape=(height, width, 3),
                weights='imagenet',
                include_top=False
            )
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            predictions = Dense(nb_classes, activation='softmax')(x)
            model = Model(base_model.input, predictions)

    elif framework == "pytorch":
        model = pretrainedmodels.__dict__['xception'](
            num_classes=nb_classes, pretrained='imagenet'
        )

    return model
