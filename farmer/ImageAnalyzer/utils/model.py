from keras.models import Model
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D


def xception(nb_classes, img_width=299, img_height=299):
    base_model = Xception(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(base_model.input, predictions)

    return model
