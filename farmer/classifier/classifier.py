from ncc.models import Model3D, Model2D
from ncc.history import save_history
from ncc.preprocessing import preprocess_input
from ncc.metrics import show_matrix

from keras.callbacks import EarlyStopping

import numpy as np


def fit_from_array(x_train, y_train, x_test=None, y_test=None, class_names=None):
    # parameters
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    epochs = 30
    batch_size = 32

    # prepare data
    x_train, y_train = preprocess_input(x_train, y_train)
    print(x_train.shape, y_train.shape)
    if x_test is not None and y_test is not None:
        x_test, y_test = preprocess_input(x_test, y_test)

    # data profile
    if class_names is None:
        class_names = list(np.arange(y_train.shape[1]))

    num_classes = len(class_names)
    input_shape = x_train.shape[1:]

    # build model
    if len(input_shape) == 3:  # (height, width, channel)
        model = Model2D(input_shape=input_shape, num_classes=num_classes)
    elif len(input_shape) == 4:  # (depth, height, width, channel)
        model = Model3D(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError('input shape is invalid.')

    # compile and fit
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[EarlyStopping(patience=5)],
                        validation_data=(x_test, y_test)
                        )

    # save results
    save_history(history)
    model.save_weights('classification_model.h5')

    # validation
    if x_test is not None and y_test is not None:
        y_prediction = model.predict(x_test)
        y_prediction = np.argmax(y_prediction, axis=1)  # from one hot to class index
        y_test = np.argmax(y_test, axis=1)  # from one hot to class index
        show_matrix(y_test, y_prediction, class_names)
