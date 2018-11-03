from ncc.models import Model3D, Model2D
from ncc.preprocessing import preprocess_input
from ncc.validations import save_show_results, evaluate

from keras.callbacks import EarlyStopping

import numpy as np


class Classifier(object):
    def __init__(self, optimizer='sgd', loss='categorical_crossentropy', metrics='acc', epochs=100, batch_size=32):
        # parameters
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = list(metrics)
        self.epochs = epochs
        self.batch_size = batch_size

    def fit_from_array(self, x_train, y_train, x_test=None, y_test=None, class_names=None):
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
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()
        history = model.fit(x_train, y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            callbacks=[EarlyStopping(patience=5)],
                            validation_data=(x_test, y_test)
                            )

        # save and eval model
        save_show_results(history, model)
        evaluate(model, x_test, y_train, class_names)
