import numpy as np
from ..metrics import show_matrix
from ..history import save_history, show_history


def evaluate(model, x_test, y_test, class_names):
    if x_test is None or y_test is None:
        return
    y_prediction = model.predict(x_test)
    # from one hot to class index
    y_prediction = np.argmax(y_prediction, axis=1)
    y_test = np.argmax(y_test, axis=1)  # from one hot to class index
    show_matrix(y_test, y_prediction, class_names)


def save_show_results(history, model):
    save_history(history, 'history.csv')
    model.save_weights('classification_model.h5')
    show_history('acc', False, 'history.csv')
