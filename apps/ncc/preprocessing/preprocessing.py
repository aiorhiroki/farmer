import numpy as np


def preprocess_input(x_array, y_array=None, one_hot=True):

    if len(x_array.shape) == 3:  # (height, width, channel)
        x_array = np.expand_dims(x_array, axis=0)  # (1, height, width, channel)

    x_array = x_array.astype('float32')
    x_array /= 255

    if y_array is None:
        return x_array

    y_array = y_array.ravel()  # (num_samples, 1) => (num_samples, )

    if one_hot and len(y_array.shape) == 1:  # (num_samples, )
        num_classes = np.max(y_array) + 1
        y_array = np.eye(num_classes)[y_array]  # one hot: (num_samples, num_classes)

    y_array = y_array.astype('float32')

    return x_array, y_array
