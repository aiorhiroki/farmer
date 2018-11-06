import numpy as np
from keras.datasets import mnist
from farmer import classifier

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

# fit farmer classification
classifier.fit_from_array(x_train, y_train)
