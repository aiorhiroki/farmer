from keras.datasets import cifar10
from farmer import classifier

# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# fit farmer classification
classifier.fit_from_array(x_train, y_train, x_test, y_test)
